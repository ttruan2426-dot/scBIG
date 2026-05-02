#!/usr/bin/env python
"""Trainable GCAE implementation used by scBIG.

The public entrypoint is ``scbig.models.gcae.GCAEWrapper``.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Sequence, Optional, Dict, Tuple, Any, List
import optax
from functools import partial
import gzip
import pickle
from collections.abc import Mapping


# ============================================================================
# Component 1: Biological Positional Encoding
# ============================================================================

def create_sinusoidal_pe(seq_len: int, dim: int) -> np.ndarray:
    """
    Create fixed sinusoidal positional encoding (like original Transformer).
    
    When genes are reordered by biological similarity (scGPT clusters),
    sinusoidal PE now encodes "biological distance" - nearby positions
    mean biologically similar genes.
    
    Args:
        seq_len: Sequence length (number of genes or chunks)
        dim: Embedding dimension
    
    Returns:
        pe: (seq_len, dim) positional encoding matrix
    """
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(dim)[None, :]
    
    # Scale factor for different frequencies
    angles = positions / (10000 ** (2 * (dims // 2) / dim))
    
    pe = np.zeros((seq_len, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe


class BiologicalPositionalEncoding(nn.Module):
    """
    Fixed positional encoding from biological priors.
    
    Three modes:
    1. sinusoidal: Fixed sinusoidal PE (requires gene reordering to be meaningful)
    2. scgpt: Fixed PE structure from scGPT embeddings (trainable projection, 512-dim)
    3. genecompass: Fixed PE structure from GeneCompass embeddings (trainable projection, 768-dim)
    
    The key insight: the STRUCTURE is fixed (from biology), 
    only the PROJECTION to model dimension is learned.
    """
    embed_dim: int
    pe_type: str = 'scgpt'  # 'sinusoidal', 'scgpt', or 'genecompass'
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, scgpt_pe: Optional[jnp.ndarray] = None, 
                 training: bool = False):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            scgpt_pe: Pre-computed biological PE (seq_len, embed_dim) - works for both scGPT and GeneCompass
            training: Whether in training mode
        
        Returns:
            x + positional_encoding
        """
        batch_size, seq_len, _ = x.shape
        
        if self.pe_type == 'sinusoidal':
            # Fixed sinusoidal PE - meaningful when genes are reordered
            pe = create_sinusoidal_pe(seq_len, self.embed_dim)
            pe = jnp.array(pe, dtype=x.dtype)
            # No learnable parameters for sinusoidal PE structure
            x = x + pe[None, :, :]
            
        elif self.pe_type in ['scgpt', 'genecompass']:
            if scgpt_pe is None:
                # Fallback to sinusoidal if embeddings not provided
                pe = create_sinusoidal_pe(seq_len, self.embed_dim)
                pe = jnp.array(pe, dtype=x.dtype)
                x = x + pe[None, :, :]
            else:
                # Fixed structure from biological embeddings, learned projection
                # scgpt_pe: (seq_len, bio_embed_dim) - 512 for scGPT, 768 for GeneCompass
                # Project to model dimension - this is the ONLY learned part
                projection_name = f'{self.pe_type}_projection'
                bio_pe = nn.Dense(self.embed_dim, use_bias=False, 
                                  name=projection_name)(scgpt_pe)
                # Normalize the PE
                bio_pe = nn.LayerNorm()(bio_pe)
                x = x + bio_pe[None, :, :]
        
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


# ============================================================================
# Component 2: PPI Sparse Attention
# ============================================================================

def build_ppi_attention_mask(gene_names: List[str], ppi_edges: List[Tuple], 
                             k_hop: int = 2, 
                             include_self: bool = True) -> np.ndarray:
    """
    Build sparse attention mask from PPI network.
    
    Gene i can only attend to gene j if:
    - They directly interact in PPI (1-hop)
    - They share a common interactor (2-hop)
    - Include self-attention
    
    Args:
        gene_names: List of gene names in order
        ppi_edges: List of (gene1, gene2, score) tuples
        k_hop: Number of hops to consider (1 or 2)
        include_self: Whether to include self-attention
    
    Returns:
        mask: (n_genes, n_genes) binary mask, 1 = can attend, 0 = cannot
    """
    n_genes = len(gene_names)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # Build adjacency matrix (1-hop)
    adj_1hop = np.zeros((n_genes, n_genes), dtype=np.float32)
    
    for g1, g2, score in ppi_edges:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            adj_1hop[i, j] = 1
            adj_1hop[j, i] = 1  # Symmetric
    
    # Start with 1-hop neighbors
    mask = adj_1hop.copy()
    
    # Add 2-hop neighbors if requested
    if k_hop >= 2:
        # A @ A gives 2-hop connectivity
        adj_2hop = (adj_1hop @ adj_1hop) > 0
        mask = np.logical_or(mask > 0, adj_2hop).astype(np.float32)
    
    # Add self-attention
    if include_self:
        np.fill_diagonal(mask, 1)
    
    return mask


def build_chunk_ppi_mask(gene_mask: np.ndarray, chunk_size: int,
                         threshold: float = 0.1) -> np.ndarray:
    """
    Aggregate gene-level PPI mask to chunk level.
    
    Chunk i can attend to chunk j if at least `threshold` fraction
    of gene pairs between them have PPI interactions.
    
    Args:
        gene_mask: (n_genes, n_genes) gene-level PPI mask
        chunk_size: Size of each chunk
        threshold: Fraction of gene pairs needed for chunk attention
    
    Returns:
        chunk_mask: (n_chunks, n_chunks) chunk-level mask
    """
    n_genes = gene_mask.shape[0]
    n_chunks = (n_genes + chunk_size - 1) // chunk_size
    
    # Pad if necessary
    if n_genes % chunk_size != 0:
        padding = chunk_size - (n_genes % chunk_size)
        gene_mask = np.pad(gene_mask, ((0, padding), (0, padding)))
    
    chunk_mask = np.zeros((n_chunks, n_chunks), dtype=np.float32)
    
    for i in range(n_chunks):
        for j in range(n_chunks):
            block = gene_mask[
                i * chunk_size:(i + 1) * chunk_size,
                j * chunk_size:(j + 1) * chunk_size
            ]
            # Fraction of gene pairs with interactions
            interaction_density = block.mean()
            chunk_mask[i, j] = 1.0 if interaction_density >= threshold else 0.0
    
    # Always allow self-attention for chunks
    np.fill_diagonal(chunk_mask, 1)
    
    return chunk_mask


class PPISparseAttention(nn.Module):
    """
    Multi-head attention with HARD PPI-based sparsity mask.
    
    Unlike additive bias (which the model can learn to ignore),
    this uses multiplicative masking - genes/chunks CANNOT attend
    to non-interacting partners.
    
    This is the key inductive bias: only biologically relevant
    gene-gene relationships can be learned.
    """
    num_heads: int = 8
    dropout_rate: float = 0.1
    use_hard_mask: bool = True  # Hard mask vs soft bias
    
    @nn.compact
    def __call__(self, x, ppi_mask: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            ppi_mask: Attention mask (seq_len, seq_len), 1=attend, 0=block
            training: Training mode
        
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        
        # QKV projection
        qkv = nn.Dense(3 * embed_dim, use_bias=False)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale
        
        # Apply PPI mask - THIS IS THE KEY DIFFERENCE
        if ppi_mask is not None and self.use_hard_mask:
            # Hard mask: set non-interacting pairs to -inf
            # ppi_mask: (seq_len, seq_len)
            mask_value = -1e9
            ppi_mask_expanded = ppi_mask[None, None, :, :]  # (1, 1, seq, seq)
            attn_weights = jnp.where(ppi_mask_expanded > 0, attn_weights, mask_value)
        
        # Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Dropout
        if training and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(dropout_mask, attn_weights / keep_prob, 0)
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # Reshape and project output
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        attn_output = nn.Dense(embed_dim)(attn_output)
        
        return attn_output


# ============================================================================
# Component 3: Local Window Attention
# ============================================================================

def build_local_attention_mask(seq_len: int, window_size: int) -> np.ndarray:
    """
    Build local attention mask for sliding window attention.
    
    When genes are reordered by biological similarity, local attention
    means attending to biologically similar genes.
    
    Args:
        seq_len: Sequence length
        window_size: Size of attention window (one side)
    
    Returns:
        mask: (seq_len, seq_len) binary mask
    """
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1.0
    
    return mask


class LocalWindowAttention(nn.Module):
    """
    Local sliding window attention (like Longformer).
    
    When genes are reordered by scGPT cluster similarity,
    local attention = attending to biologically similar genes.
    
    This provides the inductive bias that nearby genes (in the
    reordered sequence) should have correlated expression patterns.
    """
    num_heads: int = 8
    window_size: int = 64  # Attend to genes within this window
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, additional_mask: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            additional_mask: Optional additional mask to combine with local
            training: Training mode
        
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        
        # Build local attention mask
        local_mask = build_local_attention_mask(seq_len, self.window_size)
        local_mask = jnp.array(local_mask)
        
        # Combine with additional mask if provided
        if additional_mask is not None:
            # Both must be satisfied (AND operation)
            combined_mask = local_mask * additional_mask
        else:
            combined_mask = local_mask
        
        # QKV projection
        qkv = nn.Dense(3 * embed_dim, use_bias=False)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale
        
        # Apply local mask
        mask_value = -1e9
        mask_expanded = combined_mask[None, None, :, :]
        attn_weights = jnp.where(mask_expanded > 0, attn_weights, mask_value)
        
        # Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Dropout
        if training and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(dropout_mask, attn_weights / keep_prob, 0)
        
        # Apply attention
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # Reshape and project
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        attn_output = nn.Dense(embed_dim)(attn_output)
        
        return attn_output


# ============================================================================
# Component 3b: Perceiver-style Induced Attention
# ============================================================================

class ModuleInducedAttention(nn.Module):
    """
    Perceiver-style attention with learnable inducing points.
    
    The inducing points serve as an information bottleneck, learning to
    represent "gene functional modules". This provides:
    1. Efficient O(C*M + M^2) complexity instead of O(C^2)
    2. Interpretable intermediate representations
    3. Global information flow through the bottleneck
    
    Architecture:
        Step 1: Inducing points (queries) attend to gene chunks (keys/values)
        Step 2: Self-attention among inducing points
        Step 3: Gene chunks (queries) attend to inducing points (keys/values)
    
    Similar to Set Transformer's ISAB (Induced Set Attention Block).
    
    When return_attention=True, returns attention weights for biological discovery:
    - attn_chunks_to_inducing: (B, H, M, C) - how inducing points aggregate from chunks
    - attn_inducing_self: (B, H, M, M) - inter-module communication
    - attn_inducing_to_chunks: (B, H, C, M) - how chunks query modules
    """
    num_inducing: int = 16  # Number of learnable "gene modules"
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, attention_mask: Optional[jnp.ndarray] = None,
                 training: bool = False, return_attention: bool = False):
        """
        Args:
            x: Input (batch, n_chunks, embed_dim)
            attention_mask: Optional mask (unused for inducing attention)
            training: Training mode
            return_attention: If True, return attention weights for analysis
        
        Returns:
            output: (batch, n_chunks, embed_dim)
            attention_dict: (optional) Dict of attention weights if return_attention=True
                - 'chunks_to_inducing': (B, H, M, C) - Step 1 attention
                - 'inducing_self': (B, H, M, M) - Step 2 attention
                - 'inducing_to_chunks': (B, H, C, M) - Step 3 attention
                - 'inducing_points': (M, embed_dim) - Learned inducing point embeddings
        """
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        
        # Learnable inducing points (shared across batch)
        # These learn to represent "gene functional modules"
        inducing_points = self.param(
            'inducing_points',
            nn.initializers.normal(stddev=0.02),
            (self.num_inducing, embed_dim)
        )
        
        # Expand for batch: (M, d) -> (B, M, d)
        I = jnp.broadcast_to(inducing_points[None, :, :], 
                             (batch_size, self.num_inducing, embed_dim))
        
        # ================================================================
        # Step 1: Inducing points aggregate gene information
        # Cross-attention: Q=I, K=x, V=x
        # ================================================================
        # I queries, x provides keys and values
        q1 = nn.Dense(embed_dim, use_bias=False, name='q1')(I)  # (B, M, d)
        k1 = nn.Dense(embed_dim, use_bias=False, name='k1')(x)  # (B, C, d)
        v1 = nn.Dense(embed_dim, use_bias=False, name='v1')(x)  # (B, C, d)
        
        # Reshape for multi-head attention
        q1 = q1.reshape(batch_size, self.num_inducing, self.num_heads, head_dim)
        k1 = k1.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v1 = v1.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        q1 = jnp.transpose(q1, (0, 2, 1, 3))  # (B, H, M, d)
        k1 = jnp.transpose(k1, (0, 2, 1, 3))  # (B, H, C, d)
        v1 = jnp.transpose(v1, (0, 2, 1, 3))  # (B, H, C, d)
        
        # Attention: (B, H, M, d) @ (B, H, d, C) -> (B, H, M, C)
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn1 = jnp.einsum('bhmd,bhcd->bhmc', q1, k1) / scale
        attn1_weights = jax.nn.softmax(attn1, axis=-1)  # Save for return
        attn1 = attn1_weights
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            keep_prob = 1.0 - self.dropout_rate
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn1.shape)
            attn1 = jnp.where(dropout_mask, attn1 / keep_prob, 0)
        
        # (B, H, M, C) @ (B, H, C, d) -> (B, H, M, d)
        I_updated = jnp.einsum('bhmc,bhcd->bhmd', attn1, v1)
        I_updated = jnp.transpose(I_updated, (0, 2, 1, 3))  # (B, M, H, d)
        I_updated = I_updated.reshape(batch_size, self.num_inducing, embed_dim)
        I_updated = nn.Dense(embed_dim, name='proj1')(I_updated)
        I_updated = nn.LayerNorm(name='ln1')(I + I_updated)  # Residual
        
        # ================================================================
        # Step 2: Self-attention among inducing points
        # ================================================================
        q2 = nn.Dense(embed_dim, use_bias=False, name='q2')(I_updated)
        k2 = nn.Dense(embed_dim, use_bias=False, name='k2')(I_updated)
        v2 = nn.Dense(embed_dim, use_bias=False, name='v2')(I_updated)
        
        q2 = q2.reshape(batch_size, self.num_inducing, self.num_heads, head_dim)
        k2 = k2.reshape(batch_size, self.num_inducing, self.num_heads, head_dim)
        v2 = v2.reshape(batch_size, self.num_inducing, self.num_heads, head_dim)
        
        q2 = jnp.transpose(q2, (0, 2, 1, 3))
        k2 = jnp.transpose(k2, (0, 2, 1, 3))
        v2 = jnp.transpose(v2, (0, 2, 1, 3))
        
        attn2 = jnp.einsum('bhmd,bhnd->bhmn', q2, k2) / scale
        attn2_weights = jax.nn.softmax(attn2, axis=-1)  # Save for return
        attn2 = attn2_weights
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn2.shape)
            attn2 = jnp.where(dropout_mask, attn2 / keep_prob, 0)
        
        I_self = jnp.einsum('bhmn,bhnd->bhmd', attn2, v2)
        I_self = jnp.transpose(I_self, (0, 2, 1, 3))
        I_self = I_self.reshape(batch_size, self.num_inducing, embed_dim)
        I_self = nn.Dense(embed_dim, name='proj2')(I_self)
        I_final = nn.LayerNorm(name='ln2')(I_updated + I_self)
        
        # ================================================================
        # Step 3: Gene chunks query inducing points
        # Cross-attention: Q=x, K=I_final, V=I_final
        # ================================================================
        q3 = nn.Dense(embed_dim, use_bias=False, name='q3')(x)
        k3 = nn.Dense(embed_dim, use_bias=False, name='k3')(I_final)
        v3 = nn.Dense(embed_dim, use_bias=False, name='v3')(I_final)
        
        q3 = q3.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k3 = k3.reshape(batch_size, self.num_inducing, self.num_heads, head_dim)
        v3 = v3.reshape(batch_size, self.num_inducing, self.num_heads, head_dim)
        
        q3 = jnp.transpose(q3, (0, 2, 1, 3))  # (B, H, C, d)
        k3 = jnp.transpose(k3, (0, 2, 1, 3))  # (B, H, M, d)
        v3 = jnp.transpose(v3, (0, 2, 1, 3))  # (B, H, M, d)
        
        # (B, H, C, d) @ (B, H, d, M) -> (B, H, C, M)
        attn3 = jnp.einsum('bhcd,bhmd->bhcm', q3, k3) / scale
        attn3_weights = jax.nn.softmax(attn3, axis=-1)  # Save for return
        attn3 = attn3_weights
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn3.shape)
            attn3 = jnp.where(dropout_mask, attn3 / keep_prob, 0)
        
        # (B, H, C, M) @ (B, H, M, d) -> (B, H, C, d)
        x_updated = jnp.einsum('bhcm,bhmd->bhcd', attn3, v3)
        x_updated = jnp.transpose(x_updated, (0, 2, 1, 3))  # (B, C, H, d)
        x_updated = x_updated.reshape(batch_size, seq_len, embed_dim)
        x_updated = nn.Dense(embed_dim, name='proj3')(x_updated)
        
        # Final output with residual
        output = nn.LayerNorm(name='ln3')(x + x_updated)
        
        if return_attention:
            attention_dict = {
                'chunks_to_inducing': attn1_weights,  # (B, H, M, C) - How modules aggregate from chunks
                'inducing_self': attn2_weights,       # (B, H, M, M) - Inter-module communication
                'inducing_to_chunks': attn3_weights,  # (B, H, C, M) - How chunks query modules
                'inducing_points': inducing_points,   # (M, embed_dim) - Learned module embeddings
                'inducing_final': I_final,            # (B, M, embed_dim) - Final module representations
            }
            return output, attention_dict
        
        return output


class AttentionPooling(nn.Module):
    """Single-query attention pooling over module states."""
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        query = self.param(
            'query',
            nn.initializers.normal(stddev=0.02),
            (self.embed_dim,),
        )
        scale = jnp.sqrt(jnp.asarray(self.embed_dim, dtype=x.dtype))
        scores = jnp.einsum('bkd,d->bk', x, query) / scale
        weights = jax.nn.softmax(scores, axis=1)
        return jnp.einsum('bk,bkd->bd', weights, x)


class SetTransformerPooling(nn.Module):
    """PMA-style set pooling with a single learnable seed."""
    embed_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        batch_size = x.shape[0]
        seed = self.param(
            'seed',
            nn.initializers.normal(stddev=0.02),
            (1, self.embed_dim),
        )
        seed = jnp.broadcast_to(seed[None, :, :], (batch_size, 1, self.embed_dim))

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name='seed_attention',
        )(seed, x, deterministic=not training)
        pooled = nn.LayerNorm(name='ln1')(seed + attn_out)

        mlp_out = nn.Dense(self.embed_dim * 2, name='mlp_fc1')(pooled)
        mlp_out = jax.nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.embed_dim, name='mlp_fc2')(mlp_out)
        mlp_out = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not training,
            name='mlp_dropout',
        )(mlp_out)
        pooled = nn.LayerNorm(name='ln2')(pooled + mlp_out)
        return pooled[:, 0, :]


# ============================================================================
# Component 3c: Pathway-Guided Cross Attention
# ============================================================================

class PathwayCrossAttention(nn.Module):
    """
    Pathway-guided cross attention using biological pathway structure.
    
    Uses pathway representations as explicit queries to aggregate gene
    information, then genes query pathway representations for global context.
    
    This provides strong biological inductive bias by explicitly using
    known pathway memberships to guide attention patterns.
    
    Architecture:
        Step 1: Pathway embeddings query gene chunks
        Step 2: Self-attention among pathway representations
        Step 3: Gene chunks query pathway representations
    """
    num_heads: int = 8
    n_pathway_queries: int = 128  # Use top-k pathways or learnable subset
    dropout_rate: float = 0.1
    use_learnable_pathway_queries: bool = True
    
    @nn.compact
    def __call__(self, x, 
                 pathway_init_embed: Optional[jnp.ndarray] = None,
                 attention_mask: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Input gene chunks (batch, n_chunks, embed_dim)
            pathway_init_embed: Initial pathway embeddings (n_pathways, embed_dim)
                               If None, uses learnable embeddings
            attention_mask: Optional attention mask
            training: Training mode
        
        Returns:
            output: (batch, n_chunks, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        n_pathways = self.n_pathway_queries
        
        # Initialize pathway queries
        if self.use_learnable_pathway_queries or pathway_init_embed is None:
            # Learnable pathway embeddings
            pathway_embed = self.param(
                'pathway_queries',
                nn.initializers.normal(stddev=0.02),
                (n_pathways, embed_dim)
            )
        else:
            # Use provided pathway embeddings (e.g., from pathway aggregation)
            # Take top-k pathways if more than n_pathway_queries
            if pathway_init_embed.shape[0] > n_pathways:
                pathway_embed = pathway_init_embed[:n_pathways]
            else:
                pathway_embed = pathway_init_embed
                n_pathways = pathway_embed.shape[0]
        
        # Expand for batch: (K, d) -> (B, K, d)
        P = jnp.broadcast_to(pathway_embed[None, :, :],
                             (batch_size, n_pathways, embed_dim))
        
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        keep_prob = 1.0 - self.dropout_rate
        
        # ================================================================
        # Step 1: Pathways query genes
        # ================================================================
        q1 = nn.Dense(embed_dim, use_bias=False, name='pw_q1')(P)
        k1 = nn.Dense(embed_dim, use_bias=False, name='pw_k1')(x)
        v1 = nn.Dense(embed_dim, use_bias=False, name='pw_v1')(x)
        
        q1 = q1.reshape(batch_size, n_pathways, self.num_heads, head_dim)
        k1 = k1.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v1 = v1.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        q1 = jnp.transpose(q1, (0, 2, 1, 3))  # (B, H, K, d)
        k1 = jnp.transpose(k1, (0, 2, 1, 3))  # (B, H, C, d)
        v1 = jnp.transpose(v1, (0, 2, 1, 3))
        
        attn1 = jnp.einsum('bhkd,bhcd->bhkc', q1, k1) / scale
        attn1 = jax.nn.softmax(attn1, axis=-1)
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn1.shape)
            attn1 = jnp.where(dropout_mask, attn1 / keep_prob, 0)
        
        P_updated = jnp.einsum('bhkc,bhcd->bhkd', attn1, v1)
        P_updated = jnp.transpose(P_updated, (0, 2, 1, 3))
        P_updated = P_updated.reshape(batch_size, n_pathways, embed_dim)
        P_updated = nn.Dense(embed_dim, name='pw_proj1')(P_updated)
        P_updated = nn.LayerNorm(name='pw_ln1')(P + P_updated)
        
        # ================================================================
        # Step 2: Self-attention among pathways
        # ================================================================
        q2 = nn.Dense(embed_dim, use_bias=False, name='pw_q2')(P_updated)
        k2 = nn.Dense(embed_dim, use_bias=False, name='pw_k2')(P_updated)
        v2 = nn.Dense(embed_dim, use_bias=False, name='pw_v2')(P_updated)
        
        q2 = q2.reshape(batch_size, n_pathways, self.num_heads, head_dim)
        k2 = k2.reshape(batch_size, n_pathways, self.num_heads, head_dim)
        v2 = v2.reshape(batch_size, n_pathways, self.num_heads, head_dim)
        
        q2 = jnp.transpose(q2, (0, 2, 1, 3))
        k2 = jnp.transpose(k2, (0, 2, 1, 3))
        v2 = jnp.transpose(v2, (0, 2, 1, 3))
        
        attn2 = jnp.einsum('bhkd,bhld->bhkl', q2, k2) / scale
        attn2 = jax.nn.softmax(attn2, axis=-1)
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn2.shape)
            attn2 = jnp.where(dropout_mask, attn2 / keep_prob, 0)
        
        P_self = jnp.einsum('bhkl,bhld->bhkd', attn2, v2)
        P_self = jnp.transpose(P_self, (0, 2, 1, 3))
        P_self = P_self.reshape(batch_size, n_pathways, embed_dim)
        P_self = nn.Dense(embed_dim, name='pw_proj2')(P_self)
        P_final = nn.LayerNorm(name='pw_ln2')(P_updated + P_self)
        
        # ================================================================
        # Step 3: Genes query pathways
        # ================================================================
        q3 = nn.Dense(embed_dim, use_bias=False, name='pw_q3')(x)
        k3 = nn.Dense(embed_dim, use_bias=False, name='pw_k3')(P_final)
        v3 = nn.Dense(embed_dim, use_bias=False, name='pw_v3')(P_final)
        
        q3 = q3.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k3 = k3.reshape(batch_size, n_pathways, self.num_heads, head_dim)
        v3 = v3.reshape(batch_size, n_pathways, self.num_heads, head_dim)
        
        q3 = jnp.transpose(q3, (0, 2, 1, 3))
        k3 = jnp.transpose(k3, (0, 2, 1, 3))
        v3 = jnp.transpose(v3, (0, 2, 1, 3))
        
        attn3 = jnp.einsum('bhcd,bhkd->bhck', q3, k3) / scale
        attn3 = jax.nn.softmax(attn3, axis=-1)
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn3.shape)
            attn3 = jnp.where(dropout_mask, attn3 / keep_prob, 0)
        
        x_updated = jnp.einsum('bhck,bhkd->bhcd', attn3, v3)
        x_updated = jnp.transpose(x_updated, (0, 2, 1, 3))
        x_updated = x_updated.reshape(batch_size, seq_len, embed_dim)
        x_updated = nn.Dense(embed_dim, name='pw_proj3')(x_updated)
        
        output = nn.LayerNorm(name='pw_ln3')(x + x_updated)
        
        return output


# ============================================================================
# Component 4: Hierarchical Pathway Pooling
# ============================================================================

class PathwayPooling(nn.Module):
    """
    Hierarchical pooling based on pathway membership.
    
    Level 1: Aggregate genes within each pathway
    Level 2: Cross-pathway attention
    Level 3: Global pooling to cell state
    
    This captures the biological hierarchy:
    Genes -> Pathways -> Cell State
    """
    embed_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, pathway_matrix: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Gene representations (batch, n_genes, embed_dim)
            pathway_matrix: (n_pathways, n_genes) binary membership matrix
            training: Training mode
        
        Returns:
            pathway_reps: (batch, n_pathways, embed_dim) if pathway_matrix provided
            cell_state: (batch, embed_dim) global representation
        """
        batch_size, n_genes, embed_dim = x.shape
        
        if pathway_matrix is not None:
            n_pathways = pathway_matrix.shape[0]
            
            # Level 1: Aggregate genes within each pathway
            # pathway_matrix: (n_pathways, n_genes)
            # x: (batch, n_genes, embed_dim)
            
            # Compute pathway gene counts for normalization
            pathway_sizes = pathway_matrix.sum(axis=1, keepdims=True)  # (n_pathways, 1)
            pathway_sizes = jnp.maximum(pathway_sizes, 1.0)  # Avoid division by zero
            
            # Weighted sum of genes in each pathway
            # (n_pathways, n_genes) @ (batch, n_genes, embed_dim)
            # -> (batch, n_pathways, embed_dim)
            pathway_reps = jnp.einsum('pn,bnd->bpd', pathway_matrix, x)
            pathway_reps = pathway_reps / pathway_sizes.T[None, :, :]
            
            # Level 2: Cross-pathway attention
            # Self-attention among pathway representations
            pathway_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(pathway_reps, pathway_reps)
            
            pathway_reps = nn.LayerNorm()(pathway_reps + pathway_attn)
            
            # MLP
            pathway_mlp = nn.Dense(self.embed_dim * 4)(pathway_reps)
            pathway_mlp = nn.gelu(pathway_mlp)
            pathway_mlp = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(pathway_mlp)
            pathway_mlp = nn.Dense(self.embed_dim)(pathway_mlp)
            
            pathway_reps = nn.LayerNorm()(pathway_reps + pathway_mlp)
            
            # Level 3: Global pooling
            # Attention-based pooling over pathways
            # Learn query for global aggregation
            global_query = self.param('global_query', 
                                      nn.initializers.normal(stddev=0.02),
                                      (1, 1, self.embed_dim))
            global_query = jnp.tile(global_query, (batch_size, 1, 1))
            
            cell_state = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(global_query, pathway_reps)
            
            cell_state = cell_state.squeeze(1)  # (batch, embed_dim)
            
            return pathway_reps, cell_state
        
        else:
            # Fallback: simple global average pooling
            cell_state = jnp.mean(x, axis=1)
            return None, cell_state


# ============================================================================
# Component 5: Bio-Inductive Transformer Block
# ============================================================================

class BioInductiveTransformerBlock(nn.Module):
    """
    Transformer block with biological inductive biases.
    
    Combines:
    - PPI sparse attention OR local window attention OR perceiver OR pathway_cross
    - Standard feed-forward network
    - Residual connections and layer norm
    """
    embed_dim: int
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    attention_type: str = 'ppi_sparse'  # 'ppi_sparse', 'local_window', 'full', 'perceiver', 'pathway_cross'
    window_size: int = 64  # For local_window attention
    num_inducing: int = 16  # For perceiver attention
    n_pathway_queries: int = 128  # For pathway_cross attention
    
    @nn.compact
    def __call__(self, x, attention_mask: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            attention_mask: PPI mask for sparse attention
            training: Training mode
        
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # Choose attention type
        if self.attention_type == 'ppi_sparse':
            attn_output = PPISparseAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                use_hard_mask=True
            )(x, ppi_mask=attention_mask, training=training)
            
        elif self.attention_type == 'local_window':
            attn_output = LocalWindowAttention(
                num_heads=self.num_heads,
                window_size=self.window_size,
                dropout_rate=self.dropout_rate
            )(x, additional_mask=attention_mask, training=training)
            
        elif self.attention_type == 'full':
            # Standard full attention (for comparison)
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x, x)
        
        elif self.attention_type == 'perceiver':
            # Perceiver-style induced attention with learnable gene modules
            perceiver_result = ModuleInducedAttention(
                num_inducing=self.num_inducing,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, attention_mask=attention_mask, training=training, 
              return_attention=getattr(self, '_return_attention', False))
            
            # Handle return_attention case
            if isinstance(perceiver_result, tuple):
                attn_output, attention_dict = perceiver_result
            else:
                attn_output = perceiver_result
                attention_dict = None
            
            # Perceiver already includes residual and LayerNorm, so skip the add&norm below
            x = attn_output
            # Skip to MLP directly
            mlp_output = nn.Dense(self.mlp_dim)(x)
            mlp_output = nn.gelu(mlp_output)
            mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
            mlp_output = nn.Dense(self.embed_dim)(mlp_output)
            mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
            output = nn.LayerNorm()(x + mlp_output)
            
            if attention_dict is not None:
                return output, attention_dict
            return output
        
        elif self.attention_type == 'pathway_cross':
            # Pathway-guided cross attention
            attn_output = PathwayCrossAttention(
                num_heads=self.num_heads,
                n_pathway_queries=self.n_pathway_queries,
                dropout_rate=self.dropout_rate,
                use_learnable_pathway_queries=True
            )(x, pathway_init_embed=None, attention_mask=attention_mask, training=training)
            # PathwayCrossAttention already includes residual and LayerNorm
            x = attn_output
            mlp_output = nn.Dense(self.mlp_dim)(x)
            mlp_output = nn.gelu(mlp_output)
            mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
            mlp_output = nn.Dense(self.embed_dim)(mlp_output)
            mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
            return nn.LayerNorm()(x + mlp_output)
        
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        # Add & Norm (for ppi_sparse, local_window, full)
        x = nn.LayerNorm()(x + attn_output)
        
        # Feed-forward network
        mlp_output = nn.Dense(self.mlp_dim)(x)
        mlp_output = nn.gelu(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
        mlp_output = nn.Dense(self.embed_dim)(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
        
        # Add & Norm
        x = nn.LayerNorm()(x + mlp_output)
        
        return x


# ============================================================================
# Component 6: Full Bio-Inductive Transformer Encoder
# ============================================================================

class BioInductiveTransformerEncoder(nn.Module):
    """
    Full Transformer encoder with biological inductive biases.
    
    Combines:
    1. Fixed PE from scGPT (or sinusoidal for reordered genes)
    2. PPI sparse attention / Local window attention
    3. Hierarchical pathway pooling
    """
    latent_dim: int = 50
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    chunk_size: int = 64
    pe_type: str = 'scgpt'  # 'sinusoidal', 'scgpt', or 'genecompass'
    attention_type: str = 'ppi_sparse'  # 'ppi_sparse', 'local_window', 'full', 'perceiver', 'pathway_cross'
    window_size: int = 64  # For local_window attention
    num_inducing: int = 16  # For perceiver attention
    n_pathway_queries: int = 128  # For pathway_cross attention
    use_pathway_pooling: bool = True
    pooling_type: str = 'mean'
    
    @nn.compact
    def __call__(self, x, 
                 ppi_mask: Optional[jnp.ndarray] = None,
                 scgpt_pe: Optional[jnp.ndarray] = None,
                 pathway_matrix: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Input (batch, n_genes)
            ppi_mask: PPI attention mask (n_chunks, n_chunks) or (n_genes, n_genes)
            scgpt_pe: scGPT-based positional encoding (n_chunks, scgpt_dim)
            pathway_matrix: Pathway membership (n_pathways, n_genes)
            training: Training mode
        
        Returns:
            latent: (batch, latent_dim)
        """
        batch_size, n_genes = x.shape
        n_chunks = (n_genes + self.chunk_size - 1) // self.chunk_size
        
        # Pad if necessary
        if n_genes % self.chunk_size != 0:
            padding = self.chunk_size - (n_genes % self.chunk_size)
            x = jnp.pad(x, ((0, 0), (0, padding)), mode='constant')
        
        # Reshape into chunks: (batch, n_chunks, chunk_size)
        x = x.reshape(batch_size, n_chunks, self.chunk_size)
        
        # Project to embedding dimension
        x = nn.Dense(self.embed_dim)(x)
        
        # Add biological positional encoding
        x = BiologicalPositionalEncoding(
            embed_dim=self.embed_dim,
            pe_type=self.pe_type,
            dropout_rate=self.dropout_rate
        )(x, scgpt_pe=scgpt_pe, training=training)
        
        # Apply transformer blocks with biological attention
        for i in range(self.num_layers):
            x = BioInductiveTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_type=self.attention_type,
                window_size=self.window_size,
                num_inducing=self.num_inducing,
                n_pathway_queries=self.n_pathway_queries,
                name=f'encoder_block_{i}'
            )(x, attention_mask=ppi_mask, training=training)
        
        # Pooling
        if self.use_pathway_pooling and pathway_matrix is not None:
            # Need to unflatten chunks back to gene-level for pathway pooling
            # x: (batch, n_chunks, embed_dim)
            # Project each chunk to full gene-level representation
            # This is a simplification - in full implementation, 
            # would work at gene level
            
            # For now, use chunk representations with modified pathway matrix
            # that maps pathways to chunks instead of genes
            _, cell_state = PathwayPooling(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, pathway_matrix=None, training=training)  # Use global pooling
        else:
            if self.pooling_type == 'mean':
                cell_state = jnp.mean(x, axis=1)
            elif self.pooling_type == 'attention':
                cell_state = AttentionPooling(
                    embed_dim=self.embed_dim,
                    name='module_attention_pool',
                )(x)
            elif self.pooling_type == 'set_transformer':
                cell_state = SetTransformerPooling(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    name='module_set_pool',
                )(x, training=training)
            else:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
        
        # Project to latent dimension
        latent = nn.Dense(self.latent_dim)(cell_state)
        
        return latent


# ============================================================================
# Component 7: Full Bio-Inductive Transformer Decoder
# ============================================================================

class BioInductiveTransformerDecoder(nn.Module):
    """
    Transformer decoder with biological inductive biases.
    
    Mirrors the encoder structure for reconstruction.
    """
    output_dim: int
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    chunk_size: int = 64
    pe_type: str = 'scgpt'
    attention_type: str = 'ppi_sparse'
    window_size: int = 64
    num_inducing: int = 16
    n_pathway_queries: int = 128
    
    @nn.compact
    def __call__(self, x,
                 ppi_mask: Optional[jnp.ndarray] = None,
                 scgpt_pe: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Latent representation (batch, latent_dim)
            ppi_mask: PPI attention mask
            scgpt_pe: scGPT-based positional encoding
            training: Training mode
        
        Returns:
            reconstructed: (batch, output_dim)
        """
        batch_size = x.shape[0]
        n_chunks = (self.output_dim + self.chunk_size - 1) // self.chunk_size
        
        # Project latent to embedding dimension
        x = nn.Dense(self.embed_dim)(x)
        
        # Expand to sequence: (batch, n_chunks, embed_dim)
        x = jnp.tile(x[:, None, :], (1, n_chunks, 1))
        
        # Add biological positional encoding
        x = BiologicalPositionalEncoding(
            embed_dim=self.embed_dim,
            pe_type=self.pe_type,
            dropout_rate=self.dropout_rate
        )(x, scgpt_pe=scgpt_pe, training=training)
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            x = BioInductiveTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_type=self.attention_type,
                window_size=self.window_size,
                num_inducing=self.num_inducing,
                n_pathway_queries=self.n_pathway_queries,
                name=f'decoder_block_{i}'
            )(x, attention_mask=ppi_mask, training=training)
        
        # Project each chunk back to chunk_size
        x = nn.Dense(self.chunk_size)(x)  # (batch, n_chunks, chunk_size)
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Trim to exact output dimension
        x = x[:, :self.output_dim]
        
        return x


# ============================================================================
# Full AutoEncoder
# ============================================================================

class BioInductiveAutoEncoder(nn.Module):
    """
    Combined encoder-decoder with biological inductive biases.
    """
    latent_dim: int = 50
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    chunk_size: int = 64
    output_dim: int = None
    pe_type: str = 'scgpt'
    attention_type: str = 'ppi_sparse'
    window_size: int = 64
    num_inducing: int = 16
    n_pathway_queries: int = 128
    use_pathway_pooling: bool = False
    pooling_type: str = 'mean'
    
    def setup(self):
        if self.output_dim is None:
            raise ValueError("output_dim must be set")
        
        self.encoder = BioInductiveTransformerEncoder(
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            chunk_size=self.chunk_size,
            pe_type=self.pe_type,
            attention_type=self.attention_type,
            window_size=self.window_size,
            num_inducing=self.num_inducing,
            n_pathway_queries=self.n_pathway_queries,
            use_pathway_pooling=self.use_pathway_pooling,
            pooling_type=self.pooling_type
        )
        
        self.decoder = BioInductiveTransformerDecoder(
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            chunk_size=self.chunk_size,
            pe_type=self.pe_type,
            attention_type=self.attention_type,
            window_size=self.window_size,
            num_inducing=self.num_inducing,
            n_pathway_queries=self.n_pathway_queries
        )
    
    def __call__(self, x,
                 ppi_mask: Optional[jnp.ndarray] = None,
                 scgpt_pe: Optional[jnp.ndarray] = None,
                 pathway_matrix: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
            x: Input (batch, n_genes)
            ppi_mask: PPI attention mask
            scgpt_pe: scGPT positional encoding
            pathway_matrix: Pathway membership matrix
            training: Training mode
        
        Returns:
            latent: (batch, latent_dim)
            reconstructed: (batch, n_genes)
        """
        latent = self.encoder(x, ppi_mask, scgpt_pe, pathway_matrix, training)
        reconstructed = self.decoder(latent, ppi_mask, scgpt_pe, training)
        return latent, reconstructed


# ============================================================================
# Utility functions for loading priors
# ============================================================================

def load_ppi_network(ppi_info_path: str, ppi_links_path: str, min_score: int = 700):
    """Load STRING PPI network."""
    ensp_to_gene = {}
    with gzip.open(ppi_info_path, 'rt') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            ensp_id = parts[0]
            gene_name = parts[1]
            ensp_to_gene[ensp_id] = gene_name
    
    ppi_edges = []
    with gzip.open(ppi_links_path, 'rt') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            g1 = ensp_to_gene.get(parts[0])
            g2 = ensp_to_gene.get(parts[1])
            score = int(parts[2])
            if g1 and g2 and score >= min_score:
                ppi_edges.append((g1, g2, score))
    
    return ensp_to_gene, ppi_edges


def load_scgpt_embeddings(embedding_path: str, gene_names: list):
    """Load scGPT gene embeddings."""
    with open(embedding_path, 'rb') as f:
        scgpt_emb = pickle.load(f)
    
    embed_dim = len(next(iter(scgpt_emb.values())))
    n_genes = len(gene_names)
    
    gene_embeddings = np.zeros((n_genes, embed_dim), dtype=np.float32)
    found_count = 0
    
    for idx, gene in enumerate(gene_names):
        if gene in scgpt_emb:
            gene_embeddings[idx] = np.array(scgpt_emb[gene])
            found_count += 1
    
    coverage = found_count / n_genes
    return gene_embeddings, coverage


def load_genecompass_embeddings(embedding_path: str, gene_names: list):
    """
    Load GeneCompass gene embeddings.
    
    GeneCompass embedding file structure:
    - gc_data['gene_symbol_embeddings'][gene_symbol] = np.array(768,)
    - gc_data['embedding_dim'] = 768
    
    Args:
        embedding_path: Path to GeneCompass embedding pickle file
        gene_names: List of gene names to load embeddings for
    
    Returns:
        gene_embeddings: (n_genes, 768) array of embeddings
        coverage: Fraction of genes found in embedding file
    """
    with open(embedding_path, 'rb') as f:
        gc_data = pickle.load(f)
    
    # Extract gene symbol embeddings dict
    gene_emb_dict = gc_data['gene_symbol_embeddings']
    embed_dim = gc_data.get('embedding_dim', 768)
    
    n_genes = len(gene_names)
    gene_embeddings = np.zeros((n_genes, embed_dim), dtype=np.float32)
    found_count = 0
    
    for idx, gene in enumerate(gene_names):
        if gene in gene_emb_dict:
            gene_embeddings[idx] = np.array(gene_emb_dict[gene])
            found_count += 1
    
    coverage = found_count / n_genes
    return gene_embeddings, coverage


def load_pathway_gene_matrix(gmt_path: str, gene_names: list, is_zip: bool = False):
    """Load pathway gene sets from GMT file."""
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    n_genes = len(gene_names)
    
    pathway_names = []
    pathway_gene_lists = []
    
    if is_zip:
        import subprocess
        result = subprocess.run(['unzip', '-p', gmt_path], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
    else:
        with open(gmt_path, 'r') as f:
            lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        pathway_name = parts[0]
        genes = parts[2:]
        
        gene_indices = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(gene_indices) > 0:
            pathway_names.append(pathway_name)
            pathway_gene_lists.append(gene_indices)
    
    n_pathways = len(pathway_names)
    pathway_matrix = np.zeros((n_pathways, n_genes), dtype=np.float32)
    
    for i, gene_indices in enumerate(pathway_gene_lists):
        for idx in gene_indices:
            pathway_matrix[i, idx] = 1.0
    
    return pathway_matrix, pathway_names


def aggregate_to_chunk_level(gene_array: np.ndarray, chunk_size: int,
                             aggregation: str = 'mean') -> np.ndarray:
    """
    Aggregate gene-level array to chunk level.
    
    Args:
        gene_array: (n_genes, dim) or (n_genes, n_genes) array
        chunk_size: Chunk size
        aggregation: 'mean', 'max', or 'sum'
    
    Returns:
        chunk_array: Aggregated array at chunk level
    """
    n_genes = gene_array.shape[0]
    n_chunks = (n_genes + chunk_size - 1) // chunk_size
    
    # Pad if necessary
    if n_genes % chunk_size != 0:
        padding = chunk_size - (n_genes % chunk_size)
        if gene_array.ndim == 1:
            gene_array = np.pad(gene_array, (0, padding))
        elif gene_array.ndim == 2:
            if gene_array.shape[0] == gene_array.shape[1]:
                # Square matrix (adjacency)
                gene_array = np.pad(gene_array, ((0, padding), (0, padding)))
            else:
                # Gene embeddings
                gene_array = np.pad(gene_array, ((0, padding), (0, 0)))
    
    if gene_array.ndim == 2 and gene_array.shape[0] != gene_array.shape[1]:
        # Gene embeddings: (n_genes, dim) -> (n_chunks, dim)
        gene_array = gene_array.reshape(n_chunks, chunk_size, -1)
        if aggregation == 'mean':
            return gene_array.mean(axis=1)
        elif aggregation == 'max':
            return gene_array.max(axis=1)
        elif aggregation == 'sum':
            return gene_array.sum(axis=1)
    
    elif gene_array.ndim == 2:
        # Adjacency matrix: (n_genes, n_genes) -> (n_chunks, n_chunks)
        chunk_array = np.zeros((n_chunks, n_chunks), dtype=np.float32)
        for i in range(n_chunks):
            for j in range(n_chunks):
                block = gene_array[
                    i * chunk_size:(i + 1) * chunk_size,
                    j * chunk_size:(j + 1) * chunk_size
                ]
                if aggregation == 'mean':
                    chunk_array[i, j] = block.mean()
                elif aggregation == 'max':
                    chunk_array[i, j] = block.max()
                elif aggregation == 'sum':
                    chunk_array[i, j] = block.sum()
        return chunk_array
    
    else:
        raise ValueError(f"Unsupported array shape: {gene_array.shape}")


# ============================================================================
# Training utilities
# ============================================================================

def create_bio_train_state(model, rng, input_shape, learning_rate=1e-4,
                           ppi_mask=None, scgpt_pe=None, pathway_matrix=None):
    """Create training state for bio-inductive model."""
    init_rngs = {'params': rng, 'dropout': rng}
    
    variables = model.init(
        init_rngs,
        jnp.ones(input_shape),
        ppi_mask=ppi_mask,
        scgpt_pe=scgpt_pe,
        pathway_matrix=pathway_matrix,
        training=False
    )
    
    optimizer = optax.adam(learning_rate)
    
    from flax.training import train_state
    
    class TrainState(train_state.TrainState):
        batch_stats: dict
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables.get('batch_stats', {})
    )
    
    return state


def train_step_bio(state, batch, rng, ppi_mask, scgpt_pe, pathway_matrix=None,
                   pathway_loss_weight=0.1, training=True):
    """Single training step."""
    
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        
        (latent, reconstructed), new_variables = state.apply_fn(
            variables,
            batch,
            ppi_mask=ppi_mask,
            scgpt_pe=scgpt_pe,
            pathway_matrix=pathway_matrix,
            training=training,
            rngs={'dropout': rng} if training else {},
            mutable=['batch_stats'] if training else []
        )
        
        # MSE reconstruction loss
        recon_loss = jnp.mean((reconstructed - batch) ** 2)
        
        # Pathway auxiliary loss
        if pathway_matrix is not None:
            pathway_true = jnp.dot(batch, pathway_matrix.T)
            pathway_pred = jnp.dot(reconstructed, pathway_matrix.T)
            pathway_loss = jnp.mean((pathway_true - pathway_pred) ** 2)
        else:
            pathway_loss = 0.0
        
        total_loss = recon_loss + pathway_loss_weight * pathway_loss
        
        return total_loss, (recon_loss, pathway_loss, latent, reconstructed, new_variables)
    
    (total_loss, (recon_loss, pathway_loss, latent, reconstructed, new_variables)), grads = \
        jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    if training and new_variables:
        state = state.replace(batch_stats=new_variables['batch_stats'])
    
    return state, total_loss, recon_loss, pathway_loss, latent, reconstructed


# ============================================================================
# Wrapper class for CellFlow integration
# ============================================================================

class GCAEWrapper:
    """
    Wrapper for bio-inductive encoder with PCA-like interface.
    
    Supports multiple configurations for ablation studies:
    - pe_type: 'sinusoidal' (fixed) or 'scgpt' (fixed structure, learned projection)
    - attention_type: 'full', 'ppi_sparse', 'local_window'
    - use_pathway_pooling: True/False
    """
    
    def __init__(self, latent_dim=50,
                 embed_dim=256, num_layers=3, num_heads=8, mlp_dim=512,
                 dropout_rate=0.1, chunk_size=64,
                 pe_type='scgpt',  # 'sinusoidal', 'scgpt', or 'genecompass'
                 attention_type='ppi_sparse',  # 'full', 'ppi_sparse', 'local_window', 'perceiver', 'pathway_cross'
                 window_size=64,  # For local_window attention
                 num_inducing=16,  # Backward-compatible alias for num_modules
                 num_modules=None,
                 n_pathway_queries=128,  # For pathway_cross attention
                 pooling_type='mean',
                 use_pathway_pooling=False,
                 use_pathway_loss=False, pathway_loss_weight=0.1):
        """
        Initialize wrapper with configuration for ablation studies.
        """
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.chunk_size = chunk_size
        self.pe_type = pe_type
        self.attention_type = attention_type
        self.window_size = window_size
        if num_modules is not None:
            num_inducing = num_modules
        self.num_inducing = num_inducing
        self.num_modules = num_inducing
        self.n_pathway_queries = n_pathway_queries
        self.pooling_type = pooling_type
        self.use_pathway_pooling = use_pathway_pooling
        self.use_pathway_loss = use_pathway_loss
        self.pathway_loss_weight = pathway_loss_weight
        
        self.model = None
        self.state = None
        self.input_dim = None
        self.X_mean = None
        
        # Prior data
        self.ppi_mask = None
        self.scgpt_pe = None
        self.pathway_matrix = None
        self.gene_names = None
    
    def set_priors(self, gene_names: list,
                   ppi_info_path: str = None, ppi_links_path: str = None,
                   ppi_min_score: int = 700, ppi_k_hop: int = 2,
                   scgpt_path: str = None,
                   genecompass_path: str = None,
                   pathway_gmt_path: str = None, pathway_is_zip: bool = False):
        """
        Load and set biological priors.
        
        Args:
            gene_names: List of gene names
            ppi_info_path: Path to STRING PPI info file
            ppi_links_path: Path to STRING PPI links file
            ppi_min_score: Minimum PPI score threshold
            ppi_k_hop: Number of hops for PPI mask
            scgpt_path: Path to scGPT gene embeddings (for pe_type='scgpt')
            genecompass_path: Path to GeneCompass gene embeddings (for pe_type='genecompass')
            pathway_gmt_path: Path to pathway GMT file
            pathway_is_zip: Whether pathway file is zipped
        """
        self.gene_names = gene_names
        n_genes = len(gene_names)
        n_chunks = (n_genes + self.chunk_size - 1) // self.chunk_size
        
        print(f"Setting up biological priors for {n_genes} genes ({n_chunks} chunks)...")
        
        # Load PPI and build attention mask
        if ppi_info_path and ppi_links_path and self.attention_type == 'ppi_sparse':
            print(f"  Loading PPI network (min_score={ppi_min_score}, k_hop={ppi_k_hop})...")
            _, ppi_edges = load_ppi_network(ppi_info_path, ppi_links_path, ppi_min_score)
            
            # Build gene-level PPI mask
            gene_ppi_mask = build_ppi_attention_mask(gene_names, ppi_edges, 
                                                      k_hop=ppi_k_hop, include_self=True)
            
            # Aggregate to chunk level
            self.ppi_mask = build_chunk_ppi_mask(gene_ppi_mask, self.chunk_size, 
                                                  threshold=0.05)
            self.ppi_mask = jnp.array(self.ppi_mask)
            
            n_edges = (self.ppi_mask > 0).sum()
            sparsity = 1 - n_edges / (n_chunks * n_chunks)
            print(f"  [OK] PPI mask: {self.ppi_mask.shape}, {n_edges} edges, {sparsity*100:.1f}% sparse")
        else:
            self.ppi_mask = None
            if self.attention_type == 'ppi_sparse':
                print(f"  ⚠ PPI not loaded, falling back to full attention")
                self.attention_type = 'full'
        
        # Load gene embeddings for positional encoding
        # Support both scGPT and GeneCompass embeddings
        if self.pe_type == 'genecompass' and genecompass_path:
            print(f"  Loading GeneCompass embeddings for positional encoding...")
            gene_emb, coverage = load_genecompass_embeddings(genecompass_path, gene_names)
            print(f"  [OK] GeneCompass coverage: {coverage*100:.1f}% (dim={gene_emb.shape[1]})")
            
            # Aggregate to chunk level
            self.scgpt_pe = aggregate_to_chunk_level(gene_emb, self.chunk_size, 'mean')
            self.scgpt_pe = jnp.array(self.scgpt_pe)
            print(f"  [OK] GeneCompass PE: {self.scgpt_pe.shape}")
        elif self.pe_type == 'scgpt' and scgpt_path:
            print(f"  Loading scGPT embeddings for positional encoding...")
            gene_emb, coverage = load_scgpt_embeddings(scgpt_path, gene_names)
            print(f"  [OK] scGPT coverage: {coverage*100:.1f}% (dim={gene_emb.shape[1]})")
            
            # Aggregate to chunk level
            self.scgpt_pe = aggregate_to_chunk_level(gene_emb, self.chunk_size, 'mean')
            self.scgpt_pe = jnp.array(self.scgpt_pe)
            print(f"  [OK] scGPT PE: {self.scgpt_pe.shape}")
        elif self.pe_type in ['scgpt', 'genecompass']:
            self.scgpt_pe = None
            print(f"  ⚠ {self.pe_type} embeddings not loaded, using sinusoidal PE")
            self.pe_type = 'sinusoidal'
        else:
            self.scgpt_pe = None
        
        # Load pathway matrix
        if pathway_gmt_path and (self.use_pathway_pooling or self.use_pathway_loss):
            print(f"  Loading pathway gene sets...")
            self.pathway_matrix, pathway_names = load_pathway_gene_matrix(
                pathway_gmt_path, gene_names, pathway_is_zip
            )
            self.pathway_matrix = jnp.array(self.pathway_matrix)
            print(f"  [OK] Pathway matrix: {self.pathway_matrix.shape} ({len(pathway_names)} pathways)")
        else:
            self.pathway_matrix = None
        
        print("  Priors setup complete!")
        print(f"  Configuration: PE={self.pe_type}, Attention={self.attention_type}")
    
    def fit(self, X, X_test=None, n_epochs=50, batch_size=256, learning_rate=1e-4,
            verbose=True, save_best_path=None, eval_every_epochs=5,
            early_stop_patience=0, early_stop_min_delta=0.0, eval_label='validation'):
        """
        Fit bio-inductive encoder/decoder.
        
        Args:
            X: Training data (n_samples, n_genes)
            X_test: Test data for early stopping
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print progress
            save_best_path: Path to save best encoder weights (optional)
            eval_every_epochs: Validation interval on X_test; <=0 disables validation
            early_stop_patience: Stop after this many evals without improvement; 0 disables
            early_stop_min_delta: Minimum validation-loss improvement to count as better
            eval_label: Human-readable name for the validation subset
        """
        X_np = np.asarray(X, dtype=np.float32)
        self.X_mean = jnp.array(X_np.mean(axis=0, keepdims=True), dtype=jnp.float32)
        X_mean_np = np.asarray(self.X_mean)
        X_centered_np = X_np - X_mean_np
        del X_np

        if X_test is not None:
            X_test_np = np.asarray(X_test, dtype=np.float32)
            X_test_centered_np = X_test_np - X_mean_np
            n_test_samples = X_test_np.shape[0]
            del X_test_np

        self.input_dim = X_centered_np.shape[1]
        n_samples = X_centered_np.shape[0]
        
        # Create model
        self.model = BioInductiveAutoEncoder(
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            chunk_size=self.chunk_size,
            output_dim=self.input_dim,
            pe_type=self.pe_type,
            attention_type=self.attention_type,
            window_size=self.window_size,
            num_inducing=self.num_inducing,
            n_pathway_queries=self.n_pathway_queries,
            use_pathway_pooling=self.use_pathway_pooling,
            pooling_type=self.pooling_type
        )
        
        # Initialize model
        rng = random.PRNGKey(0)
        self.state = create_bio_train_state(
            self.model, rng, (batch_size, self.input_dim), learning_rate,
            self.ppi_mask, self.scgpt_pe, self.pathway_matrix
        )
        
        if verbose:
            print(f"\nTraining Bio-Inductive Transformer...")
            print(f"  Input dim: {self.input_dim}, Latent dim: {self.latent_dim}")
            print(f"  PE type: {self.pe_type}")
            print(f"  Attention type: {self.attention_type}")
            print(f"  Module pooling: {self.pooling_type}")
            print(f"  Use pathway pooling: {self.use_pathway_pooling}")
            print(f"  Use pathway loss: {self.use_pathway_loss}")
            print(f"  Epochs: {n_epochs}, Batch size: {batch_size}")
            if X_test is not None and eval_every_epochs is not None and eval_every_epochs > 0:
                print(f"  Eval subset: {eval_label}")
                if early_stop_patience and early_stop_patience > 0:
                    print(
                        f"  Early stopping: patience={early_stop_patience} evals, "
                        f"min_delta={early_stop_min_delta}"
                    )
        
        from tqdm import tqdm
        
        best_test_loss = float('inf')
        best_state = None
        best_epoch = 0
        bad_eval_count = 0
        stopped_early = False
        
        # JIT compile training step
        @partial(jax.jit, static_argnames=['training'])
        def jit_train_step(state, batch, rng, ppi_mask, scgpt_pe, pathway_mat,
                           pw_weight, training):
            return train_step_bio(
                state, batch, rng, ppi_mask, scgpt_pe, pathway_mat,
                pw_weight, training
            )
        
        for epoch in range(n_epochs):
            epoch_losses = []
            epoch_recon_losses = []
            
            # Shuffle indices only (memory efficient - avoid copying entire dataset)
            rng, rng_perm = random.split(rng)
            perm = np.asarray(jax.random.permutation(rng_perm, n_samples))  # Convert to numpy
            
            n_batches = (n_samples + batch_size - 1) // batch_size
            pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}",
                       disable=not verbose)
            
            for i in pbar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                # Load batch using shuffled indices (avoids full dataset copy)
                batch_indices = perm[start_idx:end_idx]
                batch = jnp.array(X_centered_np[batch_indices], dtype=jnp.float32)
                
                rng, rng_step = random.split(rng)
                
                pathway_mat = self.pathway_matrix if self.use_pathway_loss else None
                
                self.state, total_loss, recon_loss, _, _, _ = jit_train_step(
                    self.state, batch, rng_step,
                    self.ppi_mask, self.scgpt_pe,
                    pathway_mat, self.pathway_loss_weight, True
                )
                
                epoch_losses.append(float(total_loss))
                epoch_recon_losses.append(float(recon_loss))
                
                if i % 10 == 0:
                    pbar.set_postfix({'loss': f'{np.mean(epoch_losses[-10:]):.6f}'})
            
            avg_train_loss = np.mean(epoch_losses)
            
            should_eval = (
                X_test is not None and
                eval_every_epochs is not None and
                eval_every_epochs > 0 and
                (epoch + 1) % eval_every_epochs == 0
            )

            if should_eval:
                test_losses = []
                n_test_batches = (n_test_samples + batch_size - 1) // batch_size
                
                for i in range(n_test_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_test_samples)
                    test_batch = jnp.array(X_test_centered_np[start_idx:end_idx], dtype=jnp.float32)
                    
                    rng, rng_eval = random.split(rng)
                    pathway_mat = self.pathway_matrix if self.use_pathway_loss else None
                    
                    _, test_loss, _, _, _, _ = jit_train_step(
                        self.state, test_batch, rng_eval,
                        self.ppi_mask, self.scgpt_pe,
                        pathway_mat, self.pathway_loss_weight, False
                    )
                    test_losses.append(float(test_loss))
                
                avg_test_loss = np.mean(test_losses)
                
                if verbose:
                    print(f"  Epoch {epoch+1}: train={avg_train_loss:.6f}, test={avg_test_loss:.6f}", end='')
                
                improved = avg_test_loss < (best_test_loss - early_stop_min_delta)
                if improved:
                    best_test_loss = avg_test_loss
                    best_state = self.state
                    best_epoch = epoch + 1
                    bad_eval_count = 0
                    if verbose:
                        print(f" [OK] Best!")
                    # Save best model if path provided
                    if save_best_path is not None:
                        self.save(save_best_path)
                else:
                    bad_eval_count += 1
                    if verbose:
                        suffix = f" (best: epoch {best_epoch}, {best_test_loss:.6f})"
                        if early_stop_patience and early_stop_patience > 0:
                            suffix += f" [{bad_eval_count}/{early_stop_patience}]"
                        print(suffix)
                    if early_stop_patience and early_stop_patience > 0 and bad_eval_count >= early_stop_patience:
                        stopped_early = True
                        if verbose:
                            print(
                                f"  Early stopping triggered at epoch {epoch + 1} "
                                f"on {eval_label}; restoring best epoch {best_epoch}"
                            )
                        break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: train={avg_train_loss:.6f}")
        
        if verbose:
            if X_test is not None and eval_every_epochs is not None and eval_every_epochs > 0 and best_state is not None:
                status = "Early-stopped" if stopped_early else "Training complete"
                print(f"[OK] {status}. Best: epoch {best_epoch}, {eval_label}_loss={best_test_loss:.6f}")
                self.state = best_state
            else:
                print(f"[OK] Training complete. Final loss: {avg_train_loss:.6f}")

        if save_best_path is not None:
            if X_test is not None and eval_every_epochs is not None and eval_every_epochs > 0 and best_state is not None:
                # The best validated checkpoint has already been saved during training.
                pass
            else:
                self.save(save_best_path)
    
    def transform(self, X, batch_size=1000):
        """Encode data to latent space."""
        if self.state is None:
            raise ValueError("Model not fitted!")
        
        if isinstance(X, jnp.ndarray):
            X_np = np.array(X)
        else:
            X_np = np.array(X, dtype=np.float32)
        
        n_samples = X_np.shape[0]
        latent_list = []
        variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = jnp.array(X_np[i:end_idx], dtype=jnp.float32)
            X_centered = X_batch - self.X_mean
            
            latent_batch, _ = self.model.apply(
                variables, X_centered,
                ppi_mask=self.ppi_mask,
                scgpt_pe=self.scgpt_pe,
                pathway_matrix=self.pathway_matrix if self.use_pathway_pooling else None,
                training=False, mutable=False
            )
            latent_list.append(np.array(latent_batch))
        
        return np.concatenate(latent_list, axis=0)
    
    def transform_with_attention(self, X, batch_size=256):
        """
        Encode data to latent space AND return attention weights.
        
        This is the key method for analyzing inter-module communication!
        
        Args:
            X: Input gene expression data (n_samples, n_genes)
            batch_size: Batch size for processing (smaller for attention analysis)
            
        Returns:
            latent: Encoded latent representation (n_samples, latent_dim)
            attention_dict: Dictionary containing attention weights:
                - 'chunks_to_inducing': (B, H, M, C) - How modules aggregate from chunks
                - 'inducing_self': (B, H, M, M) - Inter-module communication
                - 'inducing_to_chunks': (B, H, C, M) - How chunks query modules
                - 'inducing_points': (M, embed_dim) - Learned module embeddings
        
        Example:
            >>> encoder = GCAEWrapper.load('best_encoder.pkl')
            >>> latent, attn = encoder.transform_with_attention(X_test)
            >>> ip_self_attn = attn['inducing_self']  # (B, H, 32, 32) for 32 modules
            >>> # Average over samples and heads
            >>> mean_attn = ip_self_attn.mean(axis=(0, 1))  # (32, 32) module interaction matrix
        """
        if self.state is None:
            raise ValueError("Model not fitted!")
        
        if isinstance(X, jnp.ndarray):
            X_np = np.array(X)
        else:
            X_np = np.array(X, dtype=np.float32)
        
        n_samples = X_np.shape[0]
        latent_list = []
        attention_list = {
            'chunks_to_inducing': [],
            'inducing_self': [],
            'inducing_to_chunks': [],
        }
        inducing_points = None
        
        variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        
        # Create a modified model with return_attention=True
        # We need to call encoder blocks directly with return_attention flag
        def encode_with_attention(model_instance, x, ppi_mask, scgpt_pe, pathway_matrix):
            """Forward pass that returns attention weights."""
            # This follows the same logic as BioInductiveAutoEncoder.__call__
            # but with return_attention=True
            
            # Step 1: Input projection
            x_proj = model_instance.input_proj(x)
            
            # Step 2: Chunk into windows
            n_genes = x_proj.shape[-1]
            chunk_size = model_instance.encoder.chunk_size
            n_chunks = (n_genes + chunk_size - 1) // chunk_size
            pad_size = n_chunks * chunk_size - n_genes
            
            if pad_size > 0:
                x_padded = jnp.pad(x_proj, ((0, 0), (0, pad_size)))
            else:
                x_padded = x_proj
            
            x_chunks = x_padded.reshape(x_padded.shape[0], n_chunks, chunk_size)
            
            # Step 3: Chunk embedding
            x_chunks = model_instance.encoder.chunk_embed(x_chunks)
            
            # Step 4: Add positional encoding if available
            if scgpt_pe is not None and hasattr(model_instance.encoder, 'pe_proj'):
                pe_proj = model_instance.encoder.pe_proj(scgpt_pe)
                x_chunks = x_chunks + pe_proj
            
            # Step 5: Encoder blocks with attention
            all_attentions = []
            for block in model_instance.encoder.encoder_blocks:
                # For perceiver attention blocks, get attention weights
                if hasattr(block, 'attention') and hasattr(block.attention, 'num_inducing'):
                    # This is a perceiver attention block
                    x_norm = block.ln1(x_chunks)
                    attn_result = block.attention(
                        x_norm, 
                        attention_mask=None, 
                        training=False,
                        return_attention=True  # KEY: Request attention weights
                    )
                    if isinstance(attn_result, tuple):
                        attn_out, attn_dict = attn_result
                        all_attentions.append(attn_dict)
                        x_chunks = x_chunks + attn_out
                    else:
                        x_chunks = x_chunks + attn_result
                    
                    # MLP part
                    x_norm2 = block.ln2(x_chunks)
                    x_chunks = x_chunks + block.mlp(x_norm2)
                else:
                    # Regular transformer block
                    x_chunks = block(x_chunks, training=False)
            
            # Step 6: Pooling to get latent
            x_pooled = x_chunks.mean(axis=1)  # Global average pooling
            latent = model_instance.encoder.latent_proj(x_pooled)
            
            # Combine attention from all layers
            combined_attn = {}
            if all_attentions:
                # Use attention from last layer (most refined)
                combined_attn = all_attentions[-1]
            
            return latent, combined_attn
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = jnp.array(X_np[i:end_idx], dtype=jnp.float32)
            X_centered = X_batch - self.X_mean
            
            # Call with attention
            latent_batch, attn_dict = self.model.apply(
                variables, X_centered,
                ppi_mask=self.ppi_mask,
                scgpt_pe=self.scgpt_pe,
                pathway_matrix=self.pathway_matrix if self.use_pathway_pooling else None,
                method=encode_with_attention, 
                mutable=False
            )
            
            latent_list.append(np.array(latent_batch))
            
            # Collect attention weights
            if attn_dict:
                for key in ['chunks_to_inducing', 'inducing_self', 'inducing_to_chunks']:
                    if key in attn_dict:
                        attention_list[key].append(np.array(attn_dict[key]))
                if 'inducing_points' in attn_dict and inducing_points is None:
                    inducing_points = np.array(attn_dict['inducing_points'])
        
        # Concatenate results
        latent = np.concatenate(latent_list, axis=0)
        
        # Concatenate attention weights
        final_attention = {}
        for key, val_list in attention_list.items():
            if val_list:
                final_attention[key] = np.concatenate(val_list, axis=0)
        
        if inducing_points is not None:
            final_attention['inducing_points'] = inducing_points
        
        return latent, final_attention
    
    def inverse_transform(self, X_latent, batch_size=1000):
        """Decode latent representation to gene space."""
        if self.state is None:
            raise ValueError("Model not fitted!")
        
        if isinstance(X_latent, jnp.ndarray):
            X_latent_np = np.array(X_latent)
        else:
            X_latent_np = np.array(X_latent, dtype=np.float32)
        
        n_samples = X_latent_np.shape[0]
        reconstructed_list = []
        variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        
        def decode_fn(model_instance, latent):
            return model_instance.decoder(latent, self.ppi_mask, self.scgpt_pe, 
                                          training=False)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_latent_batch = jnp.array(X_latent_np[i:end_idx], dtype=jnp.float32)
            
            decoder_output = self.model.apply(
                variables, X_latent_batch,
                method=decode_fn, mutable=False
            )
            
            reconstructed_batch = decoder_output + self.X_mean
            reconstructed_list.append(np.array(reconstructed_batch))
        
        return np.concatenate(reconstructed_list, axis=0)
    
    def save(self, path: str):
        """
        Save encoder state to disk.
        
        Args:
            path: Path to save (will create .pkl file)
        """
        if self.state is None:
            raise ValueError("Model not fitted, cannot save!")
        
        save_dict = {
            'params': self.state.params,
            'batch_stats': self.state.batch_stats,
            'X_mean': np.array(self.X_mean),
            'input_dim': self.input_dim,
            'config': {
                'latent_dim': self.latent_dim,
                'embed_dim': self.embed_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'mlp_dim': self.mlp_dim,
                'dropout_rate': self.dropout_rate,
                'chunk_size': self.chunk_size,
                'pe_type': self.pe_type,
                'attention_type': self.attention_type,
                'window_size': self.window_size,
                'num_inducing': self.num_inducing,
                'num_modules': self.num_modules,
                'n_pathway_queries': self.n_pathway_queries,
                'pooling_type': self.pooling_type,
                'use_pathway_pooling': self.use_pathway_pooling,
                'use_pathway_loss': self.use_pathway_loss,
                'pathway_loss_weight': self.pathway_loss_weight,
            },
            'gene_names': self.gene_names,
        }
        
        # Save priors as numpy arrays
        if self.ppi_mask is not None:
            save_dict['ppi_mask'] = np.array(self.ppi_mask)
        if self.scgpt_pe is not None:
            save_dict['scgpt_pe'] = np.array(self.scgpt_pe)
        if self.pathway_matrix is not None:
            save_dict['pathway_matrix'] = np.array(self.pathway_matrix)
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"[OK] Encoder saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'GCAEWrapper':
        """
        Load encoder from disk.
        
        Args:
            path: Path to saved encoder (.pkl file)
        
        Returns:
            Loaded GCAEWrapper instance
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        config = save_dict['config']
        if 'num_modules' in config and 'num_inducing' not in config:
            config['num_inducing'] = config['num_modules']
        
        # Infer num_inducing from params if not in config (backward compatibility)
        if 'num_inducing' not in config:
            # Try to infer from saved params
            try:
                params = save_dict['params']
                def find_inducing_points(tree):
                    if isinstance(tree, Mapping):
                        if 'inducing_points' in tree and hasattr(tree['inducing_points'], 'shape'):
                            return tree['inducing_points']
                        for value in tree.values():
                            found = find_inducing_points(value)
                            if found is not None:
                                return found
                    return None

                inducing_points = find_inducing_points(params)
                if inducing_points is not None:
                    config['num_inducing'] = inducing_points.shape[0]
                    print(f"  Inferred num_inducing={config['num_inducing']} from saved params")
                else:
                    config['num_inducing'] = 16
            except Exception as e:
                print(f"  Warning: Could not infer num_inducing: {e}")
                config['num_inducing'] = 16  # Default fallback
        config.setdefault('n_pathway_queries', 128)
        config.setdefault('pooling_type', 'mean')
        
        encoder = cls(**config)
        
        encoder.input_dim = save_dict['input_dim']
        encoder.X_mean = jnp.array(save_dict['X_mean'])
        encoder.gene_names = save_dict.get('gene_names')
        
        # Load priors
        if 'ppi_mask' in save_dict:
            encoder.ppi_mask = jnp.array(save_dict['ppi_mask'])
        if 'scgpt_pe' in save_dict:
            encoder.scgpt_pe = jnp.array(save_dict['scgpt_pe'])
        if 'pathway_matrix' in save_dict:
            encoder.pathway_matrix = jnp.array(save_dict['pathway_matrix'])
        
        # Recreate model
        # Handle new parameters with defaults for backward compatibility
        num_inducing = getattr(encoder, 'num_inducing', 16)
        n_pathway_queries = getattr(encoder, 'n_pathway_queries', 128)
        
        encoder.model = BioInductiveAutoEncoder(
            latent_dim=encoder.latent_dim,
            embed_dim=encoder.embed_dim,
            num_layers=encoder.num_layers,
            num_heads=encoder.num_heads,
            mlp_dim=encoder.mlp_dim,
            dropout_rate=encoder.dropout_rate,
            chunk_size=encoder.chunk_size,
            output_dim=encoder.input_dim,
            pe_type=encoder.pe_type,
            attention_type=encoder.attention_type,
            window_size=encoder.window_size,
            num_inducing=num_inducing,
            n_pathway_queries=n_pathway_queries,
            use_pathway_pooling=encoder.use_pathway_pooling,
            pooling_type=encoder.pooling_type
        )
        
        # Recreate state
        from flax.training import train_state
        
        class TrainState(train_state.TrainState):
            batch_stats: dict
        
        # Dummy optimizer (not needed for inference)
        optimizer = optax.adam(1e-4)
        
        encoder.state = TrainState.create(
            apply_fn=encoder.model.apply,
            params=save_dict['params'],
            tx=optimizer,
            batch_stats=save_dict['batch_stats']
        )
        
        print(f"[OK] Encoder loaded from {path}")
        return encoder


# ============================================================================
# Ablation experiment configurations
# ============================================================================

ABLATION_CONFIGS = {
    'baseline_learned_pe_full_attn': {
        'pe_type': 'sinusoidal',  # Using sinusoidal as proxy for "no bio PE"
        'attention_type': 'full',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'Baseline: No biological inductive bias'
    },
    'scgpt_pe_full_attn': {
        'pe_type': 'scgpt',
        'attention_type': 'full',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'scGPT PE only: Fixed biological positional encoding'
    },
    'sinusoidal_pe_ppi_sparse': {
        'pe_type': 'sinusoidal',
        'attention_type': 'ppi_sparse',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'PPI sparse attention only: Hard attention mask from PPI'
    },
    'sinusoidal_pe_local_window': {
        'pe_type': 'sinusoidal',
        'attention_type': 'local_window',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'Local window attention: For reordered genes'
    },
    'scgpt_pe_ppi_sparse': {
        'pe_type': 'scgpt',
        'attention_type': 'ppi_sparse',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'scGPT PE + PPI sparse: Two inductive biases'
    },
    'scgpt_pe_local_window': {
        'pe_type': 'scgpt',
        'attention_type': 'local_window',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'scGPT PE + Local window: For reordered genes'
    },
    'full_bio_inductive': {
        'pe_type': 'scgpt',
        'attention_type': 'ppi_sparse',
        'use_pathway_pooling': True,
        'use_pathway_loss': True,
        'description': 'Full model: All biological inductive biases'
    },
    'full_bio_local_window': {
        'pe_type': 'scgpt',
        'attention_type': 'local_window',
        'use_pathway_pooling': True,
        'use_pathway_loss': True,
        'description': 'Full model with local window: For reordered genes'
    },
    # Window size ablations
    'scgpt_pe_local_window_w8': {
        'pe_type': 'scgpt',
        'attention_type': 'local_window',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'window_size': 8,
        'description': 'scGPT PE + Local window (w=8): Small window'
    },
    'scgpt_pe_local_window_w16': {
        'pe_type': 'scgpt',
        'attention_type': 'local_window',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'window_size': 16,
        'description': 'scGPT PE + Local window (w=16): Medium-small window'
    },
    # Chunk size ablations
    'scgpt_pe_local_window_c32': {
        'pe_type': 'scgpt',
        'attention_type': 'local_window',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'chunk_size': 32,
        'window_size': 32,
        'description': 'scGPT PE + Local window (chunk=32): Fine-grained chunking'
    },
    'scgpt_pe_local_window_c128': {
        'pe_type': 'scgpt',
        'attention_type': 'local_window',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'chunk_size': 128,
        'window_size': 128,
        'description': 'scGPT PE + Local window (chunk=128): Coarse-grained chunking'
    },
    'scgpt_pe_full_attn_no_chunk': {
        'pe_type': 'scgpt',
        'attention_type': 'full',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'chunk_size': 1,
        'description': 'scGPT PE + Full attention (no chunking): Each gene is a token'
    },
    # GeneCompass embedding configurations
    'genecompass_pe_full_attn': {
        'pe_type': 'genecompass',
        'attention_type': 'full',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'GeneCompass PE only: Fixed biological positional encoding (768-dim)'
    },
    'genecompass_pe_local_window': {
        'pe_type': 'genecompass',
        'attention_type': 'local_window',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'GeneCompass PE + Local window: For reordered genes'
    },
    'genecompass_pe_ppi_sparse': {
        'pe_type': 'genecompass',
        'attention_type': 'ppi_sparse',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'GeneCompass PE + PPI sparse: Two inductive biases'
    },
    'full_bio_genecompass': {
        'pe_type': 'genecompass',
        'attention_type': 'ppi_sparse',
        'use_pathway_pooling': True,
        'use_pathway_loss': True,
        'description': 'Full model with GeneCompass: All biological inductive biases'
    },
    'full_bio_genecompass_local_window': {
        'pe_type': 'genecompass',
        'attention_type': 'local_window',
        'use_pathway_pooling': True,
        'use_pathway_loss': True,
        'description': 'Full model with GeneCompass + local window: For reordered genes'
    },
    # Module-induced attention configurations
    'genecompass_pe_module': {
        'pe_type': 'genecompass',
        'attention_type': 'perceiver',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'num_inducing': 16,
        'description': 'GeneCompass PE + module-induced attention'
    },
    'genecompass_pe_perceiver': {
        'pe_type': 'genecompass',
        'attention_type': 'perceiver',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'num_inducing': 16,
        'description': 'Deprecated alias for genecompass_pe_module'
    },
    'scgpt_pe_perceiver': {
        'pe_type': 'scgpt',
        'attention_type': 'perceiver',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'num_inducing': 16,
        'description': 'scGPT PE + Perceiver Induced Attention: Learnable gene modules'
    },
    'genecompass_pe_perceiver_m32': {
        'pe_type': 'genecompass',
        'attention_type': 'perceiver',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'num_inducing': 32,
        'description': 'GeneCompass PE + Perceiver (M=32): More inducing points'
    },
    # Pathway-Guided Cross Attention configurations
    'genecompass_pe_pathway_cross': {
        'pe_type': 'genecompass',
        'attention_type': 'pathway_cross',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'n_pathway_queries': 128,
        'description': 'GeneCompass PE + Pathway Cross Attention: Pathway-guided attention'
    },
    'scgpt_pe_pathway_cross': {
        'pe_type': 'scgpt',
        'attention_type': 'pathway_cross',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'n_pathway_queries': 128,
        'description': 'scGPT PE + Pathway Cross Attention: Pathway-guided attention'
    },
    'genecompass_pe_pathway_cross_k256': {
        'pe_type': 'genecompass',
        'attention_type': 'pathway_cross',
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'n_pathway_queries': 256,
        'description': 'GeneCompass PE + Pathway Cross (K=256): More pathway queries'
    },
    # ============================================
    # Ablation: Default Transformer (no PE, no Inducing)
    # ============================================
    'default_transformer': {
        'pe_type': 'sinusoidal',  # Standard sinusoidal, not biological
        'attention_type': 'full',  # Standard full attention, no inducing
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'Default Transformer: sinusoidal PE + full attention (no bio PE, no Inducing)'
    },
    'default_transformer_local': {
        'pe_type': 'sinusoidal',
        'attention_type': 'local_window',  # Local window attention
        'use_pathway_pooling': False,
        'use_pathway_loss': False,
        'description': 'Default Transformer with local window: sinusoidal PE (no bio PE, no Inducing)'
    },
}


def get_encoder_for_ablation(config_name: str, latent_dim: int = 50,
                             **kwargs) -> GCAEWrapper:
    """
    Get encoder wrapper configured for specific ablation experiment.
    
    Args:
        config_name: Name of ablation config from ABLATION_CONFIGS
        latent_dim: Latent dimension
        **kwargs: Additional arguments to override config
    
    Returns:
        Configured GCAEWrapper
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ABLATION_CONFIGS.keys())}")
    
    config = ABLATION_CONFIGS[config_name].copy()
    description = config.pop('description')
    
    # Override with kwargs
    config.update(kwargs)
    
    print(f"Creating encoder for ablation: {config_name}")
    print(f"  Description: {description}")
    print(f"  Config: {config}")
    
    return GCAEWrapper(latent_dim=latent_dim, **config)
