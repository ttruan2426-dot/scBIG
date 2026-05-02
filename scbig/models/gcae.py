"""
Gene-Cluster Aware Encoder (GCAE)

Transformer encoder for single-cell gene expression with:
- Relational PE from GeneCompass embeddings
- Module-induced attention (Perceiver-style)
- Symmetric encoder-decoder
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Optional, Tuple, Dict, Any
import optax
from functools import partial
import pickle


def create_sinusoidal_pe(seq_len: int, dim: int) -> np.ndarray:
    """Standard sinusoidal positional encoding."""
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(dim)[None, :]
    
    angles = positions / (10000 ** (2 * (dims // 2) / dim))
    
    pe = np.zeros((seq_len, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe


class RelationalPositionalEncoding(nn.Module):
    """
    PE using biological priors. Structure is fixed (from GeneCompass),
    only the projection to model dim is learned.
    """
    embed_dim: int
    pe_type: str = 'genecompass'
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, 
                 bio_pe: Optional[jnp.ndarray] = None,
                 training: bool = False) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        
        if self.pe_type == 'sinusoidal':
            pe = create_sinusoidal_pe(seq_len, self.embed_dim)
            pe = jnp.array(pe, dtype=x.dtype)
            x = x + pe[None, :, :]
            
        elif self.pe_type in ['genecompass', 'scgpt']:
            if bio_pe is None:
                pe = create_sinusoidal_pe(seq_len, self.embed_dim)
                pe = jnp.array(pe, dtype=x.dtype)
                x = x + pe[None, :, :]
            else:
                projected_pe = nn.Dense(
                    self.embed_dim, 
                    use_bias=False,
                    name=f'{self.pe_type}_projection'
                )(bio_pe)
                projected_pe = nn.LayerNorm()(projected_pe)
                x = x + projected_pe[None, :, :]
        
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


class ModuleInducedAttention(nn.Module):
    """
    Perceiver-style attention with learnable gene modules as inducing points.
    
    3-step process:
    1. Modules aggregate from gene chunks (cross-attn M <- C)
    2. Inter-module communication (self-attn M <-> M)
    3. Gene chunks query modules (cross-attn C <- M)
    
    Complexity: O(C*M + M^2) instead of O(C^2)
    """
    num_modules: int = 16
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 training: bool = False,
                 return_attention: bool = False):
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        
        # learnable module embeddings
        module_embeddings = self.param(
            'module_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_modules, embed_dim)
        )
        
        M = jnp.broadcast_to(
            module_embeddings[None, :, :],
            (batch_size, self.num_modules, embed_dim)
        )
        
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        keep_prob = 1.0 - self.dropout_rate
        
        # Step 1: modules aggregate gene chunks
        q1 = nn.Dense(embed_dim, use_bias=False, name='q_agg')(M)
        k1 = nn.Dense(embed_dim, use_bias=False, name='k_agg')(x)
        v1 = nn.Dense(embed_dim, use_bias=False, name='v_agg')(x)
        
        q1 = q1.reshape(batch_size, self.num_modules, self.num_heads, head_dim)
        k1 = k1.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v1 = v1.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        q1 = jnp.transpose(q1, (0, 2, 1, 3))
        k1 = jnp.transpose(k1, (0, 2, 1, 3))
        v1 = jnp.transpose(v1, (0, 2, 1, 3))
        
        attn1 = jnp.einsum('bhmd,bhcd->bhmc', q1, k1) / scale
        attn1_weights = jax.nn.softmax(attn1, axis=-1)
        attn1 = attn1_weights
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn1.shape)
            attn1 = jnp.where(dropout_mask, attn1 / keep_prob, 0)
        
        M_updated = jnp.einsum('bhmc,bhcd->bhmd', attn1, v1)
        M_updated = jnp.transpose(M_updated, (0, 2, 1, 3))
        M_updated = M_updated.reshape(batch_size, self.num_modules, embed_dim)
        M_updated = nn.Dense(embed_dim, name='proj_agg')(M_updated)
        M_updated = nn.LayerNorm(name='ln_agg')(M + M_updated)
        
        # Step 2: inter-module self-attention
        q2 = nn.Dense(embed_dim, use_bias=False, name='q_self')(M_updated)
        k2 = nn.Dense(embed_dim, use_bias=False, name='k_self')(M_updated)
        v2 = nn.Dense(embed_dim, use_bias=False, name='v_self')(M_updated)
        
        q2 = q2.reshape(batch_size, self.num_modules, self.num_heads, head_dim)
        k2 = k2.reshape(batch_size, self.num_modules, self.num_heads, head_dim)
        v2 = v2.reshape(batch_size, self.num_modules, self.num_heads, head_dim)
        
        q2 = jnp.transpose(q2, (0, 2, 1, 3))
        k2 = jnp.transpose(k2, (0, 2, 1, 3))
        v2 = jnp.transpose(v2, (0, 2, 1, 3))
        
        attn2 = jnp.einsum('bhmd,bhnd->bhmn', q2, k2) / scale
        attn2_weights = jax.nn.softmax(attn2, axis=-1)
        attn2 = attn2_weights
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn2.shape)
            attn2 = jnp.where(dropout_mask, attn2 / keep_prob, 0)
        
        M_self = jnp.einsum('bhmn,bhnd->bhmd', attn2, v2)
        M_self = jnp.transpose(M_self, (0, 2, 1, 3))
        M_self = M_self.reshape(batch_size, self.num_modules, embed_dim)
        M_self = nn.Dense(embed_dim, name='proj_self')(M_self)
        M_final = nn.LayerNorm(name='ln_self')(M_updated + M_self)
        
        # Step 3: gene chunks query modules
        q3 = nn.Dense(embed_dim, use_bias=False, name='q_query')(x)
        k3 = nn.Dense(embed_dim, use_bias=False, name='k_query')(M_final)
        v3 = nn.Dense(embed_dim, use_bias=False, name='v_query')(M_final)
        
        q3 = q3.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k3 = k3.reshape(batch_size, self.num_modules, self.num_heads, head_dim)
        v3 = v3.reshape(batch_size, self.num_modules, self.num_heads, head_dim)
        
        q3 = jnp.transpose(q3, (0, 2, 1, 3))
        k3 = jnp.transpose(k3, (0, 2, 1, 3))
        v3 = jnp.transpose(v3, (0, 2, 1, 3))
        
        attn3 = jnp.einsum('bhcd,bhmd->bhcm', q3, k3) / scale
        attn3_weights = jax.nn.softmax(attn3, axis=-1)
        attn3 = attn3_weights
        
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn3.shape)
            attn3 = jnp.where(dropout_mask, attn3 / keep_prob, 0)
        
        x_updated = jnp.einsum('bhcm,bhmd->bhcd', attn3, v3)
        x_updated = jnp.transpose(x_updated, (0, 2, 1, 3))
        x_updated = x_updated.reshape(batch_size, seq_len, embed_dim)
        x_updated = nn.Dense(embed_dim, name='proj_query')(x_updated)
        
        output = nn.LayerNorm(name='ln_out')(x + x_updated)
        
        if return_attention:
            attention_dict = {
                'chunks_to_modules': attn1_weights,
                'module_interaction': attn2_weights,
                'modules_to_chunks': attn3_weights,
                'module_embeddings': module_embeddings,
                'module_final': M_final,
            }
            return output, attention_dict
        
        return output


class GCAETransformerBlock(nn.Module):
    """Transformer block with module-induced or standard attention."""
    embed_dim: int
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    attention_type: str = 'module'
    num_modules: int = 16
    
    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 training: bool = False):
        if self.attention_type == 'module':
            attn_result = ModuleInducedAttention(
                num_modules=self.num_modules,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, attention_mask=attention_mask, training=training)
            x = attn_result
        else:
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x, x)
            x = nn.LayerNorm()(x + attn_output)
        
        # FFN
        mlp_output = nn.Dense(self.mlp_dim)(x)
        mlp_output = nn.gelu(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
        mlp_output = nn.Dense(self.embed_dim)(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(mlp_output)
        
        output = nn.LayerNorm()(x + mlp_output)
        return output


class GeneClusterAwareEncoder(nn.Module):
    """
    GCAE Encoder: gene expression -> latent.
    
    Pipeline: chunk -> embed -> PE -> transformer blocks -> pool -> latent
    """
    latent_dim: int = 50
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    chunk_size: int = 64
    pe_type: str = 'genecompass'
    attention_type: str = 'module'
    num_modules: int = 16
    
    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 bio_pe: Optional[jnp.ndarray] = None,
                 training: bool = False) -> jnp.ndarray:
        batch_size, n_genes = x.shape
        n_chunks = (n_genes + self.chunk_size - 1) // self.chunk_size
        
        # pad if needed
        if n_genes % self.chunk_size != 0:
            padding = self.chunk_size - (n_genes % self.chunk_size)
            x = jnp.pad(x, ((0, 0), (0, padding)), mode='constant')
        
        x = x.reshape(batch_size, n_chunks, self.chunk_size)
        x = nn.Dense(self.embed_dim)(x)
        
        x = RelationalPositionalEncoding(
            embed_dim=self.embed_dim,
            pe_type=self.pe_type,
            dropout_rate=self.dropout_rate
        )(x, bio_pe=bio_pe, training=training)
        
        for i in range(self.num_layers):
            x = GCAETransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_type=self.attention_type,
                num_modules=self.num_modules,
                name=f'encoder_block_{i}'
            )(x, training=training)
        
        cell_state = jnp.mean(x, axis=1)
        latent = nn.Dense(self.latent_dim)(cell_state)
        
        return latent


class GeneClusterAwareDecoder(nn.Module):
    """
    GCAE Decoder: latent -> gene expression (symmetric to encoder).
    """
    output_dim: int
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    chunk_size: int = 64
    pe_type: str = 'genecompass'
    attention_type: str = 'module'
    num_modules: int = 16
    
    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 bio_pe: Optional[jnp.ndarray] = None,
                 training: bool = False) -> jnp.ndarray:
        batch_size = x.shape[0]
        n_chunks = (self.output_dim + self.chunk_size - 1) // self.chunk_size
        
        x = nn.Dense(self.embed_dim)(x)
        x = jnp.tile(x[:, None, :], (1, n_chunks, 1))
        
        x = RelationalPositionalEncoding(
            embed_dim=self.embed_dim,
            pe_type=self.pe_type,
            dropout_rate=self.dropout_rate
        )(x, bio_pe=bio_pe, training=training)
        
        for i in range(self.num_layers):
            x = GCAETransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_type=self.attention_type,
                num_modules=self.num_modules,
                name=f'decoder_block_{i}'
            )(x, training=training)
        
        x = nn.Dense(self.chunk_size)(x)
        x = x.reshape(batch_size, -1)
        x = x[:, :self.output_dim]
        
        return x


class GeneClusterAwareAutoEncoder(nn.Module):
    """Combined GCAE encoder-decoder."""
    latent_dim: int = 50
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    chunk_size: int = 64
    output_dim: int = None
    pe_type: str = 'genecompass'
    attention_type: str = 'module'
    num_modules: int = 16
    
    def setup(self):
        if self.output_dim is None:
            raise ValueError("output_dim must be specified")
        
        self.encoder = GeneClusterAwareEncoder(
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            chunk_size=self.chunk_size,
            pe_type=self.pe_type,
            attention_type=self.attention_type,
            num_modules=self.num_modules
        )
        
        self.decoder = GeneClusterAwareDecoder(
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            chunk_size=self.chunk_size,
            pe_type=self.pe_type,
            attention_type=self.attention_type,
            num_modules=self.num_modules
        )
    
    def __call__(self, x: jnp.ndarray,
                 bio_pe: Optional[jnp.ndarray] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        latent = self.encoder(x, bio_pe, training)
        reconstructed = self.decoder(latent, bio_pe, training)
        return latent, reconstructed
    
    def encode(self, x: jnp.ndarray,
               bio_pe: Optional[jnp.ndarray] = None,
               training: bool = False) -> jnp.ndarray:
        return self.encoder(x, bio_pe, training)
    
    def decode(self, latent: jnp.ndarray,
               bio_pe: Optional[jnp.ndarray] = None,
               training: bool = False) -> jnp.ndarray:
        return self.decoder(latent, bio_pe, training)


def load_genecompass_pe(embedding_path: str, 
                        gene_names: list,
                        chunk_size: int = 64) -> np.ndarray:
    """Load GeneCompass embeddings and aggregate to chunk level."""
    with open(embedding_path, 'rb') as f:
        gc_data = pickle.load(f)
    
    gene_emb_dict = gc_data.get('gene_symbol_embeddings', {})
    embed_dim = gc_data.get('embedding_dim', 768)
    
    n_genes = len(gene_names)
    gene_embeddings = np.zeros((n_genes, embed_dim), dtype=np.float32)
    
    for idx, gene in enumerate(gene_names):
        if gene in gene_emb_dict:
            gene_embeddings[idx] = np.array(gene_emb_dict[gene])
    
    n_chunks = (n_genes + chunk_size - 1) // chunk_size
    
    if n_genes % chunk_size != 0:
        padding = chunk_size - (n_genes % chunk_size)
        gene_embeddings = np.pad(gene_embeddings, ((0, padding), (0, 0)))
    
    gene_embeddings = gene_embeddings.reshape(n_chunks, chunk_size, -1)
    chunk_pe = gene_embeddings.mean(axis=1)
    
    return chunk_pe


# The trainable wrapper used by the Norman additive experiment lives in a
# separate implementation module to keep this public namespace compact.
from scbig.models._gcae_impl import (  # noqa: E402
    GCAEWrapper,
    ABLATION_CONFIGS as GCAE_CONFIGS,
    get_encoder_for_ablation as get_gcae_for_config,
    build_ppi_attention_mask,
    load_ppi_network,
    load_pathway_gene_matrix,
    load_genecompass_embeddings,
)


__all__ = [
    "RelationalPositionalEncoding",
    "ModuleInducedAttention",
    "GCAETransformerBlock",
    "GeneClusterAwareEncoder",
    "GeneClusterAwareDecoder",
    "GeneClusterAwareAutoEncoder",
    "GCAEWrapper",
    "GCAE_CONFIGS",
    "get_gcae_for_config",
    "build_ppi_attention_mask",
    "load_ppi_network",
    "load_pathway_gene_matrix",
    "load_genecompass_embeddings",
    "load_genecompass_pe",
]
