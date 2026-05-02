#!/usr/bin/env python
"""
Gene-Level Loss Functions for CellFlow with Biological Inductive Biases

This module provides principled loss functions that enforce gene-gene relationships
in the predicted perturbation responses.

Key Loss Functions:
1. Gene-Gene Correlation Consistency Loss - preserves regulatory relationships
2. Pathway-Level Optimal Transport Loss - matches distributions in biological quotient space
3. Graph Smoothness Loss - Laplacian regularization on PPI graph

References:
- Covariance matching in score-based models
- Moment-matching flows
- Optimal transport in generative modeling
- Physics-informed neural networks
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Optional, Tuple


# ============================================================================
# Loss 1: Gene-Gene Correlation Consistency Loss
# ============================================================================

def compute_correlation_matrix(X: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Compute Pearson correlation matrix between genes.
    
    Args:
        X: (N, G) gene expression matrix where N = cells, G = genes
        eps: Small constant for numerical stability (increased from 1e-8 to 1e-6)
    
    Returns:
        corr: (G, G) correlation matrix
    """
    # Center the data
    X_centered = X - jnp.mean(X, axis=0, keepdims=True)
    
    # Compute standard deviation with better numerical stability
    std = jnp.std(X, axis=0, keepdims=True)
    std = jnp.maximum(std, eps)  # More robust than std + eps
    
    # Normalize
    X_normalized = X_centered / std
    
    # Clip to prevent extreme values from causing NaN
    X_normalized = jnp.clip(X_normalized, -10.0, 10.0)
    
    # Compute correlation as dot product of normalized features
    n_samples = X.shape[0]
    corr = jnp.dot(X_normalized.T, X_normalized) / n_samples
    
    return corr


def compute_covariance_matrix(X: jnp.ndarray) -> jnp.ndarray:
    """
    Compute covariance matrix between genes.
    
    Args:
        X: (N, G) gene expression matrix
    
    Returns:
        cov: (G, G) covariance matrix
    """
    X_centered = X - jnp.mean(X, axis=0, keepdims=True)
    n_samples = X.shape[0]
    cov = jnp.dot(X_centered.T, X_centered) / (n_samples - 1)
    return cov


def gene_correlation_loss(X_pred: jnp.ndarray, 
                          X_true: jnp.ndarray,
                          use_correlation: bool = True) -> jnp.ndarray:
    """
    Gene-Gene Correlation Consistency Loss.
    
    Computes Frobenius norm between correlation/covariance matrices.
    
    This loss ensures that predicted expression preserves the gene-gene
    regulatory relationships present in true perturbation responses.
    
    Args:
        X_pred: (N, G) predicted gene expression
        X_true: (N, G) true gene expression
        use_correlation: If True, use correlation; if False, use covariance
    
    Returns:
        loss: Scalar loss value
    """
    if use_correlation:
        mat_pred = compute_correlation_matrix(X_pred)
        mat_true = compute_correlation_matrix(X_true)
    else:
        mat_pred = compute_covariance_matrix(X_pred)
        mat_true = compute_covariance_matrix(X_true)
    
    # Frobenius norm
    diff = mat_pred - mat_true
    loss = jnp.sqrt(jnp.sum(diff ** 2))
    
    # Normalize by matrix size for stability
    n_genes = X_pred.shape[1]
    loss = loss / n_genes
    
    return loss


def ppi_masked_correlation_loss(X_pred: jnp.ndarray,
                                 X_true: jnp.ndarray,
                                 ppi_mask: jnp.ndarray,
                                 use_correlation: bool = True) -> jnp.ndarray:
    """
    PPI-Masked Gene-Gene Correlation Loss.
    
    Only computes correlation difference for gene pairs that have
    known protein-protein interactions. This focuses the loss on
    biologically meaningful relationships.
    
    Args:
        X_pred: (N, G) predicted gene expression
        X_true: (N, G) true gene expression
        ppi_mask: (G, G) binary mask, 1 = has PPI interaction
        use_correlation: If True, use correlation; if False, use covariance
    
    Returns:
        loss: Scalar loss value
    """
    if use_correlation:
        mat_pred = compute_correlation_matrix(X_pred)
        mat_true = compute_correlation_matrix(X_true)
    else:
        mat_pred = compute_covariance_matrix(X_pred)
        mat_true = compute_covariance_matrix(X_true)
    
    # Apply mask - only consider PPI-connected pairs
    diff = jnp.abs(mat_pred - mat_true) * ppi_mask
    
    # Average over masked elements
    n_edges = jnp.maximum(jnp.sum(ppi_mask), 1.0)
    loss = jnp.sum(diff) / n_edges
    
    return loss


# ============================================================================
# Loss 1b: Cluster-Level Correlation Loss (Efficient Version)
# ============================================================================

def aggregate_by_cluster(X: jnp.ndarray, 
                         cluster_labels: jnp.ndarray,
                         n_clusters: int) -> jnp.ndarray:
    """
    Aggregate gene expression to cluster level.
    
    Args:
        X: (N, G) gene expression matrix
        cluster_labels: (G,) cluster assignment for each gene (0 to n_clusters-1)
        n_clusters: Number of clusters
    
    Returns:
        Z: (N, K) cluster-level expression (mean per cluster)
    """
    N, G = X.shape
    
    # Create one-hot encoding of cluster assignments
    # cluster_labels: (G,) -> one_hot: (G, K)
    one_hot = jax.nn.one_hot(cluster_labels, n_clusters)  # (G, K)
    
    # Count genes per cluster for normalization
    cluster_sizes = jnp.sum(one_hot, axis=0)  # (K,)
    cluster_sizes = jnp.maximum(cluster_sizes, 1.0)  # Avoid division by zero
    
    # Aggregate: X @ one_hot / cluster_sizes
    # X: (N, G), one_hot: (G, K) -> Z: (N, K)
    Z = jnp.dot(X, one_hot) / cluster_sizes[None, :]
    
    return Z


def cluster_correlation_loss(X_pred: jnp.ndarray,
                             X_true: jnp.ndarray,
                             cluster_labels: jnp.ndarray,
                             n_clusters: int = 128,
                             use_correlation: bool = True) -> jnp.ndarray:
    """
    Cluster-Level Correlation Consistency Loss.
    
    Instead of computing full G x G correlation matrix (O(G^2)),
    first aggregate genes to K clusters, then compute K x K correlation.
    
    Complexity: O(N * K^2) instead of O(N * G^2), ~256x faster for K=128, G=2051
    
    The cluster assignments should match those used for gene reordering,
    ensuring consistency between model structure and loss function.
    
    Args:
        X_pred: (N, G) predicted gene expression
        X_true: (N, G) true gene expression
        cluster_labels: (G,) cluster assignment for each gene
        n_clusters: Number of clusters (should match clustering in preprocessing)
        use_correlation: If True, use correlation; if False, use covariance
    
    Returns:
        loss: Scalar loss value
    """
    # Aggregate to cluster level
    Z_pred = aggregate_by_cluster(X_pred, cluster_labels, n_clusters)  # (N, K)
    Z_true = aggregate_by_cluster(X_true, cluster_labels, n_clusters)  # (N, K)
    
    # Compute cluster-level correlation/covariance matrices
    if use_correlation:
        mat_pred = compute_correlation_matrix(Z_pred)  # (K, K)
        mat_true = compute_correlation_matrix(Z_true)  # (K, K)
    else:
        mat_pred = compute_covariance_matrix(Z_pred)
        mat_true = compute_covariance_matrix(Z_true)
    
    # Frobenius norm
    diff = mat_pred - mat_true
    loss = jnp.sqrt(jnp.sum(diff ** 2))
    
    # Normalize by number of clusters
    loss = loss / n_clusters
    
    return loss


# ============================================================================
# Loss 2: Pathway-Level Optimal Transport Loss
# ============================================================================

def sinkhorn_distance(X: jnp.ndarray, 
                      Y: jnp.ndarray,
                      epsilon: float = 0.1,
                      n_iterations: int = 50) -> jnp.ndarray:
    """
    Compute Sinkhorn distance (entropy-regularized OT) between two distributions.
    
    This is a differentiable approximation to the Wasserstein distance.
    
    Args:
        X: (N, D) samples from first distribution
        Y: (M, D) samples from second distribution
        epsilon: Entropy regularization strength
        n_iterations: Number of Sinkhorn iterations
    
    Returns:
        distance: Scalar Sinkhorn distance
    """
    n = X.shape[0]
    m = Y.shape[0]
    
    # Compute cost matrix (squared Euclidean distance)
    # C[i,j] = ||X[i] - Y[j]||^2
    X_sqnorm = jnp.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
    Y_sqnorm = jnp.sum(Y ** 2, axis=1, keepdims=True)  # (M, 1)
    C = X_sqnorm + Y_sqnorm.T - 2 * jnp.dot(X, Y.T)  # (N, M)
    C = jnp.maximum(C, 0)  # Numerical stability
    
    # Compute kernel K = exp(-C / epsilon)
    K = jnp.exp(-C / epsilon)
    
    # Initialize uniform marginals
    a = jnp.ones(n) / n
    b = jnp.ones(m) / m
    
    # Sinkhorn iterations
    u = jnp.ones(n)
    v = jnp.ones(m)
    
    for _ in range(n_iterations):
        u = a / (jnp.dot(K, v) + 1e-10)
        v = b / (jnp.dot(K.T, u) + 1e-10)
    
    # Compute transport plan
    P = u[:, None] * K * v[None, :]
    
    # Compute distance
    distance = jnp.sum(P * C)
    
    return distance


def pathway_aggregate(X: jnp.ndarray, 
                      pathway_matrix: jnp.ndarray,
                      normalize: bool = True) -> jnp.ndarray:
    """
    Aggregate gene expression to pathway level.
    
    Args:
        X: (N, G) gene expression
        pathway_matrix: (K, G) pathway membership matrix
        normalize: Whether to normalize by pathway size
    
    Returns:
        Z: (N, K) pathway-level representation
    """
    # pathway_matrix: (K, G), X: (N, G)
    # Z = X @ pathway_matrix.T -> (N, K)
    Z = jnp.dot(X, pathway_matrix.T)
    
    if normalize:
        # Normalize by pathway size
        pathway_sizes = jnp.maximum(jnp.sum(pathway_matrix, axis=1), 1.0)  # (K,)
        Z = Z / pathway_sizes[None, :]
    
    return Z


def pathway_ot_loss(X_pred: jnp.ndarray,
                    X_true: jnp.ndarray,
                    pathway_matrix: jnp.ndarray,
                    epsilon: float = 0.1,
                    n_sinkhorn_iters: int = 50) -> jnp.ndarray:
    """
    Pathway-Level Optimal Transport Loss.
    
    Computes Sinkhorn distance between predicted and true distributions
    in the pathway-aggregated space.
    
    This provides biologically meaningful distributional matching:
    "We match distributions in a biologically meaningful quotient space."
    
    Args:
        X_pred: (N, G) predicted gene expression
        X_true: (N, G) true gene expression  
        pathway_matrix: (K, G) pathway membership
        epsilon: Sinkhorn regularization
        n_sinkhorn_iters: Number of Sinkhorn iterations
    
    Returns:
        loss: Scalar OT loss
    """
    # Aggregate to pathway level
    Z_pred = pathway_aggregate(X_pred, pathway_matrix, normalize=True)
    Z_true = pathway_aggregate(X_true, pathway_matrix, normalize=True)
    
    # Compute Sinkhorn distance
    loss = sinkhorn_distance(Z_pred, Z_true, epsilon, n_sinkhorn_iters)
    
    # Normalize by number of pathways
    n_pathways = pathway_matrix.shape[0]
    loss = loss / n_pathways
    
    return loss


def pathway_mmd_loss(X_pred: jnp.ndarray,
                     X_true: jnp.ndarray,
                     pathway_matrix: jnp.ndarray,
                     kernel_bandwidth: float = 1.0) -> jnp.ndarray:
    """
    Pathway-Level MMD Loss (Maximum Mean Discrepancy).
    
    Alternative to OT, faster but less principled.
    
    Args:
        X_pred: (N, G) predicted gene expression
        X_true: (N, G) true gene expression
        pathway_matrix: (K, G) pathway membership
        kernel_bandwidth: RBF kernel bandwidth
    
    Returns:
        loss: Scalar MMD loss
    """
    # Aggregate to pathway level
    Z_pred = pathway_aggregate(X_pred, pathway_matrix, normalize=True)
    Z_true = pathway_aggregate(X_true, pathway_matrix, normalize=True)
    
    # RBF kernel
    def rbf_kernel(X, Y, sigma=1.0):
        X_sqnorm = jnp.sum(X ** 2, axis=1, keepdims=True)
        Y_sqnorm = jnp.sum(Y ** 2, axis=1, keepdims=True)
        sq_dist = X_sqnorm + Y_sqnorm.T - 2 * jnp.dot(X, Y.T)
        return jnp.exp(-sq_dist / (2 * sigma ** 2))
    
    n = Z_pred.shape[0]
    m = Z_true.shape[0]
    
    K_pp = rbf_kernel(Z_pred, Z_pred, kernel_bandwidth)
    K_tt = rbf_kernel(Z_true, Z_true, kernel_bandwidth)
    K_pt = rbf_kernel(Z_pred, Z_true, kernel_bandwidth)
    
    # MMD^2 = E[K(p,p)] + E[K(t,t)] - 2*E[K(p,t)]
    mmd_sq = (jnp.sum(K_pp) / (n * n) + 
              jnp.sum(K_tt) / (m * m) - 
              2 * jnp.sum(K_pt) / (n * m))
    
    return jnp.maximum(mmd_sq, 0.0)


# ============================================================================
# Loss 3: Graph Smoothness Loss (Laplacian Regularization)
# ============================================================================

def compute_graph_laplacian(adj_matrix: jnp.ndarray, 
                            normalized: bool = True) -> jnp.ndarray:
    """
    Compute graph Laplacian from adjacency matrix.
    
    Args:
        adj_matrix: (G, G) adjacency matrix
        normalized: If True, compute normalized Laplacian
    
    Returns:
        L: (G, G) Laplacian matrix
    """
    # Degree matrix
    degree = jnp.sum(adj_matrix, axis=1)
    
    if normalized:
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        d_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree + 1e-10), 0.0)
        D_inv_sqrt = jnp.diag(d_inv_sqrt)
        L = jnp.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    else:
        # Unnormalized Laplacian: L = D - A
        L = jnp.diag(degree) - adj_matrix
    
    return L


def graph_smoothness_loss(X_pred: jnp.ndarray,
                          ppi_adjacency: jnp.ndarray,
                          normalized_laplacian: bool = True) -> jnp.ndarray:
    """
    Graph Smoothness Loss (Laplacian Regularization).
    
    Encourages predictions to be smooth on the PPI graph:
    connected genes should have similar expression changes.
    
    L_graph = trace(X^T L X) = sum_{(i,j) in E} w_ij (x_i - x_j)^2
    
    This extends the encoder's PPI inductive bias to the decoder output.
    
    Args:
        X_pred: (N, G) predicted gene expression
        ppi_adjacency: (G, G) PPI adjacency matrix
        normalized_laplacian: Whether to use normalized Laplacian
    
    Returns:
        loss: Scalar smoothness loss
    """
    # Compute Laplacian
    L = compute_graph_laplacian(ppi_adjacency, normalized=normalized_laplacian)
    
    # Compute smoothness: mean over cells of x^T L x
    # For each cell x (1, G): x^T @ L @ x
    # Batch: (N, G) @ (G, G) @ (G, N) -> diagonal
    
    # X_pred: (N, G), L: (G, G)
    # x^T L x for each row
    XL = jnp.dot(X_pred, L)  # (N, G)
    smoothness = jnp.sum(XL * X_pred, axis=1)  # (N,) - trace for each cell
    
    # Average over cells
    loss = jnp.mean(smoothness)
    
    return loss


def delta_graph_smoothness_loss(X_pred: jnp.ndarray,
                                 X_ctrl: jnp.ndarray,
                                 ppi_adjacency: jnp.ndarray) -> jnp.ndarray:
    """
    Graph Smoothness on Perturbation Delta.
    
    Applies smoothness constraint to the predicted change (delta),
    not the absolute expression.
    
    Biological interpretation: Perturbation effects should propagate
    smoothly along the PPI network.
    
    Args:
        X_pred: (N, G) predicted perturbed expression
        X_ctrl: (M, G) control expression (will use mean)
        ppi_adjacency: (G, G) PPI adjacency matrix
    
    Returns:
        loss: Scalar smoothness loss on delta
    """
    # Compute delta (perturbation effect)
    ctrl_mean = jnp.mean(X_ctrl, axis=0, keepdims=True)  # (1, G)
    delta = X_pred - ctrl_mean  # (N, G)
    
    # Apply smoothness to delta
    L = compute_graph_laplacian(ppi_adjacency, normalized=True)
    
    delta_L = jnp.dot(delta, L)
    smoothness = jnp.sum(delta_L * delta, axis=1)
    
    return jnp.mean(smoothness)


# ============================================================================
# Combined Loss Function
# ============================================================================

def combined_gene_loss(X_pred: jnp.ndarray,
                       X_true: jnp.ndarray,
                       ppi_mask: Optional[jnp.ndarray] = None,
                       ppi_adjacency: Optional[jnp.ndarray] = None,
                       pathway_matrix: Optional[jnp.ndarray] = None,
                       X_ctrl: Optional[jnp.ndarray] = None,
                       lambda_corr: float = 0.1,
                       lambda_ot: float = 0.1,
                       lambda_smooth: float = 0.01,
                       use_ppi_masked_corr: bool = True) -> Tuple[jnp.ndarray, dict]:
    """
    Combined Gene-Level Loss Function.
    
    Combines:
    1. Gene-Gene Correlation Consistency
    2. Pathway-Level OT
    3. Graph Smoothness
    
    Args:
        X_pred: (N, G) predicted expression
        X_true: (N, G) true expression
        ppi_mask: (G, G) PPI mask for correlation loss
        ppi_adjacency: (G, G) PPI adjacency for smoothness
        pathway_matrix: (K, G) pathway membership
        X_ctrl: (M, G) control cells for delta smoothness
        lambda_corr: Weight for correlation loss
        lambda_ot: Weight for OT loss
        lambda_smooth: Weight for smoothness loss
        use_ppi_masked_corr: Use PPI-masked or full correlation
    
    Returns:
        total_loss: Scalar total loss
        loss_dict: Dictionary of individual losses
    """
    loss_dict = {}
    total_loss = 0.0
    
    # 1. Correlation loss
    if lambda_corr > 0:
        if use_ppi_masked_corr and ppi_mask is not None:
            corr_loss = ppi_masked_correlation_loss(X_pred, X_true, ppi_mask)
        else:
            corr_loss = gene_correlation_loss(X_pred, X_true)
        
        total_loss = total_loss + lambda_corr * corr_loss
        loss_dict['corr_loss'] = corr_loss
    
    # 2. Pathway OT loss
    if lambda_ot > 0 and pathway_matrix is not None:
        ot_loss = pathway_ot_loss(X_pred, X_true, pathway_matrix)
        total_loss = total_loss + lambda_ot * ot_loss
        loss_dict['ot_loss'] = ot_loss
    
    # 3. Graph smoothness loss
    if lambda_smooth > 0 and ppi_adjacency is not None:
        if X_ctrl is not None:
            smooth_loss = delta_graph_smoothness_loss(X_pred, X_ctrl, ppi_adjacency)
        else:
            smooth_loss = graph_smoothness_loss(X_pred, ppi_adjacency)
        
        total_loss = total_loss + lambda_smooth * smooth_loss
        loss_dict['smooth_loss'] = smooth_loss
    
    loss_dict['total_gene_loss'] = total_loss
    
    return total_loss, loss_dict


# ============================================================================
# Perturbation-Invariant Gene Relation Loss (Bonus)
# ============================================================================

def perturbation_invariant_relation_loss(X_pred: jnp.ndarray,
                                          X_true: jnp.ndarray,
                                          X_ctrl: jnp.ndarray,
                                          alpha: float = 0.5) -> jnp.ndarray:
    """
    Perturbation-Invariant Gene Relation Loss.
    
    Core idea: Perturbation changes gene values but should not completely
    destroy regulatory relationships.
    
    The loss encourages:
    1. Predicted relations match true perturbed relations
    2. Both should preserve some structure from control
    
    Args:
        X_pred: (N, G) predicted perturbed expression
        X_true: (N, G) true perturbed expression
        X_ctrl: (M, G) control expression
        alpha: Balance between matching true vs preserving control structure
    
    Returns:
        loss: Scalar loss
    """
    # Compute relation matrices
    R_pred = compute_correlation_matrix(X_pred)
    R_true = compute_correlation_matrix(X_true)
    R_ctrl = compute_correlation_matrix(X_ctrl)
    
    # Loss 1: Predicted relations should match true
    match_loss = jnp.mean((R_pred - R_true) ** 2)
    
    # Loss 2: Both should preserve control structure to some degree
    # (Perturbation changes, but not everything)
    preserve_loss = jnp.mean((R_pred - R_ctrl) ** 2) + jnp.mean((R_true - R_ctrl) ** 2)
    
    # Combined
    loss = match_loss + alpha * preserve_loss
    
    return loss


# ============================================================================
# Direction Consistency Loss
# ============================================================================

def direction_loss(X_pred: jnp.ndarray,
                   X_true: jnp.ndarray,
                   X_ctrl: jnp.ndarray) -> jnp.ndarray:
    """
    Perturbation Direction Consistency Loss.
    
    Ensures the predicted perturbation direction matches the true direction.
    
    Args:
        X_pred: (N, G) predicted perturbed expression
        X_true: (N, G) true perturbed expression
        X_ctrl: (M, G) control expression
    
    Returns:
        loss: 1 - cosine_similarity (so lower is better)
    """
    ctrl_mean = jnp.mean(X_ctrl, axis=0)
    
    delta_pred = jnp.mean(X_pred, axis=0) - ctrl_mean
    delta_true = jnp.mean(X_true, axis=0) - ctrl_mean
    
    # Cosine similarity
    cos_sim = jnp.dot(delta_pred, delta_true) / (
        jnp.linalg.norm(delta_pred) * jnp.linalg.norm(delta_true) + 1e-10
    )
    
    # Loss: 1 - similarity (so 0 is perfect alignment)
    loss = 1.0 - cos_sim
    
    return loss


# ============================================================================
# Helper: Build PPI adjacency from mask
# ============================================================================

def ppi_mask_to_adjacency(ppi_mask: jnp.ndarray, 
                          gene_level_mask: bool = True) -> jnp.ndarray:
    """
    Convert PPI mask (possibly chunk-level) to adjacency matrix.
    
    Args:
        ppi_mask: Binary mask
        gene_level_mask: If True, mask is gene-level; if False, chunk-level
    
    Returns:
        adjacency: Symmetric adjacency matrix
    """
    # Ensure symmetric
    adj = (ppi_mask + ppi_mask.T) / 2
    adj = jnp.where(adj > 0, 1.0, 0.0)
    
    # Remove self-loops for Laplacian
    adj = adj - jnp.diag(jnp.diag(adj))
    
    return adj


# ============================================================================
# JIT-compiled versions for efficiency
# ============================================================================

@partial(jax.jit, static_argnames=['use_correlation'])
def jit_gene_correlation_loss(X_pred, X_true, use_correlation=True):
    return gene_correlation_loss(X_pred, X_true, use_correlation)


@partial(jax.jit, static_argnames=['use_correlation'])
def jit_ppi_masked_correlation_loss(X_pred, X_true, ppi_mask, use_correlation=True):
    return ppi_masked_correlation_loss(X_pred, X_true, ppi_mask, use_correlation)


@partial(jax.jit, static_argnames=['n_clusters', 'use_correlation'])
def jit_cluster_correlation_loss(X_pred, X_true, cluster_labels, n_clusters=128, use_correlation=True):
    return cluster_correlation_loss(X_pred, X_true, cluster_labels, n_clusters, use_correlation)


@partial(jax.jit, static_argnames=['epsilon', 'n_sinkhorn_iters'])
def jit_pathway_ot_loss(X_pred, X_true, pathway_matrix, epsilon=0.1, n_sinkhorn_iters=50):
    return pathway_ot_loss(X_pred, X_true, pathway_matrix, epsilon, n_sinkhorn_iters)


@partial(jax.jit, static_argnames=['normalized_laplacian'])
def jit_graph_smoothness_loss(X_pred, ppi_adjacency, normalized_laplacian=True):
    return graph_smoothness_loss(X_pred, ppi_adjacency, normalized_laplacian)


@jax.jit
def jit_direction_loss(X_pred, X_true, X_ctrl):
    return direction_loss(X_pred, X_true, X_ctrl)


# ============================================================================
# Test function
# ============================================================================

def test_losses():
    """Test all loss functions with dummy data."""
    print("Testing gene-level loss functions...")
    
    # Create dummy data
    rng = np.random.default_rng(42)
    N = 100  # cells
    G = 500  # genes
    K = 50   # pathways
    
    X_pred = jnp.array(rng.normal(0, 1, (N, G)), dtype=jnp.float32)
    X_true = jnp.array(rng.normal(0, 1, (N, G)), dtype=jnp.float32)
    X_ctrl = jnp.array(rng.normal(0, 1, (N, G)), dtype=jnp.float32)
    
    # Random PPI mask (sparse)
    ppi_mask = jnp.array(rng.random((G, G)) < 0.1, dtype=jnp.float32)
    ppi_mask = (ppi_mask + ppi_mask.T) / 2
    
    # Random pathway matrix
    pathway_matrix = jnp.array(rng.random((K, G)) < 0.05, dtype=jnp.float32)
    
    print("\n1. Gene Correlation Loss:")
    corr_loss = gene_correlation_loss(X_pred, X_true)
    print(f"   Full correlation loss: {corr_loss:.6f}")
    
    masked_corr_loss = ppi_masked_correlation_loss(X_pred, X_true, ppi_mask)
    print(f"   PPI-masked correlation loss: {masked_corr_loss:.6f}")
    
    print("\n2. Pathway OT Loss:")
    ot_loss = pathway_ot_loss(X_pred, X_true, pathway_matrix)
    print(f"   Pathway OT loss: {ot_loss:.6f}")
    
    mmd_loss = pathway_mmd_loss(X_pred, X_true, pathway_matrix)
    print(f"   Pathway MMD loss: {mmd_loss:.6f}")
    
    print("\n3. Graph Smoothness Loss:")
    ppi_adj = ppi_mask_to_adjacency(ppi_mask)
    smooth_loss = graph_smoothness_loss(X_pred, ppi_adj)
    print(f"   Graph smoothness loss: {smooth_loss:.6f}")
    
    delta_smooth = delta_graph_smoothness_loss(X_pred, X_ctrl, ppi_adj)
    print(f"   Delta graph smoothness: {delta_smooth:.6f}")
    
    print("\n4. Direction Loss:")
    dir_loss = direction_loss(X_pred, X_true, X_ctrl)
    print(f"   Direction loss: {dir_loss:.6f}")
    
    print("\n5. Combined Loss:")
    total, loss_dict = combined_gene_loss(
        X_pred, X_true,
        ppi_mask=ppi_mask,
        ppi_adjacency=ppi_adj,
        pathway_matrix=pathway_matrix,
        X_ctrl=X_ctrl,
        lambda_corr=0.1,
        lambda_ot=0.1,
        lambda_smooth=0.01
    )
    print(f"   Total gene loss: {total:.6f}")
    for k, v in loss_dict.items():
        print(f"   - {k}: {v:.6f}")
    
    print("\n[OK] All tests passed!")


if __name__ == '__main__':
    test_losses()
