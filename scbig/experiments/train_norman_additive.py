#!/usr/bin/env python
"""Train scBIG on the Norman additive GRC split.

This is the open-source entrypoint for the released Norman additive result.
It intentionally exposes only the `norman_additive_grc` dataset.
"""

import os
import sys
import argparse
import re
import glob
import random as py_random
from pathlib import Path

# Add the repository and optional CellFlow backend checkout to path.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELLFLOW_SRC = ""
CELLFLOW_SRC = Path(os.environ.get("SCBIG_CELLFLOW_SRC", DEFAULT_CELLFLOW_SRC)).expanduser()
sys.path.insert(0, str(REPO_ROOT))
if CELLFLOW_SRC.exists():
    sys.path.insert(0, str(CELLFLOW_SRC))

import pickle
import json
import time
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import pearsonr
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

import torch
import ot
from sklearn.cluster import KMeans

# JAX
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn
from flax.training import train_state

print(f"JAX Platform: {jax.default_backend()}")
print(f"JAX Devices: {jax.devices()}")

if not hasattr(jax.core, 'ConcreteArray'):
    jax.core.ConcreteArray = jax.core.ShapedArray

import cellflow

# GCAE encoder and biological-prior utilities
from scbig.models.gcae import (
    GCAEWrapper,
    GCAE_CONFIGS,
    get_gcae_for_config,
    load_ppi_network,
    build_ppi_attention_mask,
    load_pathway_gene_matrix,
    load_genecompass_embeddings,
)

from scbig.losses.gene_losses import (
    gene_correlation_loss,
    ppi_masked_correlation_loss,
    cluster_correlation_loss,
    pathway_ot_loss,
    graph_smoothness_loss,
    direction_loss,
    ppi_mask_to_adjacency,
)
from scipy.stats import spearmanr, ranksums


# ============================================================================
# Training-time metric helpers
# ============================================================================

PRECOMPUTED_DE_PATHS = {}

CONTROL_MATCH_CANDIDATES = ('cell_line',)


def load_precomputed_de_genes(dataset_name, gene_names, top_n=20):
    """Load precomputed DE genes from JSON file and convert to indices.
    
    Args:
        dataset_name: Name of dataset to look up file path
        gene_names: List of gene names in the dataset (for name->index mapping)
        top_n: Number of top DE genes to use (default 20)
        
    Returns:
        Dict mapping condition -> array of gene indices
    """
    if dataset_name not in PRECOMPUTED_DE_PATHS:
        print(f"  No precomputed DE genes for {dataset_name}")
        return None
    
    de_path = PRECOMPUTED_DE_PATHS[dataset_name]
    if not os.path.exists(de_path):
        print(f"  DE genes file not found: {de_path}")
        return None
    
    import json
    with open(de_path, 'r') as f:
        de_data = json.load(f)
    
    print(f"  Loading precomputed DE genes from {de_path}")
    print(f"  Conditions in file: {len(de_data)}")
    
    # Build gene name -> index mapping
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    
    de_genes_dict = {}
    
    for condition, de_info in de_data.items():
        # Handle both rich DE dictionaries and direct lists of gene names.
        
        if isinstance(de_info, dict):
            # Norman format - use top20 or top50
            if 'top20' in de_info:
                de_gene_names = de_info['top20'][:top_n]
            elif 'top50' in de_info:
                de_gene_names = de_info['top50'][:top_n]
            elif 'all' in de_info:
                de_gene_names = de_info['all'][:top_n]
            else:
                continue
        elif isinstance(de_info, list):
            de_gene_names = de_info[:top_n]
        else:
            continue
        
        # Convert gene names to indices
        indices = []
        for gene in de_gene_names:
            if gene in gene_to_idx:
                indices.append(gene_to_idx[gene])
        
        if indices:
            de_genes_dict[condition] = np.array(indices)
    
    print(f"  Loaded DE genes for {len(de_genes_dict)} conditions")
    return de_genes_dict


def to_dense_array(x):
    """Convert sparse/dense matrices to a dense numpy array without changing shape semantics."""
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def build_obs_focus_mask(adata, focus_cell_line=None, controls_only=False):
    """Build a boolean mask for optional oversampling of a focused training subset."""
    use_focus = focus_cell_line is not None or controls_only
    if not use_focus:
        return None

    mask = np.ones(adata.n_obs, dtype=bool)

    if focus_cell_line is not None:
        if 'cell_line' not in adata.obs.columns:
            raise ValueError("focus_cell_line requested but 'cell_line' column is missing from adata.obs")
        mask &= adata.obs['cell_line'].astype(str).to_numpy() == str(focus_cell_line)

    if controls_only:
        if 'is_control' not in adata.obs.columns:
            raise ValueError("focus_controls_only requested but 'is_control' column is missing from adata.obs")
        control_values = adata.obs['is_control']
        if hasattr(control_values, 'to_numpy'):
            control_values = control_values.to_numpy()
        mask &= np.asarray(control_values).astype(bool)

    return mask


def build_encoder_eval_mask(adata, target_cell_line=None, controls_only=True):
    """Build a tolerant mask for selecting encoder validation cells."""
    if adata is None:
        return None

    mask = np.ones(adata.n_obs, dtype=bool)

    if target_cell_line is not None:
        if 'cell_line' not in adata.obs.columns:
            return None
        mask &= adata.obs['cell_line'].astype(str).to_numpy() == str(target_cell_line)

    if controls_only:
        if 'is_control' not in adata.obs.columns:
            return None
        control_values = adata.obs['is_control']
        if hasattr(control_values, 'to_numpy'):
            control_values = control_values.to_numpy()
        mask &= np.asarray(control_values).astype(bool)

    return mask


def sample_encoder_eval_subset(adata, split_name, target_cell_line=None,
                               controls_only=True, max_cells=300, seed=0):
    """Sample a fixed encoder-validation subset from an AnnData split."""
    mask = build_encoder_eval_mask(
        adata,
        target_cell_line=target_cell_line,
        controls_only=controls_only
    )
    if mask is None:
        return None

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None

    if max_cells is not None and max_cells > 0 and indices.size > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_cells, replace=False))

    subset = adata[indices]
    return {
        'split': split_name,
        'indices': indices,
        'n_cells': int(indices.size),
        'controls_only': bool(controls_only),
        'target_cell_line': target_cell_line,
        'X': np.asarray(to_dense_array(subset.X), dtype=np.float32),
    }


def prepare_encoder_eval_subset(adata_train, adata_val, adata_test,
                                target_cell_line=None, controls_only=True,
                                max_cells=300, seed=0):
    """
    Build the encoder validation subset.

    Priority:
    1. val split
    2. held-out train subset (removed from encoder pretraining only)
    3. test split as a last-resort fallback only

    If no focused clean subset exists, fall back to a generic capped validation
    subset when possible.
    """
    strict_candidates = [
        ('val', adata_val, False),
        ('train_holdout', adata_train, True),
        ('test_fallback', adata_test, False),
    ]
    for offset, (split_name, split_adata, uses_train_holdout) in enumerate(strict_candidates):
        sampled = sample_encoder_eval_subset(
            split_adata,
            split_name=split_name,
            target_cell_line=target_cell_line,
            controls_only=controls_only,
            max_cells=max_cells,
            seed=seed + offset,
        )
        if sampled is not None:
            sampled['uses_train_holdout'] = uses_train_holdout
            sampled['label'] = (
                f"{split_name}:{target_cell_line or 'any'}:"
                f"{'control' if controls_only else 'all'} ({sampled['n_cells']} cells)"
            )
            return sampled

    fallback_split = adata_val if adata_val is not None else adata_test
    fallback_name = 'val' if adata_val is not None else 'test_fallback'
    if fallback_split is not None:
        fallback_indices = np.arange(fallback_split.n_obs)
        if max_cells is not None and max_cells > 0 and fallback_indices.size > max_cells:
            rng = np.random.default_rng(seed + 97)
            fallback_indices = np.sort(rng.choice(fallback_indices, size=max_cells, replace=False))
        fallback_subset = fallback_split[fallback_indices]
        return {
            'split': fallback_name,
            'indices': fallback_indices,
            'n_cells': int(fallback_indices.size),
            'controls_only': False,
            'target_cell_line': None,
            'uses_train_holdout': False,
            'label': f"{fallback_name}:fallback_all ({len(fallback_indices)} cells)",
            'fallback_reason': (
                f"No {target_cell_line or 'target'} "
                f"{'control ' if controls_only else ''}cells found in val/train/test."
            ),
            'X': np.asarray(to_dense_array(fallback_subset.X), dtype=np.float32),
        }

    return None


def augment_matrix_with_focus_rows(X, focus_mask, multiplier):
    """Duplicate a focused subset inside a dense training matrix."""
    X_dense = to_dense_array(X)
    if focus_mask is None or multiplier <= 1:
        return X_dense

    focus_rows = X_dense[focus_mask]
    if focus_rows.shape[0] == 0:
        return X_dense

    pieces = [X_dense] + [focus_rows] * (multiplier - 1)
    return np.concatenate(pieces, axis=0)


def augment_adata_with_focus_rows(adata, focus_mask, multiplier, suffix_prefix='focusdup'):
    """Duplicate a focused subset inside an AnnData object while keeping obs names unique."""
    if focus_mask is None or multiplier <= 1:
        return adata

    focus_subset = adata[focus_mask].copy()
    if focus_subset.n_obs == 0:
        return adata

    pieces = [adata]
    for rep in range(multiplier - 1):
        extra = focus_subset.copy()
        extra.obs_names = pd.Index(
            [f"{name}__{suffix_prefix}{rep + 1}" for name in extra.obs_names],
            dtype='object'
        )
        pieces.append(extra)

    return ad.concat(pieces, axis=0, join='outer', merge='same', uns_merge='same')


def normalize_obs_value(value):
    """Normalize pandas/numpy scalar values into a stable string key."""
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def detect_control_match_columns(adata_ref, adata_ctrl):
    """Detect obs columns that should route controls for matched-context evaluation."""
    if adata_ref is None or adata_ctrl is None:
        return []
    return [
        column for column in CONTROL_MATCH_CANDIDATES
        if column in adata_ref.obs.columns and column in adata_ctrl.obs.columns
    ]


def build_control_group_stats(adata_ctrl, match_columns=None):
    """
    Precompute global and per-group control means.

    For datasets with matched contexts, evaluate each condition against the
    matching control pool instead of mixing all controls.
    """
    match_columns = list(match_columns or [])
    control_expr = to_dense_array(adata_ctrl.X)
    if control_expr.ndim == 1:
        control_expr = control_expr[None, :]

    stats = {
        'match_columns': match_columns,
        'global_expr': control_expr,
        'global_mean': np.asarray(control_expr.mean(axis=0)).flatten(),
        'groups': {},
    }

    if not match_columns:
        return stats

    group_indices = (
        adata_ctrl.obs[match_columns]
        .astype(str)
        .groupby(match_columns, sort=True, observed=False)
        .indices
    )

    for group_key, indices in group_indices.items():
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        normalized_key = tuple(normalize_obs_value(value) for value in group_key)
        idx_arr = np.asarray(list(indices), dtype=int)
        group_expr = control_expr[idx_arr]
        stats['groups'][normalized_key] = {
            'expr': group_expr,
            'mean': np.asarray(group_expr.mean(axis=0)).flatten(),
            'n_cells': int(group_expr.shape[0]),
        }

    return stats


def get_control_stats_for_obs(obs_subset, control_stats):
    """Return the matched control group statistics for a condition obs subset."""
    if not control_stats:
        return None

    match_columns = control_stats.get('match_columns', [])
    if not match_columns:
        return {
            'expr': control_stats['global_expr'],
            'mean': control_stats['global_mean'],
            'n_cells': int(control_stats['global_expr'].shape[0]),
            'matched': False,
        }

    key = tuple(normalize_obs_value(obs_subset[column].iloc[0]) for column in match_columns)
    if key in control_stats['groups']:
        group_stats = control_stats['groups'][key]
        return {
            'expr': group_stats['expr'],
            'mean': group_stats['mean'],
            'n_cells': group_stats['n_cells'],
            'matched': True,
        }

    return {
        'expr': control_stats['global_expr'],
        'mean': control_stats['global_mean'],
        'n_cells': int(control_stats['global_expr'].shape[0]),
        'matched': False,
    }


def add_core_metric_aliases(metrics):
    """
    Add reviewer-friendly aliases.

    Note: scBIG's core all/de metrics in this script are already computed on
    control-relative deltas, so the plain and delta aliases intentionally share
    the same values.
    """
    base_pairs = [
        ('Pearson_all', 'r_all'),
        ('Pearson_de', 'r_de'),
        ('Acc_all', 'acc_all'),
        ('Acc_de', 'acc_de'),
        ('Pearson_delta_all', 'r_all'),
        ('Pearson_delta_de', 'r_de'),
        ('Acc_delta_all', 'acc_all'),
        ('Acc_delta_de', 'acc_de'),
    ]
    for alias_key, source_key in base_pairs:
        if source_key in metrics and alias_key not in metrics:
            metrics[alias_key] = metrics[source_key]
    return metrics


def compute_top_de_genes_inline(adata_test, control_reference, top_n=20):
    """Compute top N DE genes by absolute mean difference for each condition."""
    top_de_genes = {}
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() == 0:
            continue
        obs_subset = adata_test.obs[cond_mask]
        if isinstance(control_reference, dict) and 'global_mean' in control_reference:
            control_mean = get_control_stats_for_obs(obs_subset, control_reference)['mean']
        else:
            control_mean = np.asarray(control_reference).flatten()
        cond_mean = adata_test.X[cond_mask].mean(axis=0)
        if sparse.issparse(cond_mean):
            cond_mean = np.asarray(cond_mean).flatten()
        delta = np.abs(cond_mean - control_mean)
        top_indices = np.argsort(delta)[-top_n:]
        top_de_genes[condition] = top_indices
    return top_de_genes


def compute_de_genes_wilcoxon_inline(adata_test, control_reference, p_threshold=0.05):
    """Compute DE genes using Wilcoxon rank-sum test."""
    de_genes = {}
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() == 0:
            continue
        obs_subset = adata_test.obs[cond_mask]
        if isinstance(control_reference, dict) and 'global_expr' in control_reference:
            control_expr = get_control_stats_for_obs(obs_subset, control_reference)['expr']
        else:
            control_expr = to_dense_array(control_reference)
        
        cond_expr = adata_test.X[cond_mask]
        if sparse.issparse(cond_expr):
            cond_expr = cond_expr.toarray()
        
        pvalues = []
        for gene_idx in range(adata_test.shape[1]):
            ctrl_vals = control_expr[:, gene_idx]
            cond_vals = cond_expr[:, gene_idx]
            if np.std(ctrl_vals) == 0 and np.std(cond_vals) == 0:
                pvalues.append(1.0)
                continue
            try:
                _, pval = ranksums(cond_vals, ctrl_vals)
                pvalues.append(pval)
            except:
                pvalues.append(1.0)
        
        pvalues = np.array(pvalues)
        adjusted_pvals = np.minimum(pvalues * len(pvalues), 1.0)
        significant_genes = np.where(adjusted_pvals < p_threshold)[0]
        
        de_genes[condition] = {
            'significant_indices': significant_genes,
            'n_significant': len(significant_genes)
        }
    return de_genes


def compute_l2_mean_inline(predictions, adata_test):
    """Compute L2 mean distance (perturbation-level)."""
    l2_distances = []
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() == 0:
            continue
        true_mean = adata_test.X[cond_mask].mean(axis=0)
        if sparse.issparse(true_mean):
            true_mean = np.asarray(true_mean).flatten()
        if condition not in predictions:
            continue
        pred_mean = predictions[condition]
        l2 = np.linalg.norm(true_mean - pred_mean)
        l2_distances.append(l2)
    
    return np.mean(l2_distances) if l2_distances else 0.0


def compute_mse_mae_inline(predictions, adata_test):
    """Compute MSE and MAE at pseudobulk level."""
    mse_list, mae_list = [], []
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() == 0:
            continue
        true_mean = adata_test.X[cond_mask].mean(axis=0)
        if sparse.issparse(true_mean):
            true_mean = np.asarray(true_mean).flatten()
        if condition not in predictions:
            continue
        pred_mean = predictions[condition]
        diff = true_mean - pred_mean
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        mse_list.append(mse)
        mae_list.append(mae)
    
    return {
        'MSE': np.mean(mse_list) if mse_list else 0.0,
        'MAE': np.mean(mae_list) if mae_list else 0.0,
    }


def compute_de_spearman_sig_inline(predictions, adata_test, control_reference, de_genes_wilcoxon):
    """Compute Spearman correlation on significant DE genes."""
    spearman_list = []
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    eps = 1e-10
    
    for condition in conditions:
        if condition not in de_genes_wilcoxon:
            continue
        sig_genes = de_genes_wilcoxon[condition]['significant_indices']
        if not isinstance(sig_genes, np.ndarray):
            sig_genes = np.asarray(sig_genes)
        sig_genes = sig_genes.astype(int).flatten()
        if len(sig_genes) < 5:
            continue
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() == 0:
            continue
        obs_subset = adata_test.obs[cond_mask]
        true_mean = to_dense_array(adata_test.X[cond_mask].mean(axis=0)).reshape(-1)
        if condition not in predictions:
            continue
        pred_mean = to_dense_array(predictions[condition]).reshape(-1)
        if isinstance(control_reference, dict) and 'global_expr' in control_reference:
            ctrl = to_dense_array(get_control_stats_for_obs(obs_subset, control_reference)['mean']).reshape(-1)
        else:
            ctrl = to_dense_array(control_reference).reshape(-1)

        if true_mean.size == 0 or pred_mean.size == 0 or ctrl.size == 0:
            continue
        if not (true_mean.size == pred_mean.size == ctrl.size):
            continue
        
        true_lfc = np.log2((true_mean[sig_genes] + eps) / (ctrl[sig_genes] + eps))
        pred_lfc = np.log2((pred_mean[sig_genes] + eps) / (ctrl[sig_genes] + eps))
        
        valid_mask = np.isfinite(true_lfc) & np.isfinite(pred_lfc)
        if valid_mask.sum() < 5:
            continue
        rho, _ = spearmanr(true_lfc[valid_mask], pred_lfc[valid_mask])
        if not np.isnan(rho):
            spearman_list.append(rho)
    
    return np.mean(spearman_list) if spearman_list else 0.0


def compute_pds_inline(predictions, adata_test, gene_names):
    """Compute Perturbation Discrimination Score (PDS)."""
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    
    true_pseudobulks = {}
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() > 0:
            mean_val = adata_test.X[cond_mask].mean(axis=0)
            if sparse.issparse(mean_val):
                mean_val = np.asarray(mean_val).flatten()
            true_pseudobulks[condition] = mean_val
    
    # Get perturbation genes for exclusion
    pert_genes_by_cond = {}
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() > 0:
            pert1 = adata_test.obs.loc[cond_mask, 'pert1'].iloc[0]
            pert2 = adata_test.obs.loc[cond_mask, 'pert2'].iloc[0]
            pert_genes = set()
            if pert1 != 'ctrl' and pert1 in gene_names:
                pert_genes.add(gene_names.index(pert1))
            if pert2 != 'ctrl' and pert2 in gene_names:
                pert_genes.add(gene_names.index(pert2))
            pert_genes_by_cond[condition] = pert_genes
    
    N = len(true_pseudobulks)
    if N < 2:
        return 0.0
    
    pds_list = []
    for p_cond in predictions.keys():
        if p_cond not in true_pseudobulks:
            continue
        pred_mean = predictions[p_cond]
        exclude_genes = pert_genes_by_cond.get(p_cond, set())
        n_genes = len(pred_mean)
        include_mask = np.ones(n_genes, dtype=bool)
        for g_idx in exclude_genes:
            if g_idx < n_genes:
                include_mask[g_idx] = False
        
        distances = []
        target_list = []
        for t_cond, true_pb in true_pseudobulks.items():
            d = np.sum(np.abs(pred_mean[include_mask] - true_pb[include_mask]))
            distances.append(d)
            target_list.append(t_cond)
        
        sorted_indices = np.argsort(distances)
        sorted_targets = [target_list[i] for i in sorted_indices]
        
        try:
            rank_p = sorted_targets.index(p_cond) + 1
        except ValueError:
            continue
        
        pds_p = 1 - (rank_p - 1) / (N - 1)
        pds_list.append(pds_p)
    
    return np.mean(pds_list) if pds_list else 0.0


def compute_pearson_delta_inline(predictions, adata_test, control_reference, train_pert_means=None, top_n_variance=None):
    """Compute Pearson delta metric."""
    pearson_list = []
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() == 0:
            continue
        obs_subset = adata_test.obs[cond_mask]
        if isinstance(control_reference, dict) and 'global_mean' in control_reference:
            default_train_mean = get_control_stats_for_obs(obs_subset, control_reference)['mean']
        else:
            default_train_mean = np.asarray(control_reference).flatten()
        if train_pert_means and condition in train_pert_means:
            train_mean = np.asarray(train_pert_means[condition]).flatten()
        else:
            train_mean = default_train_mean
        
        true_cells = adata_test.X[cond_mask]
        if sparse.issparse(true_cells):
            true_cells = true_cells.toarray()
        
        if condition not in predictions:
            continue
        
        # For predictions, we have pseudobulk mean
        pred_mean = predictions[condition]
        true_mean = true_cells.mean(axis=0)
        
        if top_n_variance:
            true_residuals = true_cells - train_mean
            variances = np.var(true_residuals, axis=0)
            top_genes = np.argsort(variances)[-top_n_variance:]
        else:
            top_genes = np.arange(len(pred_mean))
        
        delta_true = true_mean[top_genes] - train_mean[top_genes]
        delta_pred = pred_mean[top_genes] - train_mean[top_genes]
        
        r = safe_pearson(delta_true, delta_pred)
        if np.isfinite(r):
            pearson_list.append(r)
    
    return safe_mean(pearson_list)


def safe_pearson(x, y):
    """Pearson wrapper that returns NaN for degenerate inputs instead of raising or propagating invalid values."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size < 2 or y.size < 2:
        return np.nan
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    r, _ = pearsonr(x, y)
    return r if np.isfinite(r) else np.nan


def safe_mean(values, default=0.0):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float(default)


def print_delta_metric_block(metrics, split='test', indent=''):
    """Print the 8 core metrics used in reviewer-facing tables and logs."""
    metric_lines = [
        ('Pearson_all', metrics.get(f'{split}_Pearson_all', metrics.get(f'{split}_r_all', 0.0))),
        ('Pearson_de', metrics.get(f'{split}_Pearson_de', metrics.get(f'{split}_r_de', 0.0))),
        ('Acc_all', metrics.get(f'{split}_Acc_all', metrics.get(f'{split}_acc_all', 0.0))),
        ('Acc_de', metrics.get(f'{split}_Acc_de', metrics.get(f'{split}_acc_de', 0.0))),
        ('Pearson_delta_all', metrics.get(f'{split}_Pearson_delta_all', metrics.get(f'{split}_r_all', 0.0))),
        ('Pearson_delta_de', metrics.get(f'{split}_Pearson_delta_de', metrics.get(f'{split}_r_de', 0.0))),
        ('Acc_delta_all', metrics.get(f'{split}_Acc_delta_all', metrics.get(f'{split}_acc_all', 0.0))),
        ('Acc_delta_de', metrics.get(f'{split}_Acc_delta_de', metrics.get(f'{split}_acc_de', 0.0))),
    ]
    for label, value in metric_lines:
        print(f"{indent}{label:<18} {value:.4f}")


def print_eval_metric_block(metrics, split='test', indent=''):
    """Print core delta metrics first, then optional coverage and extra metrics."""
    print_delta_metric_block(metrics, split=split, indent=indent)

    n_pred = metrics.get(f'{split}_n_conditions_predicted')
    n_total = metrics.get(f'{split}_n_conditions_total')
    coverage = metrics.get(f'{split}_prediction_coverage')
    if n_pred is not None and n_total is not None and coverage is not None:
        print(f"{indent}Coverage:          {n_pred}/{n_total} ({coverage:.1%})")

    extra_parts = []
    l2_key = f'{split}_L2_mean'
    pds_key = f'{split}_PDS'
    if l2_key in metrics:
        extra_parts.append(f"L2_mean={metrics.get(l2_key, 0.0):.4f}")
    if pds_key in metrics:
        extra_parts.append(f"PDS={metrics.get(pds_key, 0.0):.4f}")
    if extra_parts:
        print(f"{indent}{', '.join(extra_parts)}")


def compute_full_11_metrics(predictions, adata_test, adata_train_control, control_mean, gene_names, top_de_genes=None):
    """Compute all 11 evaluation metrics."""
    metrics = {}
    
    control_match_columns = detect_control_match_columns(adata_test, adata_train_control)
    control_stats = build_control_group_stats(adata_train_control, control_match_columns)
    
    # Compute top DE genes if not provided
    if top_de_genes is None:
        top_de_genes = compute_top_de_genes_inline(adata_test, control_stats, top_n=20)
    
    # 1-4: r_all, r_de, acc_all, acc_de
    r_all_list, r_de_list = [], []
    acc_all_list, acc_de_list = [], []
    
    conditions = adata_test.obs['condition'].unique()
    conditions = [c for c in conditions if c != 'control']
    
    for condition in conditions:
        cond_mask = (adata_test.obs['condition'] == condition).values
        if cond_mask.sum() == 0:
            continue
        obs_subset = adata_test.obs[cond_mask]
        matched_control_stats = get_control_stats_for_obs(obs_subset, control_stats)
        control_mean = matched_control_stats['mean']
        true_mean = adata_test.X[cond_mask].mean(axis=0)
        if sparse.issparse(true_mean):
            true_mean = np.asarray(true_mean).flatten()
        if condition not in predictions:
            continue
        pred_mean = predictions[condition]
        
        true_delta = true_mean - control_mean
        pred_delta = pred_mean - control_mean
        
        r = safe_pearson(true_delta, pred_delta)
        if np.isfinite(r):
            r_all_list.append(r)
        
        if condition in top_de_genes:
            de_idx = top_de_genes[condition]
            r_de = safe_pearson(true_delta[de_idx], pred_delta[de_idx])
            if np.isfinite(r_de):
                r_de_list.append(r_de)
        
        correct = ((true_delta > 0) == (pred_delta > 0)).astype(float)
        acc_all_list.append(correct.mean())
        
        if condition in top_de_genes:
            de_idx = top_de_genes[condition]
            correct_de = ((true_delta[de_idx] > 0) == (pred_delta[de_idx] > 0)).astype(float)
            acc_de_list.append(correct_de.mean())
    
    metrics['r_all'] = safe_mean(r_all_list)
    metrics['r_de'] = safe_mean(r_de_list)
    metrics['acc_all'] = safe_mean(acc_all_list)
    metrics['acc_de'] = safe_mean(acc_de_list)
    
    # 5: L2_mean
    metrics['L2_mean'] = compute_l2_mean_inline(predictions, adata_test)
    
    # 6-7: MSE, MAE
    mse_mae = compute_mse_mae_inline(predictions, adata_test)
    metrics.update(mse_mae)
    
    # 8: DE_Spearman_Sig
    de_genes_wilcoxon = compute_de_genes_wilcoxon_inline(adata_test, control_stats, p_threshold=0.05)
    metrics['DE_Spearman_Sig'] = compute_de_spearman_sig_inline(predictions, adata_test, control_stats, de_genes_wilcoxon)
    
    # 9: PDS
    metrics['PDS'] = compute_pds_inline(predictions, adata_test, gene_names)
    
    # 10-11: Pearson_delta, Pearson_delta_20
    metrics['Pearson_delta'] = compute_pearson_delta_inline(predictions, adata_test, control_stats, None, None)
    metrics['Pearson_delta_20'] = compute_pearson_delta_inline(predictions, adata_test, control_stats, None, 20)
    add_core_metric_aliases(metrics)
    
    return metrics


# ============================================================================
# Gene Loss Configurations
# ============================================================================
GENE_LOSS_CONFIGS = {
    'no_gene_loss': {
        'description': 'Baseline: No gene-level loss',
        'lambda_corr': 0.0,
        'lambda_ot': 0.0,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    'corr_only': {
        'description': 'Correlation consistency loss only',
        'lambda_corr': 0.1,
        'lambda_ot': 0.0,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    'ot_only': {
        'description': 'Pathway OT loss only',
        'lambda_corr': 0.0,
        'lambda_ot': 0.1,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    'corr_ot': {
        'description': 'Correlation + Pathway OT losses',
        'lambda_corr': 0.1,
        'lambda_ot': 0.1,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    # Loss weight ablation configs
    'corr_ot_c01_o01': {
        'description': 'Corr+OT: lambda_corr=0.01, lambda_ot=0.01',
        'lambda_corr': 0.01,
        'lambda_ot': 0.01,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    'corr_ot_c01_o1': {
        'description': 'Corr+OT: lambda_corr=0.01, lambda_ot=0.1',
        'lambda_corr': 0.01,
        'lambda_ot': 0.1,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    'corr_ot_c1_o01': {
        'description': 'Corr+OT: lambda_corr=0.1, lambda_ot=0.01',
        'lambda_corr': 0.1,
        'lambda_ot': 0.01,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    # Cluster-level correlation loss (efficient version)
    'grc_corr_pathway_ot': {
        'description': 'GRC cluster correlation + pathway OT losses',
        'lambda_corr': 0.1,
        'lambda_ot': 0.1,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
        'use_cluster_corr': True,
    },
    'cluster_corr_ot': {
        'description': 'Alias for grc_corr_pathway_ot',
        'lambda_corr': 0.1,
        'lambda_ot': 0.1,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
        'use_cluster_corr': True,
    },
    'cluster_corr_only': {
        'description': 'Cluster-level Correlation loss only (efficient)',
        'lambda_corr': 0.1,
        'lambda_ot': 0.0,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
        'use_cluster_corr': True,
    },
    'corr_ot_c1_o1': {
        'description': 'Corr+OT: lambda_corr=1.0, lambda_ot=1.0',
        'lambda_corr': 1.0,
        'lambda_ot': 1.0,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    'corr_ot_c05_o05': {
        'description': 'Corr+OT: lambda_corr=0.05, lambda_ot=0.05',
        'lambda_corr': 0.05,
        'lambda_ot': 0.05,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': False,
    },
    'ppi_corr_ot': {
        'description': 'PPI-masked Correlation + Pathway OT losses',
        'lambda_corr': 0.1,
        'lambda_ot': 0.1,
        'lambda_smooth': 0.0,
        'lambda_dir': 0.0,
        'use_ppi_masked_corr': True,
    },
    'full_gene_loss': {
        'description': 'All gene-level losses',
        'lambda_corr': 0.1,
        'lambda_ot': 0.1,
        'lambda_smooth': 0.01,
        'lambda_dir': 0.05,
        'use_ppi_masked_corr': True,
    },
}


# ============================================================================
# Command-line arguments
# ============================================================================
parser = argparse.ArgumentParser(description='scBIG Norman additive training')
parser.add_argument('--encoder_config', type=str, default='genecompass_pe_module',
                    choices=['genecompass_pe_module', 'genecompass_pe_perceiver'],
                    help='Released encoder configuration')
parser.add_argument('--gene_loss_config', type=str, default='grc_corr_pathway_ot',
                    choices=['grc_corr_pathway_ot', 'cluster_corr_ot'],
                    help='Released gene-level loss configuration')
parser.add_argument('--dataset', type=str, default='norman_additive_grc',
                    choices=['norman_additive_grc'],
                    help='Dataset to use')
parser.add_argument('--use_reordered', action='store_true', default=True,
                    help='Use reordered (cluster-sorted) gene data')
parser.add_argument('--no_reorder', action='store_true', default=False,
                    help='Disable gene reordering (use original order)')
parser.add_argument('--cluster_type', type=str, default='grc',
                    choices=['grc'],
                    help='Use the pre-sorted GRC files in the Norman additive dataset')
parser.add_argument('--n_iterations', type=int, default=500000,
                    help='Number of training iterations')
parser.add_argument('--valid_freq', type=int, default=100000,
                    help='Validation frequency')
parser.add_argument('--gene_loss_freq', type=int, default=1000,
                    help='Frequency of gene loss gradient updates (0=disabled)')
parser.add_argument('--n_gene_loss_samples', type=int, default=3,
                    help='Number of conditions to sample for gene loss')
parser.add_argument('--gene_loss_batch_size', type=int, default=64,
                    help='Batch size for gene loss computation')
parser.add_argument('--control_identity_weight', type=float, default=0.0,
                    help='Auxiliary weight for clean+ctrl->clean identity loss during gene-loss updates')
parser.add_argument('--control_identity_batch_size', type=int, default=None,
                    help='Batch size for control identity updates (default: gene_loss_batch_size)')
parser.add_argument('--batch_size', type=int, default=int(os.environ.get('CELLFLOW_BATCH_SIZE', '1024')),
                    help='Main flow-matching batch size')
parser.add_argument('--n_pcs', type=int, default=50,
                    help='Number of principal components / latent dimensions')
parser.add_argument('--encoder_batch_size', type=int, default=256,
                    help='Batch size for encoder training')
parser.add_argument('--embed_dim', type=int, default=256,
                    help='Embedding dimension for encoder')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=8,
                    help='Number of attention heads')
parser.add_argument('--pretrained_encoder', type=str, default=None,
                    help='Path to pre-trained encoder .pkl file (skip encoder training)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Custom output directory (overrides default naming)')
parser.add_argument('--data_dir_override', type=str, default=None,
                    help='Override dataset registry data directory')
parser.add_argument('--cellflow_src', type=str, default=str(CELLFLOW_SRC),
                    help='Optional CellFlow backend src/ directory. You can also set SCBIG_CELLFLOW_SRC before launch.')
parser.add_argument('--resource_dir', type=str, default=str(REPO_ROOT / 'resources'),
                    help='Directory containing external resource files such as PPI, pathways, and ESM2 embeddings')
parser.add_argument('--ppi_info_path', type=str, default=None,
                    help='Path to STRING 9606.protein.info.v12.0.txt.gz')
parser.add_argument('--ppi_links_path', type=str, default=None,
                    help='Path to STRING 9606.protein.links.v12.0.txt.gz')
parser.add_argument('--scgpt_path', type=str, default=None,
                    help='Optional scGPT gene embedding pickle')
parser.add_argument('--genecompass_path', type=str,
                    default=str(REPO_ROOT / 'resources' / 'genecompass_gene_embeddings_full.pkl'),
                    help='Path to GeneCompass gene embeddings pickle')
parser.add_argument('--pathway_path', type=str, default=None,
                    help='Path to ReactomePathways.gmt.zip')
parser.add_argument('--esm2_path', type=str, default=None,
                    help='Path to ESM2 perturbation embedding .pt file')
parser.add_argument('--precomputed_de_path', type=str, default=None,
                    help='Optional precomputed Norman additive DE-gene JSON for validation-time metrics')
# Loss weight override arguments (take precedence over gene_loss_config)
parser.add_argument('--lambda_corr', type=float, default=None,
                    help='Override lambda_corr weight (if specified)')
parser.add_argument('--lambda_ot', type=float, default=None,
                    help='Override lambda_ot weight (if specified)')
# Architecture override arguments
parser.add_argument('--num_modules', type=int, default=None,
                    help='Number of learned GCAE module tokens')
parser.add_argument('--num_inducing', type=int, default=None,
                    help='Deprecated alias for --num_modules')
parser.add_argument('--chunk_size', type=int, default=None,
                    help='Override chunk_size for gene chunking (default: use dataset config)')
parser.add_argument('--module_pooling', type=str, default='mean',
                    choices=['mean'],
                    help='Released module pooling strategy')
# Enhanced evaluation and saving arguments
parser.add_argument('--full_metrics', action='store_true', default=False,
                    help='Compute all 11 metrics during evaluation (slower but comprehensive)')
parser.add_argument('--save_best_model', action='store_true', default=True,
                    help='Save best model checkpoint (encoder + flow)')
parser.add_argument('--save_best_predictions', action='store_true', default=False,
                    help='Save predictions CSV for best epoch')
parser.add_argument('--n_cells_per_condition', type=int, default=500,
                    help='Max cells per condition for testing (default: 500)')
parser.add_argument('--skip_checkpoints', action='store_true', default=False,
                    help='Skip saving per-epoch checkpoints to save disk space')
parser.add_argument('--encoder_n_epochs', type=int, default=100,
                    help='Number of epochs for encoder pretraining (default: 100)')
parser.add_argument('--encoder_eval_every_epochs', type=int, default=5,
                    help='Validation interval for encoder pretraining; 0 disables validation')
parser.add_argument('--encoder_eval_max_cells', type=int, default=300,
                    help='Max cells for encoder validation subset (default: 300)')
parser.add_argument('--encoder_eval_cell_line', type=str, default=None,
                    help='Optional preferred cell_line for encoder validation subset')
parser.add_argument('--encoder_eval_controls_only', type=int, default=1, choices=[0, 1],
                    help='Use only clean/control cells for encoder validation when possible (default: 1)')
parser.add_argument('--encoder_early_stop_patience', type=int, default=4,
                    help='Stop encoder pretraining after this many validation checks without improvement; 0 disables')
parser.add_argument('--encoder_early_stop_min_delta', type=float, default=0.0,
                    help='Minimum validation-loss improvement required to reset encoder early stopping')
parser.add_argument('--oversample_cell_line', type=str, default=None,
                    help='Optional cell_line to oversample during training')
parser.add_argument('--oversample_multiplier', type=int, default=1,
                    help='Total weight for the focused subset; 1 disables oversampling')
parser.add_argument('--oversample_controls_only', action='store_true', default=False,
                    help='Only oversample control cells within the focused subset')
parser.add_argument('--oversample_stages', type=str, default='both',
                    choices=['encoder', 'flow', 'both'],
                    help='Apply focused oversampling to encoder pretraining, flow training, or both')
parser.add_argument('--finetune_strategy', type=str, default='flow_only',
                    choices=['flow_only'],
                    help='Released gene-loss finetuning strategy')
parser.add_argument('--resume_from', type=str, default=None,
                    help='Path to a pretrained model directory to resume training from.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for training and evaluation sampling')
parser.add_argument('--skip_encoder_pretrain', action='store_true', default=False,
                    help='Skip encoder pretraining (use when resuming from a pretrained model)')
parser.add_argument('--exit_after_encoder', action='store_true', default=False,
                    help='Exit immediately after encoder load/pretraining, before latent transform and flow setup')
parser.add_argument('--benchmark_full_augmented_steps', type=int, default=0,
                    help='If > 0, benchmark full augmented iterations (flow step + biological loss block) and exit')
parser.add_argument('--benchmark_warmup_steps', type=int, default=3,
                    help='Warmup augmented iterations before timing in benchmark mode')
parser.add_argument('--benchmark_output_path', type=str, default=None,
                    help='Optional JSON path for benchmark_full_augmented_steps output')
parser.add_argument('--dynamic_clustering', action='store_true', default=False,
                    help='Enable dynamic scaffold refresh during training')
parser.add_argument('--dynamic_refresh_start_epoch', type=int, default=2,
                    help='First epoch after which to refresh dynamic clustering')
parser.add_argument('--dynamic_refresh_freq_epochs', type=int, default=1,
                    help='Refresh frequency in epochs once dynamic clustering starts')
parser.add_argument('--dynamic_refresh_max_refreshes', type=int, default=2,
                    help='Maximum number of dynamic clustering refreshes')
parser.add_argument('--dynamic_refresh_n_conditions', type=int, default=64,
                    help='Number of train perturbation conditions sampled per refresh')
parser.add_argument('--dynamic_refresh_max_cells', type=int, default=256,
                    help='Maximum predicted cells per sampled condition for dynamic refresh')
parser.add_argument('--dynamic_static_weight', type=float, default=0.7,
                    help='Weight for static GeneCompass features during dynamic clustering')
parser.add_argument('--dynamic_response_weight', type=float, default=0.3,
                    help='Weight for model-conditioned response signatures during dynamic clustering')
parser.add_argument('--dynamic_ppi_weight', type=float, default=0.3,
                    help='Weight for PPI bonus during dynamic clustering')
parser.add_argument('--dynamic_refresh_history_path', type=str, default=None,
                    help='Optional JSON path to save dynamic clustering refresh history')

args = parser.parse_args()

if args.cellflow_src and Path(args.cellflow_src).expanduser().exists():
    cellflow_src_arg = str(Path(args.cellflow_src).expanduser())
    if cellflow_src_arg not in sys.path:
        sys.path.insert(0, cellflow_src_arg)

if args.precomputed_de_path:
    PRECOMPUTED_DE_PATHS['norman_additive_grc'] = args.precomputed_de_path


# ============================================================================
# Configuration
# ============================================================================
DATASET = args.dataset
ENCODER_CONFIG = args.encoder_config
GENE_LOSS_CONFIG = args.gene_loss_config
CLUSTER_TYPE = args.cluster_type
# Determine if we use reordered data based on cluster_type
if CLUSTER_TYPE == 'none':
    USE_REORDERED = False
else:
    USE_REORDERED = args.use_reordered and not args.no_reorder
NUM_ITERATIONS = args.n_iterations
VALID_FREQ = args.valid_freq
GENE_LOSS_FREQ = args.gene_loss_freq
N_GENE_LOSS_SAMPLES = args.n_gene_loss_samples
GENE_LOSS_BATCH_SIZE = args.gene_loss_batch_size
CONTROL_IDENTITY_WEIGHT = args.control_identity_weight
CONTROL_IDENTITY_BATCH_SIZE = args.control_identity_batch_size or args.gene_loss_batch_size
FINETUNE_STRATEGY = args.finetune_strategy
RESUME_FROM = args.resume_from
SEED = args.seed
OVERSAMPLE_CELL_LINE = args.oversample_cell_line
OVERSAMPLE_MULTIPLIER = args.oversample_multiplier
OVERSAMPLE_CONTROLS_ONLY = args.oversample_controls_only
OVERSAMPLE_STAGES = args.oversample_stages
SKIP_ENCODER_PRETRAIN = args.skip_encoder_pretrain
EXIT_AFTER_ENCODER = args.exit_after_encoder
BENCHMARK_FULL_AUGMENTED_STEPS = args.benchmark_full_augmented_steps
BENCHMARK_WARMUP_STEPS = args.benchmark_warmup_steps
BENCHMARK_OUTPUT_PATH = args.benchmark_output_path
ENCODER_EVAL_MAX_CELLS = args.encoder_eval_max_cells
ENCODER_EVAL_CELL_LINE = args.encoder_eval_cell_line
ENCODER_EVAL_CONTROLS_ONLY = bool(args.encoder_eval_controls_only)
ENCODER_EARLY_STOP_PATIENCE = args.encoder_early_stop_patience
ENCODER_EARLY_STOP_MIN_DELTA = args.encoder_early_stop_min_delta

# Dataset configurations
DATASET_CONFIGS = {
    'norman_additive_grc': {
        'data_dir': str(REPO_ROOT / 'data' / 'norman_additive_grc'),
        'train_file': 'train.h5ad',
        'test_file': 'test.h5ad',
        'val_file': 'val.h5ad',
        'train_file_sorted': 'train.h5ad',
        'test_file_sorted': 'test.h5ad',
        'val_file_sorted': 'val.h5ad',
        'n_clusters': 32,
        'chunk_size': 64,
    },
}


GRC_DATASET_PATTERN = re.compile(r'^(?P<base>.+)_grc(?P<k>\d+)?$')
GRC_CLUSTER_TYPE_PATTERN = re.compile(r'^grc(?P<k>\d+)?$')


def extract_grc_cluster_count(name):
    """Extract cluster count from dataset or cluster type names like *_grc64."""
    text = name or ''
    match = GRC_DATASET_PATTERN.match(text)
    if match:
        return int(match.group('k')) if match.group('k') else None
    match = GRC_CLUSTER_TYPE_PATTERN.match(text)
    if match:
        return int(match.group('k')) if match.group('k') else None
    return None


def get_grc_base_dataset(name):
    """Strip the _grc{K} suffix when present."""
    match = GRC_DATASET_PATTERN.match(name or '')
    if match:
        return match.group('base')
    return None


def is_grc_dataset_name(name):
    return bool(GRC_DATASET_PATTERN.match(name or '') or GRC_CLUSTER_TYPE_PATTERN.match(name or ''))


def is_clustered_dataset_name(name):
    return (
        is_grc_dataset_name(name) or
        name.endswith('_kmeans32') or
        name.endswith('_kmeans_unbalanced') or
        name.endswith('_random32') or
        name.endswith('_esm2_32') or
        name.endswith('_go_32')
    )


def _dedupe_paths(paths):
    seen = set()
    ordered = []
    for path in paths:
        if path and path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def build_cluster_label_candidates(data_dir, dataset_name, cluster_type, dataset_grc_clusters=None,
                                   cluster_type_grc_clusters=None, default_n_clusters=128):
    """Build an ordered list of cluster metadata files to try for clustered datasets."""
    candidates = []

    if dataset_grc_clusters is not None:
        base_dataset = get_grc_base_dataset(dataset_name) or dataset_name
        candidates.extend([
            os.path.join(data_dir, f'{base_dataset}_grc{dataset_grc_clusters}_metadata.pkl'),
            os.path.join(data_dir, f'{base_dataset}_gene_clusters_{dataset_grc_clusters}_grc.pkl'),
        ])
    elif cluster_type_grc_clusters is not None:
        candidates.extend([
            os.path.join(data_dir, f'{dataset_name}_grc{cluster_type_grc_clusters}_metadata.pkl'),
            os.path.join(data_dir, f'{dataset_name}_gene_clusters_{cluster_type_grc_clusters}_grc.pkl'),
        ])
    elif cluster_type in ['genecompassV', 'genecompassG', 'genecompassH']:
        candidates.extend([
            os.path.join(data_dir, f'{dataset_name}_{cluster_type}_metadata.pkl'),
            os.path.join(data_dir, f'{dataset_name}_gene_clusters_{default_n_clusters}_{cluster_type}.pkl'),
        ])
    elif dataset_name.endswith(('_random32', '_kmeans32', '_kmeans_unbalanced')):
        candidates.extend([
            os.path.join(data_dir, 'metadata.pkl'),
            os.path.join(data_dir, 'cluster_info.pkl'),
            os.path.join(data_dir, f'{dataset_name}_metadata.pkl'),
        ])
    else:
        candidates.extend([
            os.path.join(data_dir, f'{dataset_name}_genecompass_metadata.pkl'),
            os.path.join(data_dir, f'{dataset_name}_gene_clusters_{default_n_clusters}_genecompass.pkl'),
        ])

    if dataset_grc_clusters is not None:
        candidates.extend(sorted(glob.glob(os.path.join(data_dir, f'*_grc{dataset_grc_clusters}_metadata.pkl'))))
        candidates.extend(sorted(glob.glob(os.path.join(data_dir, f'*_gene_clusters_{dataset_grc_clusters}_grc.pkl'))))

    for pattern in ['metadata.pkl', 'cluster_info.pkl', 'gene_clusters_*.pkl', '*_metadata.pkl', '*_gene_clusters_*.pkl']:
        candidates.extend(sorted(glob.glob(os.path.join(data_dir, pattern))))

    return _dedupe_paths(candidates)


def safe_l2_normalize(matrix, axis=1, eps=1e-8):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def build_ppi_gene_dict(ppi_edges):
    ppi_dict = {}
    for g1, g2, _ in ppi_edges:
        g1u = str(g1).upper()
        g2u = str(g2).upper()
        ppi_dict.setdefault(g1u, set()).add(g2u)
        ppi_dict.setdefault(g2u, set()).add(g1u)
    return ppi_dict


def infer_balanced_cluster_sizes(n_genes, n_clusters):
    base = n_genes // n_clusters
    remainder = n_genes % n_clusters
    return [base + (1 if i < remainder else 0) for i in range(n_clusters)]


def align_cluster_labels_by_overlap(old_labels, new_labels, n_clusters):
    old_labels = np.asarray(old_labels, dtype=np.int32)
    new_labels = np.asarray(new_labels, dtype=np.int32)
    overlap = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for old_c, new_c in zip(old_labels, new_labels):
        if 0 <= old_c < n_clusters and 0 <= new_c < n_clusters:
            overlap[old_c, new_c] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {int(new_c): int(old_c) for old_c, new_c in zip(row_ind, col_ind)}
    unused_old = [c for c in range(n_clusters) if c not in set(mapping.values())]
    for new_c in range(n_clusters):
        if new_c not in mapping:
            mapping[new_c] = unused_old.pop(0) if unused_old else new_c

    aligned = np.array([mapping[int(c)] for c in new_labels], dtype=np.int32)
    return aligned, mapping


def dynamic_balanced_grc_recluster(
    gene_names,
    static_embeddings,
    response_signatures,
    ppi_dict,
    previous_labels,
    n_clusters,
    cluster_sizes,
    static_weight=0.7,
    response_weight=0.3,
    ppi_weight=0.3,
    random_state=0,
):
    """
    Dynamic balanced clustering on top of the current scaffold.

    We keep the module count fixed and refresh gene-to-module membership using a
    blend of static GeneCompass similarity, model-conditioned response
    signatures, and PPI coherence.
    """
    n_genes = len(gene_names)
    if cluster_sizes is None:
        cluster_sizes = infer_balanced_cluster_sizes(n_genes, n_clusters)
    cluster_sizes = [int(x) for x in cluster_sizes]
    if sum(cluster_sizes) != n_genes:
        raise ValueError(f"Cluster size targets must sum to {n_genes}, got {sum(cluster_sizes)}")

    static_embeddings = safe_l2_normalize(static_embeddings, axis=1)
    response_block = None
    if response_signatures is not None and response_signatures.size > 0:
        response_block = safe_l2_normalize(response_signatures, axis=1)

    if response_block is None:
        combined = static_embeddings
    else:
        combined = np.concatenate(
            [
                np.sqrt(max(static_weight, 1e-8)) * static_embeddings,
                np.sqrt(max(response_weight, 1e-8)) * response_block,
            ],
            axis=1,
        )
        combined = safe_l2_normalize(combined, axis=1)

    gene_to_idx = {str(g).upper(): i for i, g in enumerate(gene_names)}
    ppi_adj = np.zeros((n_genes, n_genes), dtype=np.float32)
    if ppi_weight > 0:
        for i, gene in enumerate(gene_names):
            for neighbor in ppi_dict.get(str(gene).upper(), ()):
                j = gene_to_idx.get(neighbor)
                if j is not None:
                    ppi_adj[i, j] = 1.0

    combined_adjusted = combined.copy()
    if ppi_weight > 0:
        mix = min(0.25, 0.2 * float(ppi_weight))
        for i in range(n_genes):
            neighbor_idx = np.where(ppi_adj[i] > 0)[0]
            if len(neighbor_idx) > 0:
                combined_adjusted[i] = (1.0 - mix) * combined[i] + mix * combined[neighbor_idx].mean(axis=0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(combined_adjusted)
    centers = kmeans.cluster_centers_

    cluster_cost = cdist(combined_adjusted, centers, metric='cosine')
    if ppi_weight > 0:
        cluster_ppi_bonus = np.zeros((n_genes, n_clusters), dtype=np.float32)
        initial_labels = kmeans.labels_
        for cluster_id in range(n_clusters):
            cluster_gene_idx = np.where(initial_labels == cluster_id)[0]
            if len(cluster_gene_idx) == 0:
                continue
            denom = float(max(1, len(cluster_gene_idx)))
            cluster_ppi_bonus[:, cluster_id] = ppi_adj[:, cluster_gene_idx].sum(axis=1) / denom
        cluster_cost = cluster_cost - (0.2 * float(ppi_weight)) * cluster_ppi_bonus

    cluster_cost = np.maximum(cluster_cost, 1e-6)
    slot_to_cluster = np.repeat(np.arange(n_clusters, dtype=np.int32), np.asarray(cluster_sizes, dtype=np.int32))
    expanded_cost = cluster_cost[:, slot_to_cluster]
    row_ind, col_ind = linear_sum_assignment(expanded_cost)
    new_labels = np.zeros(n_genes, dtype=np.int32)
    new_labels[row_ind] = slot_to_cluster[col_ind]

    aligned_labels, label_mapping = align_cluster_labels_by_overlap(previous_labels, new_labels, n_clusters)
    sort_idx = np.array(sorted(range(n_genes), key=lambda i: (int(aligned_labels[i]), i)), dtype=np.int32)
    sorted_gene_names = [gene_names[i] for i in sort_idx]
    sorted_cluster_labels = aligned_labels[sort_idx]
    moved_fraction = float(np.mean(aligned_labels != np.asarray(previous_labels, dtype=np.int32)))

    refreshed_cluster_sizes = np.bincount(sorted_cluster_labels, minlength=n_clusters).astype(int).tolist()
    return {
        'sorted_gene_names': sorted_gene_names,
        'sorted_cluster_labels': sorted_cluster_labels,
        'sort_idx': sort_idx,
        'moved_fraction': moved_fraction,
        'cluster_sizes': refreshed_cluster_sizes,
        'label_mapping': label_mapping,
    }

dataset_config = DATASET_CONFIGS[DATASET]
DATA_DIR = args.data_dir_override or dataset_config['data_dir']

# Paths to biological priors
RESOURCE_DIR = Path(args.resource_dir).expanduser()
PPI_INFO_PATH = args.ppi_info_path or str(RESOURCE_DIR / '9606.protein.info.v12.0.txt.gz')
PPI_LINKS_PATH = args.ppi_links_path or str(RESOURCE_DIR / '9606.protein.links.v12.0.txt.gz')
SCGPT_PATH = args.scgpt_path or str(RESOURCE_DIR / 'gene_emb.pkl')
GENECOMPASS_PATH = args.genecompass_path
PATHWAY_PATH = args.pathway_path or str(RESOURCE_DIR / 'ReactomePathways.gmt.zip')
ESM2_PATH = args.esm2_path or str(RESOURCE_DIR / 'ESM2_pert_features.pt')

# Training parameters
BATCH_SIZE = args.batch_size
N_PCS = args.n_pcs
TOP_DE_GENES = 20
LEARNING_RATE = 5e-5
MULTI_STEPS = 20

# Encoder training parameters
ENCODER_N_EPOCHS = args.encoder_n_epochs
ENCODER_EVAL_EVERY_EPOCHS = args.encoder_eval_every_epochs
ENCODER_BATCH_SIZE = args.encoder_batch_size
EMBED_DIM = args.embed_dim
NUM_LAYERS = args.num_layers
NUM_HEADS = args.num_heads
MODULE_POOLING = args.module_pooling
ENCODER_LR = 1e-4

# Gene loss training parameters
GENE_LOSS_LR = LEARNING_RATE


def run_training():
    """Run E2E training with gene-level losses."""
    
    # Determine actual encoder config early (before printing)
    # For GRC datasets, default to GeneCompass module-induced attention.
    is_clustered_dataset = is_clustered_dataset_name(DATASET)
    dataset_grc_clusters = extract_grc_cluster_count(DATASET)
    if dataset_grc_clusters is None and is_grc_dataset_name(DATASET):
        dataset_grc_clusters = dataset_config.get('n_clusters')
    
    # Keep the legacy fallback for older command lines while using the public
    # module-induced GeneCompass config by default.
    use_encoder_config = ENCODER_CONFIG
    if is_clustered_dataset and ENCODER_CONFIG == 'scgpt_pe_local_window':
        # User didn't override, use default for clustered datasets
        use_encoder_config = 'genecompass_pe_module'
    
    # Get configurations (use actual encoder config)
    encoder_config = GCAE_CONFIGS[use_encoder_config]
    gene_loss_config = GENE_LOSS_CONFIGS[GENE_LOSS_CONFIG].copy()  # Make a copy to allow overrides
    
    # Override loss weights if specified via command line
    if args.lambda_corr is not None:
        gene_loss_config['lambda_corr'] = args.lambda_corr
        print(f"[Override] lambda_corr = {args.lambda_corr}")
    if args.lambda_ot is not None:
        gene_loss_config['lambda_ot'] = args.lambda_ot
        print(f"[Override] lambda_ot = {args.lambda_ot}")
    
    # Check if any gene loss-like fine-tuning objective is active
    any_gene_loss = (gene_loss_config['lambda_corr'] > 0 or
                     gene_loss_config['lambda_ot'] > 0 or
                     gene_loss_config['lambda_smooth'] > 0 or
                     gene_loss_config['lambda_dir'] > 0 or
                     CONTROL_IDENTITY_WEIGHT > 0)
    
    # Output directory
    cluster_suffix = f'_{CLUSTER_TYPE}' if CLUSTER_TYPE != 'scgpt' else '_reordered'
    if not USE_REORDERED:
        cluster_suffix = '_original'
    e2e_suffix = '_e2e' if any_gene_loss else ''
    pooling_suffix = '' if MODULE_POOLING == 'mean' else f'_pool_{MODULE_POOLING}'
    dynamic_suffix = '_dynamic_refresh' if args.dynamic_clustering else ''
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = str(REPO_ROOT / 'outputs' / f'{DATASET}_{use_encoder_config}_{GENE_LOSS_CONFIG}{cluster_suffix}{e2e_suffix}{pooling_suffix}{dynamic_suffix}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("scBIG Norman additive training")
    print(f"Dataset: {DATASET}")
    print(f"Encoder: {use_encoder_config} (requested: {ENCODER_CONFIG})")
    print(f"Gene Loss: {GENE_LOSS_CONFIG}")
    print(f"Cluster Type: {CLUSTER_TYPE}")
    print(f"Reordered: {USE_REORDERED}")
    print(f"Module pooling: {MODULE_POOLING}")
    gene_loss_updates_active = any_gene_loss and GENE_LOSS_FREQ > 0
    print(f"Gene Loss Config Present: {any_gene_loss}")
    print(f"Gene Loss Gradient Updates Active: {gene_loss_updates_active}")
    print(f"Gene Loss Frequency: every {GENE_LOSS_FREQ} iterations")
    print(f"Main flow batch size: {BATCH_SIZE}")
    print(f"Encoder batch size: {ENCODER_BATCH_SIZE}")
    print(f"Transform batch size: {ENCODER_BATCH_SIZE}")
    print(f"Gene loss batch size: {GENE_LOSS_BATCH_SIZE}")
    print(
        f"Control identity aux: weight={CONTROL_IDENTITY_WEIGHT}, "
        f"batch_size={CONTROL_IDENTITY_BATCH_SIZE}"
    )
    print(f"Data dir: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seed: {SEED}")
    print(
        f"Oversampling: cell_line={OVERSAMPLE_CELL_LINE}, multiplier={OVERSAMPLE_MULTIPLIER}, "
        f"controls_only={OVERSAMPLE_CONTROLS_ONLY}, stages={OVERSAMPLE_STAGES}"
    )
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dynamic clustering: {args.dynamic_clustering}")
    if args.dynamic_clustering:
        print(f"  refresh_start_epoch={args.dynamic_refresh_start_epoch}")
        print(f"  refresh_freq_epochs={args.dynamic_refresh_freq_epochs}")
        print(f"  refresh_max_refreshes={args.dynamic_refresh_max_refreshes}")
        print(f"  refresh_n_conditions={args.dynamic_refresh_n_conditions}")
        print(f"  static/response/ppi weights={args.dynamic_static_weight}/{args.dynamic_response_weight}/{args.dynamic_ppi_weight}")
    if RESUME_FROM is not None:
        print(f"Resume from arg: {RESUME_FROM} (exists={os.path.exists(RESUME_FROM)})")
    print("=" * 80)
    
    print(f"\nEncoder config: {encoder_config['description']}")
    print(f"Gene loss config: {gene_loss_config['description']}")
    print(f"  lambda_corr: {gene_loss_config['lambda_corr']}")
    print(f"  lambda_ot: {gene_loss_config['lambda_ot']}")
    print(f"  lambda_smooth: {gene_loss_config['lambda_smooth']}")
    print(f"  lambda_dir: {gene_loss_config['lambda_dir']}")

    np.random.seed(SEED)
    py_random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # ========================================================================
    # Load data
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)
    
    if USE_REORDERED:
        try:
            # Determine file suffix based on cluster type
            if CLUSTER_TYPE == 'scgpt':
                # Default scGPT sorted files
                sorted_train = os.path.join(DATA_DIR, dataset_config['train_file_sorted'])
                sorted_test = os.path.join(DATA_DIR, dataset_config['test_file_sorted'])
                sorted_val = os.path.join(DATA_DIR, dataset_config['val_file_sorted'])
            elif CLUSTER_TYPE.startswith('grc') or is_clustered_dataset_name(DATASET):
                # GRC and other clustered datasets are already sorted and preprocessed.
                sorted_train = os.path.join(DATA_DIR, dataset_config['train_file_sorted'])
                sorted_test = os.path.join(DATA_DIR, dataset_config['test_file_sorted'])
                sorted_val = os.path.join(DATA_DIR, dataset_config['val_file_sorted'])
            elif CLUSTER_TYPE in ['esm2', 'go', 'genecompass', 'genecompassV', 'genecompassG', 'genecompassH']:
                # ESM2, GO, GeneCompass, GeneCompassV/G/H sorted files - use dataset-specific naming
                # Map dataset to output prefix
                # genecompass uses 'norman' prefix, genecompassV/G/H uses 'norman_additive' prefix
                if CLUSTER_TYPE in ['genecompassV', 'genecompassG', 'genecompassH']:
                    # genecompassV/G/H uses full dataset name as prefix
                    prefix = DATASET
                else:
                    # Legacy: genecompass uses 'norman' for norman_additive
                    dataset_prefix_map = {'norman_additive': 'norman'}
                    prefix = dataset_prefix_map.get(DATASET, DATASET.lower())
                
                sorted_train = os.path.join(DATA_DIR, f'{prefix}_train_filtered_sorted_{CLUSTER_TYPE}.h5ad')
                sorted_test = os.path.join(DATA_DIR, f'{prefix}_test_filtered_sorted_{CLUSTER_TYPE}.h5ad')
                sorted_val = os.path.join(DATA_DIR, f'{prefix}_val_filtered_sorted_{CLUSTER_TYPE}.h5ad')
            else:
                sorted_train = os.path.join(DATA_DIR, dataset_config['train_file_sorted'])
                sorted_test = os.path.join(DATA_DIR, dataset_config['test_file_sorted'])
                sorted_val = os.path.join(DATA_DIR, dataset_config['val_file_sorted'])
            
            # Only require train and test sorted files, val can use original
            if all(os.path.exists(p) for p in [sorted_train, sorted_test]):
                adata_train = ad.read_h5ad(sorted_train)
                adata_test = ad.read_h5ad(sorted_test)
                print(f"[OK] Loaded reordered train/test data (cluster_type={CLUSTER_TYPE})")
                
                # Val: use sorted if available, otherwise use original with gene reordering
                if os.path.exists(sorted_val):
                    adata_val = ad.read_h5ad(sorted_val)
                    print(f"[OK] Loaded reordered val data")
                else:
                    # Load original val and reorder genes to match train
                    adata_val_orig = ad.read_h5ad(os.path.join(DATA_DIR, dataset_config['val_file']))
                    # Reorder val genes to match train gene order
                    train_genes = list(adata_train.var_names)
                    val_genes = list(adata_val_orig.var_names)
                    # Find common genes in train order
                    common_genes = [g for g in train_genes if g in val_genes]
                    adata_val = adata_val_orig[:, common_genes].copy()
                    print(f"[OK] Loaded val data with gene reordering ({len(common_genes)} genes)")
            else:
                raise FileNotFoundError(f"Sorted files not found for cluster_type={CLUSTER_TYPE}")
        except Exception as e:
            print(f"⚠ Falling back to original order: {e}")
            adata_train = ad.read_h5ad(os.path.join(DATA_DIR, dataset_config['train_file']))
            adata_test = ad.read_h5ad(os.path.join(DATA_DIR, dataset_config['test_file']))
            adata_val = ad.read_h5ad(os.path.join(DATA_DIR, dataset_config['val_file']))
    else:
        adata_train = ad.read_h5ad(os.path.join(DATA_DIR, dataset_config['train_file']))
        adata_test = ad.read_h5ad(os.path.join(DATA_DIR, dataset_config['test_file']))
        adata_val = ad.read_h5ad(os.path.join(DATA_DIR, dataset_config['val_file']))
        print(f"[OK] Loaded original order data (no clustering)")
    
    print(f"  Train: {adata_train.shape}")
    print(f"  Test: {adata_test.shape}")
    print(f"  Val: {adata_val.shape}")
    
    # Ensure dense
    if sparse.issparse(adata_train.X):
        adata_train.X = adata_train.X.toarray()
    if sparse.issparse(adata_test.X):
        adata_test.X = adata_test.X.toarray()
    if sparse.issparse(adata_val.X):
        adata_val.X = adata_val.X.toarray()

    encoder_train_adata = adata_train
    encoder_eval_subset = None
    encoder_eval_X = None
    encoder_eval_label = 'validation'
    if ENCODER_EVAL_EVERY_EPOCHS > 0:
        encoder_eval_subset = prepare_encoder_eval_subset(
            adata_train,
            adata_val,
            adata_test,
            target_cell_line=ENCODER_EVAL_CELL_LINE,
            controls_only=ENCODER_EVAL_CONTROLS_ONLY,
            max_cells=ENCODER_EVAL_MAX_CELLS,
            seed=SEED,
        )

        if encoder_eval_subset is not None:
            encoder_eval_X = encoder_eval_subset['X']
            encoder_eval_label = encoder_eval_subset['label']
            print("\n" + "=" * 80)
            print("Preparing encoder validation subset...")
            print("=" * 80)
            print(f"  Encoder validation source: {encoder_eval_label}")
            if encoder_eval_subset.get('fallback_reason'):
                print(f"  ⚠ {encoder_eval_subset['fallback_reason']}")

            if encoder_eval_subset.get('uses_train_holdout'):
                holdout_mask = np.ones(adata_train.n_obs, dtype=bool)
                holdout_mask[encoder_eval_subset['indices']] = False
                encoder_train_adata = adata_train[holdout_mask].copy()
                print(
                    f"  Held out {encoder_eval_subset['n_cells']} cells from train split "
                    f"for encoder early stopping"
                )
                print(f"  Encoder pretrain rows after holdout removal: {encoder_train_adata.n_obs}")
        else:
            print("\n" + "=" * 80)
            print("Preparing encoder validation subset...")
            print("=" * 80)
            print("  ⚠ No dedicated encoder validation subset found; using full val split")
            encoder_eval_X = np.asarray(to_dense_array(adata_val.X), dtype=np.float32)
            encoder_eval_label = 'val:fallback_full'

    original_train_n_obs = adata_train.n_obs
    encoder_train_X = to_dense_array(encoder_train_adata.X)
    encoder_oversample_focus_mask = build_obs_focus_mask(
        encoder_train_adata,
        focus_cell_line=OVERSAMPLE_CELL_LINE,
        controls_only=OVERSAMPLE_CONTROLS_ONLY
    )
    flow_oversample_focus_mask = build_obs_focus_mask(
        adata_train,
        focus_cell_line=OVERSAMPLE_CELL_LINE,
        controls_only=OVERSAMPLE_CONTROLS_ONLY
    )
    encoder_oversample_focus_count = (
        int(encoder_oversample_focus_mask.sum()) if encoder_oversample_focus_mask is not None else 0
    )
    flow_oversample_focus_count = (
        int(flow_oversample_focus_mask.sum()) if flow_oversample_focus_mask is not None else 0
    )

    if OVERSAMPLE_MULTIPLIER > 1:
        print("\n" + "=" * 80)
        print("Applying Focused Oversampling...")
        print("=" * 80)

        if OVERSAMPLE_STAGES in ('encoder', 'both'):
            if encoder_oversample_focus_mask is None or encoder_oversample_focus_count == 0:
                raise ValueError(
                    "Encoder oversampling was requested, but the focused subset is empty. "
                    f"cell_line={OVERSAMPLE_CELL_LINE}, controls_only={OVERSAMPLE_CONTROLS_ONLY}"
                )
            print(
                f"  Encoder focus subset size: {encoder_oversample_focus_count} / {encoder_train_adata.n_obs} "
                f"(cell_line={OVERSAMPLE_CELL_LINE}, controls_only={OVERSAMPLE_CONTROLS_ONLY})"
            )
            encoder_train_X = augment_matrix_with_focus_rows(
                encoder_train_adata.X, encoder_oversample_focus_mask, OVERSAMPLE_MULTIPLIER
            ).astype(np.float32, copy=False)
            print(f"  Encoder pretrain rows: {encoder_train_X.shape[0]} (x{OVERSAMPLE_MULTIPLIER} focused weight)")

        if OVERSAMPLE_STAGES in ('flow', 'both'):
            if flow_oversample_focus_mask is None or flow_oversample_focus_count == 0:
                raise ValueError(
                    "Flow oversampling was requested, but the focused subset is empty. "
                    f"cell_line={OVERSAMPLE_CELL_LINE}, controls_only={OVERSAMPLE_CONTROLS_ONLY}"
                )
            print(
                f"  Flow focus subset size: {flow_oversample_focus_count} / {adata_train.n_obs} "
                f"(cell_line={OVERSAMPLE_CELL_LINE}, controls_only={OVERSAMPLE_CONTROLS_ONLY})"
            )
            adata_train = augment_adata_with_focus_rows(
                adata_train,
                flow_oversample_focus_mask,
                OVERSAMPLE_MULTIPLIER,
                suffix_prefix='focus'
            )
            print(f"  Flow training rows: {adata_train.n_obs} (was {original_train_n_obs})")
    encoder_train_X = np.asarray(encoder_train_X, dtype=np.float32)

    ENCODER_MAX_CELLS = int(os.environ.get('ENCODER_MAX_CELLS', 200000))
    if encoder_train_X.shape[0] > ENCODER_MAX_CELLS:
        rng_sub = np.random.default_rng(SEED)
        sub_idx = rng_sub.choice(encoder_train_X.shape[0], ENCODER_MAX_CELLS, replace=False)
        print(f"  Subsampling encoder training data: {encoder_train_X.shape[0]} -> {ENCODER_MAX_CELLS}")
        encoder_train_X = encoder_train_X[sub_idx]

    gene_names = list(adata_train.var_names)
    n_genes = len(gene_names)
    print(f"  Genes: {n_genes}")
    
    # ========================================================================
    # Load precomputed DE genes (for faster evaluation)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading precomputed DE genes...")
    print("=" * 80)
    
    PRECOMPUTED_DE_DICT = None
    if args.full_metrics:
        PRECOMPUTED_DE_DICT = load_precomputed_de_genes(DATASET, gene_names, top_n=TOP_DE_GENES)
        if PRECOMPUTED_DE_DICT is not None:
            print(f"  [OK] Will use precomputed DE genes for evaluation")
        else:
            print(f"  Will compute DE genes on-the-fly during evaluation")
    
    # ========================================================================
    # Load gene-level priors for losses
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading gene-level priors for losses...")
    print("=" * 80)
    
    # PPI mask for gene-level losses
    ppi_gene_mask = None
    ppi_adjacency = None
    if gene_loss_config['use_ppi_masked_corr'] or gene_loss_config['lambda_smooth'] > 0:
        print("  Loading PPI network for gene-level losses...")
        _, ppi_edges = load_ppi_network(PPI_INFO_PATH, PPI_LINKS_PATH, min_score=700)
        ppi_gene_mask = build_ppi_attention_mask(gene_names, ppi_edges, k_hop=2, include_self=True)
        ppi_gene_mask = jnp.array(ppi_gene_mask)
        
        # Build adjacency for smoothness loss
        ppi_adjacency = ppi_mask_to_adjacency(ppi_gene_mask)
        
        n_edges = int((ppi_gene_mask > 0).sum())
        print(f"  [OK] PPI gene mask: {ppi_gene_mask.shape}, {n_edges} edges")
    
    # Pathway matrix for OT loss
    # NOTE: This uses external database pathways (Reactome/KEGG), NOT GRC clusters.
    # The pathway_ot_loss aggregates gene expression to pathway level and computes
    # Sinkhorn distance in the pathway-aggregated space.
    pathway_matrix = None
    if gene_loss_config['lambda_ot'] > 0:
        print("  Loading pathway matrix for OT loss from external database (Reactome)...")
        pathway_matrix_np, pathway_names = load_pathway_gene_matrix(
            PATHWAY_PATH, gene_names, is_zip=True
        )
        pathway_matrix = jnp.array(pathway_matrix_np)
        print(f"  [OK] Pathway matrix: {pathway_matrix.shape} ({len(pathway_names)} pathways from Reactome)")
    
    # Cluster labels for cluster-level correlation loss
    cluster_labels = None
    n_clusters = 128
    cluster_size_targets = None
    use_cluster_corr = gene_loss_config.get('use_cluster_corr', False)
    if use_cluster_corr:
        print("  Loading cluster labels for cluster-level correlation loss...")
        cluster_type_grc_clusters = extract_grc_cluster_count(CLUSTER_TYPE)
        if cluster_type_grc_clusters is None and CLUSTER_TYPE.startswith('grc'):
            cluster_type_grc_clusters = dataset_config.get('n_clusters')
        candidate_paths = build_cluster_label_candidates(
            DATA_DIR,
            DATASET,
            CLUSTER_TYPE,
            dataset_grc_clusters=dataset_grc_clusters,
            cluster_type_grc_clusters=cluster_type_grc_clusters,
            default_n_clusters=n_clusters,
        )
        cluster_metadata_path = None

        for candidate_path in candidate_paths:
            if not os.path.exists(candidate_path):
                continue
            try:
                with open(candidate_path, 'rb') as f:
                    cluster_metadata = pickle.load(f)
            except Exception as e:
                print(f"  ⚠ Failed loading cluster metadata candidate {candidate_path}: {e}")
                continue

            if not isinstance(cluster_metadata, dict):
                continue

            candidate_labels = cluster_metadata.get('sorted_cluster_labels', cluster_metadata.get('cluster_labels'))
            if candidate_labels is None:
                continue

            cluster_labels = jnp.array(candidate_labels)
            n_clusters = int(cluster_metadata.get('n_clusters', n_clusters))
            cluster_size_targets = cluster_metadata.get('cluster_sizes')
            cluster_metadata_path = candidate_path
            break

        if cluster_labels is not None:
            print(f"  [OK] Cluster labels: {len(cluster_labels)} genes, {n_clusters} clusters")
            print(f"  [OK] Cluster labels source: {cluster_metadata_path}")
            if cluster_size_targets is not None:
                print(f"  [OK] Cluster size targets loaded ({len(cluster_size_targets)} clusters)")
        else:
            if candidate_paths:
                print(f"  ⚠ Cluster labels not found in candidates: {candidate_paths[:5]}")
            else:
                print(f"  ⚠ No cluster metadata candidates found in {DATA_DIR}")
            print("  ⚠ Falling back to gene-level correlation")
            use_cluster_corr = False

    if cluster_size_targets is None and is_clustered_dataset:
        cluster_size_targets = infer_balanced_cluster_sizes(n_genes, dataset_config.get('n_clusters', n_clusters))

    dynamic_refresh_history = []
    dynamic_static_embeddings = None
    dynamic_ppi_dict = None
    dynamic_ppi_edges = None
    if args.dynamic_clustering:
        print("  Preparing static features for dynamic clustering refresh...")
        dynamic_static_embeddings, gc_coverage = load_genecompass_embeddings(GENECOMPASS_PATH, gene_names)
        dynamic_static_embeddings = np.asarray(dynamic_static_embeddings, dtype=np.float32)
        print(f"  [OK] Dynamic static embeddings ready: {dynamic_static_embeddings.shape}, coverage={gc_coverage*100:.1f}%")
        _, dynamic_ppi_edges = load_ppi_network(PPI_INFO_PATH, PPI_LINKS_PATH, min_score=700)
        dynamic_ppi_dict = build_ppi_gene_dict(dynamic_ppi_edges)
        print(f"  [OK] Dynamic PPI dictionary ready: {len(dynamic_ppi_dict)} genes")
    
    # ========================================================================
    # Create and train encoder
    # ========================================================================
    print("\n" + "=" * 80)
    print("Setting up Bio-Inductive Encoder...")
    print("=" * 80)
    
    # Get chunk_size and window_size from config, or use defaults
    # For GRC datasets, use dataset-specific chunk_size
    if 'chunk_size' in dataset_config:
        config_chunk_size = dataset_config['chunk_size']
        config_window_size = dataset_config.get('chunk_size', dataset_config['chunk_size'])
    elif is_clustered_dataset and dataset_config.get('n_clusters'):
        inferred_chunk_size = int(np.ceil(n_genes / dataset_config['n_clusters']))
        config_chunk_size = inferred_chunk_size
        config_window_size = inferred_chunk_size
        print(f"  Inferred chunk_size from n_genes/n_clusters: {inferred_chunk_size}")
    elif dataset_grc_clusters is not None:
        config_chunk_size = dataset_config.get('chunk_size', 64)
        config_window_size = config_chunk_size
    else:
        config_chunk_size = encoder_config.get('chunk_size', 64)
        config_window_size = encoder_config.get('window_size', 64)
    
    # Override chunk_size from command line if specified
    if args.chunk_size is not None:
        config_chunk_size = args.chunk_size
        config_window_size = args.chunk_size
        print(f"  [Override] chunk_size = {args.chunk_size}")
    
    # Adjust MLP dim proportionally to embed_dim
    mlp_dim = EMBED_DIM * 2
    
    # Determine module count: command line > dataset default > target default.
    requested_modules = args.num_modules if args.num_modules is not None else args.num_inducing
    if requested_modules is not None:
        num_inducing = requested_modules
        print(f"  [Override] num_modules = {requested_modules}")
    elif 'num_inducing' in dataset_config:
        num_inducing = dataset_config['num_inducing']
    elif dataset_grc_clusters is not None:
        num_inducing = dataset_config.get('num_inducing', 8)
    else:
        num_inducing = encoder_config.get('num_inducing', 8)
    
    # use_encoder_config already determined earlier - use it here
    if is_clustered_dataset and ENCODER_CONFIG == 'scgpt_pe_local_window':
        print(f"  Using default encoder for clustered dataset: {use_encoder_config}")
    else:
        print(f"  Using specified encoder config: {use_encoder_config}")
    
    encoder = get_gcae_for_config(
        use_encoder_config,
        latent_dim=N_PCS,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=mlp_dim,
        dropout_rate=0.1,
        chunk_size=config_chunk_size,
        window_size=config_window_size,
        num_inducing=num_inducing,
        pooling_type=MODULE_POOLING,
    )
    if is_clustered_dataset:
        print(f"  Clustered dataset config: chunk_size={config_chunk_size}, num_modules={num_inducing}")
    
    # Set priors for encoder
    # Determine which embedding path to use based on the actual encoder config being used
    actual_encoder_config = GCAE_CONFIGS[use_encoder_config]
    pe_type = actual_encoder_config.get('pe_type', 'sinusoidal')
    scgpt_path_for_encoder = SCGPT_PATH if pe_type == 'scgpt' else None
    genecompass_path_for_encoder = GENECOMPASS_PATH if pe_type in ['genecompass', 'genecompassV', 'genecompassG', 'genecompassH'] else None
    print(f"  PE type: {pe_type}, scgpt_path: {'Yes' if scgpt_path_for_encoder else 'No'}, genecompass_path: {'Yes' if genecompass_path_for_encoder else 'No'}")
    
    encoder.set_priors(
        gene_names=gene_names,
        ppi_info_path=PPI_INFO_PATH if actual_encoder_config['attention_type'] == 'ppi_sparse' else None,
        ppi_links_path=PPI_LINKS_PATH if actual_encoder_config['attention_type'] == 'ppi_sparse' else None,
        ppi_min_score=700,
        ppi_k_hop=2,
        scgpt_path=scgpt_path_for_encoder,
        genecompass_path=genecompass_path_for_encoder,
        pathway_gmt_path=PATHWAY_PATH if (actual_encoder_config['use_pathway_pooling'] or 
                                          actual_encoder_config['use_pathway_loss']) else None,
        pathway_is_zip=True
    )
    
    # Train or load encoder
    print("\n" + "=" * 80)
    PRETRAINED_ENCODER = args.pretrained_encoder
    
    # Check if we are resuming from a pretrained model
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        # Try to load encoder from resume directory
        resume_encoder_path = os.path.join(RESUME_FROM, 'best_encoder_e2e.pkl')
        if not os.path.exists(resume_encoder_path):
            resume_encoder_path = os.path.join(RESUME_FROM, 'best_encoder.pkl')
        
        print(f"Resuming from: {RESUME_FROM}")
        print(f"Loading encoder from: {resume_encoder_path}")
        print("=" * 80)
        encoder = GCAEWrapper.load(resume_encoder_path)
        
        # Check module-count mismatch when resuming from a checkpoint.
        loaded_num_inducing = getattr(encoder, 'num_inducing', None)
        if loaded_num_inducing is not None:
            config_num_inducing = requested_modules or encoder_config.get('num_inducing', 8)
            if loaded_num_inducing != config_num_inducing:
                print("  Module-count mismatch detected.")
                print(f"     Loaded encoder: num_modules={loaded_num_inducing}")
                print(f"     Config expects: num_modules={config_num_inducing}")
                print(f"     This may cause issues if encoder architecture changed.")
                print(f"     For flow_only finetuning, this is usually OK (encoder frozen).")
        
        print(f"[OK] Encoder loaded successfully from resume checkpoint")
        encoder_save_path = os.path.join(OUTPUT_DIR, 'best_encoder.pkl')
        import shutil
        shutil.copy(resume_encoder_path, encoder_save_path)
    elif PRETRAINED_ENCODER and os.path.exists(PRETRAINED_ENCODER):
        print(f"Loading pre-trained encoder from: {PRETRAINED_ENCODER}")
        print("=" * 80)
        # Use class method to load encoder
        encoder = GCAEWrapper.load(PRETRAINED_ENCODER)
        print(f"[OK] Encoder loaded successfully")
        encoder_save_path = os.path.join(OUTPUT_DIR, 'best_encoder.pkl')
        import shutil
        if os.path.abspath(PRETRAINED_ENCODER) != os.path.abspath(encoder_save_path):
            shutil.copy(PRETRAINED_ENCODER, encoder_save_path)
    elif SKIP_ENCODER_PRETRAIN:
        print("Skipping encoder pretraining (using initialized encoder)...")
        print("=" * 80)
        encoder_save_path = os.path.join(OUTPUT_DIR, 'best_encoder.pkl')
        try:
            encoder.save(encoder_save_path)
            print(f"[OK] Initial encoder saved to: {encoder_save_path}")
        except ValueError as e:
            print(f"  ⚠ Encoder not fitted yet, skipping initial save: {e}")
    else:
        print("Training encoder...")
        print("=" * 80)
        if ENCODER_EVAL_EVERY_EPOCHS <= 0:
            print("  Encoder validation disabled; final encoder will be saved after pretraining.")
        
        encoder_save_path = os.path.join(OUTPUT_DIR, 'best_encoder.pkl')
        encoder.fit(
            encoder_train_X,
            X_test=encoder_eval_X,
            n_epochs=ENCODER_N_EPOCHS,
            batch_size=ENCODER_BATCH_SIZE,
            learning_rate=ENCODER_LR,
            verbose=True,
            save_best_path=encoder_save_path,
            eval_every_epochs=ENCODER_EVAL_EVERY_EPOCHS,
            early_stop_patience=ENCODER_EARLY_STOP_PATIENCE,
            early_stop_min_delta=ENCODER_EARLY_STOP_MIN_DELTA,
            eval_label=encoder_eval_label,
        )
        print(f"[OK] Encoder training complete")
        if ENCODER_EVAL_EVERY_EPOCHS > 0:
            print(f"[OK] Best encoder saved to: {encoder_save_path}")
        else:
            print(f"[OK] Final encoder saved to: {encoder_save_path}")

    if EXIT_AFTER_ENCODER:
        print("\n" + "=" * 80)
        print("Exiting after encoder stage as requested.")
        print("=" * 80)
        raise SystemExit(0)

    # Transform data
    print("\nTransforming data to latent space...")
    adata_train.obsm['X_pca'] = np.array(encoder.transform(adata_train.X, batch_size=ENCODER_BATCH_SIZE))
    adata_test.obsm['X_pca'] = np.array(encoder.transform(adata_test.X, batch_size=ENCODER_BATCH_SIZE))
    adata_val.obsm['X_pca'] = np.array(encoder.transform(adata_val.X, batch_size=ENCODER_BATCH_SIZE))
    
    adata_train.varm['X_mean'] = np.array(encoder.X_mean).reshape(-1, 1)
    adata_train.varm['PCs'] = np.eye(n_genes, N_PCS)
    adata_train.uns['neural_encoder'] = encoder
    
    # ========================================================================
    # Prepare flow backend
    # ========================================================================
    print("\n" + "=" * 80)
    print("Preparing flow backend...")
    print("=" * 80)
    
    def build_one_hot_embeddings(values):
        unique_values = sorted({normalize_obs_value(value) for value in values})
        dim = len(unique_values)
        embeddings = {}
        for index, value in enumerate(unique_values):
            vec = np.zeros(dim, dtype=np.float32)
            vec[index] = 1.0
            embeddings[value] = vec
        return embeddings

    # Extract control cells
    train_control_mask = np.array(adata_train.obs['is_control'])
    adata_train_control = adata_train[train_control_mask].copy()
    adata_train_control.uns['neural_encoder'] = encoder
    print(f"  Control cells: {adata_train_control.shape[0]}")

    use_cell_line_context = (
        'cell_line' in adata_train.obs.columns and
        adata_train.obs['cell_line'].astype(str).nunique() > 1
    )
    sample_covariates = None
    sample_covariate_reps = None
    split_covariates = None
    cell_line_embeddings = None
    if use_cell_line_context:
        if 'cell_line_embeddings' in adata_train.uns and isinstance(adata_train.uns['cell_line_embeddings'], dict):
            cell_line_embeddings = {
                str(k): np.asarray(v, dtype=np.float32)
                for k, v in adata_train.uns['cell_line_embeddings'].items()
            }
            print(f"  Using pre-computed cell_line embeddings from uns (dim={next(iter(cell_line_embeddings.values())).shape[0]})")
        else:
            cell_line_values = []
            for adata_obj in [adata_train, adata_val, adata_test]:
                cell_line_values.extend(adata_obj.obs['cell_line'].astype(str).tolist())
            cell_line_embeddings = build_one_hot_embeddings(cell_line_values)
            print(f"  Fallback: using one-hot cell_line embeddings")
        sample_covariates = ['cell_line']
        sample_covariate_reps = {'cell_line': 'cell_line_embeddings'}
        split_covariates = ['cell_line']
        print(f"  Using cell_line-aware control routing: {sorted(cell_line_embeddings.keys())}")

        test_only_lines = set(adata_test.obs['cell_line'].astype(str)) - set(adata_train.obs['cell_line'].astype(str))
        if test_only_lines:
            print(f"  Cross-donor: injecting test-only donor controls into train: {sorted(test_only_lines)}")
            test_ctrl_mask = (
                adata_test.obs['is_control'].astype(bool).values &
                adata_test.obs['cell_line'].astype(str).isin(test_only_lines).values
            )
            if test_ctrl_mask.any():
                test_ctrl_cells = adata_test[test_ctrl_mask].copy()
                print(f"    Injecting {test_ctrl_cells.n_obs} test control cells for {sorted(test_only_lines)}")
                saved_uns = dict(adata_train.uns)
                adata_train = ad.concat([adata_train, test_ctrl_cells], join='inner')
                adata_train.uns.update(saved_uns)
                train_control_mask = np.array(adata_train.obs['is_control'])
                adata_train_control = adata_train[train_control_mask].copy()
                adata_train_control.uns['neural_encoder'] = encoder
                print(f"    adata_train now: {adata_train.shape}, controls: {adata_train_control.shape[0]}")
    
    # Control mean for losses
    X_ctrl_mean = jnp.array(np.mean(adata_train_control.X, axis=0), dtype=jnp.float32)

    def sample_controls_for_split(split_adata, max_controls=1000, seed_offset=0):
        """Sample train controls while preserving cell-line context when available."""
        if adata_train_control.n_obs == 0:
            return adata_train_control[:0].copy()

        if not use_cell_line_context or 'cell_line' not in split_adata.obs.columns:
            n_take = min(max_controls, adata_train_control.shape[0])
            return adata_train_control[:n_take].copy()

        split_lines = sorted(set(split_adata.obs['cell_line'].astype(str).tolist()))
        if not split_lines:
            n_take = min(max_controls, adata_train_control.shape[0])
            return adata_train_control[:n_take].copy()

        local_rng = np.random.default_rng(SEED + seed_offset)
        per_group_budget = max(1, max_controls // len(split_lines))
        control_lines = adata_train_control.obs['cell_line'].astype(str).to_numpy()
        chosen_indices = []
        for cell_line in split_lines:
            group_indices = np.where(control_lines == cell_line)[0]
            if len(group_indices) == 0:
                continue
            n_take = min(len(group_indices), per_group_budget)
            chosen_indices.extend(local_rng.choice(group_indices, size=n_take, replace=False).tolist())

        if not chosen_indices:
            n_take = min(max_controls, adata_train_control.shape[0])
            return adata_train_control[:n_take].copy()

        chosen_indices = np.array(sorted(set(chosen_indices)), dtype=int)
        return adata_train_control[chosen_indices].copy()

    # Add matched controls to val/test
    control_subset_val = sample_controls_for_split(adata_val, max_controls=1000, seed_offset=17)
    control_subset_test = sample_controls_for_split(adata_test, max_controls=1000, seed_offset=29)

    adata_val_with_control = ad.concat([adata_val, control_subset_val]) if control_subset_val.n_obs else adata_val.copy()
    adata_test_with_control = ad.concat([adata_test, control_subset_test]) if control_subset_test.n_obs else adata_test.copy()
    
    # Get perturbations
    all_perts = set()
    for adata in [adata_train, adata_test, adata_val]:
        all_perts.update(adata.obs['pert1'].unique())
        all_perts.update(adata.obs['pert2'].unique())
    all_perts.discard('ctrl')
    all_perts = sorted(list(all_perts))
    print(f"  Perturbations: {len(all_perts)}")
    
    pert_embeddings = None
    if 'pert_embeddings' in adata_train.uns:
        print("  Loading perturbation embeddings from adata.uns['pert_embeddings']...")
        source_embeddings = adata_train.uns['pert_embeddings']
        pert_embeddings = {
            str(pert): np.asarray(emb, dtype=np.float32)
            for pert, emb in source_embeddings.items()
        }
        if 'ctrl' not in pert_embeddings:
            emb_dim = next(iter(pert_embeddings.values())).shape[0]
            pert_embeddings['ctrl'] = np.zeros(emb_dim, dtype=np.float32)
        for pert in all_perts:
            if pert not in pert_embeddings:
                emb_dim = next(iter(pert_embeddings.values())).shape[0]
                pert_embeddings[pert] = np.zeros(emb_dim, dtype=np.float32)
    else:
        # Load ESM2 embeddings for gene perturbations
        print("  Loading ESM2 embeddings...")
        esm2_dict = torch.load(ESM2_PATH, map_location='cpu')
        
        pert_embeddings = {}
        for pert in all_perts:
            if pert in esm2_dict:
                pert_embeddings[pert] = np.asarray(esm2_dict[pert], dtype=np.float32)
            else:
                emb_dim = esm2_dict[list(esm2_dict.keys())[0]].shape[0]
                pert_embeddings[pert] = np.zeros(emb_dim, dtype=np.float32)
        
        emb_dim = esm2_dict[list(esm2_dict.keys())[0]].shape[0]
        pert_embeddings['ctrl'] = np.zeros(emb_dim, dtype=np.float32)
    
    for adata in [adata_train, adata_test, adata_val, adata_train_control,
                  adata_val_with_control, adata_test_with_control]:
        adata.uns['pert_embeddings'] = pert_embeddings
        if cell_line_embeddings is not None:
            adata.uns['cell_line_embeddings'] = cell_line_embeddings
    
    # Get train conditions for gene loss sampling
    train_conditions = []
    seen = set()
    for idx in adata_train.obs.index:
        cond = adata_train.obs.loc[idx, 'condition']
        condition_key = (cond,)
        if use_cell_line_context and 'cell_line' in adata_train.obs.columns:
            condition_key = (cond, normalize_obs_value(adata_train.obs.loc[idx, 'cell_line']))
        if condition_key not in seen and cond != 'control':
            seen.add(condition_key)
            train_conditions.append(cond)
    print(f"  Train conditions for gene loss: {len(train_conditions)}")
    
    # ========================================================================
    # Initialize flow model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing flow model...")
    print("=" * 80)

    cf = cellflow.model.CellFlow(adata_train, solver='otfm')

    use_slot_fusion_conditions = all(
        col in adata_train.obs.columns for col in ['dose1', 'dose2']
    )
    if use_slot_fusion_conditions:
        print("  Using slot-wise drug+dose condition fusion")
        perturbation_covariates = {'drug': ['pert1', 'pert2'], 'dose': ['dose1', 'dose2']}
        perturbation_covariate_reps = {'drug': 'pert_embeddings'}
        condition_encoder_kwargs = {
            'slot_fusion': {
                'primary_group': 'drug',
                'linked_groups': ['dose'],
                'mlp_dims': (128, 128),
                'modulation_scale': 0.5,
                'token_dim': 256,
                'layer_norm': True,
            }
        }
    else:
        perturbation_covariates = {'pert': ['pert1', 'pert2']}
        perturbation_covariate_reps = {'pert': 'pert_embeddings'}
        condition_encoder_kwargs = {}
    
    cf.prepare_data(
        sample_rep='X_pca',
        control_key='is_control',
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_covariate_reps,
        sample_covariates=sample_covariates,
        sample_covariate_reps=sample_covariate_reps,
        split_covariates=split_covariates,
        max_combination_length=2,
        null_value=0.0
    )
    
    if 'X_pca' not in adata_val_with_control.obsm:
        adata_val_with_control.obsm['X_pca'] = np.array(
            encoder.transform(adata_val_with_control.X), dtype=np.float32
        )
    
    cf.prepare_validation_data(
        adata=adata_val_with_control,
        name='val',
        n_conditions_on_log_iteration=min(10, len(adata_val.obs['condition'].unique())),
        n_conditions_on_train_end=len(adata_val.obs['condition'].unique()),
    )
    
    # Setup optimizer
    base_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale(-LEARNING_RATE)
    )
    optimizer = optax.MultiSteps(base_optimizer, MULTI_STEPS)
    
    cf.prepare_model(
        condition_mode='deterministic',
        regularization=0.0,
        pooling='attention_seed',
        condition_embedding_dim=256,
        time_encoder_dims=(1024, 1024),
        hidden_dims=(1024, 1024),
        decoder_dims=(1024, 1024),
        hidden_dropout=0.1,
        decoder_dropout=0.1,
        optimizer=optimizer,
        probability_path={"constant_noise": 0.5},
        condition_encoder_kwargs=condition_encoder_kwargs,
    )
    
    print("[OK] Flow model prepared")
    
    # Load pretrained model weights if resuming
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        resume_model_path = os.path.join(RESUME_FROM, 'best_model_CellFlow.pkl')
        if os.path.exists(resume_model_path):
            print(f"\n  Loading pretrained flow model from: {resume_model_path}")
            with open(resume_model_path, 'rb') as f:
                saved_cf = pickle.load(f)
            
            # SOLUTION: Use the COMPLETE saved model instead of just copying parameters
            # This ensures DataManager (_dm), solver, and all configurations are consistent
            # We only need to update the _dm's internal adata references with our current data
            cf = saved_cf
            cf.solver.is_trained = True
            
            # Update the DataManager's adata references to use current transformed data
            # The saved model's _dm was created with training-time data
            if hasattr(cf, '_dm') and cf._dm is not None:
                # The key insight: _dm stores references to adata objects
                # We need to ensure our current adata has the same format
                print(f"  [OK] Using complete saved flow model")
            else:
                print(f"  ⚠ Warning: Saved model has no DataManager")
            
            print(f"  is_trained: {cf.solver.is_trained}")
            print("  (Initial model testing will be done before training loop starts)")
        else:
            print(f"  ⚠ Warning: No pretrained model found at {resume_model_path}, starting fresh")
    
    # ========================================================================
    # E2E Gene Loss: Setup differentiable prediction pipeline
    # ========================================================================
    if any_gene_loss and GENE_LOSS_FREQ > 0:
        print("\n" + "=" * 80)
        print("Setting up E2E gene loss training...")
        print(f"  Finetune strategy: {FINETUNE_STRATEGY}")
        print("=" * 80)
        
        import diffrax
        from flax.core import frozen_dict
        
        # Learning rates for different parameter groups (gene loss fine-tuning)
        FINETUNE_FLOW_LR = GENE_LOSS_LR  # Use gene loss LR for flow
        FINETUNE_ENC_LR = GENE_LOSS_LR * 0.1  # Lower LR for pretrained encoder
        FINETUNE_DEC_LR = GENE_LOSS_LR * 0.1  # Lower LR for pretrained decoder
        
        # Get encoder/decoder parameters (they are separate sub-modules)
        full_enc_params = encoder.state.params
        enc_only_params = full_enc_params['encoder']  # Encoder sub-module params
        dec_only_params = full_enc_params['decoder']  # Decoder sub-module params
        encoder_X_mean = jnp.array(encoder.X_mean, dtype=jnp.float32)
        encoder_ppi_mask = encoder.ppi_mask
        encoder_scgpt_pe = encoder.scgpt_pe
        
        # Create optimizers based on finetune strategy
        # flow_only: only flow optimizer
        # all: flow + encoder + decoder optimizers
        # flow_encoder: flow + encoder optimizers
        # flow_decoder: flow + decoder optimizers
        # encoder_decoder: encoder + decoder optimizers
        # encoder_only: encoder optimizer only
        # decoder_only: decoder optimizer only
        
        flow_optimizer = optax.adam(learning_rate=FINETUNE_FLOW_LR)
        encoder_optimizer = optax.adam(learning_rate=FINETUNE_ENC_LR)
        decoder_optimizer = optax.adam(learning_rate=FINETUNE_DEC_LR)
        
        # Initialize optimizer states based on strategy
        flow_opt_state = flow_optimizer.init(cf.solver.vf_state.params)
        enc_opt_state = encoder_optimizer.init(enc_only_params)
        dec_opt_state = decoder_optimizer.init(dec_only_params)
        
        print(f"  Finetune LRs: Flow={FINETUNE_FLOW_LR}, Encoder={FINETUNE_ENC_LR}, Decoder={FINETUNE_DEC_LR}")
        
        # Create differentiable decode function using separate decoder params
        def decode_latent_to_genes_separate(latent, dec_params, X_mean, ppi_mask, scgpt_pe):
            """Decode latent representation to gene space using separate decoder params."""
            # Create full params dict with only decoder (encoder not needed for decode)
            # We need to construct a params dict that the model expects
            full_params = {'encoder': enc_only_params, 'decoder': dec_params}  # Use current enc_only_params as placeholder
            
            def decode_fn(model_instance, lat):
                return model_instance.decoder(lat, ppi_mask, scgpt_pe, training=False)
            decoded = encoder.model.apply({'params': full_params}, latent, method=decode_fn)
            return decoded + X_mean
        
        # Create differentiable encode function
        def encode_genes_to_latent_separate(genes, enc_params, X_mean, ppi_mask, scgpt_pe):
            """Encode genes to latent space using separate encoder params."""
            full_params = {'encoder': enc_params, 'decoder': dec_only_params}  # Use current dec_only_params as placeholder
            
            def encode_fn(model_instance, x):
                return model_instance.encoder(x, ppi_mask, scgpt_pe, None, training=False)
            genes_centered = genes - X_mean
            latent = encoder.model.apply({'params': full_params}, genes_centered, method=encode_fn)
            return latent
        
        # Create differentiable ODE solver for prediction
        def solve_ode_differentiable(
            x_source, condition, flow_params, 
            vf_apply_fn, probability_path_unused
        ):
            """Solve ODE to predict target (differentiable w.r.t. flow_params)."""
            noise_dim = (1, cf.solver.vf.condition_embedding_dim)
            encoder_noise = jnp.zeros(noise_dim)
            
            def vf(t, x, args):
                cond, enc_noise = args
                return vf_apply_fn({"params": flow_params}, t, x, cond, enc_noise, train=False)[0]
            
            ode_term = diffrax.ODETerm(vf)
            solver = diffrax.Tsit5()
            stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
            
            result = diffrax.diffeqsolve(
                ode_term, solver, t0=0.0, t1=1.0, dt0=None,
                y0=x_source, args=(condition, encoder_noise),
                stepsize_controller=stepsize_controller,
            )
            return result.ys[0]
        
        # Batch ODE solve
        solve_ode_batch = jax.vmap(
            solve_ode_differentiable,
            in_axes=(0, None, None, None, None)
        )
        
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static'])
        def compute_gene_loss_batch(X_pred, X_true, 
                                    pathway_matrix_arg, ppi_mask_arg,
                                    ppi_adjacency_arg, X_ctrl,
                                    cluster_labels_arg,
                                    lambda_corr, lambda_ot, 
                                    lambda_smooth, lambda_dir,
                                    use_ppi_corr, use_cluster_corr_flag,
                                    n_clusters_static):
            """Compute combined gene loss for a batch."""
            total_loss = 0.0
            loss_dict = {}
            
            if lambda_corr > 0:
                if use_cluster_corr_flag and cluster_labels_arg is not None:
                    corr_l = cluster_correlation_loss(X_pred, X_true, cluster_labels_arg, n_clusters_static)
                elif use_ppi_corr and ppi_mask_arg is not None:
                    corr_l = ppi_masked_correlation_loss(X_pred, X_true, ppi_mask_arg)
                else:
                    corr_l = gene_correlation_loss(X_pred, X_true)
                total_loss = total_loss + lambda_corr * corr_l
                loss_dict['corr'] = corr_l
            
            if lambda_ot > 0 and pathway_matrix_arg is not None:
                ot_l = pathway_ot_loss(X_pred, X_true, pathway_matrix_arg)
                total_loss = total_loss + lambda_ot * ot_l
                loss_dict['ot'] = ot_l
            
            if lambda_smooth > 0 and ppi_adjacency_arg is not None:
                smooth_l = graph_smoothness_loss(X_pred, ppi_adjacency_arg)
                total_loss = total_loss + lambda_smooth * smooth_l
                loss_dict['smooth'] = smooth_l
            
            if lambda_dir > 0:
                dir_l = direction_loss(X_pred, X_true, X_ctrl)
                total_loss = total_loss + lambda_dir * dir_l
                loss_dict['dir'] = dir_l
            
            return total_loss, loss_dict
        
        # =====================================================================
        # Strategy-specific loss functions
        # =====================================================================
        
        # Loss function with ALL trainable params: flow, encoder, decoder
        def e2e_gene_loss_fn_all(
            flow_params, enc_params, dec_params,
            X_source_genes, condition,
            X_true_genes,
            X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Full E2E: encode -> ODE -> decode -> loss. All params trainable."""
            # Encode source genes to latent
            X_source_latent = encode_genes_to_latent_separate(X_source_genes, enc_params, X_mean, ppi_mask, scgpt_pe)
            
            # Solve ODE
            pred_latent = solve_ode_batch(
                X_source_latent, condition, flow_params,
                cf.solver.vf_state.apply_fn, None
            )
            
            # Decode to gene space
            X_pred_genes = decode_latent_to_genes_separate(pred_latent, dec_params, X_mean, ppi_mask, scgpt_pe)
            
            # Compute loss
            total_loss, loss_dict = compute_gene_loss_batch(
                X_pred_genes, X_true_genes,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            return total_loss, loss_dict
        
        # Loss function for flow_only (original behavior)
        def e2e_gene_loss_fn_flow_only(
            flow_params,
            X_source_latent, condition,
            X_true_genes,
            enc_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            control_condition,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static, control_identity_weight
        ):
            """Flow only: ODE -> decode (frozen) -> loss."""
            pred_latent = solve_ode_batch(
                X_source_latent, condition, flow_params,
                cf.solver.vf_state.apply_fn, None
            )
            
            # Use frozen decoder params
            full_params = {'encoder': enc_params_frozen['encoder'], 'decoder': enc_params_frozen['decoder']}
            def decode_fn(model_instance, lat):
                return model_instance.decoder(lat, ppi_mask, scgpt_pe, training=False)
            X_pred_genes = encoder.model.apply({'params': full_params}, pred_latent, method=decode_fn) + X_mean
            
            total_loss, loss_dict = compute_gene_loss_batch(
                X_pred_genes, X_true_genes,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )

            if control_identity_weight > 0 and control_condition is not None:
                pred_ctrl_latent = solve_ode_batch(
                    X_source_latent, control_condition, flow_params,
                    cf.solver.vf_state.apply_fn, None
                )
                X_pred_ctrl_genes = encoder.model.apply({'params': full_params}, pred_ctrl_latent, method=decode_fn) + X_mean
                ctrl_identity_l = jnp.mean((X_pred_ctrl_genes - X_ctrl) ** 2)
                total_loss = total_loss + control_identity_weight * ctrl_identity_l
                loss_dict['ctrl_identity'] = ctrl_identity_l
            return total_loss, loss_dict
        
        # Loss function for flow + encoder (decoder frozen)
        def e2e_gene_loss_fn_flow_encoder(
            flow_params, enc_params,
            X_source_genes, condition,
            X_true_genes,
            dec_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Flow + Encoder trainable, decoder frozen."""
            X_source_latent = encode_genes_to_latent_separate(X_source_genes, enc_params, X_mean, ppi_mask, scgpt_pe)
            pred_latent = solve_ode_batch(X_source_latent, condition, flow_params, cf.solver.vf_state.apply_fn, None)
            X_pred_genes = decode_latent_to_genes_separate(pred_latent, dec_params_frozen, X_mean, ppi_mask, scgpt_pe)
            
            total_loss, loss_dict = compute_gene_loss_batch(
                X_pred_genes, X_true_genes, pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg, lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            return total_loss, loss_dict
        
        # Loss function for flow + decoder (encoder frozen)
        def e2e_gene_loss_fn_flow_decoder(
            flow_params, dec_params,
            X_source_latent, condition,
            X_true_genes,
            X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Flow + Decoder trainable, encoder frozen (use pre-computed latent)."""
            pred_latent = solve_ode_batch(X_source_latent, condition, flow_params, cf.solver.vf_state.apply_fn, None)
            X_pred_genes = decode_latent_to_genes_separate(pred_latent, dec_params, X_mean, ppi_mask, scgpt_pe)
            
            total_loss, loss_dict = compute_gene_loss_batch(
                X_pred_genes, X_true_genes, pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg, lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            return total_loss, loss_dict
        
        # Loss function for encoder + decoder only (flow frozen)
        def e2e_gene_loss_fn_encoder_decoder(
            enc_params, dec_params,
            X_source_genes, condition,
            X_true_genes,
            flow_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Encoder + Decoder trainable, flow frozen."""
            X_source_latent = encode_genes_to_latent_separate(X_source_genes, enc_params, X_mean, ppi_mask, scgpt_pe)
            pred_latent = solve_ode_batch(X_source_latent, condition, flow_params_frozen, cf.solver.vf_state.apply_fn, None)
            X_pred_genes = decode_latent_to_genes_separate(pred_latent, dec_params, X_mean, ppi_mask, scgpt_pe)
            
            total_loss, loss_dict = compute_gene_loss_batch(
                X_pred_genes, X_true_genes, pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg, lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            return total_loss, loss_dict
        
        # Loss function for encoder only
        def e2e_gene_loss_fn_encoder_only(
            enc_params,
            X_source_genes, condition,
            X_true_genes,
            flow_params_frozen, dec_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Encoder only trainable."""
            X_source_latent = encode_genes_to_latent_separate(X_source_genes, enc_params, X_mean, ppi_mask, scgpt_pe)
            pred_latent = solve_ode_batch(X_source_latent, condition, flow_params_frozen, cf.solver.vf_state.apply_fn, None)
            X_pred_genes = decode_latent_to_genes_separate(pred_latent, dec_params_frozen, X_mean, ppi_mask, scgpt_pe)
            
            total_loss, loss_dict = compute_gene_loss_batch(
                X_pred_genes, X_true_genes, pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg, lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            return total_loss, loss_dict
        
        # Loss function for decoder only
        def e2e_gene_loss_fn_decoder_only(
            dec_params,
            X_source_latent, condition,
            X_true_genes,
            flow_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Decoder only trainable."""
            pred_latent = solve_ode_batch(X_source_latent, condition, flow_params_frozen, cf.solver.vf_state.apply_fn, None)
            X_pred_genes = decode_latent_to_genes_separate(pred_latent, dec_params, X_mean, ppi_mask, scgpt_pe)
            
            total_loss, loss_dict = compute_gene_loss_batch(
                X_pred_genes, X_true_genes, pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg, lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            return total_loss, loss_dict
        
        # =====================================================================
        # Strategy-specific gradient step functions
        # =====================================================================
        
        # JIT-compiled gradient step for flow_only
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static', 'control_identity_weight'])
        def e2e_gene_loss_step_flow_only(
            flow_params, flow_opt_st,
            X_source_latent, condition,
            X_true_genes,
            enc_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            control_condition,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static, control_identity_weight
        ):
            """Gradient step for flow_only strategy."""
            grad_fn = jax.value_and_grad(e2e_gene_loss_fn_flow_only, argnums=0, has_aux=True)
            (loss, loss_dict), flow_grads = grad_fn(
                flow_params, X_source_latent, condition, X_true_genes,
                enc_params_frozen, X_mean, ppi_mask, scgpt_pe,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                control_condition,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static, control_identity_weight
            )
            flow_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), flow_grads)
            flow_updates, new_flow_opt_st = flow_optimizer.update(flow_grads, flow_opt_st, flow_params)
            new_flow_params = optax.apply_updates(flow_params, flow_updates)
            return new_flow_params, None, None, new_flow_opt_st, None, None, loss, loss_dict
        
        # JIT-compiled gradient step for all (flow + encoder + decoder)
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static'])
        def e2e_gene_loss_step_all(
            flow_params, enc_params, dec_params,
            flow_opt_st, enc_opt_st, dec_opt_st,
            X_source_genes, condition,
            X_true_genes,
            X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Gradient step for all strategy: flow + encoder + decoder."""
            grad_fn = jax.value_and_grad(e2e_gene_loss_fn_all, argnums=(0, 1, 2), has_aux=True)
            (loss, loss_dict), (flow_grads, enc_grads, dec_grads) = grad_fn(
                flow_params, enc_params, dec_params,
                X_source_genes, condition, X_true_genes,
                X_mean, ppi_mask, scgpt_pe,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            # Clip gradients
            flow_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), flow_grads)
            enc_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), enc_grads)
            dec_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), dec_grads)
            # Update
            flow_updates, new_flow_opt_st = flow_optimizer.update(flow_grads, flow_opt_st, flow_params)
            enc_updates, new_enc_opt_st = encoder_optimizer.update(enc_grads, enc_opt_st, enc_params)
            dec_updates, new_dec_opt_st = decoder_optimizer.update(dec_grads, dec_opt_st, dec_params)
            new_flow_params = optax.apply_updates(flow_params, flow_updates)
            new_enc_params = optax.apply_updates(enc_params, enc_updates)
            new_dec_params = optax.apply_updates(dec_params, dec_updates)
            return new_flow_params, new_enc_params, new_dec_params, new_flow_opt_st, new_enc_opt_st, new_dec_opt_st, loss, loss_dict
        
        # JIT-compiled gradient step for flow_encoder
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static'])
        def e2e_gene_loss_step_flow_encoder(
            flow_params, enc_params,
            flow_opt_st, enc_opt_st,
            X_source_genes, condition,
            X_true_genes,
            dec_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Gradient step for flow_encoder strategy."""
            grad_fn = jax.value_and_grad(e2e_gene_loss_fn_flow_encoder, argnums=(0, 1), has_aux=True)
            (loss, loss_dict), (flow_grads, enc_grads) = grad_fn(
                flow_params, enc_params,
                X_source_genes, condition, X_true_genes,
                dec_params_frozen, X_mean, ppi_mask, scgpt_pe,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            flow_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), flow_grads)
            enc_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), enc_grads)
            flow_updates, new_flow_opt_st = flow_optimizer.update(flow_grads, flow_opt_st, flow_params)
            enc_updates, new_enc_opt_st = encoder_optimizer.update(enc_grads, enc_opt_st, enc_params)
            new_flow_params = optax.apply_updates(flow_params, flow_updates)
            new_enc_params = optax.apply_updates(enc_params, enc_updates)
            return new_flow_params, new_enc_params, None, new_flow_opt_st, new_enc_opt_st, None, loss, loss_dict
        
        # JIT-compiled gradient step for flow_decoder
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static'])
        def e2e_gene_loss_step_flow_decoder(
            flow_params, dec_params,
            flow_opt_st, dec_opt_st,
            X_source_latent, condition,
            X_true_genes,
            X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Gradient step for flow_decoder strategy."""
            grad_fn = jax.value_and_grad(e2e_gene_loss_fn_flow_decoder, argnums=(0, 1), has_aux=True)
            (loss, loss_dict), (flow_grads, dec_grads) = grad_fn(
                flow_params, dec_params,
                X_source_latent, condition, X_true_genes,
                X_mean, ppi_mask, scgpt_pe,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            flow_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), flow_grads)
            dec_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), dec_grads)
            flow_updates, new_flow_opt_st = flow_optimizer.update(flow_grads, flow_opt_st, flow_params)
            dec_updates, new_dec_opt_st = decoder_optimizer.update(dec_grads, dec_opt_st, dec_params)
            new_flow_params = optax.apply_updates(flow_params, flow_updates)
            new_dec_params = optax.apply_updates(dec_params, dec_updates)
            return new_flow_params, None, new_dec_params, new_flow_opt_st, None, new_dec_opt_st, loss, loss_dict
        
        # JIT-compiled gradient step for encoder_decoder (flow frozen)
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static'])
        def e2e_gene_loss_step_encoder_decoder(
            enc_params, dec_params,
            enc_opt_st, dec_opt_st,
            X_source_genes, condition,
            X_true_genes,
            flow_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Gradient step for encoder_decoder strategy."""
            grad_fn = jax.value_and_grad(e2e_gene_loss_fn_encoder_decoder, argnums=(0, 1), has_aux=True)
            (loss, loss_dict), (enc_grads, dec_grads) = grad_fn(
                enc_params, dec_params,
                X_source_genes, condition, X_true_genes,
                flow_params_frozen, X_mean, ppi_mask, scgpt_pe,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            enc_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), enc_grads)
            dec_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), dec_grads)
            enc_updates, new_enc_opt_st = encoder_optimizer.update(enc_grads, enc_opt_st, enc_params)
            dec_updates, new_dec_opt_st = decoder_optimizer.update(dec_grads, dec_opt_st, dec_params)
            new_enc_params = optax.apply_updates(enc_params, enc_updates)
            new_dec_params = optax.apply_updates(dec_params, dec_updates)
            return None, new_enc_params, new_dec_params, None, new_enc_opt_st, new_dec_opt_st, loss, loss_dict
        
        # JIT-compiled gradient step for encoder_only
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static'])
        def e2e_gene_loss_step_encoder_only(
            enc_params, enc_opt_st,
            X_source_genes, condition,
            X_true_genes,
            flow_params_frozen, dec_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Gradient step for encoder_only strategy."""
            grad_fn = jax.value_and_grad(e2e_gene_loss_fn_encoder_only, argnums=0, has_aux=True)
            (loss, loss_dict), enc_grads = grad_fn(
                enc_params,
                X_source_genes, condition, X_true_genes,
                flow_params_frozen, dec_params_frozen, X_mean, ppi_mask, scgpt_pe,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            enc_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), enc_grads)
            enc_updates, new_enc_opt_st = encoder_optimizer.update(enc_grads, enc_opt_st, enc_params)
            new_enc_params = optax.apply_updates(enc_params, enc_updates)
            return None, new_enc_params, None, None, new_enc_opt_st, None, loss, loss_dict
        
        # JIT-compiled gradient step for decoder_only
        @partial(jax.jit, static_argnames=['lambda_corr', 'lambda_ot', 'lambda_smooth', 'lambda_dir', 'use_ppi_corr', 'use_cluster_corr_flag', 'n_clusters_static'])
        def e2e_gene_loss_step_decoder_only(
            dec_params, dec_opt_st,
            X_source_latent, condition,
            X_true_genes,
            flow_params_frozen, X_mean, ppi_mask, scgpt_pe,
            pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
            cluster_labels_arg,
            lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
            use_cluster_corr_flag, n_clusters_static
        ):
            """Gradient step for decoder_only strategy."""
            grad_fn = jax.value_and_grad(e2e_gene_loss_fn_decoder_only, argnums=0, has_aux=True)
            (loss, loss_dict), dec_grads = grad_fn(
                dec_params,
                X_source_latent, condition, X_true_genes,
                flow_params_frozen, X_mean, ppi_mask, scgpt_pe,
                pathway_matrix_arg, ppi_mask_arg, ppi_adjacency_arg, X_ctrl,
                cluster_labels_arg,
                lambda_corr, lambda_ot, lambda_smooth, lambda_dir, use_ppi_corr,
                use_cluster_corr_flag, n_clusters_static
            )
            dec_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), dec_grads)
            dec_updates, new_dec_opt_st = decoder_optimizer.update(dec_grads, dec_opt_st, dec_params)
            new_dec_params = optax.apply_updates(dec_params, dec_updates)
            return None, None, new_dec_params, None, None, new_dec_opt_st, loss, loss_dict
        
        print(f"[OK] E2E gene loss functions prepared (strategy: {FINETUNE_STRATEGY})")
        print(f"  Finetune LRs: Flow={FINETUNE_FLOW_LR}, Encoder={FINETUNE_ENC_LR}, Decoder={FINETUNE_DEC_LR}")
    
    # ========================================================================
    # Evaluation functions
    # ========================================================================
    def compute_de_genes(adata, condition, control_stats, top_n=20):
        condition_mask = (adata.obs['condition'] == condition).values
        if condition_mask.sum() == 0:
            return np.array([])

        obs_subset = adata.obs[condition_mask]
        matched_control = get_control_stats_for_obs(obs_subset, control_stats)
        if matched_control['n_cells'] == 0:
            return np.array([])

        control_mean = matched_control['mean']
        condition_mean = adata.X[condition_mask].mean(axis=0) if not sparse.issparse(adata.X) else \
                        np.array(adata.X[condition_mask].mean(axis=0)).flatten()
        
        delta = np.abs(condition_mean - control_mean)
        return np.argsort(delta)[-top_n:]

    def build_prediction_covariates(obs_subset):
        covariates = {
            'is_control': [False],
            'pert1': [obs_subset['pert1'].iloc[0]],
            'pert2': [obs_subset['pert2'].iloc[0]],
        }
        if 'dose1' in obs_subset.columns and 'dose2' in obs_subset.columns:
            covariates['dose1'] = [float(obs_subset['dose1'].iloc[0])]
            covariates['dose2'] = [float(obs_subset['dose2'].iloc[0])]
        if use_cell_line_context and 'cell_line' in obs_subset.columns:
            covariates['cell_line'] = [str(obs_subset['cell_line'].iloc[0])]
        return pd.DataFrame(covariates)

    def _obs_matches_value(obs_series, value):
        if pd.api.types.is_numeric_dtype(obs_series):
            try:
                return np.isclose(obs_series.to_numpy(dtype=np.float64), float(value), equal_nan=False)
            except (TypeError, ValueError):
                pass
        return obs_series.astype(str).to_numpy() == str(value)

    def build_gene_loss_target_mask(pert_idx):
        pert_cov_values = cf.train_data.perturbation_idx_to_covariates.get(pert_idx)
        data_manager = getattr(cf.train_data, 'data_manager', None)
        if pert_cov_values is not None and data_manager is not None:
            split_keys = list(data_manager.split_covariates)
            sample_keys = split_keys if split_keys else list(data_manager.sample_covariates)
            perturb_keys = [key for key in data_manager.perturb_covar_keys if key not in sample_keys]
            match_keys = perturb_keys + sample_keys
            if len(pert_cov_values) >= len(match_keys):
                mask = np.ones(adata_train.n_obs, dtype=bool)
                match_desc = []
                for key, value in zip(match_keys, pert_cov_values[:len(match_keys)]):
                    if key not in adata_train.obs.columns:
                        mask = None
                        break
                    mask &= _obs_matches_value(adata_train.obs[key], value)
                    match_desc.append(f"{key}={value}")
                if mask is not None:
                    return mask, ", ".join(match_desc)

        cond_name = None
        if hasattr(cf.train_data, 'perturbation_idx_to_id'):
            cond_name = cf.train_data.perturbation_idx_to_id.get(pert_idx, None)
        if cond_name is None and pert_cov_values is not None:
            cond_name = '_'.join(str(c) for c in pert_cov_values)
        if cond_name is not None:
            cond_name = str(cond_name)
            for cond_col in ('condition', 'condition_name'):
                if cond_col in adata_train.obs.columns:
                    mask = (adata_train.obs[cond_col].astype(str) == cond_name).to_numpy()
                    if mask.any():
                        return mask, f"{cond_col}={cond_name}"

        return np.zeros(adata_train.n_obs, dtype=bool), f"pert_idx={pert_idx}, covariates={pert_cov_values}"

    def build_control_identity_condition(ctrl_idx):
        """Build a ctrl-condition embedding for a given control population."""
        data_manager = getattr(cf.train_data, 'data_manager', None)
        if data_manager is None:
            return None

        split_keys = list(data_manager.split_covariates)
        split_values = cf.train_data.split_idx_to_covariates.get(int(ctrl_idx), tuple())
        split_map = {key: value for key, value in zip(split_keys, split_values)}

        covariates = {}
        for key in data_manager.perturb_covar_keys:
            if key in split_map:
                covariates[key] = split_map[key]
                continue
            if key in adata_train.obs.columns and pd.api.types.is_numeric_dtype(adata_train.obs[key]):
                covariates[key] = 0.0
            else:
                covariates[key] = 'ctrl'

        if data_manager.control_key in adata_train.obs.columns:
            covariates[data_manager.control_key] = True

        cond_df = pd.DataFrame([covariates])
        cond_data = data_manager.get_condition_data(
            covariate_data=cond_df,
            rep_dict=adata_train.uns,
        )
        return {k: jnp.array(v, dtype=jnp.float32) for k, v in cond_data.condition_data.items()}

    effective_control_identity_weight = CONTROL_IDENTITY_WEIGHT
    control_identity_conditions = {}
    if CONTROL_IDENTITY_WEIGHT > 0:
        if FINETUNE_STRATEGY != 'flow_only':
            print(
                f"  ⚠ control_identity_weight={CONTROL_IDENTITY_WEIGHT} currently only supports "
                f"finetune_strategy=flow_only; disabling auxiliary identity loss."
            )
            effective_control_identity_weight = 0.0
        else:
            print("\n" + "=" * 80)
            print("Preparing control identity auxiliary conditions...")
            print("=" * 80)
            for ctrl_idx in sorted(cf.train_data.control_to_perturbation.keys()):
                try:
                    condition_dict = build_control_identity_condition(ctrl_idx)
                    if condition_dict is not None:
                        control_identity_conditions[int(ctrl_idx)] = condition_dict
                except Exception as e:
                    print(f"  ⚠ Failed to build control identity condition for ctrl_idx={ctrl_idx}: {e}")
            if control_identity_conditions:
                print(
                    f"  [OK] Prepared {len(control_identity_conditions)} control identity conditions "
                    f"(weight={effective_control_identity_weight}, batch_size={CONTROL_IDENTITY_BATCH_SIZE})"
                )
            else:
                print("  ⚠ No control identity conditions prepared; disabling auxiliary identity loss.")
                effective_control_identity_weight = 0.0

    def save_dynamic_refresh_history():
        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            if isinstance(obj, tuple):
                return [_convert(v) for v in obj]
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        history_path = args.dynamic_refresh_history_path or os.path.join(OUTPUT_DIR, 'dynamic_refresh_history.json')
        with open(history_path, 'w') as f:
            json.dump(_convert(dynamic_refresh_history), f, indent=2)

    def attach_runtime_uns(adata_obj, include_encoder=False):
        adata_obj.uns['pert_embeddings'] = pert_embeddings
        if cell_line_embeddings is not None:
            adata_obj.uns['cell_line_embeddings'] = cell_line_embeddings
        if include_encoder:
            adata_obj.uns['neural_encoder'] = encoder

    dynamic_refresh_count = 0

    def build_dynamic_response_signatures():
        control_match_columns = detect_control_match_columns(adata_train, adata_train_control)
        control_stats = build_control_group_stats(adata_train_control, control_match_columns)
        n_take = min(args.dynamic_refresh_n_conditions, len(train_conditions))
        if n_take <= 0:
            return None, {
                'n_requested': args.dynamic_refresh_n_conditions,
                'n_used': 0,
                'n_predicted': 0,
                'n_fallback': 0,
                'sampled_conditions': [],
                'used_conditions': [],
            }

        sampled_conditions = rng_np.choice(train_conditions, size=n_take, replace=False).tolist()
        response_columns = []
        used_conditions = []
        n_predicted = 0
        n_fallback = 0
        first_error = None

        for condition in sampled_conditions:
            condition_mask = (adata_train.obs['condition'].astype(str) == str(condition)).to_numpy()
            if condition_mask.sum() == 0:
                continue

            obs_subset = adata_train.obs.loc[condition_mask]
            matched_control = get_control_stats_for_obs(obs_subset, control_stats)
            control_mean = matched_control['mean']
            response_delta = None

            try:
                cov_df = build_prediction_covariates(obs_subset)
                pred = cf.predict(
                    adata=adata_train_control,
                    covariate_data=cov_df,
                    sample_rep='X_pca'
                )
                if pred and len(pred) > 0:
                    pred_key = next(iter(pred))
                    pred_pca = np.asarray(pred[pred_key], dtype=np.float32)
                    if pred_pca.ndim == 1:
                        pred_pca = pred_pca[None, :]
                    if pred_pca.shape[0] > args.dynamic_refresh_max_cells:
                        chosen = rng_np.choice(pred_pca.shape[0], size=args.dynamic_refresh_max_cells, replace=False)
                        pred_pca = pred_pca[chosen]
                    pred_genes = np.asarray(encoder.inverse_transform(pred_pca), dtype=np.float32)
                    response_delta = pred_genes.mean(axis=0) - control_mean
                    n_predicted += 1
            except Exception as e:
                if first_error is None:
                    first_error = str(e)

            if response_delta is None or not np.all(np.isfinite(response_delta)):
                true_mean = to_dense_array(adata_train.X[condition_mask]).mean(axis=0)
                response_delta = np.asarray(true_mean - control_mean, dtype=np.float32)
                n_fallback += 1

            response_columns.append(np.asarray(response_delta, dtype=np.float32))
            used_conditions.append(str(condition))

        if first_error is not None:
            print(f"  [Dynamic] First prediction fallback reason: {first_error}")

        if not response_columns:
            return None, {
                'n_requested': args.dynamic_refresh_n_conditions,
                'n_used': 0,
                'n_predicted': 0,
                'n_fallback': 0,
                'sampled_conditions': sampled_conditions,
                'used_conditions': [],
            }

        response_matrix = np.stack(response_columns, axis=1).astype(np.float32)
        if response_matrix.shape[1] > 1:
            row_mean = response_matrix.mean(axis=1, keepdims=True)
            row_std = response_matrix.std(axis=1, keepdims=True)
            response_matrix = (response_matrix - row_mean) / np.maximum(row_std, 1e-6)

        return response_matrix, {
            'n_requested': args.dynamic_refresh_n_conditions,
            'n_used': len(used_conditions),
            'n_predicted': n_predicted,
            'n_fallback': n_fallback,
            'sampled_conditions': sampled_conditions,
            'used_conditions': used_conditions,
        }

    def refresh_dynamic_clustering(epoch):
        nonlocal adata_train
        nonlocal adata_val
        nonlocal adata_test
        nonlocal adata_train_control
        nonlocal adata_val_with_control
        nonlocal adata_test_with_control
        nonlocal gene_names
        nonlocal cluster_labels
        nonlocal pathway_matrix
        nonlocal ppi_gene_mask
        nonlocal ppi_adjacency
        nonlocal X_ctrl_mean
        nonlocal cf
        nonlocal dataloader
        nonlocal encoder_X_mean
        nonlocal encoder_ppi_mask
        nonlocal encoder_scgpt_pe
        nonlocal dynamic_static_embeddings
        nonlocal dynamic_refresh_count

        if not args.dynamic_clustering:
            return False
        if dynamic_static_embeddings is None or dynamic_ppi_dict is None:
            print("  [Dynamic] Missing static features; skipping refresh.")
            return False

        print("\n" + "=" * 80)
        print(f"Dynamic clustering refresh after epoch {epoch}")
        print("=" * 80)
        response_matrix, response_meta = build_dynamic_response_signatures()
        if response_matrix is None:
            print("  [Dynamic] No usable response signatures; skipping refresh.")
            dynamic_refresh_history.append({
                'epoch': epoch,
                'refresh_index': dynamic_refresh_count + 1,
                'status': 'skipped_no_signatures',
                **response_meta,
            })
            save_dynamic_refresh_history()
            return False

        current_cluster_labels = np.asarray(cluster_labels, dtype=np.int32) if cluster_labels is not None else np.zeros(len(gene_names), dtype=np.int32)
        refresh_result = dynamic_balanced_grc_recluster(
            gene_names=gene_names,
            static_embeddings=dynamic_static_embeddings,
            response_signatures=response_matrix,
            ppi_dict=dynamic_ppi_dict,
            previous_labels=current_cluster_labels,
            n_clusters=n_clusters,
            cluster_sizes=cluster_size_targets,
            static_weight=args.dynamic_static_weight,
            response_weight=args.dynamic_response_weight,
            ppi_weight=args.dynamic_ppi_weight,
            random_state=SEED + epoch,
        )

        new_gene_names = refresh_result['sorted_gene_names']
        sort_idx = np.asarray(refresh_result['sort_idx'], dtype=np.int32)
        moved_fraction = float(refresh_result['moved_fraction'])
        print(f"  [Dynamic] Used {response_meta['n_used']} conditions ({response_meta['n_predicted']} predicted, {response_meta['n_fallback']} fallback)")
        print(f"  [Dynamic] Gene reassignment fraction: {moved_fraction:.4f}")

        adata_train = adata_train[:, new_gene_names].copy()
        adata_val = adata_val[:, new_gene_names].copy()
        adata_test = adata_test[:, new_gene_names].copy()

        if sparse.issparse(adata_train.X):
            adata_train.X = adata_train.X.toarray()
        if sparse.issparse(adata_val.X):
            adata_val.X = adata_val.X.toarray()
        if sparse.issparse(adata_test.X):
            adata_test.X = adata_test.X.toarray()

        gene_names = list(new_gene_names)
        dynamic_static_embeddings = np.asarray(dynamic_static_embeddings[sort_idx], dtype=np.float32)
        cluster_labels = jnp.array(refresh_result['sorted_cluster_labels'])
        encoder.X_mean = jnp.array(np.take(np.asarray(encoder.X_mean), sort_idx, axis=-1), dtype=jnp.float32)

        if gene_loss_config['use_ppi_masked_corr'] or gene_loss_config['lambda_smooth'] > 0:
            ppi_gene_mask = build_ppi_attention_mask(gene_names, dynamic_ppi_edges, k_hop=2, include_self=True)
            ppi_gene_mask = jnp.array(ppi_gene_mask)
            ppi_adjacency = ppi_mask_to_adjacency(ppi_gene_mask)

        if gene_loss_config['lambda_ot'] > 0:
            pathway_matrix_np, _ = load_pathway_gene_matrix(PATHWAY_PATH, gene_names, is_zip=True)
            pathway_matrix = jnp.array(pathway_matrix_np)

        encoder.set_priors(
            gene_names=gene_names,
            ppi_info_path=PPI_INFO_PATH if actual_encoder_config['attention_type'] == 'ppi_sparse' else None,
            ppi_links_path=PPI_LINKS_PATH if actual_encoder_config['attention_type'] == 'ppi_sparse' else None,
            ppi_min_score=700,
            ppi_k_hop=2,
            scgpt_path=scgpt_path_for_encoder,
            genecompass_path=genecompass_path_for_encoder,
            pathway_gmt_path=PATHWAY_PATH if (actual_encoder_config['use_pathway_pooling'] or actual_encoder_config['use_pathway_loss']) else None,
            pathway_is_zip=True
        )

        adata_train.obsm['X_pca'] = np.asarray(encoder.transform(adata_train.X, batch_size=ENCODER_BATCH_SIZE), dtype=np.float32)
        adata_val.obsm['X_pca'] = np.asarray(encoder.transform(adata_val.X, batch_size=ENCODER_BATCH_SIZE), dtype=np.float32)
        adata_test.obsm['X_pca'] = np.asarray(encoder.transform(adata_test.X, batch_size=ENCODER_BATCH_SIZE), dtype=np.float32)
        adata_train.varm['X_mean'] = np.asarray(encoder.X_mean).reshape(-1, 1)
        adata_train.varm['PCs'] = np.eye(len(gene_names), N_PCS)
        attach_runtime_uns(adata_train, include_encoder=True)
        attach_runtime_uns(adata_val)
        attach_runtime_uns(adata_test)

        train_control_mask = np.array(adata_train.obs['is_control'])
        adata_train_control = adata_train[train_control_mask].copy()
        attach_runtime_uns(adata_train_control, include_encoder=True)
        X_ctrl_mean = jnp.array(np.mean(adata_train_control.X, axis=0), dtype=jnp.float32)

        control_subset_val = sample_controls_for_split(adata_val, max_controls=1000, seed_offset=17)
        control_subset_test = sample_controls_for_split(adata_test, max_controls=1000, seed_offset=29)
        adata_val_with_control = ad.concat([adata_val, control_subset_val]) if control_subset_val.n_obs else adata_val.copy()
        adata_test_with_control = ad.concat([adata_test, control_subset_test]) if control_subset_test.n_obs else adata_test.copy()
        if 'X_pca' not in adata_val_with_control.obsm:
            adata_val_with_control.obsm['X_pca'] = np.asarray(encoder.transform(adata_val_with_control.X, batch_size=ENCODER_BATCH_SIZE), dtype=np.float32)
        if 'X_pca' not in adata_test_with_control.obsm:
            adata_test_with_control.obsm['X_pca'] = np.asarray(encoder.transform(adata_test_with_control.X, batch_size=ENCODER_BATCH_SIZE), dtype=np.float32)
        attach_runtime_uns(adata_val_with_control)
        attach_runtime_uns(adata_test_with_control)

        cf._adata = adata_train
        cf.prepare_data(
            sample_rep='X_pca',
            control_key='is_control',
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
            split_covariates=split_covariates,
            max_combination_length=2,
            null_value=0.0
        )
        cf.prepare_validation_data(
            adata=adata_val_with_control,
            name='val',
            n_conditions_on_log_iteration=min(10, len(adata_val.obs['condition'].unique())),
            n_conditions_on_train_end=len(adata_val.obs['condition'].unique()),
        )
        dataloader = TrainSampler(data=cf.train_data, batch_size=BATCH_SIZE)
        cf.solver.is_trained = True

        encoder_X_mean = jnp.array(encoder.X_mean, dtype=jnp.float32)
        encoder_ppi_mask = encoder.ppi_mask
        encoder_scgpt_pe = encoder.scgpt_pe
        dynamic_refresh_count += 1

        refresh_entry = {
            'epoch': epoch,
            'refresh_index': dynamic_refresh_count,
            'status': 'applied',
            'moved_fraction': moved_fraction,
            'cluster_sizes': refresh_result['cluster_sizes'],
            'label_mapping': refresh_result['label_mapping'],
            **response_meta,
        }
        dynamic_refresh_history.append(refresh_entry)
        save_dynamic_refresh_history()
        print(f"  [Dynamic] Refresh complete. Train control cells: {adata_train_control.n_obs}")
        return True

    def _safe_mean(values):
        return safe_mean(values)

    def print_delta_metrics(metrics, split='test', indent=''):
        print_eval_metric_block(metrics, split=split, indent=indent)
    
    def compute_metrics(adata, predictions, adata_ctrl, split='test', top_n=20):
        results = {'r_all': [], 'r_de': [], 'acc_all': [], 'acc_de': []}
        total_conditions = int(adata.obs['condition'].astype(str).loc[adata.obs['condition'].astype(str) != 'control'].nunique())
        control_match_columns = detect_control_match_columns(adata, adata_ctrl)
        control_stats = build_control_group_stats(adata_ctrl, control_match_columns)
        
        for condition, pred in predictions.items():
            mask = (adata.obs['condition'] == condition).values
            if mask.sum() == 0:
                continue

            obs_subset = adata.obs[mask]
            matched_control = get_control_stats_for_obs(obs_subset, control_stats)
            control_mean = matched_control['mean']
            
            true_mean = adata.X[mask].mean(axis=0) if not sparse.issparse(adata.X) else \
                       np.array(adata.X[mask].mean(axis=0)).flatten()
            
            pred_arr = np.asarray(pred)
            pred_mean = pred_arr if pred_arr.ndim == 1 else pred_arr.mean(axis=0)
            
            true_delta = true_mean - control_mean
            pred_delta = pred_mean - control_mean
            
            r_all = safe_pearson(true_delta, pred_delta)
            if np.isfinite(r_all):
                results['r_all'].append(r_all)
            
            acc_all = ((true_delta > 0) == (pred_delta > 0)).mean()
            results['acc_all'].append(acc_all)
            
            de_genes = compute_de_genes(adata, condition, control_stats, top_n)
            if len(de_genes) > 0:
                r_de = safe_pearson(true_delta[de_genes], pred_delta[de_genes])
                if np.isfinite(r_de):
                    results['r_de'].append(r_de)
                acc_de = ((true_delta[de_genes] > 0) == (pred_delta[de_genes] > 0)).mean()
                results['acc_de'].append(acc_de)
        
        n_predicted = int(len(predictions))
        metrics = {
            f'{split}_r_all': _safe_mean(results['r_all']),
            f'{split}_r_de': _safe_mean(results['r_de']),
            f'{split}_acc_all': _safe_mean(results['acc_all']),
            f'{split}_acc_de': _safe_mean(results['acc_de']),
            f'{split}_n_conditions_predicted': n_predicted,
            f'{split}_n_conditions_total': total_conditions,
            f'{split}_prediction_coverage': (n_predicted / total_conditions) if total_conditions > 0 else float('nan'),
        }
        metrics[f'{split}_Pearson_all'] = metrics[f'{split}_r_all']
        metrics[f'{split}_Pearson_de'] = metrics[f'{split}_r_de']
        metrics[f'{split}_Acc_all'] = metrics[f'{split}_acc_all']
        metrics[f'{split}_Acc_de'] = metrics[f'{split}_acc_de']
        metrics[f'{split}_Pearson_delta_all'] = metrics[f'{split}_r_all']
        metrics[f'{split}_Pearson_delta_de'] = metrics[f'{split}_r_de']
        metrics[f'{split}_Acc_delta_all'] = metrics[f'{split}_acc_all']
        metrics[f'{split}_Acc_delta_de'] = metrics[f'{split}_acc_de']
        return metrics
    
    def get_conditions(adata):
        conditions = []
        seen = set()
        for idx in adata.obs.index:
            cond = adata.obs.loc[idx, 'condition']
            if cond not in seen and cond != 'control':
                seen.add(cond)
                conditions.append(cond)
        return conditions
    
    test_conditions = get_conditions(adata_test)
    val_conditions = get_conditions(adata_val)
    
    print(f"\nTest conditions: {len(test_conditions)}")
    print(f"Val conditions: {len(val_conditions)}")
    
    # ========================================================================
    # Training loop with E2E gene-level losses
    # ========================================================================
    print("\n" + "=" * 80)
    print("Starting scBIG training...")
    print("=" * 80)
    
    best_val_r_all = -np.inf
    best_metrics = {}
    training_history = []
    gene_loss_history = []
    
    rng_jax = jax.random.PRNGKey(SEED)
    rng_np = np.random.default_rng(SEED)
    
    from cellflow.data._dataloader import TrainSampler
    dataloader = TrainSampler(data=cf.train_data, batch_size=BATCH_SIZE)
    
    num_epochs = NUM_ITERATIONS // VALID_FREQ
    
    print(f"Epochs: {num_epochs}")
    print(f"Iterations per epoch: {VALID_FREQ}")
    if any_gene_loss and GENE_LOSS_FREQ > 0:
        print(f"Gene loss updates: every {GENE_LOSS_FREQ} iterations")
    
    # Compute initial metrics and loss to diagnose pretrained model state
    if RESUME_FROM and os.path.exists(RESUME_FROM) and BENCHMARK_FULL_AUGMENTED_STEPS <= 0:
        print("\n" + "=" * 80)
        print("Validating pretrained model before training...")
        print("=" * 80)
        
        # Diagnostic: Check adata_train_control state
        print(f"\n  Diagnostic - adata_train_control:")
        print(f"    Shape: {adata_train_control.shape}")
        print(f"    Has X_pca: {'X_pca' in adata_train_control.obsm}")
        if 'X_pca' in adata_train_control.obsm:
            print(f"    X_pca shape: {adata_train_control.obsm['X_pca'].shape}")
            print(f"    X_pca mean: {adata_train_control.obsm['X_pca'].mean():.4f}")
        print(f"    Has pert_embeddings: {'pert_embeddings' in adata_train_control.uns}")
        if 'pert_embeddings' in adata_train_control.uns:
            print(f"    Num pert_embeddings: {len(adata_train_control.uns['pert_embeddings'])}")
        print(f"    Has neural_encoder: {'neural_encoder' in adata_train_control.uns}")
        
        # Validate initial predictions using same logic as training evaluation
        try:
            val_predictions = {}
            val_conditions_list = list(set(adata_val.obs['condition'].values) - {'control'})[:15]
            print(f"  Validating on {len(val_conditions_list)} conditions...")
            
            first_error_printed = False
            successful_count = 0
            for condition in val_conditions_list:
                try:
                    mask = adata_val.obs['condition'] == condition
                    if mask.sum() == 0:
                        continue
                    
                    obs_subset = adata_val.obs[mask]
                    cov_df = build_prediction_covariates(obs_subset)
                    
                    pred = cf.predict(
                        adata=adata_train_control,
                        covariate_data=cov_df,
                        sample_rep='X_pca'
                    )
                    
                    if pred and len(pred) > 0:
                        key = list(pred.keys())[0]
                        pred_pca = pred[key]
                        n_cells = min(100, int(pred_pca.shape[0]))
                        sample_idx = np.random.choice(pred_pca.shape[0], n_cells, replace=False)
                        pred_pca_sample = pred_pca[sample_idx]
                        genes = np.asarray(encoder.inverse_transform(pred_pca_sample), dtype=np.float32)
                        val_predictions[condition] = genes.mean(axis=0).astype(np.float32)
                        successful_count += 1
                    else:
                        raise RuntimeError("cf.predict returned no outputs")
                except Exception as e:
                    if not first_error_printed:
                        print(f"    Error predicting {condition}: {e}")
                        import traceback
                        traceback.print_exc()
                        first_error_printed = True
                    continue
            
            print(f"  Successfully predicted {successful_count}/{len(val_conditions_list)} conditions")
            
            if len(val_predictions) > 0:
                initial_metrics = compute_metrics(adata_val, val_predictions, adata_train_control, 'val', TOP_DE_GENES)
                print(f"\n  Initial validation metrics (BEFORE any training)")
                print_delta_metrics(initial_metrics, split='val', indent='    ')
                
                if not np.isfinite(initial_metrics['val_r_all']):
                    print(f"\n  Initial val_r_all is NaN.")
                    print(f"     Stopping training because the evaluation pipeline is invalid.")
                    print("=" * 80)
                    raise RuntimeError("Initial evaluation produced NaN metrics")
                if initial_metrics['val_r_all'] < 0.3:
                    print(f"\n  Initial r_all is extremely low ({initial_metrics['val_r_all']:.4f}).")
                    print(f"     Expected: ~0.85 for pretrained model from: {RESUME_FROM}")
                    print(f"     Stopping training because the pretrained model did not load correctly.")
                    print("=" * 80)
                    raise RuntimeError("Pretrained model not loaded correctly - initial r_all too low")
                elif initial_metrics['val_r_all'] < 0.5:
                    print(f"\n  ⚠ WARNING: Initial r_all is low ({initial_metrics['val_r_all']:.4f})!")
                    print(f"     Expected: ~0.85 for a well-trained model")
                    print(f"     Proceeding with caution...")
                elif initial_metrics['val_r_all'] > 0.7:
                    print(f"\n  [OK] Initial metrics look GOOD! Pretrained model loaded correctly.")
                    print(f"     Proceeding with fine-tuning...")
            else:
                print("  Could not compute initial metrics (no successful predictions)")
                print("     Stopping training because the pretrained model could not be verified.")
                print("=" * 80)
                raise RuntimeError("Could not compute initial metrics - no successful predictions")
        except Exception as e:
            print(f"  Failed to test loaded model: {e}")
            import traceback
            traceback.print_exc()
            print("     Stopping training because the pretrained model could not be verified.")
            print("=" * 80)
            raise
        
        print("=" * 80 + "\n")

    def block_until_ready_tree(tree):
        """Synchronize a pytree by blocking on its first leaf."""
        leaves = jax.tree_util.tree_leaves(tree)
        if leaves:
            jax.block_until_ready(leaves[0])
        return tree

    if BENCHMARK_FULL_AUGMENTED_STEPS > 0:
        if not any_gene_loss or GENE_LOSS_FREQ <= 0:
            raise ValueError("benchmark_full_augmented_steps requires an active biological loss configuration")

        print("\n" + "=" * 80)
        print("Benchmarking full augmented iterations...")
        print("=" * 80)
        print(f"  Warmup steps: {BENCHMARK_WARMUP_STEPS}")
        print(f"  Measured steps: {BENCHMARK_FULL_AUGMENTED_STEPS}")
        print(f"  Finetune strategy: {FINETUNE_STRATEGY}")
        print(f"  Gene loss frequency in training config: every {GENE_LOSS_FREQ} iterations")

        bench_state = {
            'rng_jax': rng_jax,
            'flow_opt_state': flow_opt_state,
            'enc_opt_state': enc_opt_state,
            'dec_opt_state': dec_opt_state,
            'enc_only_params': enc_only_params,
            'dec_only_params': dec_only_params,
            'full_enc_params': full_enc_params,
        }

        def run_full_augmented_iteration(global_it):
            bench_state['rng_jax'], rng_step = jax.random.split(bench_state['rng_jax'])
            batch = dataloader.sample(rng_np)

            t_flow_start = time.time()
            flow_loss = cf.solver.step_fn(rng_step, batch)
            jax.block_until_ready(flow_loss)
            flow_step_sec = time.time() - t_flow_start

            cf.solver.is_trained = True
            n_perturbations = cf.train_data.n_perturbations
            sampled_pert_idxs = rng_np.choice(
                n_perturbations,
                size=min(N_GENE_LOSS_SAMPLES, n_perturbations),
                replace=False
            )

            batch_gene_losses = {'corr': [], 'ot': [], 'smooth': [], 'dir': [], 'ctrl_identity': [], 'total': []}
            n_gene_updates = 0

            t_gene_start = time.time()
            for pert_idx in sampled_pert_idxs:
                try:
                    condition_dict = {
                        k: jnp.array(v[[pert_idx], ...])
                        for k, v in cf.train_data.condition_data.items()
                    }

                    ctrl_idx_for_pert = None
                    for ctrl_idx, pert_list in cf.train_data.control_to_perturbation.items():
                        if pert_idx in pert_list:
                            ctrl_idx_for_pert = ctrl_idx
                            break

                    if ctrl_idx_for_pert is None:
                        continue

                    ctrl_mask = cf.train_data.split_covariates_mask == ctrl_idx_for_pert
                    ctrl_indices = np.where(ctrl_mask)[0]
                    n_sample = min(GENE_LOSS_BATCH_SIZE, len(ctrl_indices))
                    sampled_ctrl_idx = rng_np.choice(ctrl_indices, size=n_sample, replace=False)
                    X_ctrl_latent = jnp.array(cf.train_data.cell_data[sampled_ctrl_idx], dtype=jnp.float32)

                    X_ctrl_genes = jnp.array(
                        adata_train.X[sampled_ctrl_idx]
                        if not sparse.issparse(adata_train.X)
                        else adata_train.X[sampled_ctrl_idx].toarray(),
                        dtype=jnp.float32
                    )

                    pert_mask = cf.train_data.perturbation_covariates_mask == pert_idx
                    pert_indices = np.where(pert_mask)[0]
                    n_true = min(n_sample, len(pert_indices))
                    rng_np.choice(pert_indices, size=n_true, replace=False)

                    cond_mask, _ = build_gene_loss_target_mask(pert_idx)
                    if cond_mask.sum() == 0:
                        continue

                    true_idx = rng_np.choice(cond_mask.sum(), size=min(n_sample, cond_mask.sum()), replace=False)
                    X_true_genes = jnp.array(
                        adata_train.X[cond_mask][true_idx]
                        if not sparse.issparse(adata_train.X)
                        else adata_train.X[cond_mask][true_idx].toarray(),
                        dtype=jnp.float32
                    )

                    current_flow_params = cf.solver.vf_state.params
                    static_args = (
                        gene_loss_config['lambda_corr'],
                        gene_loss_config['lambda_ot'],
                        gene_loss_config['lambda_smooth'],
                        gene_loss_config['lambda_dir'],
                        gene_loss_config['use_ppi_masked_corr'],
                        use_cluster_corr,
                        n_clusters
                    )

                    control_identity_condition = control_identity_conditions.get(int(ctrl_idx_for_pert))

                    if FINETUNE_STRATEGY == 'flow_only':
                        new_flow, new_enc, new_dec, new_flow_opt_state, new_enc_opt_state, new_dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_flow_only(
                            current_flow_params, bench_state['flow_opt_state'],
                            X_ctrl_latent, condition_dict, X_true_genes,
                            bench_state['full_enc_params'], encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                            pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                            cluster_labels, control_identity_condition, *static_args, effective_control_identity_weight
                        )
                    elif FINETUNE_STRATEGY == 'all':
                        new_flow, new_enc, new_dec, new_flow_opt_state, new_enc_opt_state, new_dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_all(
                            current_flow_params, bench_state['enc_only_params'], bench_state['dec_only_params'],
                            bench_state['flow_opt_state'], bench_state['enc_opt_state'], bench_state['dec_opt_state'],
                            X_ctrl_genes, condition_dict, X_true_genes,
                            encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                            pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                            cluster_labels, *static_args
                        )
                    elif FINETUNE_STRATEGY == 'flow_encoder':
                        new_flow, new_enc, new_dec, new_flow_opt_state, new_enc_opt_state, new_dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_flow_encoder(
                            current_flow_params, bench_state['enc_only_params'],
                            bench_state['flow_opt_state'], bench_state['enc_opt_state'],
                            X_ctrl_genes, condition_dict, X_true_genes,
                            bench_state['dec_only_params'], encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                            pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                            cluster_labels, *static_args
                        )
                    elif FINETUNE_STRATEGY == 'flow_decoder':
                        new_flow, new_enc, new_dec, new_flow_opt_state, new_enc_opt_state, new_dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_flow_decoder(
                            current_flow_params, bench_state['dec_only_params'],
                            bench_state['flow_opt_state'], bench_state['dec_opt_state'],
                            X_ctrl_latent, condition_dict, X_true_genes,
                            encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                            pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                            cluster_labels, *static_args
                        )
                    elif FINETUNE_STRATEGY == 'encoder_decoder':
                        new_flow, new_enc, new_dec, new_flow_opt_state, new_enc_opt_state, new_dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_encoder_decoder(
                            bench_state['enc_only_params'], bench_state['dec_only_params'],
                            bench_state['enc_opt_state'], bench_state['dec_opt_state'],
                            X_ctrl_genes, condition_dict, X_true_genes,
                            current_flow_params, encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                            pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                            cluster_labels, *static_args
                        )
                    elif FINETUNE_STRATEGY == 'encoder_only':
                        new_flow, new_enc, new_dec, new_flow_opt_state, new_enc_opt_state, new_dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_encoder_only(
                            bench_state['enc_only_params'], bench_state['enc_opt_state'],
                            X_ctrl_genes, condition_dict, X_true_genes,
                            current_flow_params, bench_state['dec_only_params'], encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                            pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                            cluster_labels, *static_args
                        )
                    elif FINETUNE_STRATEGY == 'decoder_only':
                        new_flow, new_enc, new_dec, new_flow_opt_state, new_enc_opt_state, new_dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_decoder_only(
                            bench_state['dec_only_params'], bench_state['dec_opt_state'],
                            X_ctrl_latent, condition_dict, X_true_genes,
                            current_flow_params, encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                            pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                            cluster_labels, *static_args
                        )
                    else:
                        raise ValueError(f"Unsupported finetune_strategy for benchmark: {FINETUNE_STRATEGY}")

                    block_until_ready_tree(total_gene_loss)
                    if new_flow is not None:
                        block_until_ready_tree(new_flow)
                        cf.solver.vf_state = cf.solver.vf_state.replace(params=new_flow)
                        cf.solver.vf_state_inference = cf.solver.vf_state_inference.replace(params=new_flow)
                    if new_enc is not None:
                        block_until_ready_tree(new_enc)
                        bench_state['enc_only_params'] = new_enc
                    if new_dec is not None:
                        block_until_ready_tree(new_dec)
                        bench_state['dec_only_params'] = new_dec

                    bench_state['flow_opt_state'] = new_flow_opt_state
                    bench_state['enc_opt_state'] = new_enc_opt_state
                    bench_state['dec_opt_state'] = new_dec_opt_state
                    bench_state['full_enc_params'] = {
                        'encoder': bench_state['enc_only_params'],
                        'decoder': bench_state['dec_only_params']
                    }
                    encoder.state = encoder.state.replace(params=bench_state['full_enc_params'])

                    n_gene_updates += 1
                    batch_gene_losses['total'].append(float(total_gene_loss))
                    for k, v in loss_dict.items():
                        if k in batch_gene_losses:
                            batch_gene_losses[k].append(float(v))
                except Exception as e:
                    print(f"[Benchmark] Gene loss update failed for pert_idx {pert_idx}: {e}", flush=True)
                    continue

            gene_step_sec = time.time() - t_gene_start

            return {
                'iteration': global_it,
                'flow_loss': float(flow_loss),
                'flow_step_ms': flow_step_sec * 1000.0,
                'gene_block_ms': gene_step_sec * 1000.0,
                'full_augmented_step_ms': (flow_step_sec + gene_step_sec) * 1000.0,
                'effective_step_ms_at_configured_freq': (flow_step_sec + gene_step_sec / GENE_LOSS_FREQ) * 1000.0,
                'n_gene_updates': n_gene_updates,
                'gene_loss_total': float(np.mean(batch_gene_losses['total'])) if batch_gene_losses['total'] else 0.0,
                'gene_loss_corr': float(np.mean(batch_gene_losses['corr'])) if batch_gene_losses['corr'] else 0.0,
                'gene_loss_ot': float(np.mean(batch_gene_losses['ot'])) if batch_gene_losses['ot'] else 0.0,
                'gene_loss_ctrl_identity': float(np.mean(batch_gene_losses['ctrl_identity'])) if batch_gene_losses['ctrl_identity'] else 0.0,
            }

        total_steps = BENCHMARK_WARMUP_STEPS + BENCHMARK_FULL_AUGMENTED_STEPS
        benchmark_records = []
        for step_idx in range(total_steps):
            record = run_full_augmented_iteration(step_idx + 1)
            phase = 'warmup' if step_idx < BENCHMARK_WARMUP_STEPS else 'measure'
            print(
                f"[Benchmark][{phase}] step={step_idx + 1} "
                f"flow={record['flow_step_ms']:.2f} ms "
                f"gene={record['gene_block_ms']:.2f} ms "
                f"full={record['full_augmented_step_ms']:.2f} ms "
                f"updates={record['n_gene_updates']}",
                flush=True
            )
            if phase == 'measure':
                benchmark_records.append(record)

        benchmark_summary = {
            'dataset': DATASET,
            'resume_from': RESUME_FROM,
            'output_dir': OUTPUT_DIR,
            'finetune_strategy': FINETUNE_STRATEGY,
            'gene_loss_config': GENE_LOSS_CONFIG,
            'benchmark_warmup_steps': BENCHMARK_WARMUP_STEPS,
            'benchmark_measured_steps': BENCHMARK_FULL_AUGMENTED_STEPS,
            'gene_loss_freq': GENE_LOSS_FREQ,
            'batch_size_main_flow': BATCH_SIZE,
            'gene_loss_batch_size': GENE_LOSS_BATCH_SIZE,
            'mean_flow_step_ms': float(np.mean([r['flow_step_ms'] for r in benchmark_records])),
            'mean_gene_block_ms': float(np.mean([r['gene_block_ms'] for r in benchmark_records])),
            'mean_full_augmented_step_ms': float(np.mean([r['full_augmented_step_ms'] for r in benchmark_records])),
            'mean_effective_step_ms_at_configured_freq': float(np.mean([r['effective_step_ms_at_configured_freq'] for r in benchmark_records])),
            'mean_n_gene_updates': float(np.mean([r['n_gene_updates'] for r in benchmark_records])),
            'records': benchmark_records,
        }

        benchmark_output_path = BENCHMARK_OUTPUT_PATH or os.path.join(OUTPUT_DIR, 'benchmark_full_augmented_step.json')
        os.makedirs(os.path.dirname(benchmark_output_path), exist_ok=True)
        with open(benchmark_output_path, 'w') as f:
            json.dump(benchmark_summary, f, indent=2)

        print("\nBenchmark summary:")
        print(f"  Mean flow step: {benchmark_summary['mean_flow_step_ms']:.2f} ms")
        print(f"  Mean gene block: {benchmark_summary['mean_gene_block_ms']:.2f} ms")
        print(f"  Mean full augmented step: {benchmark_summary['mean_full_augmented_step_ms']:.2f} ms")
        print(f"  Mean effective step @ freq={GENE_LOSS_FREQ}: {benchmark_summary['mean_effective_step_ms_at_configured_freq']:.4f} ms")
        print(f"  Saved benchmark JSON to: {benchmark_output_path}")
        raise SystemExit(0)
    
    for epoch_idx in range(num_epochs):
        epoch = epoch_idx + 1
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        epoch_losses = []
        epoch_gene_losses = {'corr': [], 'ot': [], 'smooth': [], 'dir': [], 'ctrl_identity': [], 'total': []}
        step_times = []  # Track step times
        
        pbar = tqdm(range(VALID_FREQ), desc="Training")
        
        for it in pbar:
            global_it = epoch_idx * VALID_FREQ + it
            
            # Standard flow training step with timing
            rng_jax, rng_step = jax.random.split(rng_jax)
            batch = dataloader.sample(rng_np)
            
            t_step_start = time.time()
            loss = cf.solver.step_fn(rng_step, batch)
            step_times.append(time.time() - t_step_start)
            
            epoch_losses.append(float(loss))
            
            # E2E Gene Loss Update (periodically) - with actual gradient updates!
            if any_gene_loss and GENE_LOSS_FREQ > 0 and (global_it + 1) % GENE_LOSS_FREQ == 0:
                cf.solver.is_trained = True
                
                # Debug: First gene loss update announcement
                if global_it + 1 == GENE_LOSS_FREQ:
                    print(f"\n[Gene loss] starting updates at iteration {global_it+1}", flush=True)
                    print(f"[Gene loss] lambda_corr={gene_loss_config['lambda_corr']}, lambda_ot={gene_loss_config['lambda_ot']}", flush=True)
                
                # Sample random perturbation indices from training data
                n_perturbations = cf.train_data.n_perturbations
                sampled_pert_idxs = rng_np.choice(
                    n_perturbations, 
                    size=min(N_GENE_LOSS_SAMPLES, n_perturbations),
                    replace=False
                )
                
                batch_gene_losses = {'corr': [], 'ot': [], 'smooth': [], 'dir': [], 'ctrl_identity': [], 'total': []}
                n_gene_updates = 0
                
                for pert_idx in sampled_pert_idxs:
                    try:
                        # Get condition data from training data
                        condition_dict = {k: jnp.array(v[[pert_idx], ...]) 
                                         for k, v in cf.train_data.condition_data.items()}
                        
                        # Get control index for this perturbation
                        # Find which control this perturbation belongs to
                        ctrl_idx_for_pert = None
                        for ctrl_idx, pert_list in cf.train_data.control_to_perturbation.items():
                            if pert_idx in pert_list:
                                ctrl_idx_for_pert = ctrl_idx
                                break
                        
                        if ctrl_idx_for_pert is None:
                            if global_it + 1 == GENE_LOSS_FREQ:  # Only print for first batch
                                print(f"  [Skip] pert_idx {pert_idx}: ctrl not found", flush=True)
                            continue
                        
                        # Get control cells in latent space
                        ctrl_mask = cf.train_data.split_covariates_mask == ctrl_idx_for_pert
                        ctrl_indices = np.where(ctrl_mask)[0]
                        n_sample = min(GENE_LOSS_BATCH_SIZE, len(ctrl_indices))
                        sampled_ctrl_idx = rng_np.choice(ctrl_indices, size=n_sample, replace=False)
                        X_ctrl_latent = jnp.array(cf.train_data.cell_data[sampled_ctrl_idx], dtype=jnp.float32)
                        
                        # Get control cells in gene space
                        # Note: sampled_ctrl_idx are indices into adata_train (full training data), not adata_train_control
                        X_ctrl_genes = jnp.array(adata_train.X[sampled_ctrl_idx] if not sparse.issparse(adata_train.X) 
                                                  else adata_train.X[sampled_ctrl_idx].toarray(), dtype=jnp.float32)
                        
                        # Get target (perturbed) cells in gene space
                        pert_mask = cf.train_data.perturbation_covariates_mask == pert_idx
                        pert_indices = np.where(pert_mask)[0]
                        n_true = min(n_sample, len(pert_indices))
                        sampled_pert_cells = rng_np.choice(pert_indices, size=n_true, replace=False)
                        
                        # Get true gene expression for this perturbation from adata_train.
                        # For drug+dose datasets, perturbation ids can include dose slots
                        # while `obs['condition']` only stores the drug pair, so match on
                        # the actual perturbation covariates instead of a joined string.
                        cond_mask, cond_match_desc = build_gene_loss_target_mask(pert_idx)
                        if cond_mask.sum() == 0:
                            if global_it + 1 == GENE_LOSS_FREQ:  # Only print for first batch
                                print(f"  [Skip] pert_idx {pert_idx}: no cells matched ({cond_match_desc})", flush=True)
                            continue
                            
                        true_idx = rng_np.choice(cond_mask.sum(), size=min(n_sample, cond_mask.sum()), replace=False)
                        X_true_genes = jnp.array(adata_train.X[cond_mask][true_idx] if not sparse.issparse(adata_train.X)
                                                  else adata_train.X[cond_mask][true_idx].toarray(), dtype=jnp.float32)
                        
                        # === E2E Gradient Update (Strategy-dependent) ===
                        current_flow_params = cf.solver.vf_state.params
                        
                        # Common static args for all strategies
                        static_args = (
                            gene_loss_config['lambda_corr'],
                            gene_loss_config['lambda_ot'],
                            gene_loss_config['lambda_smooth'],
                            gene_loss_config['lambda_dir'],
                            gene_loss_config['use_ppi_masked_corr'],
                            use_cluster_corr,
                            n_clusters
                        )
                        
                        # Execute strategy-specific gradient step
                        control_identity_condition = control_identity_conditions.get(int(ctrl_idx_for_pert))

                        if FINETUNE_STRATEGY == 'flow_only':
                            # Flow only - use pre-computed latent, encoder frozen
                            new_flow, new_enc, new_dec, flow_opt_state, enc_opt_state, dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_flow_only(
                                current_flow_params, flow_opt_state,
                                X_ctrl_latent, condition_dict, X_true_genes,
                                full_enc_params, encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                                pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                                cluster_labels, control_identity_condition, *static_args, effective_control_identity_weight
                            )
                        elif FINETUNE_STRATEGY == 'all':
                            # All - update flow, encoder, decoder
                            new_flow, new_enc, new_dec, flow_opt_state, enc_opt_state, dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_all(
                                current_flow_params, enc_only_params, dec_only_params,
                                flow_opt_state, enc_opt_state, dec_opt_state,
                                X_ctrl_genes, condition_dict, X_true_genes,
                                encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                                pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                                cluster_labels, *static_args
                            )
                        elif FINETUNE_STRATEGY == 'flow_encoder':
                            # Flow + encoder, decoder frozen
                            new_flow, new_enc, new_dec, flow_opt_state, enc_opt_state, dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_flow_encoder(
                                current_flow_params, enc_only_params,
                                flow_opt_state, enc_opt_state,
                                X_ctrl_genes, condition_dict, X_true_genes,
                                dec_only_params, encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                                pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                                cluster_labels, *static_args
                            )
                        elif FINETUNE_STRATEGY == 'flow_decoder':
                            # Flow + decoder, encoder frozen (use pre-computed latent)
                            new_flow, new_enc, new_dec, flow_opt_state, enc_opt_state, dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_flow_decoder(
                                current_flow_params, dec_only_params,
                                flow_opt_state, dec_opt_state,
                                X_ctrl_latent, condition_dict, X_true_genes,
                                encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                                pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                                cluster_labels, *static_args
                            )
                        elif FINETUNE_STRATEGY == 'encoder_decoder':
                            # Encoder + decoder, flow frozen
                            new_flow, new_enc, new_dec, flow_opt_state, enc_opt_state, dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_encoder_decoder(
                                enc_only_params, dec_only_params,
                                enc_opt_state, dec_opt_state,
                                X_ctrl_genes, condition_dict, X_true_genes,
                                current_flow_params, encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                                pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                                cluster_labels, *static_args
                            )
                        elif FINETUNE_STRATEGY == 'encoder_only':
                            # Encoder only
                            new_flow, new_enc, new_dec, flow_opt_state, enc_opt_state, dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_encoder_only(
                                enc_only_params, enc_opt_state,
                                X_ctrl_genes, condition_dict, X_true_genes,
                                current_flow_params, dec_only_params, encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                                pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                                cluster_labels, *static_args
                            )
                        elif FINETUNE_STRATEGY == 'decoder_only':
                            # Decoder only (use pre-computed latent)
                            new_flow, new_enc, new_dec, flow_opt_state, enc_opt_state, dec_opt_state, total_gene_loss, loss_dict = e2e_gene_loss_step_decoder_only(
                                dec_only_params, dec_opt_state,
                                X_ctrl_latent, condition_dict, X_true_genes,
                                current_flow_params, encoder_X_mean, encoder_ppi_mask, encoder_scgpt_pe,
                                pathway_matrix, ppi_gene_mask, ppi_adjacency, X_ctrl_genes,
                                cluster_labels, *static_args
                            )
                        
                        # Update parameters based on what was changed
                        if new_flow is not None:
                            cf.solver.vf_state = cf.solver.vf_state.replace(params=new_flow)
                            cf.solver.vf_state_inference = cf.solver.vf_state_inference.replace(params=new_flow)
                        if new_enc is not None:
                            enc_only_params = new_enc
                            # Sync back to full encoder params
                            full_enc_params = {'encoder': enc_only_params, 'decoder': dec_only_params}
                            encoder.state = encoder.state.replace(params=full_enc_params)
                        if new_dec is not None:
                            dec_only_params = new_dec
                            # Sync back to full encoder params
                            full_enc_params = {'encoder': enc_only_params, 'decoder': dec_only_params}
                            encoder.state = encoder.state.replace(params=full_enc_params)
                        
                        n_gene_updates += 1
                        
                        # Store losses
                        batch_gene_losses['total'].append(float(total_gene_loss))
                        for k, v in loss_dict.items():
                            if k in batch_gene_losses:
                                batch_gene_losses[k].append(float(v))
                            
                    except Exception as e:
                        # Skip this condition if update fails
                        # Always print first 3 errors per batch, then every 50th
                        n_errors = len([x for x in batch_gene_losses['total']]) if batch_gene_losses['total'] else 0
                        total_attempts = n_gene_updates + n_errors + 1
                        if total_attempts <= 3 or pert_idx % 50 == 0:
                            import traceback
                            print(f"Gene loss update failed for pert_idx {pert_idx}: {e}", flush=True)
                            if total_attempts == 1:
                                traceback.print_exc()
                                import sys
                                sys.stdout.flush()
                        continue
                
                # Debug: Print gene loss update status
                if n_gene_updates > 0:
                    avg_loss = np.mean(batch_gene_losses['total']) if batch_gene_losses['total'] else 0
                    print(f"  [Gene Loss] iter={global_it+1}: {n_gene_updates} updates, avg_loss={avg_loss:.6f}", flush=True)
                else:
                    print(f"  [Gene Loss] iter={global_it+1}: NO updates (all perturbations failed)", flush=True)
                
                # Aggregate and store
                for key in batch_gene_losses:
                    if batch_gene_losses[key]:
                        avg = np.mean(batch_gene_losses[key])
                        epoch_gene_losses[key].append(avg)
                
                # Log
                gene_loss_entry = {
                    'iteration': global_it + 1,
                    'n_updates': n_gene_updates,
                    **{k: np.mean(v) if v else 0 for k, v in batch_gene_losses.items()}
                }
                gene_loss_history.append(gene_loss_entry)
            
            # Update progress bar
            if it % 100 == 0:
                postfix = {'loss': f'{np.mean(epoch_losses[-100:]):.4f}'}
                if epoch_gene_losses['total']:
                    postfix['gene'] = f'{np.mean(epoch_gene_losses["total"][-10:]):.4f}'
                pbar.set_postfix(postfix)
        
        avg_loss = np.mean(epoch_losses)
        avg_step_time = np.mean(step_times)
        print(f"Average flow loss: {avg_loss:.4f}")
        print(f"Average step time: {avg_step_time*1000:.2f} ms ({1.0/avg_step_time:.2f} steps/sec)")
        
        if any_gene_loss:
            gene_loss_summary = []
            for key in ['corr', 'ot', 'smooth', 'dir', 'ctrl_identity', 'total']:
                if epoch_gene_losses[key]:
                    gene_loss_summary.append(f"{key}={np.mean(epoch_gene_losses[key]):.4f}")
            if gene_loss_summary:
                print(f"Gene losses: {', '.join(gene_loss_summary)}")
        
        cf.solver.is_trained = True
        
        # Evaluate on validation set. Test metrics are intentionally not used
        # during training or best-checkpoint selection.
        n_cells_limit = args.n_cells_per_condition
        print(f"\nEvaluating on validation set (max {n_cells_limit} cells per condition)...")
        val_predictions = {}
        val_predictions_full = {}  # For saving predictions (cell-level)
        failed_val_conditions = []
        
        for condition in tqdm(val_conditions, desc="Validation"):
            mask = adata_val.obs['condition'] == condition
            if mask.sum() == 0:
                continue
            
            obs_subset = adata_val.obs[mask]
            pert1 = obs_subset['pert1'].iloc[0]
            pert2 = obs_subset['pert2'].iloc[0]
            cov_df = build_prediction_covariates(obs_subset)
            
            try:
                pred = cf.predict(
                    adata=adata_train_control,
                    covariate_data=cov_df,
                    sample_rep='X_pca'
                )
                
                if pred and len(pred) > 0:
                    key = list(pred.keys())[0]
                    pred_pca = pred[key]
                    
                    # Sample cells if exceeding limit
                    n_cells = int(pred_pca.shape[0])
                    if n_cells > n_cells_limit:
                        sample_idx = np.random.choice(n_cells, n_cells_limit, replace=False)
                        pred_pca = pred_pca[sample_idx]
                        n_cells = n_cells_limit
                    
                    chunk_size = 4096
                    sum_genes = None
                    cell_genes_list = []
                    
                    for start in range(0, n_cells, chunk_size):
                        end = min(start + chunk_size, n_cells)
                        chunk = np.array(pred_pca[start:end], dtype=np.float32)
                        genes = np.asarray(encoder.inverse_transform(chunk), dtype=np.float32)
                        cell_genes_list.append(genes)
                        chunk_sum = genes.sum(axis=0, dtype=np.float64)
                        sum_genes = chunk_sum if sum_genes is None else (sum_genes + chunk_sum)
                    
                    val_predictions[condition] = (sum_genes / n_cells).astype(np.float32)
                    
                    # Store cell-level predictions for saving
                    if args.save_best_predictions:
                        cell_genes = np.concatenate(cell_genes_list, axis=0)
                        val_predictions_full[condition] = {
                            'genes': cell_genes,
                            'pert1': pert1,
                            'pert2': pert2
                        }
                else:
                    failed_val_conditions.append(condition)
            except Exception as e:
                print(f"Failed: {condition}: {e}")
                failed_val_conditions.append(condition)
                continue

        successful_val_conditions = len(val_predictions)
        print(
            f"  Validation prediction coverage: {successful_val_conditions}/{len(val_conditions)} conditions"
        )
        if failed_val_conditions:
            print(f"  Failed validation conditions: {', '.join(failed_val_conditions)}")
        if successful_val_conditions == 0:
            raise RuntimeError("All validation condition predictions failed; metrics are invalid.")
        
        # Compute metrics
        if args.full_metrics:
            # Compute all 11 metrics
            control_mean_np = np.mean(adata_train_control.X, axis=0) if not sparse.issparse(adata_train_control.X) \
                              else np.asarray(adata_train_control.X.mean(axis=0)).flatten()
            # Use precomputed DE genes if available, otherwise compute on-the-fly
            if PRECOMPUTED_DE_DICT is not None:
                top_de_dict = PRECOMPUTED_DE_DICT
            else:
                top_de_dict = compute_top_de_genes_inline(adata_val, control_mean_np, top_n=TOP_DE_GENES)
            full_metrics = compute_full_11_metrics(
                val_predictions, adata_val, adata_train_control, 
                control_mean_np, gene_names, top_de_genes=top_de_dict
            )
            val_metrics = {f'val_{k}': v for k, v in full_metrics.items()}
        else:
            val_metrics = compute_metrics(adata_val, val_predictions, adata_train_control, 'val', TOP_DE_GENES)
        
        # Combine metrics
        all_metrics = {
            'epoch': epoch,
            'iteration': epoch * VALID_FREQ,
            'flow_loss': avg_loss,
            'avg_step_time_ms': avg_step_time * 1000,
            'steps_per_sec': 1.0 / avg_step_time,
            **val_metrics,
        }
        
        # Add gene loss metrics
        for key in ['corr', 'ot', 'smooth', 'dir', 'total']:
            if epoch_gene_losses[key]:
                all_metrics[f'gene_loss_{key}'] = np.mean(epoch_gene_losses[key])
        
        training_history.append(all_metrics)
        
        # Print metrics
        print("\nValidation delta metrics:")
        print_delta_metrics(val_metrics, split='val', indent='  ')
        
        # Save best model
        is_new_best = np.isfinite(val_metrics['val_r_all']) and val_metrics['val_r_all'] > best_val_r_all
        if is_new_best:
            best_val_r_all = val_metrics['val_r_all']
            best_metrics = all_metrics.copy()
            
            # Save best model
            if args.save_best_model:
                cf.save(OUTPUT_DIR, file_prefix='best_model', overwrite=True)
                encoder.save(os.path.join(OUTPUT_DIR, 'best_encoder_e2e.pkl'))
            else:
                cf.save(OUTPUT_DIR, file_prefix='best_model', overwrite=True)
                encoder.save(os.path.join(OUTPUT_DIR, 'best_encoder_e2e.pkl'))
            
            # Save predictions for best model
            if args.save_best_predictions and val_predictions_full:
                print("  Saving best predictions...")
                pred_rows = []
                for cond, data in val_predictions_full.items():
                    genes = data['genes']
                    for i in range(genes.shape[0]):
                        row = {'cell_name': f'{cond}_cell_{i}', 'pert1': data['pert1'], 'pert2': data['pert2']}
                        for j, gname in enumerate(gene_names):
                            row[gname] = genes[i, j]
                        pred_rows.append(row)
                pred_df = pd.DataFrame(pred_rows)
                pred_path = os.path.join(OUTPUT_DIR, 'best_val_predictions.csv')
                pred_df.to_csv(pred_path, index=False)
                print(f"  [OK] Saved {len(pred_rows)} cells to {pred_path}")
                
                # Save evaluation metrics JSON
                eval_metrics_path = os.path.join(OUTPUT_DIR, 'validation_metrics.json')
                with open(eval_metrics_path, 'w') as f:
                    json.dump({k.replace('val_', ''): v for k, v in val_metrics.items()}, f, indent=2)
            
            print("[OK] New best validation delta metrics:")
            print_delta_metrics(best_metrics, split='val', indent='  ')
        
        # Clear cell-level predictions to save memory
        val_predictions_full.clear()
        del val_predictions
        gc.collect()
        
        # Save checkpoint for each epoch (optional)
        if not args.skip_checkpoints:
            checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            cf.save(checkpoint_dir, file_prefix=f'model_epoch{epoch}', overwrite=True)
            encoder.save(os.path.join(checkpoint_dir, f'encoder_epoch{epoch}.pkl'))
            print(f"[OK] Epoch {epoch} checkpoint saved")

        should_refresh_dynamic = (
            args.dynamic_clustering and
            epoch < num_epochs and
            dynamic_refresh_count < args.dynamic_refresh_max_refreshes and
            epoch >= args.dynamic_refresh_start_epoch and
            ((epoch - args.dynamic_refresh_start_epoch) % max(args.dynamic_refresh_freq_epochs, 1) == 0)
        )
        if should_refresh_dynamic:
            refresh_dynamic_clustering(epoch)
    
    # ========================================================================
    # Save results
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    # Helper to convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(convert_for_json(training_history), f, indent=2)
    
    # Save gene loss history
    gene_loss_path = os.path.join(OUTPUT_DIR, 'gene_loss_history.json')
    with open(gene_loss_path, 'w') as f:
        json.dump(convert_for_json(gene_loss_history), f, indent=2)

    if args.dynamic_clustering:
        save_dynamic_refresh_history()
    
    # Save config
    resolved_encoder_config = {
        k: v for k, v in encoder_config.items() if not callable(v)
    }
    resolved_encoder_config.update({
        'latent_dim': N_PCS,
        'embed_dim': EMBED_DIM,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'chunk_size': config_chunk_size,
        'num_modules': num_inducing,
        'num_inducing': num_inducing,
    })

    config_info = {
        'encoder_config': ENCODER_CONFIG,
        'encoder_config_details': resolved_encoder_config,
        'gene_loss_config': GENE_LOSS_CONFIG,
        'gene_loss_config_details': gene_loss_config,
        'dataset': DATASET,
        'data_dir': DATA_DIR,
        'cluster_type': CLUSTER_TYPE,
        'use_reordered': USE_REORDERED,
        'n_iterations': NUM_ITERATIONS,
        'valid_freq': VALID_FREQ,
        'gene_loss_freq': GENE_LOSS_FREQ,
        'batch_size': BATCH_SIZE,
        'encoder_n_epochs': ENCODER_N_EPOCHS,
        'encoder_eval_every_epochs': ENCODER_EVAL_EVERY_EPOCHS,
        'encoder_eval_max_cells': ENCODER_EVAL_MAX_CELLS,
        'encoder_eval_cell_line': ENCODER_EVAL_CELL_LINE,
        'encoder_eval_controls_only': ENCODER_EVAL_CONTROLS_ONLY,
        'encoder_early_stop_patience': ENCODER_EARLY_STOP_PATIENCE,
        'encoder_early_stop_min_delta': ENCODER_EARLY_STOP_MIN_DELTA,
        'encoder_eval_source': encoder_eval_label if ENCODER_EVAL_EVERY_EPOCHS > 0 else None,
        'encoder_eval_n_cells': int(encoder_eval_X.shape[0]) if encoder_eval_X is not None else 0,
        'encoder_batch_size': ENCODER_BATCH_SIZE,
        'control_identity_weight': CONTROL_IDENTITY_WEIGHT,
        'effective_control_identity_weight': effective_control_identity_weight,
        'control_identity_batch_size': CONTROL_IDENTITY_BATCH_SIZE,
        'oversample_cell_line': OVERSAMPLE_CELL_LINE,
        'oversample_multiplier': OVERSAMPLE_MULTIPLIER,
        'oversample_controls_only': OVERSAMPLE_CONTROLS_ONLY,
        'oversample_stages': OVERSAMPLE_STAGES,
        'gene_loss_batch_size': GENE_LOSS_BATCH_SIZE,
        'n_pcs': N_PCS,
        'latent_dim': N_PCS,
        'embed_dim': EMBED_DIM,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'chunk_size': config_chunk_size,
        'num_modules': num_inducing,
        'module_pooling': MODULE_POOLING,
        'use_cell_line_context': use_cell_line_context,
        'sample_covariates': sample_covariates,
        'split_covariates': split_covariates,
        'finetune_strategy': FINETUNE_STRATEGY,
        'resume_from': RESUME_FROM,
        'seed': SEED,
        'dynamic_clustering': args.dynamic_clustering,
        'dynamic_refresh_start_epoch': args.dynamic_refresh_start_epoch,
        'dynamic_refresh_freq_epochs': args.dynamic_refresh_freq_epochs,
        'dynamic_refresh_max_refreshes': args.dynamic_refresh_max_refreshes,
        'dynamic_refresh_n_conditions': args.dynamic_refresh_n_conditions,
        'dynamic_refresh_max_cells': args.dynamic_refresh_max_cells,
        'dynamic_static_weight': args.dynamic_static_weight,
        'dynamic_response_weight': args.dynamic_response_weight,
        'dynamic_ppi_weight': args.dynamic_ppi_weight,
        'dynamic_refresh_count': dynamic_refresh_count,
        'dynamic_refresh_history': dynamic_refresh_history,
        'best_metrics': best_metrics
    }
    
    config_path = os.path.join(OUTPUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(convert_for_json(config_info), f, indent=2)
    
    # Final encoder save
    final_encoder_path = os.path.join(OUTPUT_DIR, 'final_encoder.pkl')
    try:
        encoder.save(final_encoder_path)
        print(f"[OK] Final encoder saved to: {final_encoder_path}")
    except Exception as e:
        print(f"⚠ Could not save final encoder: {e}")
    
    print(f"\nBest validation metrics (epoch {best_metrics.get('epoch', 'N/A')}):")
    print_delta_metrics(best_metrics, split='val', indent='  ')
    
    return best_metrics


if __name__ == '__main__':
    run_training()
