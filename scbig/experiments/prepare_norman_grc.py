#!/usr/bin/env python
"""Prepare the Norman additive GRC32 split used by scBIG."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np

from scbig.clustering.grc import (
    build_gene_embedding_matrix,
    formula1_grc_clustering,
    load_genecompass,
    load_string_ppi_adjacency,
    reorder_genes_by_cluster,
    validate_ppi_coherence,
)


def _lookup_embedding(gene, gene_embeddings, symbol_embeddings, symbol_to_ensembl):
    emb = symbol_embeddings.get(gene)
    if emb is None and gene in symbol_to_ensembl:
        emb = gene_embeddings.get(symbol_to_ensembl[gene])
    if emb is None:
        emb = gene_embeddings.get(gene)
    return None if emb is None else np.asarray(emb, dtype=np.float32)


def reorder_adata_file(input_path: Path, sorted_gene_names, output_path: Path) -> None:
    adata = ad.read_h5ad(input_path)
    adata = adata[:, sorted_gene_names].copy()
    adata.write_h5ad(output_path)


def build_cluster_pe(
    sorted_genes,
    sorted_labels,
    gene_embeddings,
    symbol_embeddings,
    symbol_to_ensembl,
    n_clusters,
):
    source = symbol_embeddings if symbol_embeddings else gene_embeddings
    embed_dim = len(next(iter(source.values())))
    pe = np.zeros((n_clusters, embed_dim), dtype=np.float32)

    sorted_genes = np.asarray(sorted_genes)
    sorted_labels = np.asarray(sorted_labels)
    for k in range(n_clusters):
        rows = []
        for gene in sorted_genes[sorted_labels == k]:
            emb = _lookup_embedding(gene, gene_embeddings, symbol_embeddings, symbol_to_ensembl)
            if emb is not None:
                rows.append(emb[:embed_dim])
        if rows:
            pe[k] = np.mean(rows, axis=0)

    return pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-8)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Norman additive GRC32 data")
    parser.add_argument("--data_dir", required=True, help="Directory with raw Norman additive split")
    parser.add_argument("--output_dir", required=True, help="Output directory for GRC-sorted split")
    parser.add_argument("--train_file", default="norman_train_filtered.h5ad")
    parser.add_argument("--test_file", default="norman_test_filtered.h5ad")
    parser.add_argument("--val_file", default="norman_val_filtered.h5ad")
    parser.add_argument("--genecompass_path", required=True)
    parser.add_argument("--ppi_info_path", required=True)
    parser.add_argument("--ppi_links_path", required=True)
    parser.add_argument("--dataset_name", default="norman_additive")
    parser.add_argument("--n_clusters", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0, help="Deprecated alias for --semantic_weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Deprecated alias for --ppi_weight")
    parser.add_argument("--semantic_weight", type=float, default=None, help="Weight for D_sem in formula 1")
    parser.add_argument("--ppi_weight", type=float, default=None, help="Weight for D_PPI in formula 1")
    parser.add_argument("--ppi_min_score", type=int, default=700)
    parser.add_argument("--sinkhorn_reg", type=float, default=0.1)
    parser.add_argument("--n_refinement_steps", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--use_ppi_cost",
        action="store_true",
        help="Deprecated; formula-1 GRC always includes D_PPI in the public release.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_weight = args.semantic_weight if args.semantic_weight is not None else args.alpha
    ppi_weight = args.ppi_weight if args.ppi_weight is not None else args.beta

    train_path = Path(args.data_dir) / args.train_file
    adata_train = ad.read_h5ad(train_path)
    gene_names = list(adata_train.var_names)

    gene_embeddings, symbol_embeddings, symbol_to_ensembl = load_genecompass(args.genecompass_path)
    embeddings, coverage = build_gene_embedding_matrix(
        gene_names,
        gene_embeddings,
        symbol_embeddings,
        symbol_to_ensembl,
        fill_missing=True,
        random_seed=args.random_state,
    )
    ppi_adj, ppi_edge_count = load_string_ppi_adjacency(
        gene_names,
        protein_info_path=args.ppi_info_path,
        links_path=args.ppi_links_path,
        min_score=args.ppi_min_score,
    )

    labels, _ = formula1_grc_clustering(
        embeddings,
        ppi_adj,
        n_clusters=args.n_clusters,
        n_refinement_steps=args.n_refinement_steps,
        sinkhorn_reg=args.sinkhorn_reg,
        semantic_weight=semantic_weight,
        ppi_weight=ppi_weight,
        random_state=args.random_state,
        verbose=True,
    )
    sorted_gene_names, sorted_labels, order = reorder_genes_by_cluster(
        gene_names,
        labels,
        return_order=True,
    )
    cluster_sizes = np.bincount(labels, minlength=args.n_clusters)
    validation = validate_ppi_coherence(
        sorted_labels,
        ppi_adj[np.ix_(order, order)],
        random_state=args.random_state,
    )

    for split, filename in (("train", args.train_file), ("test", args.test_file), ("val", args.val_file)):
        input_path = Path(args.data_dir) / filename
        if input_path.exists():
            reorder_adata_file(input_path, sorted_gene_names, output_dir / f"{split}.h5ad")

    pe = build_cluster_pe(
        sorted_gene_names,
        sorted_labels,
        gene_embeddings,
        symbol_embeddings,
        symbol_to_ensembl,
        args.n_clusters,
    )
    np.save(output_dir / f"{args.dataset_name}_grc{args.n_clusters}_pe.npy", pe)

    metadata = {
        "dataset": args.dataset_name,
        "method": "GRC formula 1 (GeneCompass + STRING PPI + Sinkhorn OT)",
        "formula": "C_ik = D_sem(E_i, mu_k) + D_PPI(i, k)",
        "n_genes": len(sorted_gene_names),
        "n_clusters": args.n_clusters,
        "semantic_weight": semantic_weight,
        "ppi_weight": ppi_weight,
        "alpha": semantic_weight,
        "beta": ppi_weight,
        "ppi_min_score": args.ppi_min_score,
        "sinkhorn_reg": args.sinkhorn_reg,
        "n_refinement_steps": args.n_refinement_steps,
        "random_state": args.random_state,
        "genecompass_coverage": coverage,
        "ppi_edge_count": int(ppi_edge_count),
        "sorted_gene_names": sorted_gene_names,
        "sorted_gene_indices": order.tolist(),
        "sorted_cluster_labels": sorted_labels.astype(int).tolist(),
        "gene_to_cluster": {gene: int(label) for gene, label in zip(sorted_gene_names, sorted_labels)},
        "cluster_sizes": cluster_sizes.astype(int).tolist(),
        "validation": validation,
    }

    with open(output_dir / f"{args.dataset_name}_grc{args.n_clusters}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(output_dir / f"{args.dataset_name}_gene_clusters_{args.n_clusters}_grc.pkl", "wb") as f:
        pickle.dump(
            {
                "sorted_gene_names": sorted_gene_names,
                "sorted_gene_indices": order.tolist(),
                "sorted_cluster_labels": sorted_labels.astype(int).tolist(),
                "n_clusters": args.n_clusters,
                "n_genes": len(sorted_gene_names),
                "semantic_weight": semantic_weight,
                "ppi_weight": ppi_weight,
            },
            f,
        )

    print(f"Saved GRC split to {output_dir}")
    print(f"Genes: {len(sorted_gene_names)}, clusters: {args.n_clusters}")
    print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}")
    print(f"First genes: {', '.join(sorted_gene_names[:5])}")
    print(f"PPI density improvement: {validation['improvement']:.2f}x")


if __name__ == "__main__":
    main()
