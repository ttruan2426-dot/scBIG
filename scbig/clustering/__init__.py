"""
scBIG Clustering - Gene-Relation Clustering (GRC)

Balanced gene clustering using semantic embeddings + PPI priors + OT.
"""

from scbig.clustering.grc import (
    GeneRelationClustering,
    balanced_ot_clustering,
    build_gene_embedding_matrix,
    compute_formula1_cost_matrix,
    formula1_grc_clustering,
    load_genecompass,
    load_gene_embeddings,
    load_ppi_network,
    load_string_ppi_adjacency,
    reorder_genes_by_cluster,
    validate_ppi_coherence,
)

__all__ = [
    "GeneRelationClustering",
    "balanced_ot_clustering",
    "build_gene_embedding_matrix",
    "compute_formula1_cost_matrix",
    "formula1_grc_clustering",
    "load_genecompass",
    "load_gene_embeddings",
    "load_ppi_network",
    "load_string_ppi_adjacency",
    "reorder_genes_by_cluster",
    "validate_ppi_coherence",
]
