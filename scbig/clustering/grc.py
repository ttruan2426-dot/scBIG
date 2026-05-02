from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

try:
    import ot

    HAS_OT = True
except ImportError:
    HAS_OT = False


GeneCompassStores = Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]


def load_genecompass(embedding_path: str) -> GeneCompassStores:
    with open(embedding_path, "rb") as f:
        data = pickle.load(f)
    return (
        data.get("gene_embeddings", {}),
        data.get("gene_symbol_embeddings", {}),
        data.get("symbol_to_ensembl", {}),
    )


def _infer_embedding_dim(gene_embeddings: Dict, symbol_embeddings: Dict) -> int:
    for store in (symbol_embeddings, gene_embeddings):
        if store:
            return len(next(iter(store.values())))
    raise ValueError("Could not infer GeneCompass embedding dimension")


def _lookup_embedding(
    gene: str,
    gene_embeddings: Dict[str, np.ndarray],
    symbol_embeddings: Dict[str, np.ndarray],
    symbol_to_ensembl: Dict[str, str],
) -> Optional[np.ndarray]:
    emb = symbol_embeddings.get(gene)
    if emb is None and gene in symbol_to_ensembl:
        emb = gene_embeddings.get(symbol_to_ensembl[gene])
    if emb is None:
        emb = gene_embeddings.get(gene)
    return None if emb is None else np.asarray(emb, dtype=np.float32)


def build_gene_embedding_matrix(
    gene_names: List[str],
    gene_embeddings: Dict[str, np.ndarray],
    symbol_embeddings: Dict[str, np.ndarray],
    symbol_to_ensembl: Dict[str, str],
    fill_missing: bool = True,
    random_seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Build a GeneCompass matrix aligned to ``gene_names``."""
    embed_dim = _infer_embedding_dim(gene_embeddings, symbol_embeddings)
    matrix = np.zeros((len(gene_names), embed_dim), dtype=np.float32)
    rng = np.random.default_rng(random_seed)
    found = 0

    for i, gene in enumerate(gene_names):
        emb = _lookup_embedding(gene, gene_embeddings, symbol_embeddings, symbol_to_ensembl)
        if emb is not None:
            matrix[i] = emb[:embed_dim]
            found += 1
        elif fill_missing:
            matrix[i] = rng.normal(0.0, 0.1, size=embed_dim).astype(np.float32)

    coverage = found / max(len(gene_names), 1)
    return matrix, coverage


def load_gene_embeddings(
    embedding_path: str,
    gene_names: List[str],
    fill_missing: bool = True,
    random_seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Backward-compatible helper returning the aligned GeneCompass matrix."""
    gene_embeddings, symbol_embeddings, symbol_to_ensembl = load_genecompass(embedding_path)
    return build_gene_embedding_matrix(
        gene_names,
        gene_embeddings,
        symbol_embeddings,
        symbol_to_ensembl,
        fill_missing=fill_missing,
        random_seed=random_seed,
    )


def load_string_ppi_adjacency(
    gene_names: List[str],
    protein_info_path: str,
    links_path: str,
    min_score: int = 700,
) -> Tuple[np.ndarray, int]:
    """Load STRING PPI edges as a binary adjacency matrix."""
    protein_info = pd.read_csv(protein_info_path, sep="\t")
    protein_to_gene = dict(
        zip(protein_info["#string_protein_id"], protein_info["preferred_name"])
    )

    links = pd.read_csv(links_path, sep=r"\s+", engine="python")
    links = links[links["combined_score"] >= min_score]

    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    adj = np.zeros((len(gene_names), len(gene_names)), dtype=np.float32)
    edge_count = 0

    for _, row in links.iterrows():
        g1 = protein_to_gene.get(row["protein1"])
        g2 = protein_to_gene.get(row["protein2"])
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            if adj[i, j] == 0.0:
                edge_count += 1
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return adj, edge_count


def load_ppi_network(
    ppi_path: str,
    protein_info_path: str,
    gene_names: List[str],
    min_score: int = 700,
) -> np.ndarray:
    """Backward-compatible STRING loader returning only adjacency."""
    adj, _ = load_string_ppi_adjacency(
        gene_names,
        protein_info_path=protein_info_path,
        links_path=ppi_path,
        min_score=min_score,
    )
    return adj


def _l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def compute_formula1_cost_matrix(
    embeddings_norm: np.ndarray,
    centroids_norm: np.ndarray,
    ppi_adj: np.ndarray,
    labels: np.ndarray,
    semantic_weight: float = 1.0,
    ppi_weight: float = 1.0,
) -> np.ndarray:
    d_sem = cdist(embeddings_norm, centroids_norm, metric="cosine")
    d_sem = np.nan_to_num(d_sem, nan=1.0, posinf=1.0, neginf=0.0)
    d_sem = d_sem / (d_sem.max() + 1e-8)

    n_genes, n_clusters = d_sem.shape
    d_ppi = np.ones((n_genes, n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        if len(members) > 0:
            d_ppi[:, k] = 1.0 - (ppi_adj[:, members].sum(axis=1) / len(members))
    d_ppi = np.clip(d_ppi, 0.0, 1.0)

    return semantic_weight * d_sem + ppi_weight * d_ppi


def compute_ot_cost_matrix(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    ppi_adj: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    return compute_formula1_cost_matrix(
        _l2_normalize(embeddings),
        _l2_normalize(centroids),
        ppi_adj,
        labels,
    )


def balanced_capacities(
    n_items: int,
    n_clusters: int,
    preference_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return integer capacities whose sum is ``n_items``."""
    base, remainder = divmod(n_items, n_clusters)
    capacities = np.full(n_clusters, base, dtype=np.int32)
    if remainder == 0:
        return capacities

    if preference_labels is None:
        preferred = np.arange(n_clusters)
    else:
        counts = np.bincount(preference_labels, minlength=n_clusters)
        preferred = np.lexsort((np.arange(n_clusters), -counts))
    capacities[preferred[:remainder]] += 1
    return capacities


def capacity_constrained_rounding(
    transport: np.ndarray,
    cost: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    n_genes, n_clusters = transport.shape
    labels = np.full(n_genes, -1, dtype=np.int32)
    remaining = capacities.astype(np.int32).copy()

    gene_idx = np.repeat(np.arange(n_genes), n_clusters)
    cluster_idx = np.tile(np.arange(n_clusters), n_genes)
    flat_transport = transport.reshape(-1)
    flat_cost = cost.reshape(-1)
    order = np.lexsort((cluster_idx, gene_idx, flat_cost, -flat_transport))

    for flat in order:
        gene = gene_idx[flat]
        cluster = cluster_idx[flat]
        if labels[gene] == -1 and remaining[cluster] > 0:
            labels[gene] = cluster
            remaining[cluster] -= 1
            if np.all(labels >= 0):
                break

    if np.any(labels < 0):
        for gene in np.where(labels < 0)[0]:
            available = np.where(remaining > 0)[0]
            cluster = available[np.argmin(cost[gene, available])]
            labels[gene] = cluster
            remaining[cluster] -= 1

    return labels


def _compute_centroids(
    embeddings_norm: np.ndarray,
    labels: np.ndarray,
    previous_centroids: np.ndarray,
) -> np.ndarray:
    centroids = previous_centroids.copy()
    for k in range(previous_centroids.shape[0]):
        mask = labels == k
        if mask.any():
            centroids[k] = embeddings_norm[mask].mean(axis=0)
    return _l2_normalize(centroids)


def formula1_grc_clustering(
    embeddings: np.ndarray,
    ppi_adj: np.ndarray,
    n_clusters: int = 32,
    n_refinement_steps: int = 3,
    sinkhorn_reg: float = 0.1,
    semantic_weight: float = 1.0,
    ppi_weight: float = 1.0,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_OT:
        raise RuntimeError("POT is required for GRC preprocessing: pip install POT")

    embeddings_norm = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(embeddings_norm).astype(np.int32)
    centroids = _l2_normalize(kmeans.cluster_centers_.astype(np.float32))
    capacities = balanced_capacities(len(labels), n_clusters, labels)

    a = np.ones(len(labels), dtype=np.float64) / len(labels)
    b = capacities.astype(np.float64) / capacities.sum()

    for step in range(n_refinement_steps):
        centroids = _compute_centroids(embeddings_norm, labels, centroids)
        cost = compute_formula1_cost_matrix(
            embeddings_norm,
            centroids,
            ppi_adj,
            labels,
            semantic_weight=semantic_weight,
            ppi_weight=ppi_weight,
        )

        try:
            transport = ot.sinkhorn(
                a,
                b,
                cost,
                sinkhorn_reg,
                numItermax=1000,
                stopThr=1e-9,
                warn=False,
            )
            if not np.isfinite(transport).all():
                raise FloatingPointError("Sinkhorn returned non-finite values")
        except Exception:
            transport = ot.emd(a, b, cost)

        new_labels = capacity_constrained_rounding(transport, cost, capacities)
        changed = int(np.sum(new_labels != labels))
        labels = new_labels

        if verbose:
            sizes = np.bincount(labels, minlength=n_clusters)
            print(
                f"  GRC step {step + 1}/{n_refinement_steps}: "
                f"changed={changed}, sizes=[{sizes.min()}-{sizes.max()}]"
            )

    centroids = _compute_centroids(embeddings_norm, labels, centroids)
    return labels, centroids


def balanced_ot_clustering(
    embeddings: np.ndarray,
    ppi_adj: np.ndarray,
    n_clusters: int,
    n_iterations: int = 3,
    ot_reg: float = 0.1,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return formula1_grc_clustering(
        embeddings,
        ppi_adj,
        n_clusters=n_clusters,
        n_refinement_steps=n_iterations,
        sinkhorn_reg=ot_reg,
        random_state=random_seed,
        verbose=verbose,
    )


def reorder_genes_by_cluster(
    gene_names: List[str],
    labels: np.ndarray,
    return_order: bool = False,
):
    order = np.argsort(labels, kind="stable")
    sorted_gene_names = [gene_names[i] for i in order]
    sorted_labels = labels[order]
    if return_order:
        return sorted_gene_names, sorted_labels, order
    return sorted_gene_names, sorted_labels


def validate_ppi_coherence(
    sorted_labels: np.ndarray,
    sorted_ppi_adj: np.ndarray,
    random_state: int = 42,
    n_random: int = 100,
) -> Dict[str, float]:
    n_clusters = int(np.max(sorted_labels)) + 1
    densities = []
    for k in range(n_clusters):
        mask = sorted_labels == k
        n = int(mask.sum())
        if n > 1:
            block = sorted_ppi_adj[np.ix_(mask, mask)]
            densities.append(float(block.sum() / (n * (n - 1))))

    rng = np.random.default_rng(random_state)
    random_densities = []
    for _ in range(n_random):
        random_labels = rng.permutation(sorted_labels)
        trial = []
        for k in range(n_clusters):
            mask = random_labels == k
            n = int(mask.sum())
            if n > 1:
                block = sorted_ppi_adj[np.ix_(mask, mask)]
                trial.append(float(block.sum() / (n * (n - 1))))
        random_densities.append(float(np.mean(trial)))

    ppi_density = float(np.mean(densities)) if densities else 0.0
    random_baseline = float(np.mean(random_densities)) if random_densities else 0.0
    random_std = float(np.std(random_densities)) if random_densities else 0.0
    return {
        "ppi_density": ppi_density,
        "random_baseline": random_baseline,
        "z_score": float((ppi_density - random_baseline) / (random_std + 1e-8)),
        "improvement": float(ppi_density / (random_baseline + 1e-8)),
    }


@dataclass
class GeneRelationClustering:

    n_clusters: int = 32
    n_refinement_steps: int = 3
    sinkhorn_reg: float = 0.1
    ppi_min_score: int = 700
    semantic_weight: float = 1.0
    ppi_weight: float = 1.0
    random_state: int = 42

    embeddings: Optional[np.ndarray] = None
    ppi_adj: Optional[np.ndarray] = None
    gene_names: Optional[List[str]] = None
    labels: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None
    genecompass_coverage: Optional[float] = None
    ppi_edge_count: int = 0

    def load_priors(
        self,
        embedding_path: str,
        ppi_path: str,
        protein_info_path: str,
        gene_names: List[str],
    ) -> None:
        """Load GeneCompass embeddings and STRING PPI."""
        self.gene_names = list(gene_names)
        self.embeddings, self.genecompass_coverage = load_gene_embeddings(
            embedding_path,
            self.gene_names,
            fill_missing=True,
            random_seed=self.random_state,
        )
        self.ppi_adj, self.ppi_edge_count = load_string_ppi_adjacency(
            self.gene_names,
            protein_info_path=protein_info_path,
            links_path=ppi_path,
            min_score=self.ppi_min_score,
        )

    def fit(self, verbose: bool = True) -> np.ndarray:
        """Run formula-1 GRC clustering."""
        if self.embeddings is None or self.ppi_adj is None:
            raise ValueError("Call load_priors() first")

        self.labels, self.centroids = formula1_grc_clustering(
            self.embeddings,
            self.ppi_adj,
            n_clusters=self.n_clusters,
            n_refinement_steps=self.n_refinement_steps,
            sinkhorn_reg=self.sinkhorn_reg,
            semantic_weight=self.semantic_weight,
            ppi_weight=self.ppi_weight,
            random_state=self.random_state,
            verbose=verbose,
        )
        return self.labels

    def reorder_genes(self, return_order: bool = False):
        """Return genes and labels ordered by cluster."""
        if self.gene_names is None or self.labels is None:
            raise ValueError("Call fit() first")
        return reorder_genes_by_cluster(self.gene_names, self.labels, return_order=return_order)

    def ppi_validation(self) -> Dict[str, float]:
        """Return within-cluster PPI validation statistics."""
        if self.ppi_adj is None or self.labels is None:
            raise ValueError("Call fit() first")
        _, sorted_labels, order = self.reorder_genes(return_order=True)
        return validate_ppi_coherence(
            sorted_labels,
            self.ppi_adj[np.ix_(order, order)],
            random_state=self.random_state,
        )

    def save_results(self, output_path: str) -> None:
        """Save clustering state."""
        if self.labels is None:
            raise ValueError("Call fit() first")
        results = {
            "n_clusters": self.n_clusters,
            "n_refinement_steps": self.n_refinement_steps,
            "sinkhorn_reg": self.sinkhorn_reg,
            "semantic_weight": self.semantic_weight,
            "ppi_weight": self.ppi_weight,
            "ppi_min_score": self.ppi_min_score,
            "random_state": self.random_state,
            "labels": self.labels,
            "centroids": self.centroids,
            "gene_names": self.gene_names,
            "cluster_sizes": np.bincount(self.labels, minlength=self.n_clusters),
            "genecompass_coverage": self.genecompass_coverage,
            "ppi_edge_count": self.ppi_edge_count,
        }
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

    @classmethod
    def load_results(cls, path: str) -> "GeneRelationClustering":
        """Load a saved GRC state."""
        with open(path, "rb") as f:
            results = pickle.load(f)
        grc = cls(
            n_clusters=results["n_clusters"],
            n_refinement_steps=results.get("n_refinement_steps", 3),
            sinkhorn_reg=results.get("sinkhorn_reg", 0.1),
            ppi_min_score=results.get("ppi_min_score", 700),
            semantic_weight=results.get("semantic_weight", 1.0),
            ppi_weight=results.get("ppi_weight", 1.0),
            random_state=results.get("random_state", 42),
        )
        grc.labels = results["labels"]
        grc.centroids = results.get("centroids")
        grc.gene_names = results.get("gene_names")
        grc.genecompass_coverage = results.get("genecompass_coverage")
        grc.ppi_edge_count = results.get("ppi_edge_count", 0)
        return grc
