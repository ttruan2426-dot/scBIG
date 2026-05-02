#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_DIR}/data/norman_additive_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/data/norman_additive_grc}"
RESOURCE_DIR="${RESOURCE_DIR:-${REPO_DIR}/resources}"
GENECOMPASS_PATH="${GENECOMPASS_PATH:-${RESOURCE_DIR}/genecompass_gene_embeddings_full.pkl}"
PPI_INFO_PATH="${PPI_INFO_PATH:-${RESOURCE_DIR}/9606.protein.info.v12.0.txt.gz}"
PPI_LINKS_PATH="${PPI_LINKS_PATH:-${RESOURCE_DIR}/9606.protein.links.v12.0.txt.gz}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${PYTHON}" -m scbig.experiments.prepare_norman_grc \
  --data_dir "${RAW_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --genecompass_path "${GENECOMPASS_PATH}" \
  --ppi_info_path "${PPI_INFO_PATH}" \
  --ppi_links_path "${PPI_LINKS_PATH}" \
  --dataset_name norman_additive \
  --n_clusters 32 \
  --semantic_weight 1.0 \
  --ppi_weight 1.0 \
  --ppi_min_score 700 \
  --sinkhorn_reg 0.1 \
  --n_refinement_steps 3 \
  --random_state 42
