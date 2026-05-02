#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU="${GPU:-5}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/norman_additive_grc}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/norman_additive_grc}"
RESOURCE_DIR="${RESOURCE_DIR:-${REPO_DIR}/resources}"
GENECOMPASS_PATH="${GENECOMPASS_PATH:-${RESOURCE_DIR}/genecompass_gene_embeddings_full.pkl}"
CELLFLOW_SRC="${CELLFLOW_SRC:-}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${CELLFLOW_SRC}" ]]; then
  export PYTHONPATH="${REPO_DIR}:${CELLFLOW_SRC}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
fi
export SCBIG_CELLFLOW_SRC="${CELLFLOW_SRC}"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting scBIG Norman additive training"
  echo "Repo: ${REPO_DIR}"
  echo "Data: ${DATA_DIR}"
  echo "Output: ${OUTPUT_DIR}"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m scbig.experiments.train_norman_additive \
    --dataset norman_additive_grc \
    --data_dir_override "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --resource_dir "${RESOURCE_DIR}" \
    --genecompass_path "${GENECOMPASS_PATH}" \
    --cellflow_src "${CELLFLOW_SRC}" \
    --encoder_config genecompass_pe_module \
    --gene_loss_config grc_corr_pathway_ot \
    --encoder_n_epochs 100 \
    --n_iterations 500000 \
    --valid_freq 100000 \
    --gene_loss_freq 1000 \
    --n_gene_loss_samples 3 \
    --gene_loss_batch_size 64 \
    --batch_size 1024 \
    --encoder_batch_size 256 \
    --n_pcs 50 \
    --embed_dim 256 \
    --num_layers 3 \
    --num_heads 8 \
    --lambda_corr 0.1 \
    --lambda_ot 0.1 \
    --num_modules 8 \
    --chunk_size 64 \
    --save_best_model \
    --skip_checkpoints
} 2>&1 | tee -a "${LOG_FILE}"
