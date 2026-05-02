<p align="center">
  <a href="https://github.com/ttruan2426-dot/scBIG">
    <img src="https://raw.githubusercontent.com/ttruan2426-dot/scBIG/main/assets/logo.png" alt="scBIG logo" width="500" />
  </a>
</p>

# scBIG: Beyond Independent Genes: Learning Module-Inductive Representations for Gene Perturbation Prediction (ICML 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2602.04901-b31b1b?logo=arxiv)](https://arxiv.org/abs/2602.04901)
[![Codebase](https://img.shields.io/badge/Codebase-GitHub-181717?logo=github)](https://github.com/ttruan2426-dot/scBIG)

Official implementation for the paper
[scBIG](https://arxiv.org/abs/2602.04901), ICML 2026.

----

## 🔬 Overview

scBIG is a module-inductive framework for single-cell perturbation prediction.
It first induces coherent gene programs with **Gene-Relation Clustering (GRC)**,
then models program-aware transcriptional responses with a
**Gene-Cluster-Aware Encoder (GCAE)** and conditional flow matching.

Framework of scBIG:

<p align="center">
  <a href="https://github.com/ttruan2426-dot/scBIG/blob/main/assets/pipeline.png">
    <img src="https://raw.githubusercontent.com/ttruan2426-dot/scBIG/main/assets/pipeline.png" alt="scBIG framework" width="760" />
  </a>
</p>

## 🛠️ Installation

```bash
conda env create -f environment.yml
conda activate scbig
```

## 🗂️ Data and resources

We provide the processed datasets and biological prior resources via Google
Drive:

- **Datasets**: [Google Drive](https://drive.google.com/drive/folders/1Q7xNeuqIp3nrMlsWnXX0c0SU90cnv9YQ?usp=sharing)
- **Prior resources**: [Google Drive](https://drive.google.com/drive/folders/1FSq5HIReZP42N2BseI8FKcCbrTdaSbhB?usp=sharing)

The dataset folder contains the Norman Additive, Norman Holdout, and RPE1 splits.
The resource folder contains the PPI network, GeneCompass embeddings, pathway
annotations, and ESM2 perturbation embeddings.

Prepare the Norman additive split and external resources locally:

```text
scBIG_submit/
├─ data/
│  ├─ norman_additive_raw/
│  │  ├─ norman_train_filtered.h5ad
│  │  ├─ norman_val_filtered.h5ad
│  │  └─ norman_test_filtered.h5ad
│  └─ norman_additive_grc/
└─ resources/
   ├─ genecompass_gene_embeddings_full.pkl
   ├─ 9606.protein.info.v12.0.txt.gz
   ├─ 9606.protein.links.v12.0.txt.gz
   ├─ ReactomePathways.gmt.zip
   └─ ESM2_pert_features.pt
```

## 🧭 GRC preprocessing

```bash
bash scripts/prepare_norman_grc.sh
```

## 🚀 Training

Run the Norman additive training script:

```bash
bash scripts/train_norman_additive.sh
```


## ✍️ Citation

If you find this work useful, please cite:

```bibtex
@article{ruan2026beyond,
  title={Beyond Independent Genes: Learning Module-Inductive Representations for Gene Perturbation Prediction},
  author={Ruan, Jiafa and Quan, Ruijie and Yang, Zongxin and Xu, Liyang and Yang, Yi},
  journal={arXiv preprint arXiv:2602.04901},
  year={2026}
}
```
