# PsychAD GRN Atlas

Gene regulatory network analysis of Alzheimer's disease and cognitive resilience in the PsychAD cohort.

**Manuscript:** *TBDe*

## Overview

This repository contains the core analytical pipeline used to construct and characterize gene regulatory networks (GRNs) from single-nucleus RNA-seq data in the PsychAD cohort (687 individuals, 1.7 million nuclei, 27 cell types). Networks are inferred independently for three phenotypic groups — Alzheimer's Disease (AD), Control, and Cognitively Resilient (RES) — and compared to identify regulatory programs associated with disease pathogenesis and resilience.

The pipeline addresses six questions:

1. **Which transcription factors regulate which genes?** (*GRN inference*)
2. **How active is each regulon in each cell type?** (*AUCell scoring*)
3. **Which regulons are cell-type–specific?** (*Regulon specificity and RRA*)
4. **Which regulons change activity with disease severity?** (*Dreamlet mixed models*)
5. **Do disease-associated DEGs concentrate in specific regulons?** (*Fisher's enrichment*)
6. **How does the regulatory network topology change between conditions?** (*Network rewiring*)

---

## Installation

### Prerequisites

- Python ≥ 3.10
- R ≥ 4.2

**R packages** (install via Bioconductor):
```r
BiocManager::install(c("dreamlet", "SingleCellExperiment", "zellkonverter", "limma"))
install.packages("optparse")
```

### Python environment

```bash
git clone https://github.com/ctspencer/psychad-grn-atlas
cd psychad-grn-atlas
conda env create -f environment.yml
conda activate psychad-grn
pip install -e .
```

> **pySCENIC note:** pySCENIC 0.12.1 requires older anndata versions that
> may conflict with scanpy 1.9.3. If you encounter dependency errors during
> GRN inference, create a separate environment:
> ```bash
> conda create -n psychad-scenic python=3.8
> pip install pyscenic==0.12.1 arboreto==0.1.6
> ```
> Run `grn_inference.py` in that environment, then all downstream steps
> in `psychad-grn`.

---

## Usage

### Quick start — full pipeline

```bash
bash scripts/run_pipeline.sh --config configs/default_config.yaml
```

Update `configs/default_config.yaml` with your data paths before running.

### Individual modules

#### 1. GRN Inference

```bash
python -m psychad_grn.grn_inference \
  --h5ad-dir data/h5ad/ \
  --output-dir results/GRNs/ \
  --tf-list data/pyscenic/allTFs_hg38.txt \
  --db-glob "data/pyscenic/*.feather" \
  --motif-file data/pyscenic/motifs-v9-nr.hgnc-m0.001-o0.0.tbl \
  --phenotypes AD CTRL RES \
  --n-runs 5
```

#### 2. AUCell Scoring

```bash
python -m psychad_grn.aucell_scoring \
  --loom-file data/h5ad/AD.loom \
  --reg-file results/GRNs/AD/regulons/AD_regulons.csv \
  --h5ad-file data/h5ad/AD.h5ad \
  --output-dir results/aucell/AD/
```

#### 3. Regulon Specificity

```bash
python -m psychad_grn.regulon_specificity \
  --aucell-file results/aucell/AD/aucell.csv \
  --h5ad-file data/h5ad/AD.h5ad \
  --output-dir results/regulon_specificity/ \
  --plot
```

#### 4. Dreamlet Analysis

```bash
python -m psychad_grn.dreamlet_analysis \
  --h5ad-file results/aucell/aucell_auc.h5ad \
  --output-dir results/dreamlet/ \
  --contrasts AD Braak CERAD dementia n07x \
  --anno-levels subclass class
```

#### 5. Fisher's Enrichment

```bash
python -m psychad_grn.fishers_enrichment \
  --grn-file results/GRNs/AD/AD_GRN.csv \
  --deg-file results/dreamlet/dreamlet_results_processed.csv \
  --h5ad-file data/h5ad/AD.h5ad \
  --output-dir results/fishers/
```

#### 6. Network Rewiring

```bash
python -m psychad_grn.network_rewiring \
  --ad-adj   results/GRNs/AD/AD_GRN.csv \
  --ctrl-adj results/GRNs/CTRL/CTRL_GRN.csv \
  --res-adj  results/GRNs/RES/RES_GRN.csv \
  --go-gmt   data/pathways/GO_Biological_Process_2023.gmt \
  --output-dir results/network_rewiring/ \
  --plot
```

## Repository Structure

```
psychad-grn-atlas/
├── psychad_grn/
│   ├── grn_inference.py        # Analysis 1: pySCENIC GRN construction
│   ├── aucell_scoring.py       # Analysis 2: AUCell + concordance
│   ├── regulon_specificity.py  # Analysis 3: RSS + RRA
│   ├── dreamlet_analysis.py    # Analysis 4: dreamlet Python wrapper
│   ├── fishers_enrichment.py   # Analysis 5: Fisher's exact test
│   ├── network_rewiring.py     # Analysis 6: differential network
│   └── utils/
│       ├── io.py               # Data loading/saving
│       ├── stats.py            # Statistical functions (z-score, RRA, Stouffer's)
│       └── plotting.py         # Shared visualization functions
├── r/
│   └── dreamlet_analysis.R     # Standalone R script (dreamlet package)
├── configs/
│   └── default_config.yaml     # All parameters and paths
└── scripts/
    └── run_pipeline.sh         # End-to-end pipeline runner
```

---

## Citation

> Spencer CS. *Decoding Gene Regulatory Programs of Resilience and Pathogenesis
> in Alzheimer's Disease.* PhD Dissertation, [Institution], [Year].

