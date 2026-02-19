#!/usr/bin/env bash
# run_pipeline.sh
#
# End-to-end runner for the PsychAD GRN Atlas analysis pipeline.
# Executes all six analyses in dependency order.
#
# Usage:
#   bash scripts/run_pipeline.sh --config configs/default_config.yaml
#
# Individual modules can be re-run by commenting out completed steps.
# All outputs go to results/ (or the output_dir set in your config).
#
# Prerequisites:
#   conda activate psychad-grn
#   pip install -e .

set -euo pipefail

CONFIG="configs/default_config.yaml"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"
echo "Pipeline started: $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"

run_step() {
  local step_name="$1"
  local cmd="${@:2}"
  echo "" | tee -a "${LOG_FILE}"
  echo "=== STEP: ${step_name} ===" | tee -a "${LOG_FILE}"
  echo "Command: ${cmd}" | tee -a "${LOG_FILE}"
  eval "${cmd}" 2>&1 | tee -a "${LOG_FILE}"
  echo "Completed: ${step_name}" | tee -a "${LOG_FILE}"
}

# ── Step 1: GRN Inference ────────────────────────────────────────────────────
# NOTE: pySCENIC may require a separate conda environment (see environment.yml).
# If so, run grn_inference.py in the psychad-scenic env and resume here from
# Step 2 using the resulting *_GRN.csv files.
run_step "GRN Inference" \
  python -m psychad_grn.grn_inference \
    --config "${CONFIG}" \
    --output-dir results/GRNs/

# ── Step 2: AUCell Scoring ───────────────────────────────────────────────────
run_step "AUCell Scoring" \
  python -m psychad_grn.aucell_scoring \
    --config "${CONFIG}" \
    --output-dir results/aucell/

# ── Step 3: Regulon Specificity ──────────────────────────────────────────────
run_step "Regulon Specificity (RSS + RRA)" \
  python -m psychad_grn.regulon_specificity \
    --config "${CONFIG}" \
    --aucell-file results/aucell/aucell.csv \
    --h5ad-file data/h5ad/psychad.h5ad \
    --output-dir results/regulon_specificity/ \
    --plot

# ── Step 4: Dreamlet Analysis ────────────────────────────────────────────────
# Runs R (dreamlet package) then Python post-processing.
run_step "Dreamlet Differential Analysis" \
  python -m psychad_grn.dreamlet_analysis \
    --config "${CONFIG}" \
    --h5ad-file results/aucell/aucell_auc.h5ad \
    --output-dir results/dreamlet/

# ── Step 5: Fisher's Enrichment ──────────────────────────────────────────────
run_step "Fisher's Exact Test Enrichment" \
  python -m psychad_grn.fishers_enrichment \
    --config "${CONFIG}" \
    --grn-file results/GRNs/AD/AD_GRN.csv \
    --deg-file results/dreamlet/dreamlet_results_processed.csv \
    --h5ad-file data/h5ad/psychad.h5ad \
    --output-dir results/fishers/

# ── Step 6: Network Rewiring ─────────────────────────────────────────────────
run_step "Network Rewiring and GO Enrichment" \
  python -m psychad_grn.network_rewiring \
    --config "${CONFIG}" \
    --ad-adj   results/GRNs/AD/AD_GRN.csv \
    --ctrl-adj results/GRNs/CTRL/CTRL_GRN.csv \
    --res-adj  results/GRNs/RES/RES_GRN.csv \
    --go-gmt   data/pathways/GO_Biological_Process_2023.gmt \
    --output-dir results/network_rewiring/ \
    --plot

echo "" | tee -a "${LOG_FILE}"
echo "Pipeline complete: $(date)" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
