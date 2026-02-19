#!/usr/bin/env Rscript
# dreamlet_analysis.R
#
# Differential regulon activity analysis using the dreamlet framework.
#
# This script applies pseudobulk aggregation followed by a linear mixed model
# to test for association between regulon activity (from AUCell scores) and
# Alzheimer's disease-related clinical variables across cell types.
#
# Model: ~ {contrast} + sex + scale(age) + log(n_genes) + Brain_bank + (1|Subject)
#
# The random effect (1|Subject) accounts for repeated observations per donor
# across cell types, while fixed effects control for known confounders. Empirical
# Bayes shrinkage (via limma) stabilizes variance estimates across regulons with
# few observations per cell type.
#
# Inputs:
#   --h5ad-file   : h5ad file containing AUCell scores in X and metadata in obs.
#                   The 'X' slot should contain AUCell values (non-count data).
#   --output-dir  : Directory where per-contrast CSV results are written.
#   --contrasts   : Comma-separated list of clinical variables to test.
#   --anno-levels : Comma-separated annotation levels (e.g., subclass,class).
#   --subject-col : Column in obs identifying the donor/subject (for random effect).
#   --celltype-col: Column in obs identifying cell type.
#
# Outputs:
#   output-dir/{contrast}/{anno_level}/{anno_level}_dreamlet.csv
#   Each CSV contains: regulon ID, logFC, t-statistic, P.Value, adj.P.Val,
#   cell type (assay), contrast, z.std.
#
# Usage:
#   Rscript r/dreamlet_analysis.R \
#     --h5ad-file results/aucell/aucell_auc.h5ad \
#     --output-dir results/dreamlet/ \
#     --contrasts AD,Braak,CERAD,dementia,n07x \
#     --anno-levels subclass,class \
#     --subject-col SubID

suppressPackageStartupMessages({
  library(optparse)
  library(dreamlet)
  library(SingleCellExperiment)
  library(zellkonverter)
  library(limma)
})

# ── CLI argument parsing ──────────────────────────────────────────────────────

option_list <- list(
  make_option("--h5ad-file",    type = "character", help = "Path to AUCell h5ad file."),
  make_option("--output-dir",   type = "character", help = "Root output directory for results."),
  make_option("--contrasts",    type = "character", default = "AD,Braak,CERAD,dementia",
              help = "Comma-separated list of contrast variables [default: %default]."),
  make_option("--anno-levels",  type = "character", default = "subclass,class",
              help = "Comma-separated annotation levels [default: %default]."),
  make_option("--subject-col",  type = "character", default = "SubID",
              help = "Column identifying subjects (random effect) [default: %default]."),
  make_option("--celltype-col", type = "character", default = "subclass",
              help = "Column identifying cell types [default: %default].")
)

opt <- parse_args(OptionParser(
  option_list = option_list,
  description = "Differential regulon activity analysis with dreamlet."
))

if (is.null(opt[["h5ad-file"]]) || is.null(opt[["output-dir"]])) {
  stop("--h5ad-file and --output-dir are required.", call. = FALSE)
}

contrasts   <- strsplit(opt[["contrasts"]], ",")[[1]]
anno_levels <- strsplit(opt[["anno-levels"]], ",")[[1]]
subject_col <- opt[["subject-col"]]

cat("Loading h5ad:", opt[["h5ad-file"]], "\n")
sce <- readH5AD(opt[["h5ad-file"]], use_hdf5 = FALSE)

# ── Core dreamlet function ────────────────────────────────────────────────────

run_dreamlet_contrast <- function(sce, contrast_var, save_dir, anno_levels, subject_col) {
  # Build the model formula with the contrast as the primary fixed effect.
  # All confounders are included: sex, age (scaled), library size proxy
  # (log of detected genes), and brain bank (batch effect). The random effect
  # (1|Subject) accounts for correlated observations across cell types within
  # the same donor.
  form <- as.formula(paste0(
    "~ ", contrast_var,
    " + sex + scale(age) + log(n_genes) + Brain_bank + (1|", subject_col, ")"
  ))
  cat("Formula:", deparse(form), "\n")

  for (level in anno_levels) {
    cat("  Annotation level:", level, "\n")

    # aggregateNonCountSignal is used instead of the standard aggregateToPseudoBulk
    # because AUCell scores are continuous values (not integer counts). This function
    # computes per-pseudobulk means rather than sums, preserving the AUCell scale.
    pb <- tryCatch(
      aggregateNonCountSignal(sce, assay = "X", by = subject_col, col = level, verbose = FALSE),
      error = function(e) {
        warning(paste("aggregateNonCountSignal failed for", level, ":", e$message))
        return(NULL)
      }
    )
    if (is.null(pb)) next

    # Fit the mixed model. dreamlet applies variancePartition's dream() to each
    # regulon (feature) independently, with empirical Bayes shrinkage via limma
    # to stabilize variance estimates in cell types with few pseudobulk replicates.
    fit <- tryCatch(
      dreamlet(pb, form, BPPARAM = BiocParallel::SerialParam()),
      error = function(e) {
        warning(paste("dreamlet failed for", contrast_var, "/", level, ":", e$message))
        return(NULL)
      }
    )
    if (is.null(fit)) next

    # Extract results for the primary contrast coefficient
    res <- tryCatch(
      topTable(fit, coef = contrast_var, number = Inf),
      error = function(e) {
        warning(paste("topTable failed for", contrast_var, ":", e$message))
        return(NULL)
      }
    )
    if (is.null(res)) next

    df <- as.data.frame(res)

    # Write results to disk
    out_dir <- file.path(save_dir, contrast_var, level)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    out_path <- file.path(out_dir, paste0(level, "_dreamlet.csv"))
    write.csv(df, out_path, row.names = TRUE)
    cat("    Saved:", out_path, "(", nrow(df), "regulons ×", length(unique(df$assay)),
        "cell types )\n")
  }
}

# ── Run all contrasts ─────────────────────────────────────────────────────────

for (contrast in contrasts) {
  cat("\n=== Contrast:", contrast, "===\n")
  run_dreamlet_contrast(
    sce        = sce,
    contrast_var = contrast,
    save_dir   = opt[["output-dir"]],
    anno_levels = anno_levels,
    subject_col = subject_col
  )
}

cat("\nDone. Results written to:", opt[["output-dir"]], "\n")
