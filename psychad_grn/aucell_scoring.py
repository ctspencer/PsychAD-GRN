"""Regulon activity scoring using AUCell and cross-cohort concordance.

AUCell (Area Under the Curve) scores quantify how active each regulon is in
each individual cell. For a given regulon, AUCell ranks all genes by their
expression in a cell and computes the AUC for the regulon's target gene set
in that ranking — a high AUC indicates the target genes are disproportionately
among the most highly expressed genes.

Pipeline:
  1. Run AUCell via pySCENIC CLI to produce a cells × regulons activity matrix.
  2. Normalize AUCell scores to z-scores per cell type for cross-cell-type
     comparison. Formula: z = (celltype_mean − global_mean) / global_std.
  3. Assess cross-cohort concordance by computing Pearson correlations of
     z-score profiles between dataset pairs (MSSM, RADC, ROSMAP).
  4. Combine cohort-level p-values using Stouffer's weighted z-score method.

Usage:
    python -m psychad_grn.aucell_scoring --config configs/default_config.yaml \\
        --loom-file data/CTRL.loom --reg-file results/CTRL/regulons/CTRL_regulons.csv \\
        --h5ad-file data/CTRL.h5ad --output-dir results/aucell/
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils.io import load_aucell, load_config, load_h5ad
from .utils.stats import apply_bh_correction, compute_zscore_matrix, stouffer_meta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── AUCell scoring ────────────────────────────────────────────────────────────

def run_aucell(
    loom_path: str | Path,
    reg_path: str | Path,
    output_path: str | Path,
    n_workers: int = 8,
) -> None:
    """Score regulon activity in each cell using AUCell (pySCENIC aucell).

    AUCell estimates per-cell regulon activity without requiring gene
    binarization. It operates on the expression-ranked gene list per cell,
    so it is robust to differences in sequencing depth and normalization.
    All default pySCENIC AUCell parameters are used.

    Args:
        loom_path: Loom file with expression data (cells × genes).
        reg_path: Regulon CSV from cisTarget pruning step.
        output_path: Output path for the AUCell CSV (cells × regulons).
        n_workers: Number of CPU workers.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pyscenic", "aucell",
        str(loom_path),
        str(reg_path),
        "--output", str(output_path),
        "--num_workers", str(n_workers),
    ]
    log.info("Running AUCell → %s", output_path)
    subprocess.run(cmd, check=True)


# ── Z-score normalization ─────────────────────────────────────────────────────

def compute_zscore(
    aucell_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    celltype_col: str = "subclass",
) -> pd.DataFrame:
    """Normalize AUCell scores to cell-type z-scores.

    Computes the mean AUCell score for each regulon within each cell type,
    then normalizes relative to the global mean and standard deviation across
    all cells. This allows comparison of regulon enrichment patterns across
    cell types with different baseline activity levels.

    Args:
        aucell_df: AUCell matrix of shape (n_cells × n_regulons).
        obs_df: Cell metadata DataFrame (index = cell IDs) containing at
            minimum the celltype_col column.
        celltype_col: Column in obs_df with cell type labels.

    Returns:
        Z-score DataFrame of shape (n_celltypes × n_regulons).
    """
    return compute_zscore_matrix(aucell_df, celltype_col, obs_df=obs_df)


# ── Cross-cohort concordance ──────────────────────────────────────────────────

def pearson_concordance(
    z_df_a: pd.DataFrame,
    z_df_b: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of regulon z-scores between cohorts.

    For each pair of cell types (one from each cohort), computes the Pearson
    correlation of their regulon z-score vectors across the shared regulon set.
    High correlation indicates that the same regulons are enriched in the same
    cell types across datasets — evidence of cross-cohort reproducibility.

    Args:
        z_df_a: Z-score matrix from cohort A (n_celltypes × n_regulons).
        z_df_b: Z-score matrix from cohort B (n_celltypes × n_regulons).

    Returns:
        Correlation matrix of shape (n_celltypes_A × n_celltypes_B).
    """
    common_regulons = list(set(z_df_a.columns) & set(z_df_b.columns))
    if not common_regulons:
        raise ValueError("No shared regulons between the two z-score matrices.")

    a = z_df_a[common_regulons]
    b = z_df_b[common_regulons]

    result = pd.DataFrame(index=a.index, columns=b.index, dtype=float)
    for ct_a in a.index:
        for ct_b in b.index:
            r = a.loc[ct_a].corr(b.loc[ct_b], method="pearson")
            result.loc[ct_a, ct_b] = r

    return result


def stouffer_meta_analysis(
    pvalue_df: pd.DataFrame,
    weights: Optional[list[float]] = None,
) -> pd.Series:
    """Combine cohort-level p-values per regulon using Stouffer's method.

    Each column of pvalue_df is a cohort; each row is a regulon. Produces
    a combined p-value that integrates evidence across cohorts.

    Args:
        pvalue_df: DataFrame of shape (n_regulons × n_cohorts) with p-values.
        weights: Optional per-cohort weights (e.g., sqrt of sample size).
            Defaults to equal weights.

    Returns:
        Series of combined p-values indexed by regulon name.
    """
    combined = pvalue_df.apply(
        lambda row: stouffer_meta(row.dropna().tolist(), weights=weights),
        axis=1,
    )
    return combined


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_aucell_pipeline(
    loom_path: str | Path,
    reg_path: str | Path,
    h5ad_path: str | Path,
    output_dir: str | Path,
    celltype_col: str = "subclass",
    n_workers: int = 8,
    concordance_pairs: Optional[list[tuple[str, str]]] = None,
    aucell_paths: Optional[dict[str, str | Path]] = None,
) -> dict:
    """Run AUCell scoring, z-score normalization, and optional concordance.

    Args:
        loom_path: Loom file for AUCell input.
        reg_path: Regulon file from cisTarget.
        h5ad_path: h5ad with cell metadata (for cell type labels).
        output_dir: Directory for all outputs.
        celltype_col: Cell type column name in AnnData obs.
        n_workers: CPU workers for AUCell.
        concordance_pairs: Optional list of (cohort_A_name, cohort_B_name) tuples
            for concordance analysis. Requires aucell_paths.
        aucell_paths: Optional dict mapping cohort names to pre-computed AUCell
            CSV paths. Used when running concordance across cohorts.

    Returns:
        Dict with keys: 'aucell' (DataFrame), 'zscore' (DataFrame),
        and optionally 'concordance' (dict of correlation matrices).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aucell_path = output_dir / "aucell.csv"
    run_aucell(loom_path, reg_path, aucell_path, n_workers=n_workers)

    adata = load_h5ad(h5ad_path)
    aucell_df = load_aucell(aucell_path)
    z_df = compute_zscore(aucell_df, adata.obs, celltype_col=celltype_col)
    z_df.to_csv(output_dir / "aucell_zscore.csv")
    log.info("Z-score matrix saved: %s", output_dir / "aucell_zscore.csv")

    result = {"aucell": aucell_df, "zscore": z_df}

    if concordance_pairs and aucell_paths:
        z_matrices = {}
        for cohort, auc_path in aucell_paths.items():
            auc = load_aucell(auc_path)
            z_matrices[cohort] = compute_zscore(auc, adata.obs, celltype_col)

        concordance = {}
        for name_a, name_b in concordance_pairs:
            corr = pearson_concordance(z_matrices[name_a], z_matrices[name_b])
            label = f"{name_a}_vs_{name_b}"
            concordance[label] = corr
            corr.to_csv(output_dir / f"concordance_{label}.csv")
            log.info("Concordance saved: %s", output_dir / f"concordance_{label}.csv")

        result["concordance"] = concordance

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score regulon activity with AUCell and assess cross-cohort concordance."
    )
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--loom-file", help="Input loom file for AUCell.")
    parser.add_argument("--reg-file", help="Regulon CSV from cisTarget.")
    parser.add_argument("--h5ad-file", help="h5ad file with cell metadata.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--celltype-col", default="subclass", help="Cell type column name.")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument(
        "--aucell-files",
        nargs="+",
        metavar="COHORT=PATH",
        help="Pre-computed AUCell CSVs per cohort, e.g. MSSM=results/MSSM/aucell.csv",
    )
    parser.add_argument(
        "--concordance-pairs",
        nargs="+",
        metavar="A:B",
        help="Cohort pairs for concordance, e.g. MSSM:RADC MSSM:ROSMAP",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    auc_cfg = cfg.get("aucell_scoring", {})
    paths_cfg = cfg.get("paths", {})

    aucell_paths = None
    if args.aucell_files:
        aucell_paths = dict(item.split("=", 1) for item in args.aucell_files)

    concordance_pairs = None
    if args.concordance_pairs:
        concordance_pairs = [tuple(p.split(":")) for p in args.concordance_pairs]

    run_aucell_pipeline(
        loom_path=args.loom_file or paths_cfg.get("loom_file", ""),
        reg_path=args.reg_file or paths_cfg.get("reg_file", ""),
        h5ad_path=args.h5ad_file or paths_cfg.get("h5ad_file", ""),
        output_dir=args.output_dir,
        celltype_col=auc_cfg.get("celltype_col", args.celltype_col),
        n_workers=args.n_workers,
        concordance_pairs=concordance_pairs,
        aucell_paths=aucell_paths,
    )


if __name__ == "__main__":
    main()
