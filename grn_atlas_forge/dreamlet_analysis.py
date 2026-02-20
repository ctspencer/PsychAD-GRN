"""Differential regulon activity analysis using the dreamlet R package.

dreamlet is an R package that extends the variancePartition/limma framework
to single-cell pseudobulk data. This module serves as the Python interface:
it prepares inputs, invokes the R script as a subprocess, and post-processes
the resulting CSV files.

Model tested for each regulon × cell type × clinical variable:
  AUCell_score ~ clinical_var + sex + scale(age) + log(n_genes) + Brain_bank + (1|Subject)

Clinical variables tested:
  - AD      : case-control diagnosis (binary)
  - Braak   : Braak neurofibrillary tangle stage (0–6)
  - CERAD   : CERAD neuritic plaque score (1–4)
  - dementia: dementia status at death (binary)
  - n07x    : tau/amyloid cognitive resilience score

Post-processing (Python side):
  - Concatenate per-contrast / per-annotation-level CSVs.
  - Apply Benjamini-Hochberg FDR correction within each cell type × contrast.
  - Add log2FC and -log10(FDR) columns.
  - Optionally filter to protein-coding autosomal genes.

Usage:
    python -m psychad_grn.dreamlet_analysis --config configs/default_config.yaml \\
        --h5ad-file results/aucell/aucell_auc.h5ad \\
        --output-dir results/dreamlet/
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils.io import load_config
from .utils.stats import apply_bh_correction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Path to the R script, relative to the package root
_DEFAULT_R_SCRIPT = Path(__file__).parent.parent / "r" / "dreamlet_analysis.R"


# ── R subprocess interface ────────────────────────────────────────────────────

def run_dreamlet_r(
    h5ad_file: str | Path,
    output_dir: str | Path,
    contrasts: list[str],
    anno_levels: list[str],
    subject_col: str = "SubID",
    rscript_path: str = "Rscript",
    r_script: str | Path = _DEFAULT_R_SCRIPT,
    timeout: int = 7200,
) -> str:
    """Invoke dreamlet_analysis.R as a subprocess.

    The R script performs pseudobulk aggregation and linear mixed model
    fitting for each contrast × annotation level combination. Results are
    written as CSVs to output_dir/{contrast}/{anno_level}/.

    Args:
        h5ad_file: Path to the AUCell h5ad file (X slot = AUCell scores).
        output_dir: Directory where per-contrast result CSVs will be written.
        contrasts: List of clinical variable names to test (e.g., ['AD', 'Braak']).
        anno_levels: Annotation levels (e.g., ['subclass', 'class']).
        subject_col: Column identifying donors (used for random effect).
        rscript_path: Path to the Rscript binary.
        r_script: Path to dreamlet_analysis.R.
        timeout: Maximum run time in seconds (default 2 hours).

    Returns:
        Captured stdout from the R script.

    Raises:
        RuntimeError: If the R script exits with a nonzero return code.
    """
    cmd = [
        str(rscript_path),
        str(r_script),
        "--h5ad-file",   str(h5ad_file),
        "--output-dir",  str(output_dir),
        "--contrasts",   ",".join(contrasts),
        "--anno-levels", ",".join(anno_levels),
        "--subject-col", subject_col,
    ]
    log.info("Launching R: %s", " ".join(cmd))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"dreamlet_analysis.R timed out after {timeout}s.")

    if proc.returncode != 0:
        log.error("R stderr:\n%s", stderr)
        raise RuntimeError(
            f"dreamlet_analysis.R exited with code {proc.returncode}. "
            "See stderr above for details."
        )
    log.info("R script completed successfully.")
    return stdout


# ── Post-processing ───────────────────────────────────────────────────────────

def aggregate_dreamlet_results(
    results_dir: str | Path,
    contrasts: list[str],
    anno_levels: list[str],
) -> pd.DataFrame:
    """Read and concatenate per-contrast dreamlet CSV files.

    Expects CSVs at: results_dir/{contrast}/{anno_level}/{anno_level}_dreamlet.csv

    Args:
        results_dir: Root directory containing per-contrast subdirectories.
        contrasts: List of contrast variable names.
        anno_levels: List of annotation levels.

    Returns:
        Concatenated DataFrame with added columns 'contrast' and 'anno_level'.
    """
    results_dir = Path(results_dir)
    dfs = []
    for contrast in contrasts:
        for level in anno_levels:
            csv_path = results_dir / contrast / level / f"{level}_dreamlet.csv"
            if not csv_path.exists():
                log.warning("Missing result file: %s", csv_path)
                continue
            df = pd.read_csv(csv_path, index_col=0)
            df["contrast"] = contrast
            df["anno_level"] = level
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No dreamlet result CSVs found under {results_dir}. "
            "Did the R script complete successfully?"
        )

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Aggregated %d rows from %d files.", len(combined), len(dfs))
    return combined


def filter_protein_coding(
    df: pd.DataFrame,
    pc_genes_file: str | Path,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Remove sex-chromosome and non-protein-coding genes from dreamlet results.

    Filters results to protein-coding autosomal genes. Sex-chromosome genes
    (chrX, chrY) are excluded to avoid confounding by sex differences.

    Args:
        df: Dreamlet results DataFrame with a column of gene/regulon identifiers.
        pc_genes_file: Path to a CSV/TSV with at minimum a 'gene_name' column
            (and optionally 'chromosome') listing protein-coding autosomal genes.
        id_col: Column in df containing the regulon/gene name to filter on.
            Regulon names like 'MEF2C(+)' are matched by stripping the suffix.

    Returns:
        Filtered DataFrame.
    """
    pc = pd.read_csv(pc_genes_file)
    pc_genes = set(pc["gene_name"].dropna())

    if "chromosome" in pc.columns:
        sex_chrs = {"chrX", "chrY", "X", "Y"}
        pc_genes -= set(pc.loc[pc["chromosome"].isin(sex_chrs), "gene_name"])

    def gene_from_regulon(name: str) -> str:
        return name.split("(")[0] if isinstance(name, str) else name

    mask = df[id_col].apply(gene_from_regulon).isin(pc_genes)
    n_removed = (~mask).sum()
    if n_removed > 0:
        log.info("Removed %d non-protein-coding / sex-chromosome regulons.", n_removed)
    return df[mask].copy()


def process_dreamlet_results(
    df: pd.DataFrame,
    pvalue_col: str = "P.Value",
    logfc_col: str = "logFC",
) -> pd.DataFrame:
    """Standardize and add derived columns to dreamlet results.

    Applies BH FDR correction within each cell type × contrast group,
    and adds 'log2FC' and 'neg_log10_FDR' columns.

    Args:
        df: Aggregated dreamlet DataFrame.
        pvalue_col: Column name for raw p-values.
        logfc_col: Column name for log fold change.

    Returns:
        Processed DataFrame with 'FDR', 'neg_log10_FDR', and 'log2FC' added.
    """
    df = df.copy()
    df["log2FC"] = df[logfc_col] / (
        1.0 if logfc_col == "log2FC" else 1.0  # logFC from limma is already log2
    )

    group_cols = [c for c in ["assay", "contrast", "anno_level"] if c in df.columns]
    df = apply_bh_correction(df, pvalue_col=pvalue_col, group_cols=group_cols or None)
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_dreamlet_pipeline(
    h5ad_file: str | Path,
    output_dir: str | Path,
    contrasts: list[str],
    anno_levels: list[str],
    subject_col: str = "SubID",
    pc_genes_file: Optional[str | Path] = None,
    rscript_path: str = "Rscript",
    r_script: str | Path = _DEFAULT_R_SCRIPT,
) -> pd.DataFrame:
    """Run the complete dreamlet pipeline: R model fitting + Python post-processing.

    Args:
        h5ad_file: AUCell h5ad input.
        output_dir: Root output directory.
        contrasts: Clinical variables to test.
        anno_levels: Annotation levels.
        subject_col: Donor ID column.
        pc_genes_file: Optional protein-coding gene list for filtering.
        rscript_path: Path to Rscript binary.
        r_script: Path to dreamlet_analysis.R.

    Returns:
        Processed DataFrame with all results, FDR corrections, and derived columns.
    """
    run_dreamlet_r(
        h5ad_file=h5ad_file,
        output_dir=output_dir,
        contrasts=contrasts,
        anno_levels=anno_levels,
        subject_col=subject_col,
        rscript_path=rscript_path,
        r_script=r_script,
    )

    df = aggregate_dreamlet_results(output_dir, contrasts, anno_levels)

    if pc_genes_file:
        df = filter_protein_coding(df, pc_genes_file)

    df = process_dreamlet_results(df)

    output_path = Path(output_dir) / "dreamlet_results_processed.csv"
    df.to_csv(output_path, index=False)
    log.info("Processed results saved: %s", output_path)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run differential regulon activity analysis via dreamlet (R) + post-processing."
    )
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--h5ad-file", required=True, help="AUCell h5ad file.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--contrasts", nargs="+", default=["AD", "Braak", "CERAD", "dementia", "n07x"]
    )
    parser.add_argument("--anno-levels", nargs="+", default=["subclass", "class"])
    parser.add_argument("--subject-col", default="SubID")
    parser.add_argument("--pc-genes-file", default=None, help="Protein-coding gene list CSV.")
    parser.add_argument("--rscript-path", default="Rscript")
    parser.add_argument("--r-script", default=str(_DEFAULT_R_SCRIPT))
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    dm_cfg = cfg.get("dreamlet_analysis", {})

    run_full_dreamlet_pipeline(
        h5ad_file=args.h5ad_file,
        output_dir=args.output_dir,
        contrasts=dm_cfg.get("contrasts", args.contrasts),
        anno_levels=dm_cfg.get("anno_levels", args.anno_levels),
        subject_col=dm_cfg.get("subject_col", args.subject_col),
        pc_genes_file=args.pc_genes_file,
        rscript_path=args.rscript_path,
        r_script=args.r_script,
    )


if __name__ == "__main__":
    main()
