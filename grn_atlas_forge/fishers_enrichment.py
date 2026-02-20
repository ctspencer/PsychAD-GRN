"""Fisher's exact test for DEG enrichment in regulon target gene sets.

Tests whether the target genes of each SCENIC regulon are significantly
enriched among differentially expressed genes (DEGs) identified by dreamlet.
A significant enrichment (high odds ratio, low FDR) indicates that the
regulon's TF likely drives or is correlated with the differential expression
pattern associated with AD, Braak stage, CERAD, or dementia.

For each combination of (cell type, clinical contrast, TF, DEG direction),
a 2×2 contingency table is constructed:

                      In regulon   | Not in regulon
  In DEG set (sig)       a         |      b
  Not in DEG set         c         |      d

  a = regulon_targets ∩ directional_DEGs ∩ significant_DEGs
  b = all significant DEGs (full DEG set)
  c = regulon_targets − a
  d = HVG_background − regulon_targets − b

Background: Highly variable genes (HVG) identified during SCENIC preprocessing,
matching the gene universe used for GRN inference.

DEG directions tested:
  - 'up':          genes with positive effect estimate (logFC > 0)
  - 'down':        genes with negative effect estimate (logFC < 0)
  - 'significant': all genes passing FDR threshold, regardless of direction

Fisher's test is one-tailed (alternative='greater'), testing whether regulon
targets are more enriched in DEGs than expected by chance.

Usage:
    python -m psychad_grn.fishers_enrichment --config configs/default_config.yaml \\
        --grn-file results/AD/AD_GRN.csv \\
        --deg-file results/dreamlet/dreamlet_results_processed.csv \\
        --h5ad-file data/AD.h5ad \\
        --output-dir results/fishers/
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .utils.io import load_adj, load_config, load_h5ad
from .utils.stats import apply_bh_correction, jaccard_similarity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Contingency table helpers ─────────────────────────────────────────────────

def get_regulon_targets(grn_df: pd.DataFrame, tf: str) -> set:
    """Return the set of target genes for a given TF.

    Args:
        grn_df: Adjacency DataFrame with columns ['TF', 'target', 'importance'].
        tf: Transcription factor name.

    Returns:
        Set of target gene names.
    """
    return set(grn_df.loc[grn_df["TF"] == tf, "target"])


def build_contingency(
    regulon_targets: set,
    deg_directional: set,
    deg_significant: set,
    background: set,
) -> tuple[int, int, int, int]:
    """Build the 2×2 Fisher's exact test contingency table.

    The table tests whether regulon target genes are over-represented
    among significant DEGs in the specified direction.

    a = regulon targets that are directional AND significant DEGs
    b = all significant DEGs
    c = regulon targets not in the significant DEG set
    d = background genes not in the regulon and not in the DEG set

    Args:
        regulon_targets: Set of target genes for one TF.
        deg_directional: Set of genes with expression change in the tested
            direction (up or down). Use deg_significant for the 'any' direction.
        deg_significant: Set of all significant DEGs (FDR < threshold).
        background: Full set of background genes (HVG universe).

    Returns:
        Tuple (a, b, c, d) for the 2×2 table.
    """
    a = len(regulon_targets & deg_directional & deg_significant)
    b = len(deg_significant)
    c = len(regulon_targets - deg_significant)
    d = len(background - regulon_targets - deg_significant)
    return a, b, c, d


def run_fisher_test(
    a: int, b: int, c: int, d: int, alternative: str = "greater"
) -> tuple[float, float]:
    """Run a one-tailed Fisher's exact test.

    Args:
        a, b, c, d: Cells of the 2×2 contingency table.
        alternative: Tail direction ('greater', 'less', or 'two-sided').

    Returns:
        Tuple of (odds_ratio, p_value).
    """
    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = fisher_exact(table, alternative=alternative)
    return float(odds_ratio), float(p_value)


# ── DEG filtering ─────────────────────────────────────────────────────────────

def filter_degs(
    dreamlet_df: pd.DataFrame,
    contrast: str,
    celltype: str,
    direction: str = "significant",
    fdr_threshold: float = 0.05,
    logfc_col: str = "logFC",
    fdr_col: str = "FDR",
    celltype_col: str = "assay",
    contrast_col: str = "contrast",
    id_col: str = "ID",
) -> tuple[set, set]:
    """Extract directional and significant DEG sets for a cell type × contrast.

    Args:
        dreamlet_df: Processed dreamlet results DataFrame.
        contrast: Clinical contrast variable (e.g., 'AD', 'Braak').
        celltype: Cell type label (e.g., 'Micro').
        direction: 'up', 'down', or 'significant' (all significant).
        fdr_threshold: FDR cutoff for significance.
        logfc_col: Column name for log fold change.
        fdr_col: Column name for FDR-corrected p-values.
        celltype_col: Column identifying cell type.
        contrast_col: Column identifying the contrast variable.
        id_col: Column with gene/regulon identifiers.

    Returns:
        Tuple of (deg_directional, deg_significant) as sets of gene names.
    """
    mask = (
        (dreamlet_df[celltype_col] == celltype) &
        (dreamlet_df[contrast_col] == contrast)
    )
    sub = dreamlet_df[mask].copy()

    # Strip regulon suffix if present (e.g., 'MEF2C(+)' → 'MEF2C')
    sub["gene"] = sub[id_col].str.split("(").str[0]

    sig = set(sub.loc[sub[fdr_col] < fdr_threshold, "gene"])

    if direction == "up":
        directional = set(sub.loc[sub[logfc_col] > 0, "gene"])
    elif direction == "down":
        directional = set(sub.loc[sub[logfc_col] < 0, "gene"])
    else:
        directional = sig  # 'significant' direction: all sig genes

    return directional, sig


# ── Main enrichment loop ──────────────────────────────────────────────────────

def run_fishers_enrichment(
    grn_df: pd.DataFrame,
    dreamlet_df: pd.DataFrame,
    background: set,
    contrasts: list[str],
    celltypes: Optional[list[str]] = None,
    directions: Optional[list[str]] = None,
    fdr_threshold: float = 0.05,
    alternative: str = "greater",
) -> pd.DataFrame:
    """Test regulon target gene enrichment in DEG sets via Fisher's exact test.

    Loops over all combinations of cell type × contrast × TF × direction.
    Applies Benjamini-Hochberg FDR correction to the accumulated p-values.

    Args:
        grn_df: GRN adjacency DataFrame (TF, target, importance).
        dreamlet_df: Processed dreamlet results with FDR column.
        background: Set of background genes (HVG universe from h5ad).
        contrasts: Clinical contrast variables to test.
        celltypes: Cell types to test. Defaults to all in dreamlet_df.
        directions: DEG directions: ['up', 'down', 'significant'].
        fdr_threshold: FDR cutoff for DEG significance.
        alternative: Fisher test tail direction.

    Returns:
        DataFrame with columns: celltype, contrast, tf, direction,
        odds_ratio, pvalue, a, b, c, d — plus 'FDR' and 'neg_log10_FDR'
        after correction.
    """
    if directions is None:
        directions = ["up", "down", "significant"]

    if celltypes is None:
        celltypes = dreamlet_df["assay"].unique().tolist()

    tf_list = grn_df["TF"].unique().tolist()
    records = []

    for ct in celltypes:
        for contrast in contrasts:
            for direction in directions:
                deg_dir, deg_sig = filter_degs(
                    dreamlet_df, contrast=contrast, celltype=ct,
                    direction=direction, fdr_threshold=fdr_threshold,
                )
                if not deg_sig:
                    continue

                for tf in tf_list:
                    targets = get_regulon_targets(grn_df, tf)
                    if not targets:
                        continue
                    a, b, c, d = build_contingency(targets, deg_dir, deg_sig, background)
                    if a == 0:
                        continue
                    odds_ratio, pvalue = run_fisher_test(a, b, c, d, alternative)
                    records.append({
                        "celltype": ct,
                        "contrast": contrast,
                        "tf": tf,
                        "direction": direction,
                        "odds_ratio": odds_ratio,
                        "pvalue": pvalue,
                        "n_overlap": a,
                        "n_sig_degs": b,
                        "n_regulon_only": c,
                        "n_background": d,
                    })

    if not records:
        log.warning("No significant overlaps found.")
        return pd.DataFrame()

    result_df = pd.DataFrame(records)
    result_df = apply_bh_correction(
        result_df, pvalue_col="pvalue",
        group_cols=["celltype", "contrast", "direction"],
    )
    log.info("Fisher test complete: %d tests, %d with FDR < 0.05",
             len(result_df), (result_df["FDR"] < fdr_threshold).sum())
    return result_df


# ── Jaccard similarity between regulon target sets ────────────────────────────

def compute_pairwise_jaccard(grn_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity between all TF target sets.

    High Jaccard similarity between two TFs indicates they regulate largely
    overlapping gene sets, which may reflect functional redundancy or
    co-regulatory modules.

    Args:
        grn_df: GRN adjacency DataFrame (TF, target, importance).

    Returns:
        Square DataFrame of Jaccard similarity scores (TFs × TFs).
    """
    tf_targets = {
        tf: set(sub["target"])
        for tf, sub in grn_df.groupby("TF")
    }
    tfs = sorted(tf_targets.keys())
    matrix = pd.DataFrame(index=tfs, columns=tfs, dtype=float)
    for i, tf_a in enumerate(tfs):
        for tf_b in tfs[i:]:
            score = jaccard_similarity(tf_targets[tf_a], tf_targets[tf_b])
            matrix.loc[tf_a, tf_b] = score
            matrix.loc[tf_b, tf_a] = score
    return matrix


def top_regulons_per_celltype(
    results_df: pd.DataFrame,
    n: int = 3,
    fdr_col: str = "FDR",
) -> pd.DataFrame:
    """Select the top-n regulons per cell type by overall enrichment.

    Ranks TFs by their minimum FDR across contrasts and directions, then
    returns the top n per cell type.

    Args:
        results_df: Fisher test results with FDR column.
        n: Number of top regulons to return per cell type.
        fdr_col: FDR column name.

    Returns:
        Filtered DataFrame with at most n TFs per cell type.
    """
    best = (
        results_df.groupby(["celltype", "tf"])[fdr_col]
        .min()
        .reset_index()
        .sort_values(["celltype", fdr_col])
    )
    top = best.groupby("celltype").head(n)
    return results_df[results_df["tf"].isin(top["tf"])].copy()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_fishers_pipeline(
    grn_path: str | Path,
    dreamlet_path: str | Path,
    h5ad_path: str | Path,
    output_dir: str | Path,
    contrasts: list[str],
    directions: Optional[list[str]] = None,
    fdr_threshold: float = 0.05,
    top_n: int = 3,
    compute_jaccard: bool = True,
) -> pd.DataFrame:
    """Run the full Fisher's enrichment pipeline.

    Args:
        grn_path: GRN adjacency CSV.
        dreamlet_path: Processed dreamlet results CSV.
        h5ad_path: h5ad file (used to extract HVG background from adata.var).
        output_dir: Output directory.
        contrasts: Clinical contrast variables.
        directions: DEG directions to test.
        fdr_threshold: FDR cutoff for significance.
        top_n: Top regulons to highlight per cell type.
        compute_jaccard: Whether to compute pairwise Jaccard similarity.

    Returns:
        Full Fisher test results DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grn_df = load_adj(grn_path)
    dreamlet_df = pd.read_csv(dreamlet_path)

    adata = load_h5ad(h5ad_path)
    background = set(adata.var_names)
    log.info("Background gene universe: %d HVG genes", len(background))

    results = run_fishers_enrichment(
        grn_df=grn_df,
        dreamlet_df=dreamlet_df,
        background=background,
        contrasts=contrasts,
        directions=directions,
        fdr_threshold=fdr_threshold,
    )

    if results.empty:
        return results

    results.to_csv(output_dir / "fishers_results.csv", index=False)

    top = top_regulons_per_celltype(results, n=top_n)
    top.to_csv(output_dir / "fishers_top_regulons.csv", index=False)
    log.info("Top regulons saved: %d entries", len(top))

    if compute_jaccard:
        jaccard_df = compute_pairwise_jaccard(grn_df)
        jaccard_df.to_csv(output_dir / "jaccard_similarity.csv")
        log.info("Jaccard similarity matrix saved.")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fisher's exact test for DEG enrichment in SCENIC regulon target sets."
    )
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--grn-file", required=True, help="GRN adjacency CSV.")
    parser.add_argument("--deg-file", required=True, help="Dreamlet results CSV.")
    parser.add_argument("--h5ad-file", required=True, help="h5ad file for HVG background.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--contrasts", nargs="+", default=["AD", "Braak", "CERAD", "dementia"]
    )
    parser.add_argument(
        "--directions", nargs="+", default=["up", "down", "significant"]
    )
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--no-jaccard", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    fe_cfg = cfg.get("fishers_enrichment", {})

    run_fishers_pipeline(
        grn_path=args.grn_file,
        dreamlet_path=args.deg_file,
        h5ad_path=args.h5ad_file,
        output_dir=args.output_dir,
        contrasts=fe_cfg.get("contrasts", args.contrasts),
        directions=fe_cfg.get("directions", args.directions),
        fdr_threshold=fe_cfg.get("fdr_threshold", args.fdr_threshold),
        top_n=fe_cfg.get("top_n_per_celltype", args.top_n),
        compute_jaccard=not args.no_jaccard,
    )


if __name__ == "__main__":
    main()
