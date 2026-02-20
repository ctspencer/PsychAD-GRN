"""Regulon Specificity Score (RSS) and Robust Rank Aggregation (RRA).

This module prioritizes regulons that are (a) specifically enriched in
particular cell types and (b) consistently ranked highly across multiple
evidence sources.

Step 1 — RSS: Computes the Regulon Specificity Score for each TF × cell type
pair. RSS measures how exclusively a regulon is active in one cell type vs.
others, analogous to the Jensen-Shannon divergence between the regulon's AUCell
distribution and the ideal cell-type-specific distribution. A high RSS for
'MEF2C in Micro' means MEF2C activity is disproportionately concentrated in
microglia.

Step 2 — RRA: Integrates three independent evidence sources via Robust Rank
Aggregation (Kolde et al. 2012 Bioinformatics). For each cell type, TFs are
ranked by:
  (a) AUCell z-score — how enriched the regulon is in that cell type
  (b) RSS score — how specifically the regulon is expressed in that cell type
  (c) Gene expression z-score (GEX) — how highly the TF gene itself is
      expressed in that cell type
RRA combines these three ranked lists using a beta distribution test.
The resulting rho score (a corrected p-value) reflects how consistently a TF
ranks near the top across all three evidence sources.

Usage:
    python -m psychad_grn.regulon_specificity --config configs/default_config.yaml \\
        --aucell-file results/aucell/aucell.csv \\
        --h5ad-file data/CTRL.h5ad \\
        --output-dir results/regulon_specificity/
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils.io import load_config, load_h5ad
from .utils.stats import aggregate_ranks, compute_zscore_matrix
from .utils.plotting import plot_rss_panel, plot_zscore_heatmap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── RSS computation ───────────────────────────────────────────────────────────

def compute_rss(
    aucell_df: pd.DataFrame,
    cell_labels: pd.Series,
) -> pd.DataFrame:
    """Compute Regulon Specificity Scores (RSS) for each regulon × cell type.

    RSS is based on the Jensen-Shannon divergence between the regulon's AUCell
    score distribution across cells and the ideal distribution where all
    activity is confined to the target cell type. A score of 1 indicates
    perfect specificity; 0 indicates no specificity.

    Implements pyscenic.rss.regulon_specificity_scores().

    Args:
        aucell_df: AUCell matrix of shape (n_cells × n_regulons).
            Index must match cell_labels.index.
        cell_labels: Series mapping cell IDs → cell type labels.

    Returns:
        RSS matrix of shape (n_celltypes × n_regulons).
    """
    from pyscenic.rss import regulon_specificity_scores

    shared_idx = aucell_df.index.intersection(cell_labels.index)
    return regulon_specificity_scores(aucell_df.loc[shared_idx], cell_labels.loc[shared_idx])


def top_regulons_by_rss(rss_matrix: pd.DataFrame, n: int = 13) -> dict[str, list]:
    """Select the top-n regulons per cell type ranked by RSS score.

    Args:
        rss_matrix: RSS matrix from compute_rss() (n_celltypes × n_regulons).
        n: Number of top regulons to select per cell type.

    Returns:
        Dict mapping cell type → list of top-n regulon names.
    """
    return {
        ct: rss_matrix.loc[ct].sort_values(ascending=False).head(n).index.tolist()
        for ct in rss_matrix.index
    }


# ── RRA integration ───────────────────────────────────────────────────────────

def integrate_rankings(
    aucell_df: pd.DataFrame,
    rss_matrix: pd.DataFrame,
    gex_zscores: pd.DataFrame,
    top_n: int = 13,
    rra_method: str = "RRA",
    score_threshold: float = 0.05,
) -> pd.DataFrame:
    """Integrate AUCell, RSS, and gene expression rankings via RRA.

    For each cell type, constructs three ranked lists of TFs ordered by:
      (a) AUCell z-score (descending) — regulon activity enrichment
      (b) RSS score (descending) — regulon cell-type specificity
      (c) GEX z-score (descending) — TF gene expression enrichment

    These three lists are aggregated using Robust Rank Aggregation (RRA),
    which assigns each TF a rho score reflecting how consistently it ranks
    highly across all three sources. Only TFs with rho < score_threshold
    are returned.

    Args:
        aucell_df: AUCell z-score matrix (n_celltypes × n_regulons).
            Produced by compute_zscore_matrix().
        rss_matrix: RSS matrix (n_celltypes × n_regulons).
            Produced by compute_rss().
        gex_zscores: Gene expression z-score matrix (n_celltypes × n_genes).
            Columns must include TF gene names.
        top_n: Number of top regulons to retain per cell type after RRA.
        rra_method: Rank aggregation method ('RRA', 'mean', 'median').
        score_threshold: Maximum rho score for inclusion (default 0.05).

    Returns:
        Long-format DataFrame with columns ['celltype', 'regulon', 'rra_score'],
        sorted by rra_score ascending within each cell type.
    """
    # Find common cell types and regulons across all three sources
    common_celltypes = (
        set(aucell_df.index) & set(rss_matrix.index) & set(gex_zscores.index)
    )
    # Regulons may be named 'MEF2C(+)' in aucell/rss but 'MEF2C' in GEX
    # Strip the '(+)'/'(-)' suffix for GEX matching
    regulon_to_gene = {r: r.split("(")[0] for r in aucell_df.columns}

    records = []
    for ct in sorted(common_celltypes):
        # Ranked list 1: AUCell z-score (descending)
        auc_ranking = aucell_df.loc[ct].sort_values(ascending=False).index.tolist()

        # Ranked list 2: RSS score (descending)
        rss_ranking = rss_matrix.loc[ct].sort_values(ascending=False).index.tolist()

        # Ranked list 3: GEX z-score for the TF gene (descending)
        # Use regulons that have a matching TF gene in GEX
        gex_row = gex_zscores.loc[ct]
        gex_ranking = [
            reg for reg in aucell_df.columns
            if regulon_to_gene.get(reg, "") in gex_row.index
        ]
        gex_ranking = sorted(
            gex_ranking,
            key=lambda r: gex_row.get(regulon_to_gene.get(r, ""), np.nan),
            reverse=True,
        )
        gex_ranking = [r for r in gex_ranking if not np.isnan(
            gex_row.get(regulon_to_gene.get(r, ""), np.nan)
        )]

        if not gex_ranking:
            # Fall back to two-source integration if GEX data is unavailable
            ranked_lists = [auc_ranking, rss_ranking]
        else:
            ranked_lists = [auc_ranking, rss_ranking, gex_ranking]

        rra_result = aggregate_ranks(ranked_lists, method=rra_method)
        rra_result = rra_result[rra_result["Score"] < score_threshold].head(top_n)

        for _, row in rra_result.iterrows():
            records.append({
                "celltype": ct,
                "regulon": row["Gene"],
                "rra_score": row["Score"],
            })

    return pd.DataFrame(records).sort_values(["celltype", "rra_score"]).reset_index(drop=True)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_regulon_specificity(
    aucell_df: pd.DataFrame,
    adata,
    output_dir: str | Path,
    celltype_col: str = "subclass",
    top_n: int = 13,
    rra_method: str = "RRA",
    score_threshold: float = 0.05,
    plot: bool = False,
) -> dict:
    """Compute RSS and RRA-integrated regulon rankings per cell type.

    Args:
        aucell_df: Raw AUCell matrix (n_cells × n_regulons).
        adata: AnnData object with cell metadata (obs) and expression (X).
        output_dir: Directory for output files.
        celltype_col: Column in adata.obs with cell type labels.
        top_n: Top regulons to report per cell type after RRA.
        rra_method: RRA aggregation method.
        score_threshold: Maximum rho score for inclusion.
        plot: Whether to generate RSS panel and z-score heatmap figures.

    Returns:
        Dict with keys: 'rss' (DataFrame), 'zscore' (DataFrame), 'rankings' (DataFrame).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_labels = adata.obs[celltype_col]
    rss = compute_rss(aucell_df, cell_labels)
    rss.to_csv(output_dir / "rss_matrix.csv")
    log.info("RSS matrix saved (%d celltypes × %d regulons)", *rss.shape)

    # AUCell z-scores (for ranking and heatmap)
    auc_zscores = compute_zscore_matrix(aucell_df, celltype_col, obs_df=adata.obs)
    auc_zscores.to_csv(output_dir / "aucell_zscore.csv")

    # Gene expression z-scores (TF gene expression across cell types)
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    gex_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )
    gex_df[celltype_col] = adata.obs[celltype_col].values
    gex_global_mean = gex_df.drop(columns=[celltype_col]).mean()
    gex_global_std = gex_df.drop(columns=[celltype_col]).std()
    gex_zscores = (
        gex_df.groupby(celltype_col).mean() - gex_global_mean
    ) / gex_global_std

    rankings = integrate_rankings(
        auc_zscores, rss, gex_zscores,
        top_n=top_n, rra_method=rra_method, score_threshold=score_threshold,
    )
    rankings.to_csv(output_dir / "rra_rankings.csv", index=False)
    log.info("RRA rankings saved: %d regulon × cell type pairs", len(rankings))

    if plot:
        # Top regulons heatmap (AUCell z-scores)
        top_regs = rankings["regulon"].unique().tolist()
        if top_regs:
            plot_zscore_heatmap(
                auc_zscores[[r for r in top_regs if r in auc_zscores.columns]],
                output_dir / "aucell_zscore_heatmap.png",
                title="Top Regulons by AUCell Z-score",
            )
        plot_rss_panel(rss, output_dir / "rss_panel.png", top_n=5)

    return {"rss": rss, "zscore": auc_zscores, "rankings": rankings}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RSS and RRA-integrated regulon rankings per cell type."
    )
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--aucell-file", required=True, help="AUCell CSV (cells × regulons).")
    parser.add_argument("--h5ad-file", required=True, help="h5ad with expression and metadata.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--celltype-col", default="subclass")
    parser.add_argument("--top-n", type=int, default=13)
    parser.add_argument("--rra-method", default="RRA", choices=["RRA", "mean", "median"])
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true", help="Generate figures.")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    rss_cfg = cfg.get("regulon_specificity", {})

    from .utils.io import load_aucell
    aucell_df = load_aucell(args.aucell_file)
    adata = load_h5ad(args.h5ad_file)

    run_regulon_specificity(
        aucell_df=aucell_df,
        adata=adata,
        output_dir=args.output_dir,
        celltype_col=rss_cfg.get("celltype_col", args.celltype_col),
        top_n=rss_cfg.get("top_n", args.top_n),
        rra_method=rss_cfg.get("rra_method", args.rra_method),
        score_threshold=args.score_threshold,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
