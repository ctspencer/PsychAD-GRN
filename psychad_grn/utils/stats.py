"""Shared statistical functions used across analysis modules."""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from statsmodels.stats.multitest import multipletests


def compute_zscore_matrix(
    aucell_df: pd.DataFrame,
    celltype_col: str,
    obs_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute per-cell-type z-scored AUCell activity.

    For each regulon, calculates the mean AUCell score within each cell type,
    then normalizes relative to the global mean and standard deviation across
    all cells. This produces a z-score that reflects how enriched a regulon is
    in a given cell type compared to the full dataset.

    Formula: z = (celltype_mean - global_mean) / global_std

    Args:
        aucell_df: DataFrame of shape (n_cells × n_regulons). Index must be
            cell IDs matching obs_df if provided.
        celltype_col: Column name in obs_df containing cell type labels. If
            obs_df is None, aucell_df must already contain this column.
        obs_df: Optional metadata DataFrame with cell type annotations.
            If provided, the celltype_col is joined onto aucell_df by index.

    Returns:
        DataFrame of shape (n_celltypes × n_regulons) with z-scores.
    """
    df = aucell_df.copy()
    if obs_df is not None:
        df[celltype_col] = df.index.map(obs_df[celltype_col])

    regulon_cols = [c for c in df.columns if c != celltype_col]
    global_mean = df[regulon_cols].mean()
    global_std = df[regulon_cols].std()

    z_matrix = (df.groupby(celltype_col)[regulon_cols].mean() - global_mean) / global_std
    return z_matrix


def apply_bh_correction(
    df: pd.DataFrame,
    pvalue_col: str = "pvalue",
    group_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to a p-value column.

    Adds 'FDR' and 'neg_log10_FDR' columns to the DataFrame. When group_cols
    is specified, correction is applied independently within each group
    (e.g., per cell type or per contrast).

    Args:
        df: DataFrame containing a column of p-values.
        pvalue_col: Name of the column containing raw p-values.
        group_cols: Optional list of column names defining groups for
            within-group correction.

    Returns:
        Copy of df with 'FDR' and 'neg_log10_FDR' columns added.
    """
    df = df.copy()
    if group_cols:
        fdr_vals = np.ones(len(df))
        for _, idx in df.groupby(group_cols).groups.items():
            pvals = df.loc[idx, pvalue_col].fillna(1.0).values
            _, fdr, _, _ = multipletests(pvals, method="fdr_bh")
            fdr_vals[idx] = fdr
        df["FDR"] = fdr_vals
    else:
        pvals = df[pvalue_col].fillna(1.0).values
        _, fdr, _, _ = multipletests(pvals, method="fdr_bh")
        df["FDR"] = fdr

    df["neg_log10_FDR"] = -np.log10(df["FDR"].clip(lower=np.finfo(float).tiny))
    return df


def stouffer_meta(
    pvalues: list[float],
    weights: Optional[list[float]] = None,
) -> float:
    """Combine p-values from independent tests using Stouffer's method.

    Converts each p-value to a z-score (inverse normal CDF), applies
    optional weights, sums, and converts back to a combined p-value.
    Used here to integrate regulon activity evidence across cohorts
    (MSSM, RADC, ROSMAP).

    Args:
        pvalues: List of p-values from independent tests (one per cohort).
        weights: Optional list of weights (e.g., sqrt(n_samples) per cohort).
            Defaults to equal weights.

    Returns:
        Combined p-value from Stouffer's weighted z-score method.
    """
    pvalues = np.clip(pvalues, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    z_scores = norm.ppf(1 - np.array(pvalues))
    if weights is None:
        weights = np.ones(len(z_scores))
    weights = np.array(weights, dtype=float)
    combined_z = np.dot(weights, z_scores) / np.sqrt(np.sum(weights**2))
    return float(norm.sf(combined_z))


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Measures overlap between two regulon target gene sets. A value of 1
    indicates identical sets; 0 indicates no overlap.

    Args:
        set_a: First set of elements.
        set_b: Second set of elements.

    Returns:
        Jaccard index in [0, 1]. Returns 0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── Robust Rank Aggregation (RRA) ─────────────────────────────────────────────
# Implements the RRA algorithm from Kolde et al. (2012) Bioinformatics.
# Used to integrate ranked regulon lists from AUCell, RSS, and gene expression.

def rank_matrix(
    glist: list[list],
    N: Optional[int] = None,
) -> tuple[np.ndarray, list]:
    """Convert a collection of ranked lists into a normalized rank matrix.

    Each column corresponds to one ranked list; each row corresponds to one
    unique element. Values are normalized ranks in (0, 1]. Missing elements
    receive NaN.

    Args:
        glist: List of ranked lists (e.g., [aucell_ranking, rss_ranking]).
        N: Total number of rankable elements. Defaults to the number of
            unique elements across all lists.

    Returns:
        Tuple of (rank_matrix, unique_elements) where rank_matrix has shape
        (n_elements × n_lists) and unique_elements is the sorted element list.
    """
    unique_elements = sorted(set(item for sublist in glist for item in sublist))
    n = len(unique_elements) if N is None else N
    rank_mat = np.full((len(unique_elements), len(glist)), np.nan)
    elem_idx = {e: i for i, e in enumerate(unique_elements)}

    for col, ranking in enumerate(glist):
        for rank, elem in enumerate(ranking, start=1):
            rank_mat[elem_idx[elem], col] = rank / n

    return rank_mat, unique_elements


def beta_scores(r: np.ndarray) -> np.ndarray:
    """Compute beta CDF p-values for a vector of normalized ranks.

    For each rank r_i in the sorted vector, computes P(Beta(i, n-i+1) <= r_i)
    where n is the number of non-NaN elements. This is the theoretical
    p-value for observing rank r_i or better by chance.

    Args:
        r: 1-D array of normalized ranks (values in (0, 1]).

    Returns:
        Array of p-values, one per rank.
    """
    r = np.sort(r[~np.isnan(r)])
    n = len(r)
    if n == 0:
        return np.array([])
    return beta.cdf(r, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)


def rho_scores(r: np.ndarray, top_cutoff: Optional[list] = None) -> float:
    """Compute the RRA rho score for a vector of normalized ranks.

    The rho score is the minimum beta p-value multiplied by the number of
    elements (correcting for multiple comparisons within the list). A small
    rho indicates that the element ranks significantly higher than expected
    by chance across all input lists.

    Args:
        r: 1-D array of normalized ranks (values in (0, 1]).
        top_cutoff: Optional per-list cutoff values for top-k elements.

    Returns:
        Rho score (a corrected p-value).
    """
    p_values = beta_scores(r)
    if top_cutoff is not None:
        valid = ~np.isnan(r)
        p_values = p_values[: sum(valid)]
        cutoff_idx = min(len(p_values), len(top_cutoff))
        p_values = np.minimum(p_values[:cutoff_idx], top_cutoff[:cutoff_idx])
    return float(np.min(p_values) * len(p_values)) if len(p_values) > 0 else 1.0


def aggregate_ranks(
    glist: list[list],
    N: Optional[int] = None,
    method: str = "RRA",
) -> pd.DataFrame:
    """Aggregate multiple ranked lists into a single ranked list.

    Used to combine AUCell activity rankings, RSS rankings, and gene
    expression rankings into a consensus regulon priority list per cell type.

    Args:
        glist: List of ranked lists (e.g., one list per scoring method).
        N: Total number of rankable elements. Defaults to unique elements.
        method: Aggregation method — 'RRA' (robust rank aggregation),
            'mean', 'median', or 'min'.

    Returns:
        DataFrame with columns ['Gene', 'Score'] sorted by Score ascending.
        For RRA, Score is the rho p-value; smaller = more consistently
        high-ranked across input lists.
    """
    rank_mat, unique_elements = rank_matrix(glist, N=N)

    if method == "RRA":
        scores = [rho_scores(row) for row in rank_mat]
    elif method == "mean":
        scores = np.nanmean(rank_mat, axis=1).tolist()
    elif method == "median":
        scores = np.nanmedian(rank_mat, axis=1).tolist()
    elif method == "min":
        scores = np.nanmin(rank_mat, axis=1).tolist()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: RRA, mean, median, min.")

    return (
        pd.DataFrame({"Gene": unique_elements, "Score": scores})
        .sort_values("Score")
        .reset_index(drop=True)
    )
