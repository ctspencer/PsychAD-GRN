"""Shared visualization functions used across analysis modules.

All plot functions accept an output_path argument and save to disk.
They do not call plt.show() — call that explicitly if running interactively.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


def plot_zscore_heatmap(
    z_df: pd.DataFrame,
    output_path: str | Path,
    cmap: str = "vlag",
    vmin: float = -2.0,
    vmax: float = 2.0,
    title: str = "",
    figsize: tuple = (25, 10),
) -> None:
    """Plot a heatmap of per-cell-type regulon z-scores.

    Rows are cell types; columns are regulons. Color encodes the z-score of
    AUCell activity relative to the global mean across all cells.

    Args:
        z_df: DataFrame of shape (n_celltypes × n_regulons) with z-scores.
        output_path: Path to save the figure (PNG and SVG).
        cmap: Colormap name (default 'vlag' — blue/white/red diverging).
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        title: Figure title.
        figsize: Figure width × height in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        z_df,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"shrink": 0.5, "aspect": 20},
    )
    ax.set_title(title)
    ax.set_xlabel("Regulon")
    ax.set_ylabel("Cell Type")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_rss_panel(
    rss_matrix: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 5,
    ncols: int = 5,
    figsize: tuple = (15, 40),
) -> None:
    """Plot per-cell-type Regulon Specificity Score (RSS) scatter panels.

    One subplot per cell type. Each point is a regulon; x-axis is regulon
    index, y-axis is RSS score. The top_n regulons are labeled.

    Args:
        rss_matrix: DataFrame of shape (n_celltypes × n_regulons) with RSS scores.
            Produced by pyscenic.rss.regulon_specificity_scores().
        output_path: Path to save the figure.
        top_n: Number of top regulons to label per cell type.
        ncols: Number of subplot columns.
        figsize: Overall figure dimensions.
    """
    from pyscenic.plotting import plot_rss

    celltypes = sorted(rss_matrix.index.tolist())
    nrows = int(np.ceil(len(celltypes) / ncols))
    fig = plt.figure(figsize=figsize)

    for i, ct in enumerate(celltypes, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        x = rss_matrix.loc[ct]
        plot_rss(rss_matrix, ct, top_n=top_n, max_n=None, ax=ax)
        ax.set_ylim(x.min() - (x.max() - x.min()) * 0.05,
                    x.max() + (x.max() - x.min()) * 0.05)
        for t in ax.texts:
            t.set_fontsize(9)
        ax.set_ylabel("")
        ax.set_xlabel("")
        adjust_text(
            ax.texts,
            autoalign="xy",
            ha="right",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color="lightgrey"),
            precision=0.001,
        )

    fig.text(0.5, 0.0, "Regulon", ha="center", size="large")
    fig.text(0.0, 0.5, "RSS", ha="center", rotation="vertical", size="large")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmap(
    sim_df: pd.DataFrame,
    output_path: str | Path,
    x_label: str = "Dataset A",
    y_label: str = "Dataset B",
    title: str = "Regulon Concordance",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "Reds",
    figsize: tuple = (12, 10),
) -> None:
    """Plot a heatmap of pairwise regulon similarity or concordance.

    Used to compare regulon activity patterns across datasets (Pearson
    correlation of z-scores) or across cell types (Jaccard similarity).

    Args:
        sim_df: Square or rectangular DataFrame of similarity scores.
        output_path: Path to save the figure.
        x_label: Label for the x-axis (columns = dataset/cell type A).
        y_label: Label for the y-axis (rows = dataset/cell type B).
        title: Figure title.
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        cmap: Colormap name.
        figsize: Figure dimensions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=0.8)
    sns.heatmap(
        sim_df,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.5},
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_go_bubble(
    enrichment_df: pd.DataFrame,
    output_path: str | Path,
    condition: str = "",
    tf_order: Optional[list] = None,
    per_tf_terms: int = 6,
    pval_thresh: float = 0.05,
    figsize: tuple = (14, 10),
) -> None:
    """Plot a bubble chart of GO term enrichment across TFs.

    Each bubble represents one GO term enriched in one TF's target gene set.
    Bubble size encodes -log10(FDR); bubble color encodes odds ratio.
    GO terms are ordered on the y-axis by Ward distance clustering.

    Args:
        enrichment_df: DataFrame with columns ['Term', 'regulon',
            'Adjusted P-value', 'Odds Ratio', 'neglog10FDR'].
        output_path: Path to save the figure.
        condition: Condition label for the figure title (e.g., 'AD-unique').
        tf_order: Ordered list of TF names for x-axis. Defaults to alphabetical.
        per_tf_terms: Maximum number of GO terms shown per TF.
        pval_thresh: FDR threshold for inclusion.
        figsize: Figure dimensions.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram

    df = enrichment_df[enrichment_df["Adjusted P-value"] <= pval_thresh].copy()
    if df.empty:
        return

    df["neglog10FDR"] = -np.log10(
        df["Adjusted P-value"].clip(lower=np.finfo(float).tiny)
    )
    df["desc"] = df["Term"].str.split("(").str[0].str.strip()

    if tf_order is None:
        tf_order = sorted(df["regulon"].unique().tolist())

    # Select top per_tf_terms terms per TF by p-value
    top_terms = (
        df.sort_values("Adjusted P-value")
        .groupby("regulon")
        .head(per_tf_terms)
    )

    # Ward-cluster the GO terms for stable y-axis ordering
    mat = top_terms.pivot_table(
        index="desc", columns="regulon", values="neglog10FDR", fill_value=0.0
    )
    if mat.shape[0] > 1:
        linked = linkage(mat.values, method="ward")
        order = dendrogram(linked, no_plot=True)["leaves"]
        term_order = mat.index[order].tolist()
    else:
        term_order = mat.index.tolist()

    top_terms["desc"] = pd.Categorical(
        top_terms["desc"], categories=term_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        x=[tf_order.index(r) if r in tf_order else 0 for r in top_terms["regulon"]],
        y=top_terms["desc"].cat.codes,
        s=top_terms["neglog10FDR"] * 20,
        c=top_terms["Odds Ratio"],
        cmap="RdYlBu_r",
        alpha=0.85,
        edgecolors="gray",
        linewidths=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="Odds Ratio", shrink=0.5)
    ax.set_xticks(range(len(tf_order)))
    ax.set_xticklabels(tf_order, rotation=45, ha="right")
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order, fontsize=8)
    ax.set_title(f"GO BP Enrichment — {condition}")
    ax.set_xlabel("Regulon")
    ax.set_ylabel("GO Term")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
