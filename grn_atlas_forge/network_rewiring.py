"""Differential GRN analysis: network rewiring, centrality, and GO enrichment.

Compares gene regulatory networks between AD, Control, and Resilient conditions
to identify transcription factors that are differentially connected (rewired)
between disease states.

Pipeline:
  1. Filter networks: retain top 25% of edges by importance (quantile ≥ 0.75),
     remove self-loops, and require each TF to have ≥ 10 targets post-filtering.
  2. Build directed NetworkX graphs for each condition.
  3. Compute per-TF rewiring metrics between conditions:
       - Δ edges: number of gained and lost TF–target edges
       - Δ importance: change in aggregated edge weight (total TF influence)
  4. Compute network centrality metrics per TF for each condition:
       - PageRank: importance based on incoming regulatory edges
       - Degree centrality: fraction of all genes regulated
       - Betweenness centrality: TF as a regulatory bottleneck/hub
       - Closeness centrality: average distance to all target genes
  5. GO Biological Process enrichment of condition-specific TF target genes,
     using gseapy against GO Biological Processes 2023.
  6. Cluster GO terms by Ward linkage on −log10(FDR) scores for visualization.

Usage:
    python -m psychad_grn.network_rewiring --config configs/default_config.yaml \\
        --ad-adj results/AD/AD_GRN.csv \\
        --ctrl-adj results/CTRL/CTRL_GRN.csv \\
        --res-adj results/RES/RES_GRN.csv \\
        --go-gmt data/pathways/GO_Biological_Process_2023.gmt \\
        --output-dir results/network_rewiring/
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import gseapy as gp
from scipy.cluster.hierarchy import dendrogram, linkage

from .utils.io import load_adj, load_config
from .utils.plotting import plot_go_bubble

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Edge filtering ────────────────────────────────────────────────────────────

def remove_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    """Remove edges where the TF regulates itself.

    Self-loops can arise from broad binding motifs in the cisTarget databases
    and are excluded as they do not represent meaningful trans-regulatory
    relationships.

    Args:
        df: Adjacency DataFrame with columns ['TF', 'target', 'importance'].

    Returns:
        Filtered DataFrame.
    """
    return df[df["TF"] != df["target"]].copy()


def filter_top_edges(df: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
    """Retain only high-confidence edges (top fraction by importance score).

    Filtering to the top 25% of edges reduces noise from low-confidence
    regulatory relationships inferred by GRNboost2 and focuses analysis
    on the most strongly supported TF–target pairs.

    Args:
        df: Adjacency DataFrame.
        quantile: Minimum importance quantile threshold (default 0.75 = top 25%).

    Returns:
        Filtered DataFrame.
    """
    threshold = df["importance"].quantile(quantile)
    return df[df["importance"] >= threshold].copy()


def filter_tf_by_min_targets(df: pd.DataFrame, min_targets: int = 10) -> pd.DataFrame:
    """Remove TFs with fewer than min_targets edges after filtering.

    TFs regulating very few genes after edge filtering are likely to produce
    noisy centrality and enrichment estimates.

    Args:
        df: Adjacency DataFrame.
        min_targets: Minimum number of target edges per TF.

    Returns:
        Filtered DataFrame.
    """
    counts = df.groupby("TF")["target"].transform("count")
    return df[counts >= min_targets].copy()


def apply_standard_filters(
    df: pd.DataFrame,
    quantile: float = 0.75,
    min_targets: int = 10,
) -> pd.DataFrame:
    """Apply the standard three-step edge filter pipeline.

    Applies in order: remove self-loops → filter top-quantile edges →
    filter TFs with insufficient target coverage.

    Args:
        df: Adjacency DataFrame.
        quantile: Top edge quantile to retain.
        min_targets: Minimum TF targets after filtering.

    Returns:
        Filtered adjacency DataFrame.
    """
    df = remove_self_loops(df)
    df = filter_top_edges(df, quantile=quantile)
    df = filter_tf_by_min_targets(df, min_targets=min_targets)
    return df


# ── NetworkX graph construction ───────────────────────────────────────────────

def construct_nx_graph(
    adj_df: pd.DataFrame,
    tf_set: Optional[set] = None,
) -> nx.DiGraph:
    """Build a directed NetworkX graph from an adjacency DataFrame.

    Each node is labeled as either a transcription factor or a target gene.
    Edge weights correspond to GRNboost2 importance scores.

    Args:
        adj_df: Adjacency DataFrame with columns ['TF', 'target', 'importance'].
        tf_set: Optional set of known TF gene names. Used to label node types.
            If None, any node that appears as a source (TF) column is labeled TF.

    Returns:
        Directed NetworkX graph with 'weight' edge attributes and 'type'
        node attributes ('TF' or 'Gene').
    """
    G = nx.DiGraph()
    for _, row in adj_df.iterrows():
        G.add_edge(row["TF"], row["target"], weight=row["importance"])

    inferred_tfs = set(adj_df["TF"].unique()) if tf_set is None else tf_set
    for node in G.nodes():
        G.nodes[node]["type"] = "TF" if node in inferred_tfs else "Gene"

    return G


# ── Centrality metrics ────────────────────────────────────────────────────────

def compute_centrality_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """Compute four centrality metrics for all nodes in the network.

    - PageRank: Importance based on the structure of incoming regulatory edges;
      a TF with high PageRank is targeted by many other influential TFs.
    - Degree centrality: Fraction of all other nodes connected to this node.
    - Betweenness centrality: Fraction of shortest paths passing through this node;
      identifies regulatory bottlenecks.
    - Closeness centrality: Inverse of average shortest path length; reflects
      how quickly a TF can influence the rest of the network.

    Args:
        G: Directed NetworkX graph.

    Returns:
        DataFrame with columns ['node', 'pagerank', 'degree', 'betweenness',
        'closeness'], one row per node.
    """
    pr = nx.pagerank(G, weight="weight")
    deg = nx.degree_centrality(G)
    btwn = nx.betweenness_centrality(G, weight="weight")
    close = nx.closeness_centrality(G)

    return pd.DataFrame({
        "node": list(pr.keys()),
        "pagerank": list(pr.values()),
        "degree": [deg[n] for n in pr],
        "betweenness": [btwn[n] for n in pr],
        "closeness": [close[n] for n in pr],
    })


# ── Rewiring metrics ──────────────────────────────────────────────────────────

def compute_rewiring_metrics(
    adj_a: pd.DataFrame,
    adj_b: pd.DataFrame,
    label_a: str = "condition_A",
    label_b: str = "condition_B",
) -> pd.DataFrame:
    """Compute TF-level rewiring metrics between two GRN conditions.

    For each TF, calculates the number of edges gained (present in A but
    not B), lost (present in B but not A), and shared, as well as the
    change in total regulatory importance.

    Args:
        adj_a: Adjacency DataFrame for condition A (e.g., AD).
        adj_b: Adjacency DataFrame for condition B (e.g., CTRL).
        label_a: Label for condition A (used in column names).
        label_b: Label for condition B (used in column names).

    Returns:
        DataFrame with per-TF rewiring metrics.
    """
    edges_a = adj_a.set_index(["TF", "target"])["importance"]
    edges_b = adj_b.set_index(["TF", "target"])["importance"]

    set_a = set(edges_a.index)
    set_b = set(edges_b.index)

    gained = set_a - set_b   # edges in A but not B
    lost = set_b - set_a     # edges in B but not A
    shared = set_a & set_b

    all_tfs = set(adj_a["TF"].unique()) | set(adj_b["TF"].unique())
    records = []
    for tf in sorted(all_tfs):
        n_gained = sum(1 for (t, g) in gained if t == tf)
        n_lost = sum(1 for (t, g) in lost if t == tf)
        n_shared = sum(1 for (t, g) in shared if t == tf)

        imp_a = edges_a[[idx for idx in edges_a.index if idx[0] == tf]].sum()
        imp_b = edges_b[[idx for idx in edges_b.index if idx[0] == tf]].sum()

        records.append({
            "TF": tf,
            f"n_edges_{label_a}": n_gained + n_shared,
            f"n_edges_{label_b}": n_lost + n_shared,
            "n_edges_gained": n_gained,
            "n_edges_lost": n_lost,
            "n_edges_shared": n_shared,
            f"importance_{label_a}": imp_a,
            f"importance_{label_b}": imp_b,
            "delta_importance": imp_a - imp_b,
        })

    return pd.DataFrame(records)


# ── GO enrichment ─────────────────────────────────────────────────────────────

def get_unique_targets(
    tf: str,
    df_focal: pd.DataFrame,
    df_others: list[pd.DataFrame],
) -> set:
    """Get target genes unique to one condition for a given TF.

    'Unique' means these targets appear in df_focal but not in any of
    the other condition DataFrames. These condition-specific targets
    are used for GO enrichment to characterize the TF's condition-specific
    regulatory program.

    Args:
        tf: Transcription factor name.
        df_focal: Adjacency DataFrame for the focal condition.
        df_others: List of adjacency DataFrames for comparison conditions.

    Returns:
        Set of target gene names unique to the focal condition.
    """
    focal_targets = set(df_focal.loc[df_focal["TF"] == tf, "target"])
    other_targets: set = set()
    for df in df_others:
        other_targets |= set(df.loc[df["TF"] == tf, "target"])
    return focal_targets - other_targets


def enrich_go(
    gene_list: list[str],
    gmt_path: str,
    cutoff: float = 1.0,
) -> Optional[pd.DataFrame]:
    """Run GO Biological Process enrichment using gseapy.

    Uses gseapy's overrepresentation analysis (ORA) against the GO
    Biological Process 2023 gene set collection.

    Args:
        gene_list: List of gene names to test.
        gmt_path: Path to or name of the GMT gene set database.
            Use 'GO_Biological_Process_2023' for the built-in collection.
        cutoff: p-value cutoff for initial inclusion (default 1.0 = include all;
            filter by FDR downstream).

    Returns:
        DataFrame with enrichment results, or None if no enrichments found.
    """
    if not gene_list:
        return None

    try:
        enr = gp.enrich(
            gene_list=gene_list,
            gene_sets=gmt_path,
            cutoff=cutoff,
            verbose=False,
        )
    except Exception as e:
        log.warning("gseapy enrichment failed: %s", e)
        return None

    if enr is None or enr.res2d is None or enr.res2d.empty:
        return None

    return enr.res2d.copy()


def run_go_enrichment_all_tfs(
    unique_targets: dict[str, list[str]],
    gmt_path: str,
    pval_thresh: float = 0.05,
) -> pd.DataFrame:
    """Run GO enrichment for condition-unique target genes of all TFs.

    Args:
        unique_targets: Dict mapping TF name → list of unique target genes.
        gmt_path: GMT gene set database path or name.
        pval_thresh: FDR threshold for filtering results.

    Returns:
        Concatenated enrichment DataFrame across all TFs, filtered by FDR.
    """
    all_results = []
    for tf, genes in unique_targets.items():
        if not genes:
            continue
        res = enrich_go(genes, gmt_path)
        if res is not None:
            res["regulon"] = tf
            all_results.append(res)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[combined["Adjusted P-value"] <= pval_thresh]
    log.info("GO enrichment: %d significant terms across %d TFs",
             len(combined), combined["regulon"].nunique())
    return combined


def cluster_go_terms_ward(
    enrichment_df: pd.DataFrame,
    term_col: str = "Term",
    regulon_col: str = "regulon",
    value_col: str = "Adjusted P-value",
) -> list[str]:
    """Order GO terms by Ward distance clustering for stable visualization.

    Computes −log10 FDR per term × TF, builds a pivot table, and applies
    Ward linkage hierarchical clustering. The resulting leaf order is used
    to arrange GO terms on the y-axis of bubble plots.

    Args:
        enrichment_df: GO enrichment results with term, regulon, and FDR columns.
        term_col: Column with GO term names.
        regulon_col: Column with TF/regulon names.
        value_col: Column with adjusted p-values.

    Returns:
        Ordered list of GO term names (leaf order from Ward dendrogram).
    """
    df = enrichment_df.copy()
    df["desc"] = df[term_col].str.split("(").str[0].str.strip()
    df["neglog10FDR"] = -np.log10(
        df[value_col].clip(lower=np.finfo(float).tiny)
    )

    mat = df.pivot_table(
        index="desc", columns=regulon_col, values="neglog10FDR", fill_value=0.0
    )

    if mat.shape[0] <= 1:
        return mat.index.tolist()

    linked = linkage(mat.values, method="ward")
    order = dendrogram(linked, no_plot=True)["leaves"]
    return mat.index[order].tolist()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_network_rewiring(
    adj_paths: dict[str, str | Path],
    output_dir: str | Path,
    gmt_path: str,
    tf_list_path: Optional[str | Path] = None,
    quantile: float = 0.75,
    min_targets: int = 10,
    pval_thresh: float = 0.05,
    plot: bool = False,
) -> dict:
    """Run the complete network rewiring analysis pipeline.

    Args:
        adj_paths: Dict mapping condition label → GRN adjacency CSV path
            (e.g., {'AD': 'results/AD_GRN.csv', 'CTRL': '...'}).
        output_dir: Root output directory.
        gmt_path: GMT database for GO enrichment.
        tf_list_path: Optional path to a TF list file for node type labeling.
        quantile: Edge importance quantile threshold.
        min_targets: Minimum TF target count after filtering.
        pval_thresh: FDR threshold for GO enrichment.
        plot: Whether to generate GO bubble plots.

    Returns:
        Dict with keys: 'centrality' (per-condition DataFrames),
        'rewiring' (per-comparison DataFrames), 'go_enrichment' (DataFrame).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter adjacency matrices
    tf_set = None
    if tf_list_path:
        tf_set = set(pd.read_csv(tf_list_path, header=None)[0].tolist())

    filtered_adjs = {}
    graphs = {}
    for condition, path in adj_paths.items():
        raw = load_adj(path)
        filt = apply_standard_filters(raw, quantile=quantile, min_targets=min_targets)
        filtered_adjs[condition] = filt
        graphs[condition] = construct_nx_graph(filt, tf_set=tf_set)
        log.info("%s: %d edges, %d TFs after filtering", condition,
                 len(filt), filt["TF"].nunique())

    # Centrality metrics per condition
    centrality = {}
    for cond, G in graphs.items():
        cent_df = compute_centrality_metrics(G)
        centrality[cond] = cent_df
        cent_df.to_csv(output_dir / f"centrality_{cond}.csv", index=False)

    # Rewiring between all condition pairs
    conditions = list(adj_paths.keys())
    rewiring = {}
    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i + 1:]:
            label = f"{cond_a}_vs_{cond_b}"
            rw = compute_rewiring_metrics(
                filtered_adjs[cond_a], filtered_adjs[cond_b],
                label_a=cond_a, label_b=cond_b,
            )
            rewiring[label] = rw
            rw.to_csv(output_dir / f"rewiring_{label}.csv", index=False)
            log.info("Rewiring %s: saved", label)

    # GO enrichment of condition-unique TF targets
    all_go = {}
    for cond, adj in filtered_adjs.items():
        others = [v for k, v in filtered_adjs.items() if k != cond]
        unique_targets = {
            tf: list(get_unique_targets(tf, adj, others))
            for tf in adj["TF"].unique()
        }
        go_df = run_go_enrichment_all_tfs(unique_targets, gmt_path, pval_thresh)
        all_go[cond] = go_df
        if not go_df.empty:
            go_df.to_csv(output_dir / f"go_enrichment_{cond}.csv", index=False)
            if plot:
                tf_order = sorted(go_df["regulon"].unique().tolist())
                plot_go_bubble(
                    go_df,
                    output_dir / f"go_bubble_{cond}.png",
                    condition=f"{cond}-unique",
                    tf_order=tf_order,
                )

    return {"centrality": centrality, "rewiring": rewiring, "go_enrichment": all_go}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differential GRN analysis: centrality, rewiring, and GO enrichment."
    )
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--ad-adj",   required=True, help="AD GRN adjacency CSV.")
    parser.add_argument("--ctrl-adj", required=True, help="CTRL GRN adjacency CSV.")
    parser.add_argument("--res-adj",  default=None, help="Resilient GRN adjacency CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--go-gmt",   required=True,
                        help="GMT gene set file or gseapy database name "
                             "(e.g., 'GO_Biological_Process_2023').")
    parser.add_argument("--tf-list",  default=None, help="TF gene list file.")
    parser.add_argument("--top-edge-quantile", type=float, default=0.75)
    parser.add_argument("--min-targets", type=int, default=10)
    parser.add_argument("--pval-threshold", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    nw_cfg = cfg.get("network_rewiring", {})

    adj_paths = {"AD": args.ad_adj, "CTRL": args.ctrl_adj}
    if args.res_adj:
        adj_paths["RES"] = args.res_adj

    run_network_rewiring(
        adj_paths=adj_paths,
        output_dir=args.output_dir,
        gmt_path=args.go_gmt,
        tf_list_path=args.tf_list,
        quantile=nw_cfg.get("top_edge_quantile", args.top_edge_quantile),
        min_targets=nw_cfg.get("min_targets_after_filter", args.min_targets),
        pval_thresh=nw_cfg.get("go_pval_threshold", args.pval_threshold),
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
