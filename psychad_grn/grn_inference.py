"""GRN construction using pySCENIC / GRNboost2.

Pipeline overview:
  1. Convert h5ad expression data to loom format (required by pySCENIC).
  2. Run GRNboost2 via pySCENIC `grn` command with a randomized seed.
     Repeat 5× per phenotype to generate an ensemble of adjacency matrices.
  3. Retain only edges that appear in ALL 5 runs (consensus); average their
     importance scores. This ensures that reported TF–target relationships
     are reproducible and not artefacts of a single random seed.
  4. Apply cisTarget pruning (`pyscenic ctx`) using the SCENIC human motif
     database v10 to filter for TFs with enriched binding motifs in target
     gene promoters. Retain only activating regulons with NES ≥ 3 and
     ≥ 10 target genes.
  5. Cross-cohort generalization: retain only TF–target edges that appear
     in networks inferred from all three cohorts (MSSM, RADC, ROSMAP),
     then average importance scores. This ensures that reported regulatory
     relationships generalize across brain banks.

Usage:
    python -m psychad_grn.grn_inference --config configs/default_config.yaml \\
        --h5ad-dir data/h5ad/ --output-dir results/GRNs/
"""

import argparse
import logging
import subprocess
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils.io import load_adj, load_config, make_loom, save_adj

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_loom(h5ad_path: str | Path, loom_path: str | Path) -> Path:
    """Ensure a loom file exists for pySCENIC input.

    Args:
        h5ad_path: Path to the source .h5ad file.
        loom_path: Desired output .loom path.

    Returns:
        Path to the loom file (created if it did not exist).
    """
    loom_path = Path(loom_path)
    if not loom_path.exists():
        log.info("Converting %s → %s", h5ad_path, loom_path)
        make_loom(h5ad_path, loom_path)
    return loom_path


# ── pySCENIC CLI wrappers ─────────────────────────────────────────────────────

def run_grnboost2(
    loom_path: str | Path,
    tf_list: str | Path,
    output_adj: str | Path,
    seed: int,
    n_workers: int = 8,
) -> None:
    """Run GRNboost2 (pySCENIC grn) to infer a TF–gene regulatory network.

    GRNboost2 uses gradient boosting to rank the regulatory importance of
    each transcription factor on every target gene. Running with multiple
    random seeds allows us to identify reproducible edges.

    Args:
        loom_path: Input loom file (cells × genes).
        tf_list: Text file listing transcription factor gene names (one per line).
        output_adj: Output path for the adjacency CSV (TF, target, importance).
        seed: Random seed for reproducibility.
        n_workers: Number of CPU workers for parallel inference.
    """
    cmd = [
        "pyscenic", "grn",
        str(loom_path),
        str(tf_list),
        "--output", str(output_adj),
        "--seed", str(seed),
        "--num_workers", str(n_workers),
    ]
    log.info("Running GRNboost2 (seed=%d): %s", seed, " ".join(cmd))
    Path(output_adj).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def run_ctx(
    adj_path: str | Path,
    loom_path: str | Path,
    db_glob: str,
    motif_path: str | Path,
    output_reg: str | Path,
    n_workers: int = 8,
) -> None:
    """Apply cisTarget pruning to an adjacency matrix (pySCENIC ctx).

    cisTarget filters TF–target edges based on enrichment of the TF's
    binding motif in the promoters of target genes, using the SCENIC
    cisTarget human motif database. This step significantly reduces
    false-positive regulatory relationships.

    Args:
        adj_path: Adjacency CSV from GRNboost2.
        loom_path: Input loom file (same as used for GRNboost2).
        db_glob: Glob pattern matching the cisTarget feather database files.
        motif_path: Path to the motif annotations table (.tbl file).
        output_reg: Output path for the pruned regulon CSV.
        n_workers: Number of CPU workers.
    """
    import glob as glob_module
    db_files = glob_module.glob(db_glob)
    if not db_files:
        raise FileNotFoundError(f"No cisTarget databases found matching: {db_glob}")

    cmd = [
        "pyscenic", "ctx",
        str(adj_path),
        *db_files,
        "--annotations_fname", str(motif_path),
        "--expression_mtx_fname", str(loom_path),
        "--output", str(output_reg),
        "--mask_dropouts",
        "--num_workers", str(n_workers),
    ]
    log.info("Running cisTarget pruning → %s", output_reg)
    Path(output_reg).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


# ── Regulon post-processing ───────────────────────────────────────────────────

def filter_regulons(
    reg_path: str | Path,
    nes_threshold: float = 3.0,
    min_targets: int = 10,
) -> pd.DataFrame:
    """Filter cisTarget regulons and extract the TF–target adjacency list.

    Keeps only activating regulons (transcriptional activators) with
    Normalized Enrichment Score (NES) ≥ 3.0 and ≥ 10 target genes. These
    thresholds correspond to high-confidence motif enrichment and a
    biologically meaningful regulon size.

    Removes self-targeting TFs (where TF == target gene) since these are
    likely to be spurious autoregulatory artefacts in the motif databases.

    Args:
        reg_path: Path to the pySCENIC ctx output CSV.
        nes_threshold: Minimum NES for regulon inclusion (default 3.0).
        min_targets: Minimum number of target genes per regulon (default 10).

    Returns:
        Adjacency DataFrame with columns ['TF', 'target', 'importance'].
    """
    df = pd.read_csv(reg_path)

    # Keep activating regulons only (transcriptional activators)
    df = df[df["Enrichment.5"].str.contains("activating", na=False)]

    # Filter by NES threshold (motif enrichment significance)
    df = df[pd.to_numeric(df["Enrichment.1"], errors="coerce") >= nes_threshold]

    # Require minimum number of target genes
    df["n_targets"] = df["Enrichment.6"].apply(
        lambda x: len(eval(x)) if pd.notnull(x) else 0
    )
    df = df[df["n_targets"] >= min_targets].drop(columns=["n_targets"])

    # Build adjacency list
    records = []
    for _, row in df.iterrows():
        tf = row.iloc[0]
        if pd.notnull(row["Enrichment.6"]):
            for target, importance in eval(row["Enrichment.6"]):
                if tf != target:  # remove self-loops
                    records.append({"TF": tf, "target": target, "importance": importance})

    return pd.DataFrame(records).drop_duplicates()


def remove_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    """Remove edges where the TF targets itself.

    Self-loops can arise from the cisTarget motif databases and represent
    autoregulatory TFs. We exclude them as they conflate direct regulation
    with indirect feedback.

    Args:
        df: Adjacency DataFrame with columns ['TF', 'target', 'importance'].

    Returns:
        Filtered DataFrame with self-loops removed.
    """
    return df[df["TF"] != df["target"]].copy()


# ── Consensus GRN ─────────────────────────────────────────────────────────────

def create_consensus_grn(
    adj_files: list[str | Path],
    edge_threshold: Optional[int] = None,
) -> pd.DataFrame:
    """Build a consensus GRN from multiple GRNboost2 runs.

    Retains only TF–target edges that appear in at least `edge_threshold`
    of the input adjacency matrices, then averages their importance scores.
    By default, edge_threshold equals the number of input files (i.e., an
    edge must appear in every run to be included in the consensus).

    This 'intersection' strategy eliminates edges that arise from random
    seed-dependent sampling noise in the boosting algorithm.

    Args:
        adj_files: List of paths to adjacency CSVs from individual GRNboost2 runs.
        edge_threshold: Minimum number of runs in which an edge must appear.
            Defaults to len(adj_files) (require consensus across all runs).

    Returns:
        Consensus adjacency DataFrame with columns ['TF', 'target', 'importance'].
    """
    adj_list = [load_adj(f) for f in adj_files]
    if not adj_list:
        raise ValueError("No valid adjacency files provided.")

    threshold = edge_threshold if edge_threshold is not None else len(adj_list)
    log.info("Building consensus from %d runs (threshold=%d)", len(adj_list), threshold)

    # Rename importance column per run before merging
    for i, df in enumerate(adj_list):
        adj_list[i] = df.rename(columns={"importance": f"importance_{i}"})

    merged = reduce(
        lambda a, b: pd.merge(a, b, on=["TF", "target"], how="outer"),
        adj_list,
    )

    imp_cols = [c for c in merged.columns if c.startswith("importance_")]
    merged["edge_count"] = merged[imp_cols].notna().sum(axis=1)
    merged = merged[merged["edge_count"] >= threshold]
    merged[imp_cols] = merged[imp_cols].fillna(0)
    merged["importance"] = merged[imp_cols].sum(axis=1) / merged["edge_count"]

    return merged[["TF", "target", "importance"]].reset_index(drop=True)


def cross_cohort_consensus(
    cohort_adjs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Retain only TF–target edges shared across all cohorts.

    Cross-cohort generalization ensures that regulatory relationships are not
    specific to one brain bank's sample composition, processing protocol, or
    demographic bias. Only edges present in MSSM, RADC, and ROSMAP networks
    are retained; their importance scores are averaged.

    Args:
        cohort_adjs: Dictionary mapping cohort name → adjacency DataFrame
            (columns ['TF', 'target', 'importance']).

    Returns:
        Adjacency DataFrame with edges common to all cohorts, with averaged
        importance.
    """
    if len(cohort_adjs) < 2:
        raise ValueError("At least two cohorts are required for cross-cohort consensus.")

    dfs = list(cohort_adjs.values())
    for cohort, df in zip(cohort_adjs.keys(), dfs):
        df.rename(columns={"importance": f"importance_{cohort}"}, inplace=True)

    merged = reduce(
        lambda a, b: pd.merge(a, b, on=["TF", "target"], how="inner"),
        dfs,
    )

    imp_cols = [c for c in merged.columns if c.startswith("importance_")]
    merged["importance"] = merged[imp_cols].mean(axis=1)
    log.info(
        "Cross-cohort consensus: %d edges across %d cohorts",
        len(merged),
        len(cohort_adjs),
    )
    return merged[["TF", "target", "importance"]].reset_index(drop=True)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    h5ad_files: dict[str, str | Path],
    tf_list: str | Path,
    db_glob: str,
    motif_path: str | Path,
    output_dir: str | Path,
    n_runs: int = 5,
    seed_start: int = 42,
    n_workers: int = 8,
    nes_threshold: float = 3.0,
    min_targets: int = 10,
) -> dict[str, pd.DataFrame]:
    """Run the full GRN inference pipeline for all phenotypes.

    For each phenotype (e.g., AD, CTRL, RES):
      1. Convert h5ad to loom.
      2. Run GRNboost2 n_runs times with different seeds.
      3. Build consensus adjacency (edges in all n_runs runs).
      4. Apply cisTarget pruning.
      5. Filter regulons (activating, NES ≥ threshold, ≥ min_targets genes).

    Cross-cohort consensus (Step 6) must be run separately once networks are
    available for all three cohorts, using cross_cohort_consensus().

    Args:
        h5ad_files: Dict mapping phenotype label → h5ad path
            (e.g., {'AD': 'data/AD.h5ad', 'CTRL': 'data/CTRL.h5ad'}).
        tf_list: Path to transcription factor list file.
        db_glob: Glob pattern for cisTarget feather database files.
        motif_path: Path to motif annotations table.
        output_dir: Root output directory. Per-phenotype results go into
            output_dir/{phenotype}/.
        n_runs: Number of GRNboost2 runs per phenotype.
        seed_start: Base random seed; actual seeds are seed_start .. seed_start+n_runs-1.
        n_workers: CPU workers for pySCENIC.
        nes_threshold: Minimum NES for cisTarget filtering.
        min_targets: Minimum target genes per regulon.

    Returns:
        Dict mapping phenotype → final consensus adjacency DataFrame.
    """
    output_dir = Path(output_dir)
    results = {}

    for phenotype, h5ad_path in h5ad_files.items():
        log.info("=== Processing phenotype: %s ===", phenotype)
        pheno_dir = output_dir / phenotype
        loom_path = pheno_dir / f"{phenotype}.loom"
        prepare_loom(h5ad_path, loom_path)

        # Run GRNboost2 with multiple seeds
        adj_files = []
        for i in range(n_runs):
            seed = seed_start + i
            adj_path = pheno_dir / "adj" / f"adj_seed{seed}.csv"
            if not adj_path.exists():
                run_grnboost2(loom_path, tf_list, adj_path, seed=seed, n_workers=n_workers)
            adj_files.append(adj_path)

        # Build consensus across runs
        consensus_adj = create_consensus_grn(adj_files, edge_threshold=n_runs)
        consensus_path = pheno_dir / "adj" / "adj_consensus.csv"
        save_adj(consensus_adj, consensus_path)

        # Apply cisTarget pruning
        reg_path = pheno_dir / "regulons" / f"{phenotype}_regulons.csv"
        run_ctx(
            consensus_path, loom_path, db_glob, motif_path, reg_path,
            n_workers=n_workers,
        )

        # Filter regulons → final adjacency
        final_adj = filter_regulons(reg_path, nes_threshold=nes_threshold,
                                    min_targets=min_targets)
        final_path = pheno_dir / f"{phenotype}_GRN.csv"
        save_adj(final_adj, final_path)
        log.info(
            "%s: %d edges across %d TFs",
            phenotype, len(final_adj), final_adj["TF"].nunique(),
        )
        results[phenotype] = final_adj

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer consensus GRNs using pySCENIC/GRNboost2."
    )
    parser.add_argument("--config", help="Path to YAML config file.")
    parser.add_argument("--h5ad-dir", help="Directory containing per-phenotype h5ad files.")
    parser.add_argument("--output-dir", required=True, help="Root output directory.")
    parser.add_argument("--tf-list", required=True, help="File listing transcription factors.")
    parser.add_argument("--db-glob", required=True, help="Glob for cisTarget .feather databases.")
    parser.add_argument("--motif-file", required=True, help="cisTarget motif annotations file.")
    parser.add_argument("--phenotypes", nargs="+", default=["AD", "CTRL", "RES"])
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--nes-threshold", type=float, default=3.0)
    parser.add_argument("--min-targets", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    grn_cfg = cfg.get("grn_inference", {})
    paths_cfg = cfg.get("paths", {})

    h5ad_dir = Path(args.h5ad_dir or paths_cfg.get("data_dir", "data/h5ad"))
    phenotypes = grn_cfg.get("phenotypes", args.phenotypes)
    h5ad_files = {p: h5ad_dir / f"{p}.h5ad" for p in phenotypes}

    run_full_pipeline(
        h5ad_files=h5ad_files,
        tf_list=args.tf_list or paths_cfg.get("tf_list"),
        db_glob=args.db_glob or paths_cfg.get("scenic_db_glob"),
        motif_path=args.motif_file or paths_cfg.get("motif_file"),
        output_dir=args.output_dir,
        n_runs=grn_cfg.get("n_runs", args.n_runs),
        seed_start=grn_cfg.get("seed_start", args.seed_start),
        n_workers=args.n_workers,
        nes_threshold=grn_cfg.get("nes_threshold", args.nes_threshold),
        min_targets=grn_cfg.get("min_targets", args.min_targets),
    )


if __name__ == "__main__":
    main()
