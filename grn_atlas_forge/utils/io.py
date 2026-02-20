"""I/O helpers for loading and saving analysis data."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
import yaml


def load_h5ad(path: str | Path) -> sc.AnnData:
    """Load an AnnData object from an h5ad file.

    Args:
        path: Path to the .h5ad file.

    Returns:
        AnnData object with cells × genes expression matrix.
    """
    return sc.read_h5ad(str(path))


def make_loom(h5ad_path: str | Path, loom_path: str | Path) -> None:
    """Convert an h5ad file to loom format for pySCENIC.

    pySCENIC requires loom format as input. This function writes
    a minimal loom file with gene names, cell IDs, and the count
    matrix.

    Args:
        h5ad_path: Path to the input .h5ad file.
        loom_path: Path where the .loom file will be written.
    """
    adata = load_h5ad(h5ad_path)
    row_attrs = {"Gene": np.array(adata.var.index)}
    col_attrs = {
        "CellID": np.array(adata.obs.index),
        "nGene": np.array(np.sum(adata.X.T > 0, axis=0)).flatten(),
        "nUMI": np.array(np.sum(adata.X.T, axis=0)).flatten(),
    }
    lp.create(str(loom_path), adata.X.T, row_attrs, col_attrs)


def load_aucell(path: str | Path) -> pd.DataFrame:
    """Load an AUCell output CSV (cells × regulons).

    Args:
        path: Path to the AUCell .csv file produced by pySCENIC.

    Returns:
        DataFrame with cell IDs as the index and regulon names as columns.
    """
    df = pd.read_csv(path, index_col=0)
    return df


def load_adj(path: str | Path) -> pd.DataFrame:
    """Load a GRN adjacency file (TF–target–importance table).

    Args:
        path: Path to a CSV with columns ['TF', 'target', 'importance'].

    Returns:
        DataFrame with columns ['TF', 'target', 'importance'].

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    required = {"TF", "target", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Adjacency file missing columns: {missing}")
    return df[["TF", "target", "importance"]]


def save_adj(df: pd.DataFrame, path: str | Path) -> None:
    """Save a GRN adjacency DataFrame to CSV.

    Args:
        df: DataFrame with columns ['TF', 'target', 'importance'].
        path: Output path for the CSV file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[["TF", "target", "importance"]].to_csv(path, index=False)


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Dictionary of configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
