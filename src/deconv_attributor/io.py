"""Data loading utilities for bulk RNA-seq and scRNA-seq references."""

from dataclasses import dataclass
import gzip
from pathlib import Path
import pandas as pd
from scipy import io as spio
from scipy import sparse


@dataclass
class SparseSingleCellData:
    """Container for sparse single-cell count matrix (genes x cells) and metadata."""

    counts: sparse.csr_matrix
    genes: pd.Index
    barcodes: pd.Index
    metadata: pd.DataFrame


def load_bulk_counts(path: str | Path) -> pd.DataFrame:
    """Load a bulk RNA-seq expression matrix (genes x samples)."""
    return pd.read_csv(path, index_col=0)


def load_scrna_reference(
    counts_path: str | Path, metadata_path: str | Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dense scRNA-seq counts (genes x cells) and cell-type annotations."""
    counts = pd.read_csv(counts_path, index_col=0)
    meta = pd.read_csv(metadata_path, index_col=0)
    if "cell_type" not in meta.columns:
        raise ValueError("Metadata must contain a 'cell_type' column.")
    return counts, meta["cell_type"]


def load_mtx_dataset(
    mtx_path: str | Path,
    genes_path: str | Path,
    barcodes_path: str | Path,
    cellinfo_path: str | Path,
) -> SparseSingleCellData:
    """Load a 10x/GEO MatrixMarket dataset (.mtx.gz + genes + barcodes + metadata)."""
    with gzip.open(genes_path, "rt") as f:
        genes = pd.Index([line.strip().split("\t")[0] for line in f if line.strip()])

    with gzip.open(barcodes_path, "rt") as f:
        barcodes = pd.Index([line.strip() for line in f if line.strip()])

    metadata = pd.read_csv(cellinfo_path, index_col=0)
    counts = spio.mmread(mtx_path).tocsr()

    if counts.shape != (len(genes), len(barcodes)):
        raise ValueError(
            f"Matrix shape {counts.shape} does not match ({len(genes)} genes, {len(barcodes)} barcodes)."
        )

    # Deduplicate gene symbols if needed by summing rows or keeping first
    if genes.has_duplicates:
        keep_mask = ~genes.duplicated(keep="first")
        counts = counts[keep_mask, :]
        genes = genes[keep_mask]

    metadata = metadata.reindex(barcodes)
    return SparseSingleCellData(
        counts=counts,
        genes=genes,
        barcodes=barcodes,
        metadata=metadata,
    )
