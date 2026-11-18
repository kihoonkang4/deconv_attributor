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


def _infer_sep(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return "\t"
    return ","


def load_bulk_counts(path: str | Path) -> pd.DataFrame:
    """Load a bulk RNA-seq expression matrix (genes x samples) from CSV or TSV/TXT."""
    sep = _infer_sep(path)
    df = pd.read_csv(path, sep=sep, index_col=0)
    # Strip standard FeatureCounts annotation columns if present
    drop_cols = [c for c in ("Chr", "Start", "End", "Strand", "Length", "Sum_ambig", "Ambigs") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if df.index.has_duplicates:
        df = df.groupby(level=0).sum()
    return df


def load_scrna_reference(
    counts_path: str | Path,
    metadata_path: str | Path,
    cell_type_col: str = "cell_type",
    sample_filter: str | None = None,
    sample_col: str = "sample",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load scRNA-seq counts (genes x cells) and cell-type annotations from CSV or TSV/TXT."""
    meta_sep = _infer_sep(metadata_path)
    meta = pd.read_csv(metadata_path, sep=meta_sep, index_col=0)

    if sample_filter is not None and sample_col in meta.columns:
        meta = meta[meta[sample_col] == sample_filter]

    if cell_type_col not in meta.columns:
        # Fallback if 'cell_type' was requested but 'maintype' or 'celltype' exists
        for alt in ("cell_type", "maintype", "celltype", "metacelltype"):
            if alt in meta.columns:
                cell_type_col = alt
                break
        else:
            raise ValueError(f"Metadata must contain '{cell_type_col}' column (found: {list(meta.columns)}).")

    meta = meta[meta[cell_type_col].notna() & (meta[cell_type_col] != "unassigned")].copy()

    counts_sep = _infer_sep(counts_path)
    # Read header of counts file first so we only load matched cell columns
    header_df = pd.read_csv(counts_path, sep=counts_sep, nrows=0)
    all_cols = list(header_df.columns)
    gene_col = all_cols[0]
    col_set = set(all_cols[1:])

    # Match either exact index or barcode suffix (e.g. 'x2_35_0_bc0013' -> 'bc0013')
    exact_matches = [c for c in meta.index.astype(str) if c in col_set]
    if exact_matches:
        matched_cols = exact_matches
        cell_types = meta.loc[matched_cols, cell_type_col]
    else:
        suffixes = meta.index.astype(str).str.split("_").str[-1]
        meta_by_suffix = pd.Series(meta[cell_type_col].to_numpy(), index=suffixes)
        meta_by_suffix = meta_by_suffix[~meta_by_suffix.index.duplicated(keep="first")]
        matched_cols = [c for c in meta_by_suffix.index if c in col_set]
        if not matched_cols:
            raise ValueError("No overlapping cell barcodes found between counts and metadata.")
        cell_types = meta_by_suffix.loc[matched_cols]

    counts = pd.read_csv(counts_path, sep=counts_sep, index_col=0, usecols=[gene_col, *matched_cols])
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()

    return counts[matched_cols], cell_types


def extract_bulk_from_counts(
    counts_a_path: str | Path,
    counts_b_path: str | Path,
    sample_a_name: str = "Bulk_A",
    sample_b_name: str = "Bulk_B",
) -> pd.DataFrame:
    """Extract or sum bulk count columns from two count tables (e.g., B1_1_0h_counts.txt & B1_3_4h_counts.txt)."""
    series_dict = {}
    for label, path in [(sample_a_name, counts_a_path), (sample_b_name, counts_b_path)]:
        sep = _infer_sep(path)
        header = list(pd.read_csv(path, sep=sep, nrows=0).columns)
        if len(header) > 1 and header[1] == "Sum_counts":
            df = pd.read_csv(path, sep=sep, index_col=0, usecols=[header[0], "Sum_counts"])
            s = df["Sum_counts"].astype(float)
        else:
            df = pd.read_csv(path, sep=sep, index_col=0)
            num_cols = [c for c in df.columns if c not in ("Sum_ambig", "Ambigs", "Chr", "Start", "End", "Strand", "Length")]
            s = df[num_cols].sum(axis=1).astype(float)
        if s.index.has_duplicates:
            s = s.groupby(level=0).sum()
        series_dict[label] = s

    return pd.DataFrame(series_dict).fillna(0.0)


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
