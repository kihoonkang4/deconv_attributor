from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def build_reference(
    sc_counts: pd.DataFrame | sparse.spmatrix,
    cell_types: pd.Series | np.ndarray | list[str],
    genes: pd.Index | list[str] | None = None,
    normalize: bool = True,
    scale_factor: float = 1e4,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute baseline cluster mean expression S_T (G x K) and marker specificity weights w (G,).

    Formula:
        w_g = Var_between(S_{k,g}) / (Var_within(X_{c,g}) + 1e-4)
        w   = w / median(w[w > 0])
    """
    if isinstance(sc_counts, pd.DataFrame):
        labels = cell_types.loc[sc_counts.columns].to_numpy() if isinstance(cell_types, pd.Series) else np.asarray(cell_types)
        gene_index = pd.Index(sc_counts.index)
        mat = sparse.csc_matrix(sc_counts.to_numpy(dtype=float))
    else:
        if genes is None:
            raise ValueError("`genes` index must be provided when `sc_counts` is a sparse matrix.")
        gene_index = pd.Index(genes)
        labels = np.asarray(cell_types)
        mat = sc_counts.tocsc().astype(float)

    # Normalize each cell to linear CP10K
    if normalize:
        col_sums = np.asarray(mat.sum(axis=0)).ravel()
        inv_sums = np.where(col_sums > 0, scale_factor / col_sums, 0.0)
        mat = mat @ sparse.diags(inv_sums)

    unique_types = np.unique(labels)
    K = len(unique_types)
    n_cells = mat.shape[1]
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    cols = np.array([type_to_idx[ct] for ct in labels])
    counts_per_type = np.bincount(cols, minlength=K).astype(float)
    weights = 1.0 / np.maximum(counts_per_type[cols], 1.0)

    indicator = sparse.csc_matrix((weights, (np.arange(n_cells), cols)), shape=(n_cells, K))
    S_T_arr = (mat @ indicator).toarray()  # (G x K) cluster means

    # Within-cell-type variance across cells
    mat_sq = mat.copy()
    mat_sq.data = mat_sq.data ** 2
    mean_sq = (mat_sq @ indicator).toarray()
    var_per_type = np.maximum(mean_sq - S_T_arr ** 2, 0.0)
    bessel = np.where(counts_per_type > 1, counts_per_type / np.maximum(counts_per_type - 1.0, 1.0), 1.0)
    vars_within = (var_per_type * bessel[np.newaxis, :]).mean(axis=1)

    # Between-cell-type variance across K cluster means
    var_between = np.var(S_T_arr, axis=1)

    # One-way ANOVA F-ratio marker specificity weight
    w = var_between / (vars_within + 1e-4)
    if np.any(w > 0):
        w = w / np.median(w[w > 0])

    S_T = pd.DataFrame(S_T_arr, index=gene_index, columns=unique_types)
    S_T.attrs["p_ref"] = pd.Series(counts_per_type / float(n_cells), index=unique_types)
    w_series = pd.Series(w, index=gene_index, name="w_prior")
    return S_T, w_series


def save_reference(S_T: pd.DataFrame, w: pd.Series, out_dir: str | Path) -> Path:
    """Save reference matrix S_T and marker weights w to disk."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    S_T.to_csv(dest / "S_T_matrix_GxK.csv.gz", compression="gzip")
    w.to_frame(name="w_prior").to_csv(dest / "marker_weights.csv")
    return dest


def load_reference(ref_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load pre-computed reference matrix S_T and marker weights w from disk."""
    src = Path(ref_dir)
    S_T = pd.read_csv(src / "S_T_matrix_GxK.csv.gz", index_col=0)
    w = pd.read_csv(src / "marker_weights.csv", index_col=0)["w_prior"]
    return S_T, w
