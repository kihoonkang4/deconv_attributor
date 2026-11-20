from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Reference:
    """Baseline single-cell reference.

    S_T:   (G x K) cell-type mean expression in linear CP10K.
    w:     (G,) marker specificity weights.
    p_ref: (K,) cell-type fractions in the reference cells, or None if unknown.
    """

    S_T: pd.DataFrame
    w: pd.Series
    p_ref: pd.Series | None = None


def build_reference(
    sc_counts: pd.DataFrame | sparse.spmatrix,
    cell_types: pd.Series | np.ndarray | list[str],
    genes: pd.Index | list[str] | None = None,
    normalize: bool = True,
    scale_factor: float = 1e4,
) -> Reference:
    """Compute baseline cluster mean expression S_T (G x K), marker weights w (G,) and cell-type fractions p_ref (K,).

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

    return Reference(
        S_T=pd.DataFrame(S_T_arr, index=gene_index, columns=unique_types),
        w=pd.Series(w, index=gene_index, name="w_prior"),
        p_ref=pd.Series(counts_per_type / float(n_cells), index=unique_types, name="p_ref"),
    )


S_T_FILE = "S_T_matrix_GxK.csv.gz"
WEIGHTS_FILE = "marker_weights.csv"
P_REF_FILE = "cell_type_fractions.csv"


def save_reference(ref: Reference, out_dir: str | Path) -> Path:
    """Save reference matrix S_T, marker weights w and cell-type fractions p_ref to disk."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ref.S_T.to_csv(dest / S_T_FILE, compression="gzip")
    ref.w.to_frame(name="w_prior").to_csv(dest / WEIGHTS_FILE)
    if ref.p_ref is not None:
        ref.p_ref.to_frame(name="p_ref").to_csv(dest / P_REF_FILE)
    return dest


def load_reference(ref_dir: str | Path) -> Reference:
    """Load a reference written by `save_reference`."""
    src = Path(ref_dir)
    S_T = pd.read_csv(src / S_T_FILE, index_col=0)
    w = pd.read_csv(src / WEIGHTS_FILE, index_col=0)["w_prior"]
    p_ref = None
    if (src / P_REF_FILE).exists():
        p_ref = pd.read_csv(src / P_REF_FILE, index_col=0)["p_ref"].reindex(S_T.columns)
    else:
        warnings.warn(
            f"{src / P_REF_FILE} not found (reference built by an older version); "
            "platform calibration will fall back to proportions estimated from the baseline bulk sample. "
            "Re-run `build-ref` to restore identical behaviour to the in-memory pipeline.",
            stacklevel=2,
        )
    return Reference(S_T=S_T, w=w, p_ref=p_ref)
