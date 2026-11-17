"""Steps 2 & 3: Robustly estimate cell-type shift Δp and attribute Δy into cell-type vs. expression change."""

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor


@dataclass
class AttributionOutput:
    """Minimal proof-of-concept output separating cell-type shift from gene expression change."""

    delta_p: pd.Series             # (K,) Estimated cell-type proportion shift
    gene_table: pd.DataFrame       # (G, 5) Per-gene decomposition and % attribution
    global_pct_cell_type: float    # Global % of change explained by cell-type shift
    global_pct_expression: float   # Global % of change explained by intrinsic gene expression


def attribute_changes(
    S_T: pd.DataFrame,
    w: pd.Series,
    y1: pd.Series,
    y2: pd.Series,
) -> AttributionOutput:
    """
    Minimal 3-step attribution between bulk sample y1 and bulk sample y2:
      1. Given reference S_T (genes x cell_types) and marker specificity weights w,
      2. Fit Δy = y2 - y1 ≈ S_T @ Δp using HuberRegressor weighted by w so stable
         marker genes lock in Δp while DE outlier genes are ignored.
      3. Subtract to split every gene's change into:
           Δy_cell_type = S_T @ Δp
           Δy_expr      = Δy - Δy_cell_type
    """
    shared_genes = S_T.index.intersection(y1.index).intersection(y2.index)
    if shared_genes.empty:
        raise ValueError("No overlapping genes between reference S_T and bulk samples.")

    S_mat = S_T.loc[shared_genes].to_numpy(dtype=float)
    w_vec = w.loc[shared_genes].to_numpy(dtype=float)
    delta_y = (y2.loc[shared_genes].astype(float) - y1.loc[shared_genes].astype(float)).to_numpy()

    # Step 2: Robustly estimate cell-type shift Δp from marker consensus
    # Normalize each gene row by its reference norm so super-abundant genes (e.g. Sftpc)
    # do not overpower other high-specificity markers for the same cell type.
    row_scale = np.linalg.norm(S_mat, axis=1) + 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        huber = HuberRegressor(fit_intercept=False, alpha=1e-2, max_iter=1000).fit(
            S_mat / row_scale[:, np.newaxis],
            delta_y / row_scale,
            sample_weight=w_vec,
        )
    delta_p = pd.Series(huber.coef_, index=S_T.columns, name="delta_p")

    # Step 3: Subtract to split Δy into Cell-Type change vs. Intrinsic Expression change
    dy_cell_type = S_mat @ huber.coef_
    dy_expr = delta_y - dy_cell_type

    abs_ct = np.abs(dy_cell_type)
    abs_ex = np.abs(dy_expr)
    denom = abs_ct + abs_ex + 1e-12

    pct_cell_type = 100.0 * abs_ct / denom
    pct_expr = 100.0 * abs_ex / denom

    total_l1 = float((abs_ct + abs_ex).sum())
    if total_l1 > 0:
        global_pct_ct = float(100.0 * abs_ct.sum() / total_l1)
        global_pct_ex = float(100.0 * abs_ex.sum() / total_l1)
    else:
        global_pct_ct, global_pct_ex = 0.0, 0.0

    gene_table = pd.DataFrame(
        {
            "delta_y_total": delta_y,
            "delta_y_cell_type": dy_cell_type,
            "delta_y_expr": dy_expr,
            "pct_cell_type": pct_cell_type,
            "pct_expr": pct_expr,
        },
        index=shared_genes,
    )

    return AttributionOutput(
        delta_p=delta_p,
        gene_table=gene_table,
        global_pct_cell_type=global_pct_ct,
        global_pct_expression=global_pct_ex,
    )
