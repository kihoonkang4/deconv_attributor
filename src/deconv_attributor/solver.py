"""Steps 2 & 3: Robustly estimate cell-type shift Δp, attribute Δy, and compute simple p-values."""

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor


@dataclass
class AttributionOutput:
    """Minimal proof-of-concept output with cell-type and gene-level p-values."""

    delta_p: pd.Series             # (K,) Estimated cell-type proportion shift
    cell_type_table: pd.DataFrame  # (K, 3) delta_p, std_err, p_value per cell type
    gene_table: pd.DataFrame       # (G, 7) Per-gene decomposition, % attribution, z_expr, pval_expr
    global_pct_cell_type: float    # Global % of change explained by cell-type shift
    global_pct_expression: float   # Global % of change explained by intrinsic gene expression


def attribute_changes(
    S_T: pd.DataFrame,
    w: pd.Series,
    y1: pd.Series,
    y2: pd.Series,
    calibrate_platform: bool = True,
) -> AttributionOutput:
    """
    Minimal 3-step attribution + textbook statistics between bulk sample y1 and y2:
      1. Given reference S_T (genes x cell_types) and marker specificity weights w,
         optionally calibrate gene capture efficiency from baseline bulk y1.
      2. Fit Δy = y2 - y1 ≈ S_T @ Δp using HuberRegressor weighted by w, and compute
         standard errors & p-values for Δp from the inlier marker genes.
      3. Subtract to split every gene's change into Δy_cell_type and Δy_expr, and compute
         robust MAD Z-scores & p-values for each gene's intrinsic expression shift.
    """
    shared_genes = S_T.index.intersection(y1.index).intersection(y2.index)
    if shared_genes.empty:
        raise ValueError("No overlapping genes between reference S_T and bulk samples.")

    S_mat = S_T.loc[shared_genes].to_numpy(dtype=float)
    w_vec = w.loc[shared_genes].to_numpy(dtype=float)
    y1_vec = y1.loc[shared_genes].to_numpy(dtype=float)
    y2_vec = y2.loc[shared_genes].to_numpy(dtype=float)
    delta_y = y2_vec - y1_vec

    # Normalize each gene row by its reference norm so super-abundant genes (e.g. Sftpc)
    # do not overpower other high-specificity markers for the same cell type.
    row_scale = np.linalg.norm(S_mat, axis=1) + 1.0

    # Automatic baseline platform calibration when cross-platform capture bias is detected
    if calibrate_platform:
        p1_weights = np.sqrt(np.clip(w_vec, 0.0, None)) / row_scale
        p1_est, _ = nnls(S_mat * p1_weights[:, np.newaxis], np.clip(y1_vec, 0.0, None) * p1_weights)
        p1_sum = float(p1_est.sum())
        p1_norm = p1_est / p1_sum if p1_sum > 0 else np.full(S_mat.shape[1], 1.0 / S_mat.shape[1])
        y1_pos = np.clip(y1_vec, 0.0, None)
        r_platform = (
            float(np.corrcoef(np.log1p(y1_pos), np.log1p(S_mat @ p1_norm))[0, 1])
            if np.std(y1_pos) > 0
            else 1.0
        )
        if r_platform < 0.90:
            p_base = (
                S_T.attrs["p_ref"].reindex(S_T.columns).to_numpy(dtype=float)
                if "p_ref" in S_T.attrs
                else p1_norm
            )
            y1_expected_sc = S_mat @ p_base
            gamma = (y1_pos + 0.1) / (y1_expected_sc + 0.1)
            S_mat = S_mat * gamma[:, np.newaxis]
            row_scale = np.linalg.norm(S_mat, axis=1) + 1.0

    # Step 2: Robustly estimate cell-type shift Δp from marker consensus
    X_scaled = S_mat / row_scale[:, np.newaxis]
    y_scaled = delta_y / row_scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        huber = HuberRegressor(fit_intercept=False, alpha=1e-2, max_iter=1000).fit(
            X_scaled,
            y_scaled,
            sample_weight=w_vec,
        )
    delta_p_arr = huber.coef_
    delta_p = pd.Series(delta_p_arr, index=S_T.columns, name="delta_p")

    # Step 3: Subtract to split Δy into Cell-Type change vs. Intrinsic Expression change
    dy_cell_type = S_mat @ delta_p_arr
    dy_expr = delta_y - dy_cell_type

    abs_ct = np.abs(dy_cell_type)
    abs_ex = np.abs(dy_expr)
    denom = abs_ct + abs_ex + 1e-12

    pct_cell_type = 100.0 * abs_ct / denom
    pct_expr = 100.0 * abs_ex / denom

    # Simple Statistics A: Cell-type shift standard errors & t-test p-values from Huber inliers
    inliers = ~huber.outliers_
    if inliers.sum() <= S_mat.shape[1]:
        inliers = np.ones(len(delta_y), dtype=bool)
    sqrt_w = np.sqrt(np.clip(w_vec[inliers], 0.0, None))
    X_in = X_scaled[inliers] * sqrt_w[:, np.newaxis]
    r_in = (dy_expr[inliers] / row_scale[inliers]) * sqrt_w
    df_resid = max(int(inliers.sum()) - S_mat.shape[1], 1)
    sigma2 = float(np.sum(r_in ** 2) / df_resid)
    cov_dp = sigma2 * np.linalg.pinv(X_in.T @ X_in)
    se_dp = np.sqrt(np.maximum(np.diag(cov_dp), 1e-16))
    pval_dp = 2.0 * stats.t.sf(np.abs(delta_p_arr / se_dp), df=df_resid)

    cell_type_table = pd.DataFrame(
        {
            "delta_p": delta_p_arr,
            "std_err": se_dp,
            "p_value": pval_dp,
        },
        index=S_T.columns,
    )

    # Simple Statistics B: Gene-level Z-scores & p-values from robust MAD of scaled residuals
    r_scaled = dy_expr / row_scale
    abs_dev = np.abs(r_scaled - np.median(r_scaled))
    tau = max(1.4826 * float(np.median(abs_dev)), 1e-4)
    z_expr = r_scaled / tau
    pval_expr = 2.0 * stats.norm.sf(np.abs(z_expr))

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
            "z_expr": z_expr,
            "pval_expr": pval_expr,
        },
        index=shared_genes,
    )

    return AttributionOutput(
        delta_p=delta_p,
        cell_type_table=cell_type_table,
        gene_table=gene_table,
        global_pct_cell_type=global_pct_ct,
        global_pct_expression=global_pct_ex,
    )
