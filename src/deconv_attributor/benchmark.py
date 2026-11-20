"""Pseudobulk benchmark with known ground truth for both Δp and the per-gene split of Δy.

Pseudobulk for a group of cells is  y = S_group @ p_group, where S_group holds the CP10K
cell-type means of that group's cells and p_group its cell-type fractions. With the
baseline profiles S_base, the true decomposition matches the solver's convention:

    Δy_cell_type_true = S_base @ Δp_true
    Δy_expr_true      = (S_target - S_base) @ p_target

and the two terms add up exactly to Δy = y_target - y_base.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse

from .reference import build_reference
from .solver import AttributionOutput, attribute_changes

# Genes with |Δy| below this (CP10K) are ignored when scoring per-gene dominant source.
MIN_ABS_DELTA_Y = 1.0


@dataclass
class BenchmarkResult:
    result: AttributionOutput
    true_delta_p: pd.Series
    true_gene_table: pd.DataFrame   # delta_y_total, delta_y_cell_type, delta_y_expr, pct_cell_type
    true_global_pct_cell_type: float
    metrics: dict[str, float]


def _l1_pct(a: np.ndarray, b: np.ndarray) -> float:
    total = float(np.abs(a).sum() + np.abs(b).sum())
    return 100.0 * float(np.abs(a).sum()) / total if total > 0 else 0.0


def run_pseudobulk_benchmark(
    counts: sparse.spmatrix,
    cell_types: np.ndarray,
    groups: np.ndarray,
    genes: pd.Index,
    baseline: str,
    target: str,
    seed: int = 42,
) -> BenchmarkResult:
    """Split baseline cells into a reference half and a held-out half, then attribute held-out baseline -> target."""
    cell_types = np.asarray(cell_types)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    base_idx = np.where(groups == baseline)[0]
    rng.shuffle(base_idx)
    ref_idx, test_idx = base_idx[: len(base_idx) // 2], base_idx[len(base_idx) // 2 :]
    target_idx = np.where(groups == target)[0]

    ref = build_reference(counts[:, ref_idx], cell_types[ref_idx], genes=genes)
    K_cols = ref.S_T.columns

    missing = set(np.unique(cell_types[target_idx])) - set(K_cols)
    if missing:
        raise ValueError(f"Target cell types absent from the reference half: {sorted(missing)}")

    def _profile(idx: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
        S = build_reference(counts[:, idx], cell_types[idx], genes=genes).S_T.reindex(columns=K_cols, fill_value=0.0)
        p = pd.Series(cell_types[idx]).value_counts(normalize=True).reindex(K_cols, fill_value=0.0)
        return S, p

    S_base, p_base = _profile(test_idx)
    S_tgt, p_tgt = _profile(target_idx)
    y_base, y_tgt = S_base @ p_base, S_tgt @ p_tgt

    res = attribute_changes(ref, y_base, y_tgt)

    true_dp = p_tgt - p_base
    true_ct = (S_base @ true_dp).loc[res.gene_table.index]
    true_ex = ((S_tgt - S_base) @ p_tgt).loc[res.gene_table.index]
    true_tot = true_ct + true_ex
    denom = true_ct.abs() + true_ex.abs() + 1e-12
    true_table = pd.DataFrame(
        {
            "delta_y_total": true_tot,
            "delta_y_cell_type": true_ct,
            "delta_y_expr": true_ex,
            "pct_cell_type": 100.0 * true_ct.abs() / denom,
        }
    )

    est = res.gene_table
    est_dp = res.delta_p.reindex(K_cols)
    changed = true_tot.abs() >= MIN_ABS_DELTA_Y
    est_ct_dominant = est["pct_cell_type"] > 50.0
    true_ct_dominant = true_table["pct_cell_type"] > 50.0

    metrics = {
        "delta_p_pearson_r": float(np.corrcoef(true_dp, est_dp)[0, 1]),
        "delta_p_mae": float(np.abs(true_dp - est_dp).mean()),
        "delta_p_sum": float(est_dp.sum()),
        "gene_ct_pearson_r": float(np.corrcoef(true_ct, est["delta_y_cell_type"])[0, 1]),
        "gene_ct_pearson_r_log": float(
            np.corrcoef(np.sign(true_ct) * np.log1p(true_ct.abs()),
                        np.sign(est["delta_y_cell_type"]) * np.log1p(est["delta_y_cell_type"].abs()))[0, 1]
        ),
        "pct_cell_type_mae_changed_genes": float((est["pct_cell_type"] - true_table["pct_cell_type"])[changed].abs().mean()),
        "dominant_source_accuracy": float((est_ct_dominant == true_ct_dominant)[changed].mean()),
        "n_changed_genes": int(changed.sum()),
    }

    return BenchmarkResult(
        result=res,
        true_delta_p=true_dp,
        true_gene_table=true_table,
        true_global_pct_cell_type=_l1_pct(true_ct.to_numpy(), true_ex.to_numpy()),
        metrics=metrics,
    )
