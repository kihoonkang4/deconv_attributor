"""End-to-end minimal pipeline connecting Step 1 (build_reference) and Steps 2-3 (attribute_changes)."""

import pandas as pd
from scipy import sparse

from .reference import build_reference
from .solver import AttributionOutput, attribute_changes


def run_attribution_pipeline(
    bulk: pd.DataFrame,
    sc_counts: pd.DataFrame | sparse.spmatrix,
    cell_types: pd.Series,
    sample_a: str,
    sample_b: str,
    genes: pd.Index | list[str] | None = None,
    normalize_linear: bool = True,
    scale_factor: float = 1e4,
) -> AttributionOutput:
    """Build reference S_T & weights w from scRNA-seq and attribute bulk changes between sample_a and sample_b."""
    S_T, w = build_reference(
        sc_counts=sc_counts,
        cell_types=cell_types,
        genes=genes,
        normalize=normalize_linear,
        scale_factor=scale_factor,
    )

    y1 = bulk[sample_a].astype(float)
    y2 = bulk[sample_b].astype(float)
    if normalize_linear:
        if y1.sum() > 0:
            y1 = y1 * (scale_factor / float(y1.sum()))
        if y2.sum() > 0:
            y2 = y2 * (scale_factor / float(y2.sum()))

    return attribute_changes(S_T=S_T, w=w, y1=y1, y2=y2)
