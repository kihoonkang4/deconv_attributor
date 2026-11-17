"""Unit tests for the minimal 3-step proof-of-concept pipeline."""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deconv_attributor.pipeline import run_attribution_pipeline
from deconv_attributor.reference import build_reference
from deconv_attributor.solver import attribute_changes


class TestMinimalPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.G = 30
        self.genes = pd.Index([f"Gene_{i:02d}" for i in range(self.G)])

        # 2 cell types: Genes 0..14 mark TypeA, Genes 15..29 mark TypeB
        S_base = np.full((self.G, 2), 2.0)
        S_base[:15, 0] = np.linspace(50.0, 150.0, 15)
        S_base[15:, 1] = np.linspace(50.0, 150.0, 15)

        cells = []
        labels = []
        for k, name in enumerate(["TypeA", "TypeB"]):
            block = S_base[:, [k]] + self.rng.normal(0.0, 0.2, size=(self.G, 10))
            cells.append(np.clip(block, 0.0, None))
            labels.extend([name] * 10)

        self.sc_df = pd.DataFrame(np.hstack(cells), index=self.genes, columns=[f"C{i}" for i in range(20)])
        self.cell_types = pd.Series(labels, index=self.sc_df.columns)
        self.S_T, self.w = build_reference(self.sc_df, self.cell_types, normalize=False)

    def test_pure_cell_type_shift(self) -> None:
        """100% cell-type shift with zero intrinsic gene expression change."""
        p1 = pd.Series({"TypeA": 0.7, "TypeB": 0.3})
        p2 = pd.Series({"TypeA": 0.3, "TypeB": 0.7})
        y1 = self.S_T @ p1
        y2 = self.S_T @ p2

        res = attribute_changes(self.S_T, self.w, y1, y2)
        np.testing.assert_allclose(res.delta_p.to_numpy(), (p2 - p1).to_numpy(), atol=1e-2)
        self.assertGreater(res.global_pct_cell_type, 99.0)
        self.assertLess(res.global_pct_expression, 1.0)

    def test_pure_expression_change_on_subset_of_genes(self) -> None:
        """0% cell-type shift with intrinsic expression spikes on 3 genes."""
        p = pd.Series({"TypeA": 0.5, "TypeB": 0.5})
        y1 = self.S_T @ p
        y2 = y1.copy()
        y2.iloc[[0, 1, 2]] += [300.0, -200.0, 400.0]

        res = attribute_changes(self.S_T, self.w, y1, y2)
        np.testing.assert_allclose(res.delta_p.to_numpy(), np.zeros(2), atol=1e-2)
        self.assertGreater(res.global_pct_expression, 98.0)

    def test_combined_cell_type_and_expression_change(self) -> None:
        """Combined cell-type shift + single-gene expression spike."""
        p1 = pd.Series({"TypeA": 0.6, "TypeB": 0.4})
        p2 = pd.Series({"TypeA": 0.2, "TypeB": 0.8})
        bulk = pd.DataFrame({"S1": self.S_T @ p1, "S2": self.S_T @ p2}, index=self.genes)
        bulk.loc["Gene_00", "S2"] += 500.0  # Massive intrinsic expression spike on Gene_00

        res = run_attribution_pipeline(bulk, self.sc_df, self.cell_types, "S1", "S2", normalize_linear=False)
        np.testing.assert_allclose(res.delta_p.to_numpy(), (p2 - p1).to_numpy(), atol=1e-2)
        self.assertGreater(float(res.gene_table.loc["Gene_00", "pct_expr"]), 90.0)
        self.assertGreater(float(res.gene_table.loc["Gene_05", "pct_cell_type"]), 95.0)


if __name__ == "__main__":
    unittest.main()
