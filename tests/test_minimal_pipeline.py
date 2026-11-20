"""Unit tests for the minimal 3-step proof-of-concept pipeline."""

from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deconv_attributor.benchmark import run_pseudobulk_benchmark
from deconv_attributor.reference import P_REF_FILE, build_reference, load_reference, save_reference
from deconv_attributor.solver import attribute_changes, to_cp10k


def _marker_profiles(G: int = 30) -> np.ndarray:
    """2 cell types: genes 0..14 mark TypeA, genes 15..29 mark TypeB."""
    S = np.full((G, 2), 2.0)
    S[: G // 2, 0] = np.linspace(50.0, 150.0, G // 2)
    S[G // 2 :, 1] = np.linspace(50.0, 150.0, G // 2)
    return S


class TestMinimalPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.G = 30
        self.genes = pd.Index([f"Gene_{i:02d}" for i in range(self.G)])
        S_base = _marker_profiles(self.G)

        cells = []
        labels = []
        for k, name in enumerate(["TypeA", "TypeB"]):
            block = S_base[:, [k]] + self.rng.normal(0.0, 0.2, size=(self.G, 10))
            cells.append(np.clip(block, 0.0, None))
            labels.extend([name] * 10)

        self.sc_df = pd.DataFrame(np.hstack(cells), index=self.genes, columns=[f"C{i}" for i in range(20)])
        self.cell_types = pd.Series(labels, index=self.sc_df.columns)
        self.ref = build_reference(self.sc_df, self.cell_types, normalize=False)
        self.S_T = self.ref.S_T

    def test_pure_cell_type_shift(self) -> None:
        """100% cell-type shift with zero intrinsic gene expression change."""
        p1 = pd.Series({"TypeA": 0.7, "TypeB": 0.3})
        p2 = pd.Series({"TypeA": 0.3, "TypeB": 0.7})
        res = attribute_changes(self.ref, self.S_T @ p1, self.S_T @ p2)
        np.testing.assert_allclose(res.delta_p.to_numpy(), (p2 - p1).to_numpy(), atol=1e-2)
        self.assertGreater(res.global_pct_cell_type, 99.0)
        self.assertLess(res.global_pct_expression, 1.0)
        self.assertFalse(res.platform_calibrated)

    def test_pure_expression_change_on_subset_of_genes(self) -> None:
        """0% cell-type shift with intrinsic expression spikes on 3 genes."""
        p = pd.Series({"TypeA": 0.5, "TypeB": 0.5})
        y1 = self.S_T @ p
        y2 = y1.copy()
        y2.iloc[[0, 1, 2]] += [300.0, -200.0, 400.0]

        res = attribute_changes(self.ref, y1, y2)
        np.testing.assert_allclose(res.delta_p.to_numpy(), np.zeros(2), atol=1e-2)
        self.assertGreater(res.global_pct_expression, 98.0)

    def test_combined_cell_type_and_expression_change(self) -> None:
        """Combined cell-type shift + single-gene expression spike."""
        p1 = pd.Series({"TypeA": 0.6, "TypeB": 0.4})
        p2 = pd.Series({"TypeA": 0.2, "TypeB": 0.8})
        y1, y2 = self.S_T @ p1, self.S_T @ p2
        y2.loc["Gene_00"] += 500.0  # Massive intrinsic expression spike on Gene_00

        res = attribute_changes(self.ref, y1, y2)
        np.testing.assert_allclose(res.delta_p.to_numpy(), (p2 - p1).to_numpy(), atol=1e-2)
        self.assertGreater(float(res.gene_table.loc["Gene_00", "pct_expr"]), 90.0)
        self.assertGreater(float(res.gene_table.loc["Gene_05", "pct_cell_type"]), 95.0)

    def test_to_cp10k(self) -> None:
        y = to_cp10k(pd.Series([1.0, 3.0]))
        self.assertAlmostEqual(float(y.sum()), 1e4)
        self.assertEqual(float(to_cp10k(pd.Series([0.0, 0.0])).sum()), 0.0)


class TestReferenceRoundTrip(unittest.TestCase):
    """Regression for p_ref being dropped when a reference was saved and reloaded (formerly stored in DataFrame.attrs)."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        G = 30
        genes = pd.Index([f"Gene_{i:02d}" for i in range(G)])
        S = _marker_profiles(G)
        # Unbalanced reference: 15 TypeA cells, 5 TypeB cells -> p_ref = (0.75, 0.25)
        cells = [np.clip(S[:, [0]] + rng.normal(0, 0.2, (G, 15)), 0, None),
                 np.clip(S[:, [1]] + rng.normal(0, 0.2, (G, 5)), 0, None)]
        sc = pd.DataFrame(np.hstack(cells), index=genes, columns=[f"C{i}" for i in range(20)])
        labels = pd.Series(["TypeA"] * 15 + ["TypeB"] * 5, index=sc.columns)
        self.ref = build_reference(sc, labels, normalize=False)

        # Bulk on a "different platform": strong per-gene capture bias forces calibration.
        capture = np.exp(rng.normal(0, 1.0, G))
        p1 = pd.Series({"TypeA": 0.4, "TypeB": 0.6})
        p2 = pd.Series({"TypeA": 0.2, "TypeB": 0.8})
        self.y1 = (self.ref.S_T @ p1) * capture
        self.y2 = (self.ref.S_T @ p2) * capture

    def test_p_ref_survives_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_reference(self.ref, tmp)
            loaded = load_reference(tmp)
        pd.testing.assert_series_equal(loaded.p_ref, self.ref.p_ref, check_names=False)
        np.testing.assert_allclose(loaded.p_ref.to_numpy(), [0.75, 0.25])

    def test_in_memory_and_reloaded_reference_give_identical_results(self) -> None:
        res_mem = attribute_changes(self.ref, self.y1, self.y2)
        self.assertTrue(res_mem.platform_calibrated)
        with tempfile.TemporaryDirectory() as tmp:
            save_reference(self.ref, tmp)
            res_disk = attribute_changes(load_reference(tmp), self.y1, self.y2)
        np.testing.assert_allclose(res_disk.delta_p.to_numpy(), res_mem.delta_p.to_numpy(), atol=1e-8)
        np.testing.assert_allclose(
            res_disk.gene_table.to_numpy(), res_mem.gene_table.to_numpy(), rtol=1e-6, atol=1e-6
        )

    def test_legacy_reference_without_p_ref_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_reference(self.ref, tmp)
            (Path(tmp) / P_REF_FILE).unlink()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_reference(tmp)
        self.assertIsNone(loaded.p_ref)
        self.assertTrue(any(P_REF_FILE in str(w.message) for w in caught))


class TestPseudobulkBenchmark(unittest.TestCase):
    def test_truth_decomposition_is_exact_and_scored(self) -> None:
        rng = np.random.default_rng(1)
        G = 40
        genes = pd.Index([f"Gene_{i:02d}" for i in range(G)])
        S = np.full((G, 2), 2.0)
        S[:20, 0] = np.linspace(50, 150, 20)
        S[20:, 1] = np.linspace(50, 150, 20)
        S_tgt = S.copy()
        S_tgt[0, 0] *= 4.0  # intrinsic change in one TypeA gene at the target

        def cells(profile: np.ndarray, n: int) -> np.ndarray:
            return rng.poisson(np.repeat(profile[:, None], n, axis=1) * 10)

        # baseline 50/50 (split in half by the benchmark), target 20/80
        blocks = [cells(S[:, 0], 200), cells(S[:, 1], 200), cells(S_tgt[:, 0], 40), cells(S_tgt[:, 1], 160)]
        counts = sparse.csr_matrix(np.hstack(blocks).astype(float))
        cell_types = np.array(["A"] * 200 + ["B"] * 200 + ["A"] * 40 + ["B"] * 160)
        groups = np.array(["base"] * 400 + ["tgt"] * 200)

        bench = run_pseudobulk_benchmark(counts, cell_types, groups, genes, "base", "tgt", seed=0)
        truth = bench.true_gene_table
        np.testing.assert_allclose(
            truth["delta_y_cell_type"] + truth["delta_y_expr"], truth["delta_y_total"], atol=1e-9
        )
        np.testing.assert_allclose(
            bench.result.gene_table["delta_y_total"], truth["delta_y_total"], atol=1e-9
        )
        self.assertAlmostEqual(float(bench.true_delta_p.sum()), 0.0, places=12)
        self.assertGreater(bench.metrics["delta_p_pearson_r"], 0.99)
        self.assertGreater(bench.metrics["dominant_source_accuracy"], 0.9)
        self.assertGreater(float(truth.loc["Gene_00", "delta_y_expr"]), 0.0)
        self.assertGreater(float(bench.result.gene_table.loc["Gene_00", "pct_expr"]), 50.0)


if __name__ == "__main__":
    unittest.main()
