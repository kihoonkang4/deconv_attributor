"""Minimal CLI for deconv-attributor (build-ref, make-bulk, solve, benchmark)."""

import argparse
from pathlib import Path
import urllib.request

from .io import (
    extract_bulk_from_counts,
    load_bulk_counts,
    load_mtx_dataset,
    load_scrna_reference,
)
from .benchmark import run_pseudobulk_benchmark
from .reference import build_reference, load_reference, save_reference
from .solver import attribute_changes, to_cp10k

GSE141259_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE141nnn/GSE141259/suppl"
GSE141259_FILES = [
    "GSE141259_WholeLung_cellinfo.csv.gz",
    "GSE141259_WholeLung_genes.txt.gz",
    "GSE141259_WholeLung_barcodes.txt.gz",
    "GSE141259_WholeLung_rawcounts.mtx.gz",
]


def _ensure_gse141259_downloaded(dest_dir: str | Path = "data/GSE141259") -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for fname in GSE141259_FILES:
        out_path = dest / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        url = f"{GSE141259_BASE_URL}/{fname}"
        print(f"[download] {url} -> {out_path} ...")
        with urllib.request.urlopen(url) as resp, open(out_path, "wb") as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
    return dest


def _load_gse141259(data_dir: Path):
    return load_mtx_dataset(
        mtx_path=data_dir / "GSE141259_WholeLung_rawcounts.mtx.gz",
        genes_path=data_dir / "GSE141259_WholeLung_genes.txt.gz",
        barcodes_path=data_dir / "GSE141259_WholeLung_barcodes.txt.gz",
        cellinfo_path=data_dir / "GSE141259_WholeLung_cellinfo.csv.gz",
    )


def _reference_from_sc_files(args: argparse.Namespace):
    counts, cell_types = load_scrna_reference(
        args.sc_counts,
        args.sc_meta,
        cell_type_col=args.cell_type_col,
        sample_filter=args.sample_filter,
    )
    return build_reference(counts, cell_types, normalize=True)


def _cmd_build_ref(args: argparse.Namespace) -> None:
    if args.sc_counts and args.sc_meta:
        ref = _reference_from_sc_files(args)
    elif args.gse141259_dir:
        sc = _load_gse141259(_ensure_gse141259_downloaded(args.gse141259_dir))
        mask = (
            (sc.metadata["grouping"] == args.baseline_group)
            & sc.metadata[args.cell_type_col].notna()
            & (sc.metadata[args.cell_type_col] != "unassigned")
        ).to_numpy()
        ref = build_reference(
            sc.counts[:, mask], sc.metadata.loc[mask, args.cell_type_col].to_numpy(), genes=sc.genes, normalize=True
        )
    else:
        raise SystemExit("Error: Provide either --sc-counts and --sc-meta, or --gse141259-dir.")

    dest = save_reference(ref, args.out_dir)
    print(f"Reference built ({ref.S_T.shape[0]:,} genes x {ref.S_T.shape[1]} cell types) -> saved to {dest}")


def _cmd_make_bulk(args: argparse.Namespace) -> None:
    bulk_df = extract_bulk_from_counts(
        counts_a_path=args.counts_a,
        counts_b_path=args.counts_b,
        sample_a_name=args.sample_a,
        sample_b_name=args.sample_b,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bulk_df.to_csv(out_path)
    print(f"Created bulk matrix ({bulk_df.shape[0]:,} genes x 2 samples: {args.sample_a}, {args.sample_b}) -> {out_path}")


def _cmd_solve(args: argparse.Namespace) -> None:
    if args.ref_dir:
        ref = load_reference(args.ref_dir)
    elif args.sc_counts and args.sc_meta:
        ref = _reference_from_sc_files(args)
    else:
        raise SystemExit("Error: Provide either --ref-dir or both --sc-counts and --sc-meta.")

    bulk = load_bulk_counts(args.bulk)
    y1 = bulk[args.sample_a].astype(float)
    y2 = bulk[args.sample_b].astype(float)
    if not args.no_normalize:
        y1, y2 = to_cp10k(y1), to_cp10k(y2)
    res = attribute_changes(ref, y1, y2)

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_genes).parent.mkdir(parents=True, exist_ok=True)
    res.delta_p.to_frame(name="delta_p").to_csv(args.out_summary)
    res.gene_table.to_csv(args.out_genes)

    print(f"Attribution ({args.sample_a} -> {args.sample_b}):")
    print(f"  Platform r = {res.platform_r:.3f} (calibration {'applied' if res.platform_calibrated else 'not applied'})")
    print(f"  Global Cell-Type Shift Attribution     : {res.global_pct_cell_type:.1f}%")
    print(f"  Global Intrinsic Expression Attribution: {res.global_pct_expression:.1f}%")
    print(f"Saved cell-type shifts to {args.out_summary}")
    print(f"Saved per-gene attribution to {args.out_genes}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    data_dir = _ensure_gse141259_downloaded(args.data_dir)
    sc = _load_gse141259(data_dir)
    meta = sc.metadata
    valid = (meta["metacelltype"].notna() & (meta["metacelltype"] != "unassigned")).to_numpy()
    bench = run_pseudobulk_benchmark(
        counts=sc.counts[:, valid],
        cell_types=meta.loc[valid, "metacelltype"].to_numpy(),
        groups=meta.loc[valid, "grouping"].to_numpy(),
        genes=sc.genes,
        baseline=args.baseline_group,
        target=args.target_group,
        seed=args.seed,
    )
    res, m = bench.result, bench.metrics
    label = f"{args.baseline_group} -> {args.target_group}"

    print("=" * 78)
    print(f"PSEUDOBULK BENCHMARK: GSE141259 Mouse Lung Bleomycin Fibrosis ({label})")
    print("=" * 78)
    print(f"Platform r = {res.platform_r:.3f} (calibration {'applied' if res.platform_calibrated else 'not applied'})\n")
    print("Cell-type shift (Δp):")
    print(f"  Pearson r vs. truth           : {m['delta_p_pearson_r']:.4f}")
    print(f"  Mean absolute error           : {m['delta_p_mae']:.4f}")
    print(f"  Sum of estimated Δp (truth 0) : {m['delta_p_sum']:+.4f}\n")
    print("Per-gene attribution:")
    print(f"  Δy_cell_type Pearson r (linear / signed-log1p): {m['gene_ct_pearson_r']:.4f} / {m['gene_ct_pearson_r_log']:.4f}")
    print(f"  Dominant source accuracy (|Δy| >= 1 CP10K, n={m['n_changed_genes']:,}): {m['dominant_source_accuracy']:.1%}")
    print(f"  %CellType mean absolute error (same genes): {m['pct_cell_type_mae_changed_genes']:.1f} pts")
    print(f"  Global %CellType  truth {bench.true_global_pct_cell_type:.1f}%  vs. estimated {res.global_pct_cell_type:.1f}%\n")

    print(f"{'Cell Type':<20} | {'True Δp':>10} | {'Est Δp':>10}")
    print("-" * 46)
    for ct in bench.true_delta_p.abs().nlargest(6).index:
        print(f"{ct:<20} | {bench.true_delta_p[ct]:>+10.4f} | {res.delta_p[ct]:>+10.4f}")
    print()

    print(f"{'Gene':<10} | {'Δy_total':>10} | {'True Δy_ct':>11} | {'Est Δy_ct':>10} | {'True %CT':>9} | {'Est %CT':>8}")
    print("-" * 74)
    for g in ["Spp1", "Arg1", "Ccl2", "Fn1", "Col1a1", "Col3a1", "C1qa", "Sftpc", "Scgb1a1"]:
        if g in res.gene_table.index:
            est, true = res.gene_table.loc[g], bench.true_gene_table.loc[g]
            print(
                f"{g:<10} | {est['delta_y_total']:>+10.2f} | {true['delta_y_cell_type']:>+11.2f} | "
                f"{est['delta_y_cell_type']:>+10.2f} | {true['pct_cell_type']:>8.1f}% | {est['pct_cell_type']:>7.1f}%"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deconv-attributor",
        description="Minimal proof-of-concept: separate bulk RNA-seq changes into cell-type vs. expression changes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ref = subparsers.add_parser("build-ref", help="Step 1: Build reference matrix S_T and marker weights w")
    p_ref.add_argument("--gse141259-dir", default=None, help="Path to GSE141259 folder (downloaded if missing)")
    p_ref.add_argument("--sc-counts", default=None, help="Path to baseline scRNA-seq counts (.csv or .txt)")
    p_ref.add_argument("--sc-meta", default=None, help="Path to baseline scRNA-seq metadata (.csv or .txt)")
    p_ref.add_argument("--sample-filter", default=None, help="Optional sample name to filter metadata (e.g. B1_1_0h)")
    p_ref.add_argument("--baseline-group", default="PBS", help="Baseline condition name (default: PBS)")
    p_ref.add_argument("--cell-type-col", default="metacelltype", help="Cell-type column name (e.g. maintype)")
    p_ref.add_argument("--out-dir", default="data/reference", help="Output directory for reference")
    p_ref.set_defaults(func=_cmd_build_ref)

    p_bulk = subparsers.add_parser("make-bulk", help="Extract or sum bulk counts from two count files into a CSV")
    p_bulk.add_argument("--counts-a", required=True, help="Path to baseline condition count file")
    p_bulk.add_argument("--counts-b", required=True, help="Path to comparison condition count file")
    p_bulk.add_argument("--sample-a", default="Bulk_0h", help="Column name for sample A (default: Bulk_0h)")
    p_bulk.add_argument("--sample-b", default="Bulk_4h", help="Column name for sample B (default: Bulk_4h)")
    p_bulk.add_argument("--out", required=True, help="Output CSV path for combined bulk table")
    p_bulk.set_defaults(func=_cmd_make_bulk)

    p_solve = subparsers.add_parser("solve", help="Steps 2-3: Estimate Δp and attribute Δy per gene")
    p_solve.add_argument("--bulk", required=True, help="Path to bulk counts CSV (genes x samples)")
    p_solve.add_argument("--ref-dir", default=None, help="Path to pre-computed reference directory")
    p_solve.add_argument("--sc-counts", default=None, help="Path to baseline scRNA-seq counts CSV/TXT")
    p_solve.add_argument("--sc-meta", default=None, help="Path to baseline scRNA-seq metadata CSV/TXT")
    p_solve.add_argument("--sample-filter", default=None, help="Optional sample name filter in metadata")
    p_solve.add_argument("--cell-type-col", default="cell_type", help="Cell-type column name in metadata")
    p_solve.add_argument("--sample-a", required=True, help="Column name of baseline bulk sample 1")
    p_solve.add_argument("--sample-b", required=True, help="Column name of comparison bulk sample 2")
    p_solve.add_argument("--no-normalize", action="store_true", help="Skip CP10K linear normalization")
    p_solve.add_argument("--out-summary", default="cell_type_shifts.csv", help="Output CSV for Δp")
    p_solve.add_argument("--out-genes", default="gene_attribution.csv", help="Output CSV for per-gene attribution")
    p_solve.set_defaults(func=_cmd_solve)

    p_bench = subparsers.add_parser("benchmark", help="Download (if needed) and run the GSE141259 Mouse Lung benchmark")
    p_bench.add_argument("--data-dir", default="data/GSE141259", help="Directory for GSE141259 data")
    p_bench.add_argument("--baseline-group", default="PBS", help="Baseline group in the `grouping` column (default: PBS)")
    p_bench.add_argument("--target-group", default="d10", help="Target group in the `grouping` column (default: d10)")
    p_bench.add_argument("--seed", type=int, default=42, help="Seed for the reference/held-out split of baseline cells")
    p_bench.set_defaults(func=_cmd_benchmark)

    parsed = parser.parse_args(argv)
    parsed.func(parsed)


if __name__ == "__main__":
    main()
