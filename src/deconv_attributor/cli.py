"""Minimal CLI for deconv-attributor (build-ref, solve, benchmark)."""

import argparse
from pathlib import Path
import subprocess
import sys

from .io import load_bulk_counts, load_mtx_dataset, load_scrna_reference
from .pipeline import run_attribution_pipeline
from .reference import build_reference, load_reference, save_reference
from .solver import attribute_changes


def _cmd_build_ref(args: argparse.Namespace) -> None:
    if args.gse141259_dir and Path(args.gse141259_dir).exists():
        d = Path(args.gse141259_dir)
        sc = load_mtx_dataset(
            mtx_path=d / "GSE141259_WholeLung_rawcounts.mtx.gz",
            genes_path=d / "GSE141259_WholeLung_genes.txt.gz",
            barcodes_path=d / "GSE141259_WholeLung_barcodes.txt.gz",
            cellinfo_path=d / "GSE141259_WholeLung_cellinfo.csv.gz",
        )
        mask = (
            (sc.metadata["grouping"] == args.baseline_group)
            & sc.metadata[args.cell_type_col].notna()
            & (sc.metadata[args.cell_type_col] != "unassigned")
        ).to_numpy()
        counts = sc.counts[:, mask]
        cell_types = sc.metadata.loc[mask, args.cell_type_col].to_numpy()
        genes = sc.genes
    elif args.sc_counts and args.sc_meta:
        counts, ct_ser = load_scrna_reference(args.sc_counts, args.sc_meta)
        cell_types = ct_ser.to_numpy()
        genes = counts.index
    else:
        raise SystemExit("Error: Provide either --gse141259-dir or both --sc-counts and --sc-meta.")

    S_T, w = build_reference(sc_counts=counts, cell_types=cell_types, genes=genes, normalize=True)
    dest = save_reference(S_T, w, args.out_dir)
    print(f"Reference built ({S_T.shape[0]:,} genes x {S_T.shape[1]} cell types) -> saved to {dest}")


def _cmd_solve(args: argparse.Namespace) -> None:
    bulk = load_bulk_counts(args.bulk)
    if args.ref_dir:
        S_T, w = load_reference(args.ref_dir)
        y1 = bulk[args.sample_a].astype(float)
        y2 = bulk[args.sample_b].astype(float)
        if not args.no_normalize:
            if y1.sum() > 0:
                y1 = y1 * (1e4 / float(y1.sum()))
            if y2.sum() > 0:
                y2 = y2 * (1e4 / float(y2.sum()))
        res = attribute_changes(S_T=S_T, w=w, y1=y1, y2=y2)
    elif args.sc_counts and args.sc_meta:
        sc_counts, cell_types = load_scrna_reference(args.sc_counts, args.sc_meta)
        res = run_attribution_pipeline(
            bulk=bulk,
            sc_counts=sc_counts,
            cell_types=cell_types,
            sample_a=args.sample_a,
            sample_b=args.sample_b,
            normalize_linear=not args.no_normalize,
        )
    else:
        raise SystemExit("Error: Provide either --ref-dir or both --sc-counts and --sc-meta.")

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_genes).parent.mkdir(parents=True, exist_ok=True)
    res.delta_p.to_frame(name="delta_p").to_csv(args.out_summary)
    res.gene_table.to_csv(args.out_genes)

    print(f"Attribution ({args.sample_a} -> {args.sample_b}):")
    print(f"  Global Cell-Type Shift Attribution     : {res.global_pct_cell_type:.1f}%")
    print(f"  Global Intrinsic Expression Attribution: {res.global_pct_expression:.1f}%")
    print(f"Saved cell-type shifts to {args.out_summary}")
    print(f"Saved per-gene attribution to {args.out_genes}")


def main(argv: list[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0].startswith("-") and args_list[0] not in ("-h", "--help"):
        args_list = ["solve", *args_list]

    parser = argparse.ArgumentParser(
        prog="deconv-attributor",
        description="Minimal proof-of-concept: separate bulk RNA-seq changes into cell-type vs. expression changes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ref = subparsers.add_parser("build-ref", help="Step 1: Build reference matrix S_T and marker weights w")
    p_ref.add_argument("--gse141259-dir", default="data/GSE141259", help="Path to GSE141259 folder")
    p_ref.add_argument("--sc-counts", default=None, help="Path to baseline scRNA-seq counts CSV")
    p_ref.add_argument("--sc-meta", default=None, help="Path to baseline scRNA-seq metadata CSV")
    p_ref.add_argument("--baseline-group", default="PBS", help="Baseline condition name (default: PBS)")
    p_ref.add_argument("--cell-type-col", default="metacelltype", help="Cell-type column name")
    p_ref.add_argument("--out-dir", default="data/reference", help="Output directory for reference")
    p_ref.set_defaults(func=_cmd_build_ref)

    p_solve = subparsers.add_parser("solve", help="Steps 2-3: Estimate Δp and attribute Δy per gene")
    p_solve.add_argument("--bulk", required=True, help="Path to bulk counts CSV (genes x samples)")
    p_solve.add_argument("--ref-dir", default=None, help="Path to pre-computed reference directory")
    p_solve.add_argument("--sc-counts", default=None, help="Path to baseline scRNA-seq counts CSV")
    p_solve.add_argument("--sc-meta", default=None, help="Path to baseline scRNA-seq metadata CSV")
    p_solve.add_argument("--sample-a", required=True, help="Column name of baseline bulk sample 1")
    p_solve.add_argument("--sample-b", required=True, help="Column name of comparison bulk sample 2")
    p_solve.add_argument("--no-normalize", action="store_true", help="Skip CP10K linear normalization")
    p_solve.add_argument("--out-summary", default="cell_type_shifts.csv", help="Output CSV for Δp")
    p_solve.add_argument("--out-genes", default="gene_attribution.csv", help="Output CSV for per-gene attribution")
    p_solve.set_defaults(func=_cmd_solve)

    p_bench = subparsers.add_parser("benchmark", help="Run the GSE141259 Mouse Lung Bleomycin proof-of-concept")
    p_bench.set_defaults(
        func=lambda _a: subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark_gse141259.py")],
            check=True,
        )
    )

    parsed = parser.parse_args(args_list)
    parsed.func(parsed)


if __name__ == "__main__":
    main()
