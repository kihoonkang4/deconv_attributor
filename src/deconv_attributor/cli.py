"""Minimal CLI for deconv-attributor (build-ref, make-bulk, solve, benchmark)."""

import argparse
from pathlib import Path
import ssl
import sys
import urllib.request
import numpy as np

from .io import (
    extract_bulk_from_counts,
    load_bulk_counts,
    load_mtx_dataset,
    load_scrna_reference,
)
from .pipeline import run_attribution_pipeline
from .reference import build_reference, load_reference, save_reference
from .solver import attribute_changes

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
    ctx = ssl._create_unverified_context()
    for fname in GSE141259_FILES:
        out_path = dest / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        url = f"{GSE141259_BASE_URL}/{fname}"
        print(f"[download] {url} -> {out_path} ...")
        with urllib.request.urlopen(url, context=ctx) as resp, open(out_path, "wb") as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
    return dest


def _cmd_build_ref(args: argparse.Namespace) -> None:
    if args.sc_counts and args.sc_meta:
        counts, ct_ser = load_scrna_reference(
            args.sc_counts,
            args.sc_meta,
            cell_type_col=args.cell_type_col,
            sample_filter=args.sample_filter,
        )
        cell_types = ct_ser.to_numpy()
        genes = counts.index
    elif args.gse141259_dir:
        d = _ensure_gse141259_downloaded(args.gse141259_dir)
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
    else:
        raise SystemExit("Error: Provide either --sc-counts and --sc-meta, or --gse141259-dir.")

    S_T, w = build_reference(sc_counts=counts, cell_types=cell_types, genes=genes, normalize=True)
    dest = save_reference(S_T, w, args.out_dir)
    print(f"Reference built ({S_T.shape[0]:,} genes x {S_T.shape[1]} cell types) -> saved to {dest}")


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
        sc_counts, cell_types = load_scrna_reference(
            args.sc_counts,
            args.sc_meta,
            cell_type_col=args.cell_type_col,
            sample_filter=args.sample_filter,
        )
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
    res.cell_type_table.to_csv(args.out_summary)
    res.gene_table.to_csv(args.out_genes)

    print(f"Attribution ({args.sample_a} -> {args.sample_b}):")
    print(f"  Global Cell-Type Shift Attribution     : {res.global_pct_cell_type:.1f}%")
    print(f"  Global Intrinsic Expression Attribution: {res.global_pct_expression:.1f}%")
    print(f"Saved cell-type shifts to {args.out_summary}")
    print(f"Saved per-gene attribution to {args.out_genes}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    data_dir = _ensure_gse141259_downloaded(args.data_dir)
    print("=" * 78)
    print("MINIMAL 3-STEP PROOF OF CONCEPT: GSE141259 Mouse Lung Bleomycin Fibrosis")
    print("=" * 78)
    sc = load_mtx_dataset(
        mtx_path=data_dir / "GSE141259_WholeLung_rawcounts.mtx.gz",
        genes_path=data_dir / "GSE141259_WholeLung_genes.txt.gz",
        barcodes_path=data_dir / "GSE141259_WholeLung_barcodes.txt.gz",
        cellinfo_path=data_dir / "GSE141259_WholeLung_cellinfo.csv.gz",
    )

    valid = (sc.metadata["metacelltype"].notna() & (sc.metadata["metacelltype"] != "unassigned")).to_numpy()
    counts = sc.counts[:, valid]
    meta = sc.metadata.loc[valid].copy()
    genes = sc.genes

    rng = np.random.default_rng(42)
    pbs_idx = np.where(meta["grouping"].to_numpy() == "PBS")[0]
    rng.shuffle(pbs_idx)
    ref_pbs_idx = pbs_idx[: len(pbs_idx) // 2]
    test_pbs_idx = pbs_idx[len(pbs_idx) // 2 :]
    d10_idx = np.where(meta["grouping"].to_numpy() == "d10")[0]

    S_T, w = build_reference(
        sc_counts=counts[:, ref_pbs_idx],
        cell_types=meta["metacelltype"].to_numpy()[ref_pbs_idx],
        genes=genes,
        normalize=True,
    )
    print(f"Step 1: Built reference S_T ({S_T.shape[0]:,} genes x {S_T.shape[1]} cell types) and weights w.\n")

    S_T_pbs, _ = build_reference(counts[:, test_pbs_idx], meta["metacelltype"].to_numpy()[test_pbs_idx], genes=genes)
    S_T_d10, _ = build_reference(counts[:, d10_idx], meta["metacelltype"].to_numpy()[d10_idx], genes=genes)
    S_T_pbs = S_T_pbs.reindex(columns=S_T.columns, fill_value=0.0)
    S_T_d10 = S_T_d10.reindex(columns=S_T.columns, fill_value=0.0)

    p_pbs = meta["metacelltype"].iloc[test_pbs_idx].value_counts(normalize=True).reindex(S_T.columns).fillna(0.0)
    p_d10 = meta["metacelltype"].iloc[d10_idx].value_counts(normalize=True).reindex(S_T.columns).fillna(0.0)

    y_pbs = S_T_pbs.dot(p_pbs)
    y_d10 = S_T_d10.dot(p_d10)

    res_real = attribute_changes(S_T, w, y_pbs, y_d10)
    true_delta_p = p_d10 - p_pbs
    r_dp = float(np.corrcoef(true_delta_p, res_real.delta_p)[0, 1])

    print(f"Real Bleomycin Injury (PBS -> d10):")
    print(f"  Estimated vs. True Cell-Type Shift (Δp) Pearson r : {r_dp:.4f}")
    print(f"  Global Cell-Type Shift Attribution                : {res_real.global_pct_cell_type:.1f}%")
    print(f"  Global Intrinsic Expression Attribution           : {res_real.global_pct_expression:.1f}%\n")

    print("Top Cell-Type Proportion Shifts (PBS -> d10):")
    print("-" * 78)
    print(f"{'Cell Type':<20} | {'True Δp':>10} | {'Est Δp':>10} | {'StdErr':>10} | {'p-value':>12}")
    print("-" * 78)
    for ct in res_real.cell_type_table["delta_p"].abs().nlargest(6).index:
        ct_row = res_real.cell_type_table.loc[ct]
        print(
            f"{ct:<20} | {true_delta_p[ct]:>+10.4f} | {ct_row['delta_p']:>+10.4f} | "
            f"{ct_row['std_err']:>10.4f} | {ct_row['p_value']:>12.2e}"
        )
    print()

    print("Key Fibrosis & Lung Marker Genes (PBS -> d10):")
    print("-" * 92)
    print(
        f"{'Gene':<10} | {'Δy_total':>9} | {'Δy_cell_type':>13} | {'Δy_expr':>9} | {'%CellType':>9} | {'%Expr':>7} | {'z_expr':>8} | {'pval_expr':>10}"
    )
    print("-" * 92)
    for g in ["Spp1", "Arg1", "Ccl2", "Fn1", "Col1a1", "Col3a1", "C1qa", "Sftpc", "Scgb1a1"]:
        if g in res_real.gene_table.index:
            row = res_real.gene_table.loc[g]
            print(
                f"{g:<10} | {row['delta_y_total']:>+9.2f} | {row['delta_y_cell_type']:>+13.2f} | "
                f"{row['delta_y_expr']:>+9.2f} | {row['pct_cell_type']:>8.1f}% | {row['pct_expr']:>6.1f}% | "
                f"{row['z_expr']:>+8.2f} | {row['pval_expr']:>10.2e}"
            )


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
    p_bench.set_defaults(func=_cmd_benchmark)

    parsed = parser.parse_args(args_list)
    parsed.func(parsed)


if __name__ == "__main__":
    main()
