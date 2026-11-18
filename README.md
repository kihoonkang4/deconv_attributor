# deconv_attributor

Minimal 3-step proof-of-concept for separating bulk RNA-seq differences ($\Delta y = y_2 - y_1$) into **cell-type proportion shifts** vs. **intrinsic gene expression changes**:

1. **Step 1 — Reference Matrix ($S^T$) & Marker Weights ($w$):**
   Computes baseline cluster mean expression $S^T \in \mathbb{R}^{G \times K}$ (`genes x cell_types`) in linear CP10K space and weights each gene by its cell-type specificity:
   $$w_g = \frac{\mathrm{Var}_{\mathrm{between}}(g)}{\mathrm{Var}_{\mathrm{within}}(g) + 10^{-4}}$$
2. **Step 2 — Robust Cell-Type Shift Estimation ($\Delta p$):**
   Fits $\Delta y \approx S^T \Delta p$ using `HuberRegressor` weighted by $w$ so the consensus of stable marker genes locks in $\Delta p$ while differentially expressed outlier genes are ignored.
3. **Step 3 — Subtract to Attribute Each Gene:**
   $$\Delta y_{\mathrm{celltype}} = S^T \Delta p, \qquad \Delta y_{\mathrm{expr}} = \Delta y - \Delta y_{\mathrm{celltype}}$$
   $$\mathrm{Pct}_{\mathrm{celltype}} = 100 \times \frac{|\Delta y_{\mathrm{celltype}}|}{|\Delta y_{\mathrm{celltype}}| + |\Delta y_{\mathrm{expr}}|}, \qquad \mathrm{Pct}_{\mathrm{expr}} = 100 - \mathrm{Pct}_{\mathrm{celltype}}$$

---

## Quick Start on Google Colab

```python
# 1. Install directly from GitHub
!pip install -q git+https://github.com/kihoonkang4/deconv_attributor.git

# 2. Run the Mouse Lung Bleomycin Fibrosis benchmark (auto-downloads GSE141259 from GEO)
!deconv-attributor benchmark
```

---

## CLI Commands & Options Reference

The `deconv-attributor` CLI provides four subcommands: `build-ref`, `make-bulk`, `solve`, and `benchmark`.

---

### 1. `deconv-attributor build-ref`
**Purpose:** Step 1 — Builds the baseline cell-type reference matrix (`S_T_matrix_GxK.csv.gz`) and per-gene marker specificity weights (`marker_weights.csv`) from baseline single-cell RNA-seq data.

```bash
deconv-attributor build-ref \
  --sc-counts path/to/B1_1_0h_counts.txt \
  --sc-meta path/to/GSE102827_meta.csv.gz \
  --sample-filter B1_1_0h \
  --cell-type-col maintype \
  --out-dir path/to/reference
```

| Option | Required? | Default | Description |
| :--- | :---: | :--- | :--- |
| `--sc-counts` | Yes (or `--gse141259-dir`) | `None` | Path to baseline scRNA-seq count matrix (`.csv`, `.tsv`, `.txt`, or `.gz`, formatted as `genes x cells`). |
| `--sc-meta` | Yes (or `--gse141259-dir`) | `None` | Path to scRNA-seq cell metadata table (`.csv`, `.tsv`, `.txt`, or `.gz`, indexed by cell barcode). |
| `--sample-filter` | Optional | `None` | Filters `--sc-meta` to rows where the `sample` column equals this value (e.g., `B1_1_0h`). Useful when a single metadata file contains multiple samples with overlapping barcode names (`bc0001`, etc.). |
| `--cell-type-col` | Optional | `metacelltype` | Column name in `--sc-meta` containing cell-type labels (e.g., `maintype`, `celltype`, or `cell_type`). |
| `--gse141259-dir` | Optional | `data/GSE141259` | Path to a 10x MatrixMarket `GSE141259` folder (used only if `--sc-counts` and `--sc-meta` are omitted; auto-downloads from GEO if missing). |
| `--baseline-group` | Optional | `PBS` | Baseline condition name in the `grouping` column when using `--gse141259-dir`. |
| `--out-dir` | Optional | `data/reference` | Output directory where `S_T_matrix_GxK.csv.gz` and `marker_weights.csv` will be saved. |

---

### 2. `deconv-attributor make-bulk`
**Purpose:** Helper utility — Extracts the `Sum_counts` column (or sums all cell columns) from two count tables (`counts-a` and `counts-b`) and combines them into a single 2-column bulk RNA-seq CSV (`genes x 2 samples`) ready for `solve`.

```bash
deconv-attributor make-bulk \
  --counts-a path/to/B1_1_0h_counts.txt \
  --counts-b path/to/B1_3_4h_counts.txt \
  --sample-a Bulk_0h \
  --sample-b Bulk_4h \
  --out path/to/bulk_0h_vs_4h.csv
```

| Option | Required? | Default | Description |
| :--- | :---: | :--- | :--- |
| `--counts-a` | **Yes** | — | Path to baseline/control count file (`.csv`, `.tsv`, or `.txt`). Uses the `Sum_counts` column if present, or sums all cell columns across each gene. |
| `--counts-b` | **Yes** | — | Path to treatment/comparison count file (`.csv`, `.tsv`, or `.txt`). |
| `--sample-a` | Optional | `Bulk_0h` | Column name to assign to sample A in the output CSV. |
| `--sample-b` | Optional | `Bulk_4h` | Column name to assign to sample B in the output CSV. |
| `--out` | **Yes** | — | Output file path for the combined 2-sample bulk CSV. |

---

### 3. `deconv-attributor solve`
**Purpose:** Steps 2 & 3 — Estimates cell-type proportion shifts ($\Delta p$) between two bulk samples and attributes each gene's bulk expression change into cell-type shift (`delta_y_cell_type`, `pct_cell_type`) vs. intrinsic expression change (`delta_y_expr`, `pct_expr`).

```bash
deconv-attributor solve \
  --bulk path/to/bulk_0h_vs_4h.csv \
  --ref-dir path/to/reference \
  --sample-a Bulk_0h \
  --sample-b Bulk_4h \
  --out-summary path/to/cell_type_shifts.csv \
  --out-genes path/to/gene_attribution.csv
```

| Option | Required? | Default | Description |
| :--- | :---: | :--- | :--- |
| `--bulk` | **Yes** | — | Path to bulk RNA-seq expression table (`.csv`, `.tsv`, or `.txt`, formatted as `genes x samples` in linear counts/TPM/CP10K). |
| `--sample-a` | **Yes** | — | Column name of the baseline/control bulk sample ($y_1$) inside `--bulk`. |
| `--sample-b` | **Yes** | — | Column name of the treatment/comparison bulk sample ($y_2$) inside `--bulk`. |
| `--ref-dir` | Yes (or `--sc-counts` + `--sc-meta`) | `None` | Path to the pre-computed reference directory created by `build-ref`. |
| `--sc-counts` | Optional | `None` | Path to baseline scRNA-seq count matrix (if running end-to-end without running `build-ref` first). |
| `--sc-meta` | Optional | `None` | Path to baseline scRNA-seq metadata table (if running end-to-end without `--ref-dir`). |
| `--sample-filter` | Optional | `None` | Sample name filter in `--sc-meta` when using `--sc-counts` + `--sc-meta`. |
| `--cell-type-col` | Optional | `cell_type` | Cell-type column name in `--sc-meta` when using `--sc-counts` + `--sc-meta`. |
| `--no-normalize` | Optional | `False` | If set, skips normalizing each bulk sample column to sum to $10{,}000$ (CP10K). |
| `--out-summary` | Optional | `cell_type_shifts.csv` | Output CSV path for the cell-type shift table (`delta_p`). |
| `--out-genes` | Optional | `gene_attribution.csv` | Output CSV path for the per-gene attribution table (`delta_y_total`, `delta_y_cell_type`, `delta_y_expr`, `pct_cell_type`, `pct_expr`). |

---

### 4. `deconv-attributor benchmark`
**Purpose:** Downloads the Strunz et al. (2020) Mouse Lung Bleomycin Fibrosis dataset (`GSE141259`) from NCBI GEO (if not already present) and runs the end-to-end benchmark (`PBS -> d10`).

```bash
deconv-attributor benchmark --data-dir data/GSE141259
```

| Option | Required? | Default | Description |
| :--- | :---: | :--- | :--- |
| `--data-dir` | Optional | `data/GSE141259` | Local directory where the `GSE141259` files are stored or downloaded. |
