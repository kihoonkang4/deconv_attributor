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

## Run on Google Colab

```python
# 1. Install directly from GitHub
!pip install -q git+https://github.com/kihoonkang4/deconv_attributor.git

# 2. Run the Mouse Lung Bleomycin Fibrosis benchmark (auto-downloads GSE141259 from GEO)
!deconv-attributor benchmark
```

---

## Step-by-Step CLI Usage

```bash
# Step 1: Build baseline reference matrix S_T and marker weights w
deconv-attributor build-ref \
  --sc-counts path/to/baseline_sc_counts.txt \
  --sc-meta path/to/sc_metadata.csv \
  --cell-type-col cell_type \
  --out-dir data/reference

# Step 2 (Optional): Combine two count files into a 2-column bulk CSV
deconv-attributor make-bulk \
  --counts-a path/to/control_counts.txt \
  --counts-b path/to/treated_counts.txt \
  --sample-a Bulk_Control \
  --sample-b Bulk_Treated \
  --out data/bulk_comparison.csv

# Step 3: Solve for cell-type shifts (Δp) and per-gene attribution with p-values
deconv-attributor solve \
  --bulk data/bulk_comparison.csv \
  --ref-dir data/reference \
  --sample-a Bulk_Control \
  --sample-b Bulk_Treated \
  --out-summary results/cell_type_shifts.csv \
  --out-genes results/gene_attribution.csv
```
