# deconv_attributor

Minimal 3-step proof-of-concept for separating bulk RNA-seq differences ($\Delta y = y_2 - y_1$) into **cell-type proportion shifts** vs. **intrinsic gene expression changes**:

1. **Step 1 — Reference Matrix ($S^T$) & Marker Weights ($w$)** ([reference.py](file:///Users/kihoonkang/projects/deconv_attributor/src/deconv_attributor/reference.py)):
   Computes baseline cluster mean expression $S^T \in \mathbb{R}^{G \times K}$ (`genes × cell_types`) in linear CP10K space and weights each gene by its cell-type specificity ($w_g = \frac{\text{Var}_{\text{between}}(g)}{\text{Var}_{\text{within}}(g) + 10^{-4}}$).
2. **Step 2 — Robust Cell-Type Shift Estimation ($\Delta p$)** ([solver.py](file:///Users/kihoonkang/projects/deconv_attributor/src/deconv_attributor/solver.py)):
   Fits $\Delta y \approx S^T \Delta p$ using `HuberRegressor` weighted by $w$ so the consensus of stable marker genes locks in $\Delta p$ while differentially expressed outlier genes are ignored.
3. **Step 3 — Subtract to Attribute Each Gene** ([solver.py](file:///Users/kihoonkang/projects/deconv_attributor/src/deconv_attributor/solver.py)):
   $$\Delta y_{\text{cell\_type}} = S^T \Delta p, \qquad \Delta y_{\text{expr}} = \Delta y - \Delta y_{\text{cell\_type}}$$
   $$\text{\% Cell-Type} = 100 \times \frac{|\Delta y_{\text{cell\_type}}|}{|\Delta y_{\text{cell\_type}}| + |\Delta y_{\text{expr}}|}, \qquad \text{\% Expression} = 100 - \text{\% Cell-Type}$$

---

## Quick Start

```bash
# 1. Build baseline reference S_T and marker weights w
./deconv-attributor build-ref \
  --gse141259-dir data/GSE141259 \
  --baseline-group PBS \
  --cell-type-col metacelltype \
  --out-dir data/reference

# 2. Attribute bulk RNA-seq changes between two samples
./deconv-attributor solve \
  --bulk path/to/bulk_counts.csv \
  --ref-dir data/reference \
  --sample-a Control_1 \
  --sample-b Treated_1 \
  --out-summary results/cell_type_shifts.csv \
  --out-genes results/gene_attribution.csv

# 3. Run the GSE141259 Mouse Lung Bleomycin proof-of-concept benchmark
./deconv-attributor benchmark
```
