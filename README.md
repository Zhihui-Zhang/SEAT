# SDM-Z similarity (all-sample vs healthy-only)

This repository provides a **minimal, reproducible** script to compute similarity metrics between:

- **All-sample** unthresholded SDM-Z map (e.g., `all*_MyTest_z.nii.gz`), and
- **Healthy-only** unthresholded SDM-Z map (e.g., `health*_MyTest_z.nii.gz`).

It is designed for **reviewer-facing code availability** (healthy-only sensitivity check). You **do not** need to publish your full SDM workflow.

---

## Metrics (what the script computes)

For each pair of SDM-Z maps (same grid/shape), the script computes:

1. **Unthresholded signed similarity**
   - `r`: voxelwise Pearson correlation within the analysis mask
   - `fisher_z`: Fisher z-transform of `r` (`arctanh(r)` with numerical safeguards)

2. **Top 10% overlap by |Z| (magnitude; positive + negative)**
   - thresholds: `thr_abs_all`, `thr_abs_healthy`
   - overlap: `dice_abs_top`, `jaccard_abs_top` + voxel counts

3. **Top 10% overlap by positive Z only (direction-consistent)**
   - thresholds: `thr_pos_all`, `thr_pos_healthy`
   - overlap: `dice_pos_top`, `jaccard_pos_top` + voxel counts

4. **Sign agreement within the Top-|Z| region**
   - computed in **union** and **intersection** of top-|Z| masks
   - excludes voxels where either map is exactly 0

### Default analysis mask (recommended)

Background zeros are excluded by default:

```
finite(all) & finite(healthy) & ((all != 0) | (healthy != 0))
```

If your maps are not zero-padded, you may include all finite voxels with `--include_zeros_in_mask`.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `nibabel`, `scipy`.

---

## Usage

### A) Single pair

```bash
python compute_similarity.py \
  --all all3_MyTest_z.nii.gz \
  --healthy health3_MyTest_z.nii.gz \
  --task APPRAISAL \
  --out similarity_appraisal.csv
```

### B) Batch mode (recommended)

Create a manifest CSV with columns:

- `task`
- `all_map`
- `healthy_map`

Then run:

```bash
python compute_similarity.py \
  --pairs_csv pairs.csv \
  --out similarity_all.csv
```

### Optional parameters

- Change top percentage (e.g., top 5%): `--top_pct 5`
- Include zero-background voxels: `--include_zeros_in_mask`
- Skip affine check (only if you know what you are doing): `--skip_affine_check`

---

## What to upload publicly

**Recommended (minimal):**

- `compute_similarity.py`, `README.md`, `requirements.txt`, `LICENSE`, `CITATION.cff`
- (Optional) `example_manifest.csv` and a schema-only `example_output.csv`

**Avoid uploading** SDM maps unless you explicitly want to share derived images.

---

## Suggested manuscript text (replace DOI)

> Python scripts used for the healthy-only sensitivity analysis and similarity metrics are available at **[DOI]**.

or (slightly more specific):

> Scripts used to quantify similarity between all-sample and healthy-only unthresholded SDM-Z maps (signed spatial correlation, top-decile overlap, and sign agreement) are available at **[DOI]**.

