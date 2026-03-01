````md
# SDM-Z Similarity (All-sample vs Healthy-only)

This repository contains a single script to compute similarity metrics between two **unthresholded SDM-Z maps**:
**all-sample** vs **healthy-only** (NIfTI: `.nii` / `.nii.gz`).

## 1) Install

```bash
pip install -r requirements.txt
````

## 2) Run (single pair)

```bash
python compute_similarity.py \
  --all /path/to/all3_MyTest_z.nii.gz \
  --healthy /path/to/health3_MyTest_z.nii.gz \
  --task APPRAISAL \
  --out similarity_appraisal.csv
```

## 3) Run (batch mode)

Create a CSV (e.g., `pairs.csv`) with columns: `task,all_map,healthy_map`

Example:

```csv
task,all_map,healthy_map
IMPLICIT_EMOTION,/path/to/all1_MyTest_z.nii.gz,/path/to/health1_MyTest_z.nii.gz
SHIFT,/path/to/all2_MyTest_z.nii.gz,/path/to/health2_withsiripada_MyTest_z.nii.gz
APPRAISAL,/path/to/all3_MyTest_z.nii.gz,/path/to/health3_MyTest_z.nii.gz
```

Run:

```bash
python compute_similarity.py --pairs_csv pairs.csv --out similarity_all_tasks.csv
```

## 4) Output

The script writes a CSV containing (per pair):

* Unthresholded voxelwise similarity: `r`, `fisher_z`
* Top-% overlap (default top 10%): Dice/Jaccard for top-|Z| and top +Z
* Sign agreement within top-|Z| regions
* Voxel counts and thresholds used

## 5) Optional flags

* Change top percentage (default 10): `--top_pct 5`
* Include zero-background voxels: `--include_zeros_in_mask`
* Skip affine check (only if intended): `--skip_affine_check`

```
```
