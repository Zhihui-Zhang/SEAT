#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDM-Z similarity metrics: all-sample vs healthy-only (unthresholded maps)

This script is a consolidated, parameterized version of per-task scripts
(e.g., appraisal/shift/implicit emotion) that differed only in input paths.

It computes four families of metrics (default top=10%):
  (1) Signed voxelwise Pearson correlation (r) + Fisher z
  (2) Overlap of top-|Z| voxels (pos + neg): Dice & Jaccard
  (3) Overlap of top-positive-Z voxels: Dice & Jaccard
  (4) Sign agreement within the top-|Z| region (union and intersection),
      excluding voxels where either map is exactly 0.

Masking:
  By default, uses a union "non-background" mask:
    finite(all) & finite(healthy) & ((all != 0) | (healthy != 0))

Usage:
  Single pair:
    python compute_similarity.py --all all3_MyTest_z.nii.gz --healthy health3_MyTest_z.nii.gz --task APPRAISAL --out out.csv

  Batch pairs:
    python compute_similarity.py --pairs_csv pairs.csv --out out.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit("Missing dependency nibabel. Install via: pip install nibabel") from e

try:
    from scipy.stats import pearsonr
except ImportError as e:
    raise SystemExit("Missing dependency scipy. Install via: pip install scipy") from e


@dataclass
class Result:
    task: str
    all_map: str
    healthy_map: str

    n_vox_union_mask: int

    r: float
    fisher_z: float

    top_pct: float

    # (2) top |Z| overlap
    thr_abs_all: float
    thr_abs_healthy: float
    dice_abs_top: float
    jaccard_abs_top: float
    n_abs_all_top: int
    n_abs_healthy_top: int
    n_abs_overlap_top: int

    # (3) top +Z overlap
    thr_pos_all: float
    thr_pos_healthy: float
    dice_pos_top: float
    jaccard_pos_top: float
    n_pos_all_top: int
    n_pos_healthy_top: int
    n_pos_overlap_top: int

    # (4) sign agreement within top |Z| region
    sign_agreement_top_abs_union: float
    n_sign_valid_top_abs_union: int
    sign_agreement_top_abs_inter: float
    n_sign_valid_top_abs_inter: int


def _safe_fisher_z(r: float) -> float:
    eps = 1e-12
    rr = float(r)
    if rr >= 1.0:
        rr = 1.0 - eps
    if rr <= -1.0:
        rr = -1.0 + eps
    return float(np.arctanh(rr))


def _dice_jacc(a_mask: np.ndarray, b_mask: np.ndarray):
    inter = int(np.logical_and(a_mask, b_mask).sum())
    union = int(np.logical_or(a_mask, b_mask).sum())
    a_n = int(a_mask.sum())
    b_n = int(b_mask.sum())
    dice = (2 * inter) / (a_n + b_n) if (a_n + b_n) > 0 else float("nan")
    jacc = inter / union if union > 0 else float("nan")
    return a_n, b_n, inter, union, float(dice), float(jacc)


def _top_pct_mask_abs(z: np.ndarray, base_mask: np.ndarray, pct: float):
    vals = np.abs(z[base_mask])
    thr = float(np.percentile(vals, 100 - pct))
    m = np.zeros(z.shape, dtype=bool)
    m[base_mask] = np.abs(z[base_mask]) >= thr
    return m, thr


def _top_pct_mask_pos(z: np.ndarray, base_mask: np.ndarray, pct: float):
    vals = z[base_mask]
    pos = vals[vals > 0]
    if pos.size < 10:
        return None, float("nan")
    thr = float(np.percentile(pos, 100 - pct))
    m = np.zeros(z.shape, dtype=bool)
    m[base_mask] = (z[base_mask] > 0) & (z[base_mask] >= thr)
    return m, thr


def _sign_agreement(a: np.ndarray, b: np.ndarray, region_mask: np.ndarray):
    sa = np.sign(a[region_mask])
    sb = np.sign(b[region_mask])
    valid = (sa != 0) & (sb != 0)
    if valid.sum() == 0:
        return float("nan"), 0
    agree = float((sa[valid] == sb[valid]).mean())
    return agree, int(valid.sum())


def _load_nii(path: str):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float64)
    return img, data


def compute_one(
    all_path: str,
    healthy_path: str,
    task: str,
    *,
    top_pct: float = 10.0,
    include_zeros_in_mask: bool = False,
    skip_affine_check: bool = False,
) -> Result:
    A_img, a = _load_nii(all_path)
    B_img, b = _load_nii(healthy_path)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: all={a.shape} vs healthy={b.shape}")

    if (not skip_affine_check) and (not np.allclose(A_img.affine, B_img.affine)):
        raise ValueError("Affine mismatch between the two images. Use --skip_affine_check if intended.")

    finite = np.isfinite(a) & np.isfinite(b)
    if include_zeros_in_mask:
        union_mask = finite
    else:
        union_mask = finite & ((a != 0) | (b != 0))

    n_vox = int(union_mask.sum())
    if n_vox < 10:
        raise ValueError("Too few voxels in analysis mask (<10). Check the images or mask settings.")

    x = a[union_mask].astype(np.float64).ravel()
    y = b[union_mask].astype(np.float64).ravel()
    r = float(pearsonr(x, y)[0])
    fz = _safe_fisher_z(r)

    # (2) top-|Z| overlap
    A_abs, thrA_abs = _top_pct_mask_abs(a, union_mask, top_pct)
    B_abs, thrB_abs = _top_pct_mask_abs(b, union_mask, top_pct)
    a_n_abs, b_n_abs, inter_abs, _uni_abs, dice_abs, jacc_abs = _dice_jacc(A_abs, B_abs)

    # (3) top-positive overlap
    A_pos, thrA_pos = _top_pct_mask_pos(a, union_mask, top_pct)
    B_pos, thrB_pos = _top_pct_mask_pos(b, union_mask, top_pct)
    if (A_pos is None) or (B_pos is None):
        dice_pos = float("nan")
        jacc_pos = float("nan")
        a_n_pos = b_n_pos = inter_pos = 0
    else:
        a_n_pos, b_n_pos, inter_pos, _uni_pos, dice_pos, jacc_pos = _dice_jacc(A_pos, B_pos)

    # (4) sign agreement within top-|Z| region
    top_union = A_abs | B_abs
    top_inter = A_abs & B_abs
    agree_u, n_u = _sign_agreement(a, b, top_union)
    agree_i, n_i = _sign_agreement(a, b, top_inter)

    return Result(
        task=task,
        all_map=os.path.abspath(all_path),
        healthy_map=os.path.abspath(healthy_path),
        n_vox_union_mask=n_vox,
        r=r,
        fisher_z=fz,
        top_pct=float(top_pct),
        thr_abs_all=thrA_abs,
        thr_abs_healthy=thrB_abs,
        dice_abs_top=dice_abs,
        jaccard_abs_top=jacc_abs,
        n_abs_all_top=a_n_abs,
        n_abs_healthy_top=b_n_abs,
        n_abs_overlap_top=inter_abs,
        thr_pos_all=thrA_pos,
        thr_pos_healthy=thrB_pos,
        dice_pos_top=dice_pos,
        jaccard_pos_top=jacc_pos,
        n_pos_all_top=a_n_pos,
        n_pos_healthy_top=b_n_pos,
        n_pos_overlap_top=inter_pos,
        sign_agreement_top_abs_union=agree_u,
        n_sign_valid_top_abs_union=n_u,
        sign_agreement_top_abs_inter=agree_i,
        n_sign_valid_top_abs_inter=n_i,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute SDM-Z similarity metrics (all-sample vs healthy-only).")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs_csv", type=str, help="CSV with columns: task,all_map,healthy_map")
    g.add_argument("--all", dest="all_map", type=str, help="Path to all-sample SDM-Z NIfTI")

    ap.add_argument("--healthy", type=str, default=None, help="Path to healthy-only SDM-Z NIfTI (required if --all is used)")
    ap.add_argument("--task", type=str, default="", help="Task label (e.g., SHIFT, APPRAISAL, IEP)")

    ap.add_argument("--top_pct", type=float, default=10.0, help="Top percentage (default 10 means top 10%%)")
    ap.add_argument("--include_zeros_in_mask", action="store_true", help="Include zero-background voxels in mask")
    ap.add_argument("--skip_affine_check", action="store_true", help="Skip affine equality check")

    ap.add_argument("--out", type=str, required=True, help="Output CSV path")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not (0 < args.top_pct < 100):
        raise SystemExit("--top_pct must be within (0,100)")

    results: List[Result] = []

    if args.pairs_csv:
        df = pd.read_csv(args.pairs_csv)
        required = {"task", "all_map", "healthy_map"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"pairs_csv missing required columns: {sorted(missing)}")

        for _, row in df.iterrows():
            res = compute_one(
                str(row["all_map"]),
                str(row["healthy_map"]),
                task=str(row.get("task", "")),
                top_pct=args.top_pct,
                include_zeros_in_mask=args.include_zeros_in_mask,
                skip_affine_check=args.skip_affine_check,
            )
            results.append(res)
    else:
        if not args.healthy:
            raise SystemExit("When using --all, you must also provide --healthy.")
        res = compute_one(
            args.all_map,
            args.healthy,
            task=args.task,
            top_pct=args.top_pct,
            include_zeros_in_mask=args.include_zeros_in_mask,
            skip_affine_check=args.skip_affine_check,
        )
        results.append(res)

    out_df = pd.DataFrame([asdict(r) for r in results])
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Wrote: {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
