from pathlib import Path
import numpy as np
import nibabel as nib

# ====== 改成你的两张“未阈值 SDM-Z 图” ======
A_PATH = Path(r"D:\SDM_overlap\r_F_top10%\all1_MyTest_z.nii.gz")          # all-sample
B_PATH = Path(r"D:\SDM_overlap\r_F_top10%\health1_MyTest_z.nii.gz")               # healthy-only
# ============================================
TOP_PCT = 10  # Top 10%

def load_data(p: Path):
    img = nib.load(str(p))
    return img, img.get_fdata()

def pearson_r(a, b, mask):
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if x.size < 10 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def dice_jacc(a_mask, b_mask):
    inter = int(np.logical_and(a_mask, b_mask).sum())
    union = int(np.logical_or(a_mask, b_mask).sum())
    a_n = int(a_mask.sum())
    b_n = int(b_mask.sum())
    dice = (2 * inter) / (a_n + b_n) if (a_n + b_n) > 0 else float("nan")
    jacc = inter / union if union > 0 else float("nan")
    return a_n, b_n, inter, union, float(dice), float(jacc)

def top_pct_mask_abs(z, base_mask, pct):
    vals = np.abs(z[base_mask])
    thr = np.percentile(vals, 100 - pct)
    m = np.zeros(z.shape, dtype=bool)
    m[base_mask] = np.abs(z[base_mask]) >= thr
    return m, float(thr)

def top_pct_mask_pos(z, base_mask, pct):
    vals = z[base_mask]
    pos = vals[vals > 0]
    if pos.size < 10:  # 太少就别算了
        return None, float("nan")
    thr = np.percentile(pos, 100 - pct)
    m = np.zeros(z.shape, dtype=bool)
    m[base_mask] = (z[base_mask] > 0) & (z[base_mask] >= thr)
    return m, float(thr)

def sign_agreement(a, b, region_mask):
    # 只在 region_mask 内计算符号一致率；排除 a 或 b 为 0 的体素
    sa = np.sign(a[region_mask])
    sb = np.sign(b[region_mask])
    valid = (sa != 0) & (sb != 0)
    if valid.sum() == 0:
        return float("nan"), 0
    agree = (sa[valid] == sb[valid]).mean()
    return float(agree), int(valid.sum())

def main():
    print("A exists:", A_PATH.exists(), A_PATH)
    print("B exists:", B_PATH.exists(), B_PATH)
    assert A_PATH.exists(), f"Missing: {A_PATH}"
    assert B_PATH.exists(), f"Missing: {B_PATH}"

    A_img, a = load_data(A_PATH)
    B_img, b = load_data(B_PATH)

    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert np.allclose(A_img.affine, B_img.affine), "Affine mismatch"

    # union mask：至少一张非零（避免脑外全 0 抬高相关）
    union_mask = np.isfinite(a) & np.isfinite(b) & ((a != 0) | (b != 0))
    N = int(union_mask.sum())

    # (1) signed Pearson r
    r = pearson_r(a, b, union_mask)
    fz = float(np.arctanh(r)) if (not np.isnan(r) and abs(r) < 1) else (float("inf") if r == 1 else float("-inf") if r == -1 else float("nan"))

    print("\n=== (1) Unthresholded similarity (signed) ===")
    print("N vox used:", N)
    print("Pearson r :", r)
    print("Fisher z  :", fz)

    # (2) Top10% |Z| overlap
    A_abs, thrA_abs = top_pct_mask_abs(a, union_mask, TOP_PCT)
    B_abs, thrB_abs = top_pct_mask_abs(b, union_mask, TOP_PCT)
    a_n, b_n, inter, uni, dice, jacc = dice_jacc(A_abs, B_abs)

    print(f"\n=== (2) Top {TOP_PCT}% overlap by |Z| (magnitude, pos+neg) ===")
    print("A |Z| threshold:", thrA_abs, "| A vox:", a_n)
    print("B |Z| threshold:", thrB_abs, "| B vox:", b_n)
    print("Intersection:", inter, "Union:", uni)
    print("Dice:", dice, "Jaccard:", jacc)

    # (3) Top10% positive Z overlap
    A_pos, thrA_pos = top_pct_mask_pos(a, union_mask, TOP_PCT)
    B_pos, thrB_pos = top_pct_mask_pos(b, union_mask, TOP_PCT)

    print(f"\n=== (3) Top {TOP_PCT}% overlap by positive Z only (direction-consistent with 'positive-only' reporting) ===")
    if (A_pos is None) or (B_pos is None):
        print("Not enough positive voxels in one map; skipping positive-only overlap.")
    else:
        a_n, b_n, inter, uni, dice_pos, jacc_pos = dice_jacc(A_pos, B_pos)
        print("A +Z threshold:", thrA_pos, "| A vox:", a_n)
        print("B +Z threshold:", thrB_pos, "| B vox:", b_n)
        print("Intersection:", inter, "Union:", uni)
        print("Dice:", dice_pos, "Jaccard:", jacc_pos)

    # (4) Sign agreement within Top10% |Z|
    top_union = A_abs | B_abs
    top_inter = A_abs & B_abs
    agree_u, n_u = sign_agreement(a, b, top_union)
    agree_i, n_i = sign_agreement(a, b, top_inter)

    print(f"\n=== (4) Sign agreement within Top {TOP_PCT}% |Z| region ===")
    print("Within union(top10%): sign-agreement =", agree_u, "| N valid vox:", n_u)
    print("Within inter(top10%): sign-agreement =", agree_i, "| N valid vox:", n_i)

if __name__ == "__main__":
    main()