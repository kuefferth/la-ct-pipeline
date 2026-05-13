"""LA refinement v2: TotalSeg LA seed -> region grow -> LV block -> radius-prune PVs.

Pipeline:
  1. Erode TotalSeg LA mask by ERODE_MM -> seed
  2. Adaptive threshold from CT intensities inside seed (percentile-based)
  3. Region grow from seed under that threshold window
  4. Subtract dilated LV mask (forbidden zone)
  5. Keep largest connected component
  6. Radius-prune: drop voxels with local radius < MIN_RADIUS_MM,
     then reconstruct full branches under original mask
     (cuts PV branches thinner than ~2*MIN_RADIUS_MM diameter)
"""

import os
os.environ["nnUNet_n_proc_DA"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
import SimpleITK as sitk
import numpy as np

# === Paths ===
NIFTI_DIR = Path("derivatives/nifti")
SEG_DIR   = Path("derivatives/seg_full")
LA_DIR    = Path("derivatives/seg_la")

# === Label indices in heartchambers_highres multi-label output ===
LA_LABEL  = 2          # heart_atrium_left
LV_LABEL  = 3          # heart_ventricle_left

# === Tunable parameters ===
ERODE_MM        = 5.0  # seed erosion before sampling intensities + growing
LV_DILATE_MM    = 2.0  # safety margin around LV before subtraction
MIN_RADIUS_MM   = 2.5  # voxels with local radius < this get pruned (~5 mm diameter cutoff)

# Glob for which cases to process. "4733.nii.gz" = single case; "*.nii.gz" = all
GLOB = "4733.nii.gz"

# === Morphology helpers ===
def erode_mm(mask, mm):
    """Binary erode by ~`mm` millimeters using image spacing."""
    spacing = mask.GetSpacing()
    return sitk.BinaryErode(mask, [max(1, int(round(mm / s))) for s in spacing])

def dilate_mm(mask, mm):
    """Binary dilate by ~`mm` millimeters using image spacing."""
    spacing = mask.GetSpacing()
    return sitk.BinaryDilate(mask, [max(1, int(round(mm / s))) for s in spacing])

# === Threshold estimation ===
def adaptive_threshold(ct, seed_mask):
    """Estimate blood-pool HU window from intensities inside eroded seed.
    Uses percentiles (robust to outliers), with generous cushions and a high cap
    because cardiac CT blood pool can hit 900+ HU.
    """
    ct_arr   = sitk.GetArrayFromImage(ct).astype(np.float32)
    seed_arr = sitk.GetArrayFromImage(seed_mask).astype(bool)
    vals = ct_arr[seed_arr]
    p2, p98 = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    lo = max(120.0,  p2  - 50.0)
    hi = min(1500.0, p98 + 150.0)
    print(f"    seed HU: p2={p2:.0f}  p98={p98:.0f}  -> threshold [{lo:.0f}, {hi:.0f}]")
    return lo, hi

# === Connected components ===
def largest_component(mask):
    """Keep only the largest connected component of a binary mask."""
    cc  = sitk.ConnectedComponent(mask)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.BinaryThreshold(rel, 1, 1, 1, 0)

# === Radius-based pruning of thin branches ===
def radius_prune(mask, min_radius_mm):
    """Drop voxels whose distance to nearest mask boundary < min_radius_mm,
    then reconstruct full branches under the original mask in one call.
    This cuts thin PV branches but preserves LA body + thick PV trunks.
    """
    # Distance map: inside positive, in mm
    dist = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=True,
                                       squaredDistance=False, useImageSpacing=True)
    # "Core" = thick interior region
    core = sitk.BinaryThreshold(dist, lowerThreshold=min_radius_mm,
                                upperThreshold=1e6, insideValue=1, outsideValue=0)
    # Geodesic reconstruction by dilation: regrow core through the mask.
    # Single call; orders of magnitude faster than iterative dilate+intersect.
    return sitk.BinaryReconstructionByDilation(core, mask)

# === Main per-case ===
def grow_one(case):
    ct_path  = NIFTI_DIR / f"{case}.nii.gz"
    seg_path = SEG_DIR   / f"{case}_chambers.nii.gz"
    if not (ct_path.exists() and seg_path.exists()):
        print(f"[{case}] missing inputs, skip"); return

    ct  = sitk.ReadImage(str(ct_path))
    seg = sitk.ReadImage(str(seg_path))

    # 1) Build LA seed
    la_full = sitk.BinaryThreshold(seg, LA_LABEL, LA_LABEL, 1, 0)
    la_seed = erode_mm(la_full, ERODE_MM)
    n_seed  = int(sitk.GetArrayFromImage(la_seed).sum())
    if n_seed == 0:
        print(f"[{case}] seed empty after erosion, skip"); return
    print(f"  seed voxels: {n_seed}")

    # 2) Threshold window from seed intensities
    lo, hi = adaptive_threshold(ct, la_seed)

    # 3) LV forbidden mask
    lv = sitk.BinaryThreshold(seg, LV_LABEL, LV_LABEL, 1, 0)
    lv_block = dilate_mm(lv, LV_DILATE_MM)
    print(f"  LV block voxels (dilated {LV_DILATE_MM}mm): {int(sitk.GetArrayFromImage(lv_block).sum())}")

    # 4) Build seed list for ConnectedThreshold (subsample for speed; result identical)
    seed_arr = sitk.GetArrayFromImage(la_seed)
    zs, ys, xs = np.where(seed_arr > 0)
    rng = np.random.default_rng(0)
    if len(zs) > 1000:
        idx = rng.choice(len(zs), 1000, replace=False)
        zs, ys, xs = zs[idx], ys[idx], xs[idx]
    seed_list = [(int(x), int(y), int(z)) for x, y, z in zip(xs, ys, zs)]

    # 5) Region grow
    grown = sitk.ConnectedThreshold(ct, seedList=seed_list,
                                    lower=float(lo), upper=float(hi), replaceValue=1)

    # 6) Subtract LV block
    grown = sitk.And(grown, sitk.Not(lv_block))

    # 7) Largest connected component (removes speckle)
    grown = largest_component(grown)
    print(f"  after LV-block + largest CC: {int(sitk.GetArrayFromImage(grown).sum())} voxels")

    # 8) Radius prune (drops thin PV branches)
    pruned = radius_prune(grown, MIN_RADIUS_MM)
    pruned = largest_component(pruned)  # safety: ensure single component after prune

    # 9) Save
    LA_DIR.mkdir(parents=True, exist_ok=True)
    n_final = int(sitk.GetArrayFromImage(pruned).sum())
    vol_ml  = n_final * np.prod(pruned.GetSpacing()) / 1000.0
    sitk.WriteImage(pruned, str(LA_DIR / f"{case}_LA.nii.gz"))
    print(f"  after radius prune (>{MIN_RADIUS_MM}mm radius): {n_final} voxels = {vol_ml:.1f} mL")
    print(f"[{case}] saved")

# === Entry point (Windows multiprocessing safety) ===
if __name__ == "__main__":
    for nii in sorted(NIFTI_DIR.glob(GLOB)):
        case = nii.name.replace(".nii.gz", "")
        print(f"\n[{case}] region-growing v2...")
        grow_one(case)