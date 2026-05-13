"""LA refinement v4: TotalSeg LA seed -> region grow -> LV block -> distance-from-centroid cap.

Pipeline:
  1. Erode TotalSeg LA mask by ERODE_MM -> seed
  2. Adaptive threshold from CT intensities inside seed (percentile-based)
  3. Region grow from seed under that threshold window
  4. Subtract LV mask (raw, no dilation -> preserves mitral plane)
  5. Keep largest connected component
  6. Cap to voxels within MAX_DIST_FROM_CENTROID_MM of LA seed centroid
     (cuts distal PV branches; radius-based prune doesn't work because LA body
     can be as thin as PV trunks)
"""

import os
os.environ["nnUNet_n_proc_DA"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# === Paths ===
NIFTI_DIR = Path("derivatives/nifti")
SEG_DIR   = Path("derivatives/seg_full")
LA_DIR    = Path("derivatives/seg_la")

# === Label indices in heartchambers_highres ===
LA_LABEL = 2   # heart_atrium_left
LV_LABEL = 3   # heart_ventricle_left

# === Tunables ===
ERODE_MM                  = 5.0   # seed erosion before sampling intensities + growing
LV_DILATE_MM              = 0.0   # raw LV (dilation cuts mitral plane)
MAX_DIST_FROM_CENTROID_MM = 60.0  # cap on distance from LA seed centroid; tune to taste
GLOB                      = "4733.nii.gz"

# === Tiny timing helper ===
class T:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *args):
        print(f"    [time] {self.label}: {time.perf_counter() - self.t0:.2f}s")

# === Morphology helpers ===
def erode_mm(mask, mm):
    """Binary erode by ~`mm` mm using image spacing."""
    spacing = mask.GetSpacing()
    return sitk.BinaryErode(mask, [max(1, int(round(mm / s))) for s in spacing])

def dilate_mm(mask, mm):
    """Binary dilate by ~`mm` mm using image spacing."""
    spacing = mask.GetSpacing()
    return sitk.BinaryDilate(mask, [max(1, int(round(mm / s))) for s in spacing])

# === Threshold from seed intensities ===
def adaptive_threshold(ct, seed_mask):
    """Estimate blood-pool HU window from intensities inside eroded seed.
    Percentile-based; robust to outliers and very bright contrast."""
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
    """Keep only the largest connected component."""
    cc  = sitk.ConnectedComponent(mask)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.BinaryThreshold(rel, 1, 1, 1, 0)

# === Distance-from-centroid cap (cuts distal PVs) ===
def distance_from_centroid_cap(mask, seed, max_dist_mm):
    """Cap mask to voxels within max_dist_mm of the seed centroid (in mm)."""
    # Centroid of seed in voxel index space
    arr = sitk.GetArrayFromImage(seed)
    zs, ys, xs = np.where(arr > 0)
    if len(zs) == 0:
        return mask
    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())  # SITK uses (x,y,z)
    centroid_phys = np.array(
        seed.TransformContinuousIndexToPhysicalPoint((cx, cy, cz))
    )
    print(f"    seed centroid (mm): "
          f"({centroid_phys[0]:.1f}, {centroid_phys[1]:.1f}, {centroid_phys[2]:.1f})")

    # Build per-voxel distance to centroid in physical mm
    size      = mask.GetSize()              # (x,y,z)
    spacing   = np.array(mask.GetSpacing()) # (x,y,z)
    origin    = np.array(mask.GetOrigin())
    direction = np.array(mask.GetDirection()).reshape(3, 3)
    # Voxel index grid in array order (z,y,x)
    iz, iy, ix = np.indices((size[2], size[1], size[0]))
    # Convert voxel idx -> physical coords: phys = origin + R @ (idx * spacing)
    idx_mm = np.stack(
        [ix * spacing[0], iy * spacing[1], iz * spacing[2]], axis=-1
    )  # shape (z,y,x,3)
    phys = origin + idx_mm @ direction.T
    dist = np.linalg.norm(phys - centroid_phys, axis=-1)  # (z,y,x)

    keep = (dist <= max_dist_mm).astype(np.uint8)
    keep_img = sitk.GetImageFromArray(keep)
    keep_img.CopyInformation(mask)
    return sitk.And(mask, keep_img)

# === Per-case ===
def grow_one(case):
    ct_path  = NIFTI_DIR / f"{case}.nii.gz"
    seg_path = SEG_DIR   / f"{case}_chambers.nii.gz"
    if not (ct_path.exists() and seg_path.exists()):
        print(f"[{case}] missing inputs, skip"); return

    with T("read CT + seg"):
        ct  = sitk.ReadImage(str(ct_path))
        seg = sitk.ReadImage(str(seg_path))

    # 1) Seed
    with T("build + erode LA seed"):
        la_full = sitk.BinaryThreshold(seg, LA_LABEL, LA_LABEL, 1, 0)
        la_seed = erode_mm(la_full, ERODE_MM)
    n_seed = int(sitk.GetArrayFromImage(la_seed).sum())
    if n_seed == 0:
        print(f"[{case}] seed empty after erosion, skip"); return
    print(f"  seed voxels: {n_seed}")

    # 2) Threshold
    lo, hi = adaptive_threshold(ct, la_seed)

    # 3) LV mask (raw to preserve mitral plane)
    with T("LV mask"):
        lv = sitk.BinaryThreshold(seg, LV_LABEL, LV_LABEL, 1, 0)
        lv_block = lv if LV_DILATE_MM <= 0 else dilate_mm(lv, LV_DILATE_MM)
    print(f"  LV block voxels (dilated {LV_DILATE_MM}mm): "
          f"{int(sitk.GetArrayFromImage(lv_block).sum())}")

    # 4) Seed list for ConnectedThreshold (subsample for speed; result identical)
    with T("seed list"):
        seed_arr = sitk.GetArrayFromImage(la_seed)
        zs, ys, xs = np.where(seed_arr > 0)
        rng = np.random.default_rng(0)
        if len(zs) > 1000:
            idx = rng.choice(len(zs), 1000, replace=False)
            zs, ys, xs = zs[idx], ys[idx], xs[idx]
        seed_list = [(int(x), int(y), int(z)) for x, y, z in zip(xs, ys, zs)]

    # 5) Region grow
    with T("region grow"):
        grown = sitk.ConnectedThreshold(ct, seedList=seed_list,
                                        lower=float(lo), upper=float(hi),
                                        replaceValue=1)
    with T("subtract LV"):
        grown = sitk.And(grown, sitk.Not(lv_block))
    with T("largest CC #1"):
        grown = largest_component(grown)
    print(f"  after LV-block + largest CC: "
          f"{int(sitk.GetArrayFromImage(grown).sum())} voxels")

    # 6) Distance-from-centroid cap (cuts distal PVs)
    with T("distance-from-centroid cap"):
        pruned = distance_from_centroid_cap(grown, la_seed,
                                            MAX_DIST_FROM_CENTROID_MM)
    with T("largest CC #2"):
        pruned = largest_component(pruned)

    # 7) Save
    with T("write output"):
        LA_DIR.mkdir(parents=True, exist_ok=True)
        n_final = int(sitk.GetArrayFromImage(pruned).sum())
        vol_ml  = n_final * np.prod(pruned.GetSpacing()) / 1000.0
        sitk.WriteImage(pruned, str(LA_DIR / f"{case}_LA.nii.gz"))
    print(f"  after distance cap (<{MAX_DIST_FROM_CENTROID_MM}mm from centroid): "
          f"{n_final} voxels = {vol_ml:.1f} mL")
    print(f"[{case}] saved")

# === Entry point ===
if __name__ == "__main__":
    t_total = time.perf_counter()
    for nii in sorted(NIFTI_DIR.glob(GLOB)):
        case = nii.name.replace(".nii.gz", "")
        print(f"\n[{case}] region-growing v4...")
        grow_one(case)
    print(f"\n[total] {time.perf_counter() - t_total:.1f}s")