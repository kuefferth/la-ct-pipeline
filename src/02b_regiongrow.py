"""LA refinement v6: TotalSeg LA seed -> region grow -> forbidden-neighbour block ->
distance-from-centroid cap -> thin-vessel removal -> closing.

v6 change: forbidden zone now includes RA + RV + PA (was LV + aorta only).
Seals region-grow leaks into the pulmonary artery and right ventricle.

Pipeline:
  1. Erode TotalSeg LA mask by ERODE_MM -> seed
  2. Adaptive threshold from CT intensities inside seed (percentile-based)
  3. Region grow from seed under that threshold window
  4. Subtract forbidden neighbours (LV, RA raw; aorta, RV, PA dilated)
  5. Keep largest connected component
  6. Cap to voxels within MAX_DIST_FROM_CENTROID_MM of LA seed centroid
  6b. Remove thin PV remnants via opening + geodesic reconstruction
  7. Closing to fill small dents/holes
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
LA_LABEL    = 2   # heart_atrium_left
LV_LABEL    = 3   # heart_ventricle_left
RA_LABEL    = 4   # heart_atrium_right
RV_LABEL    = 5   # heart_ventricle_right
AORTA_LABEL = 6   # aorta
PA_LABEL    = 7   # pulmonary_artery

# === Forbidden neighbours: label -> dilation (mm) ===
# Subtracting these seals leaks across thin shared walls. largest-CC then drops
# whatever the dilated barrier severed.
#   LV  raw : dilation would cut the mitral plane (we want LA-LV interface intact)
#   RA  raw : interatrial septum is thin; dilation would eat into LA
#   aorta   : TotalSeg under-segments the root, pad a bit
#   RV / PA : main leak targets; pad to seal the thin wall the grow jumps
# If PA leaks persist, bump PA to 2.0-2.5, but watch the left superior PV ostium.
FORBIDDEN = {
    LV_LABEL:    0.0,
    RA_LABEL:    0.0,
    AORTA_LABEL: 1.5,
    RV_LABEL:    1.5,
    PA_LABEL:    1.5,
}

# === Tunables ===
ERODE_MM                  = 5.0   # seed erosion before sampling intensities + growing
MAX_DIST_FROM_CENTROID_MM = 60.0  # cap on distance from LA seed centroid (cuts distal PVs)
THIN_OPEN_MM              = 2.5   # opening kernel; erases tubes thinner than ~5mm diameter
CLOSE_MM                  = 1.5   # closing kernel; fills small dents/holes
GLOB                      = "*.nii.gz"

# === Timing helper ===
class T:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *a): print(f"    [time] {self.label}: {time.perf_counter() - self.t0:.2f}s")

# === Morphology helpers ===
def erode_mm(mask, mm):
    """Binary erode by ~`mm` mm using image spacing."""
    spacing = mask.GetSpacing()
    return sitk.BinaryErode(mask, [max(1, int(round(mm / s))) for s in spacing])

def dilate_mm(mask, mm):
    """Binary dilate by ~`mm` mm using image spacing."""
    spacing = mask.GetSpacing()
    return sitk.BinaryDilate(mask, [max(1, int(round(mm / s))) for s in spacing])

def close_mm(mask, mm):
    """Binary closing (dilate then erode) by ~mm. Fills small dents/holes."""
    spacing = mask.GetSpacing()
    r = [max(1, int(round(mm / s))) for s in spacing]
    return sitk.BinaryErode(sitk.BinaryDilate(mask, r), r)

def open_then_reconstruct(mask, mm):
    """Opening (erode -> dilate) at `mm` kernel, then geodesic reconstruction
    under the original mask. Erases tubes thinner than ~2*mm diameter,
    preserves the LA body and thick PV trunks."""
    spacing = mask.GetSpacing()
    r = [max(1, int(round(mm / s))) for s in spacing]
    opened = sitk.BinaryDilate(sitk.BinaryErode(mask, r), r)
    return sitk.BinaryReconstructionByDilation(opened, mask)

# === Threshold from seed intensities ===
def adaptive_threshold(ct, seed_mask):
    """Blood-pool HU window from intensities inside eroded seed. Percentile-based."""
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

# === Forbidden mask builder ===
def build_forbidden(seg):
    """Union of forbidden-neighbour labels, each dilated per FORBIDDEN dict."""
    forbidden = None
    counts = {}
    for label, dmm in FORBIDDEN.items():
        m = sitk.BinaryThreshold(seg, label, label, 1, 0)
        if dmm > 0:
            m = dilate_mm(m, dmm)
        counts[label] = int(sitk.GetArrayFromImage(m).sum())
        forbidden = m if forbidden is None else sitk.Or(forbidden, m)
    return forbidden, counts

# === Distance-from-centroid cap (cuts distal PVs) ===
def distance_from_centroid_cap(mask, seed, max_dist_mm):
    """Cap mask to voxels within max_dist_mm of the seed centroid (mm)."""
    arr = sitk.GetArrayFromImage(seed)
    zs, ys, xs = np.where(arr > 0)
    if len(zs) == 0:
        return mask
    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())
    centroid_phys = np.array(
        seed.TransformContinuousIndexToPhysicalPoint((cx, cy, cz))
    )
    print(f"    seed centroid (mm): "
          f"({centroid_phys[0]:.1f}, {centroid_phys[1]:.1f}, {centroid_phys[2]:.1f})")

    size      = mask.GetSize()              # (x,y,z)
    spacing   = np.array(mask.GetSpacing())
    origin    = np.array(mask.GetOrigin())
    direction = np.array(mask.GetDirection()).reshape(3, 3)
    iz, iy, ix = np.indices((size[2], size[1], size[0]))
    idx_mm = np.stack(
        [ix * spacing[0], iy * spacing[1], iz * spacing[2]], axis=-1
    )
    phys = origin + idx_mm @ direction.T
    dist = np.linalg.norm(phys - centroid_phys, axis=-1)

    keep = (dist <= max_dist_mm).astype(np.uint8)
    keep_img = sitk.GetImageFromArray(keep)
    keep_img.CopyInformation(mask)
    return sitk.And(mask, keep_img)

# === Per-case pipeline ===
def grow_one(case):
    ct_path  = NIFTI_DIR / f"{case}.nii.gz"
    seg_path = SEG_DIR   / f"{case}_chambers.nii.gz"
    if not (ct_path.exists() and seg_path.exists()):
        print(f"[{case}] missing inputs, skip"); return

    with T("read CT + seg"):
        ct  = sitk.ReadImage(str(ct_path))
        seg = sitk.ReadImage(str(seg_path))

    # 1) Seed = TotalSeg LA eroded
    with T("build + erode LA seed"):
        la_full = sitk.BinaryThreshold(seg, LA_LABEL, LA_LABEL, 1, 0)
        la_seed = erode_mm(la_full, ERODE_MM)
    n_seed = int(sitk.GetArrayFromImage(la_seed).sum())
    if n_seed == 0:
        print(f"[{case}] seed empty after erosion, skip"); return
    print(f"  seed voxels: {n_seed}")

    # 2) Threshold
    lo, hi = adaptive_threshold(ct, la_seed)

    # 3) Forbidden zone: LV + RA (raw) + aorta + RV + PA (dilated)
    with T("forbidden mask"):
        forbidden, counts = build_forbidden(seg)
    print("  forbidden voxels: " + "  ".join(f"L{k}={v}" for k, v in counts.items()))

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
    with T("subtract forbidden"):
        grown = sitk.And(grown, sitk.Not(forbidden))
    with T("largest CC #1"):
        grown = largest_component(grown)
    print(f"  after forbidden-block + largest CC: "
          f"{int(sitk.GetArrayFromImage(grown).sum())} voxels")

    # 6) Distance-from-centroid cap (cuts distal PVs)
    with T("distance-from-centroid cap"):
        pruned = distance_from_centroid_cap(grown, la_seed,
                                            MAX_DIST_FROM_CENTROID_MM)
    with T("largest CC #2"):
        pruned = largest_component(pruned)
    print(f"  after distance cap (<{MAX_DIST_FROM_CENTROID_MM}mm from centroid): "
          f"{int(sitk.GetArrayFromImage(pruned).sum())} voxels")

    # 6b) Remove thin PV remnants
    with T(f"thin-vessel removal (open {THIN_OPEN_MM}mm)"):
        pruned = open_then_reconstruct(pruned, THIN_OPEN_MM)
    with T("largest CC after opening"):
        pruned = largest_component(pruned)
    print(f"  after thin-vessel removal: "
          f"{int(sitk.GetArrayFromImage(pruned).sum())} voxels")

    # 7) Closing to fill small dents/holes
    with T("closing"):
        pruned = close_mm(pruned, CLOSE_MM)
    with T("largest CC #3"):
        pruned = largest_component(pruned)

    # 8) Save
    with T("write output"):
        LA_DIR.mkdir(parents=True, exist_ok=True)
        n_final = int(sitk.GetArrayFromImage(pruned).sum())
        vol_ml  = n_final * np.prod(pruned.GetSpacing()) / 1000.0
        sitk.WriteImage(pruned, str(LA_DIR / f"{case}_LA.nii.gz"))
    print(f"  final: {n_final} voxels = {vol_ml:.1f} mL")
    print(f"[{case}] saved")

# === Entry point ===
if __name__ == "__main__":
    t_total = time.perf_counter()
    for nii in sorted(NIFTI_DIR.glob(GLOB)):
        case = nii.name.replace(".nii.gz", "")
        print(f"\n[{case}] region-growing v6...")
        grow_one(case)
    print(f"\n[total] {time.perf_counter() - t_total:.1f}s")