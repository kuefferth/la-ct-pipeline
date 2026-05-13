"""LA refinement via region-growing from eroded TotalSegmentator LA seed.
v1: adaptive threshold, no forbidden mask, largest CC.
Inputs:  derivatives/nifti/<case>.nii.gz
         derivatives/seg_full/<case>_chambers.nii.gz   (heartchambers_highres, label 2 = LA)
Output:  derivatives/seg_la/<case>_LA.nii.gz   (overwrites the prior TotalSeg-derived mask)
"""

import os
os.environ["nnUNet_n_proc_DA"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
import SimpleITK as sitk
import numpy as np

NIFTI_DIR = Path("derivatives/nifti")
SEG_DIR   = Path("derivatives/seg_full")
LA_DIR    = Path("derivatives/seg_la")

LA_LABEL = 2          # heartchambers_highres: heart_atrium_left
ERODE_MM = 5.0        # erode seed by 5 mm before sampling intensities + growing
GLOB     = "4733.nii.gz"   # one case first; switch to "*.nii.gz" later

def erode_mm(mask: sitk.Image, mm: float) -> sitk.Image:
    """Binary erode by `mm` (uses image spacing). Returns binary mask."""
    spacing = mask.GetSpacing()                                # (x,y,z) mm
    radius_vox = [max(1, int(round(mm / s))) for s in spacing] # per-axis voxel radii
    return sitk.BinaryErode(mask, radius_vox)

def adaptive_threshold(ct: sitk.Image, seed_mask: sitk.Image, k: float = 3.0):
    """Estimate threshold range from CT intensities inside the eroded LA seed.
    Returns (lo, hi) clipped to safe HU bounds for contrast-enhanced blood pool.
    """
    ct_arr   = sitk.GetArrayFromImage(ct).astype(np.float32)
    seed_arr = sitk.GetArrayFromImage(seed_mask).astype(bool)
    vals = ct_arr[seed_arr]
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    # k*sd window around mean; clip so we don't grab fat (<0) or bone (>800)
    lo = max(80.0,  mu - k * sd)
    hi = min(800.0, mu + k * sd)
    print(f"    seed HU: mean={mu:.0f}  sd={sd:.0f}  -> threshold [{lo:.0f}, {hi:.0f}]")
    return lo, hi

def largest_component(mask: sitk.Image) -> sitk.Image:
    """Keep only the largest connected component of a binary mask."""
    cc = sitk.ConnectedComponent(mask)
    # RelabelComponent sorts by size; label 1 is largest
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.BinaryThreshold(rel, lowerThreshold=1, upperThreshold=1,
                                insideValue=1, outsideValue=0)

def grow_one(case: str):
    ct_path  = NIFTI_DIR / f"{case}.nii.gz"
    seg_path = SEG_DIR   / f"{case}_chambers.nii.gz"
    if not (ct_path.exists() and seg_path.exists()):
        print(f"[{case}] missing inputs, skip"); return

    ct  = sitk.ReadImage(str(ct_path))
    seg = sitk.ReadImage(str(seg_path))
    # Sanity: same geometry (TotalSeg writes in CT space)
    assert ct.GetSize() == seg.GetSize(), f"size mismatch for {case}"

    # 1) Build LA seed = (TotalSeg LA) eroded by ERODE_MM
    la_full = sitk.BinaryThreshold(seg, lowerThreshold=LA_LABEL, upperThreshold=LA_LABEL,
                                   insideValue=1, outsideValue=0)
    la_seed = erode_mm(la_full, ERODE_MM)
    n_seed  = int(sitk.GetArrayFromImage(la_seed).sum())
    if n_seed == 0:
        print(f"[{case}] seed empty after erosion, skip"); return
    print(f"  seed voxels: {n_seed}")

    # 2) Adaptive threshold from CT intensities under seed
    lo, hi = adaptive_threshold(ct, la_seed)

    # 3) Convert seed voxels -> list of (x,y,z) indices for SITK ConnectedThreshold
    seed_arr = sitk.GetArrayFromImage(la_seed)               # shape (z,y,x)
    zs, ys, xs = np.where(seed_arr > 0)
    # Subsample seeds for speed (1000 random seeds plenty; growth is identical)
    rng = np.random.default_rng(0)
    if len(zs) > 1000:
        idx = rng.choice(len(zs), 1000, replace=False)
        zs, ys, xs = zs[idx], ys[idx], xs[idx]
    seed_list = [(int(x), int(y), int(z)) for x, y, z in zip(xs, ys, zs)]

    # 4) Region grow: include voxels in [lo, hi] connected to seeds
    grown = sitk.ConnectedThreshold(
        image1=ct, seedList=seed_list,
        lower=float(lo), upper=float(hi),
        replaceValue=1,
    )

    # 5) Largest connected component (drops noise islands)
    grown = largest_component(grown)

    # 6) Save
    LA_DIR.mkdir(parents=True, exist_ok=True)
    out = LA_DIR / f"{case}_LA.nii.gz"
    sitk.WriteImage(grown, str(out))
    n_out = int(sitk.GetArrayFromImage(grown).sum())
    vol_ml = n_out * np.prod(grown.GetSpacing()) / 1000.0
    print(f"[{case}] wrote {out.name}  voxels={n_out}  vol={vol_ml:.1f} mL")

if __name__ == "__main__":
    for nii in sorted(NIFTI_DIR.glob(GLOB)):
        case = nii.name.replace(".nii.gz", "")
        print(f"\n[{case}] region-growing...")
        grow_one(case)