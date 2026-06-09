"""Smoothing sweep for QC: render a few cases at several smoothing levels so we
can eyeball crispness vs the (crisp) segmentation in Slicer.

The meshes looked over-smoothed vs the masks. Two knobs drive it:
  GAUSS_VAR    : Gaussian blur of the binary mask BEFORE marching cubes. Rounds
                 the staircase but also eats genuine detail (and can fuse carinas).
  TAUBIN_ITER  : surface smoothing iterations after marching cubes. Volume-
                 preserving, cannot re-fuse, but high counts wash out fine detail.

This sweep separates them: the g0.0_* row isolates Taubin (no blur); the
g*_t* rows add blur. Outputs FULL-RES STL (no decimation) so decimation doesn't
confound the comparison -- pick a level here, then we bake it into 03_mesh.py.

Output: derivatives/_scrap/smooth_test/<case>_<config>.stl   (gitignored)
Run:    python src/_smooth_sweep.py [case_stem ...]
"""
import sys
import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pyvista as pv

LA_DIR  = Path("derivatives/seg_la")
OUT_DIR = Path("derivatives/_scrap/smooth_test")
PAD_VOXELS = 4   # same border pad as 03_mesh.py (keeps geometry identical)

# Default cases: crisp pcct, varied size (large / PV-heavy / smaller-high-contrast).
DEFAULT_CASES = [
    "854_0___do92008832__pcct",
    "540_0___do92008590__pcct",
    "2643_0___do92008629__pcct",
]

# (name, GAUSS_VAR mm^2, TAUBIN_ITER, TAUBIN_PASS_BAND)
CONFIGS = [
    ("g0.0_t0",   0.0,   0, 0.05),   # rawest: pure marching cubes
    ("g0.0_t30",  0.0,  30, 0.05),   # Taubin only, light
    ("g0.0_t100", 0.0, 100, 0.05),   # Taubin only, current iters (no blur)
    ("g0.5_t30",  0.5,  30, 0.05),   # light blur + light Taubin
    ("g0.7_t60",  0.7,  60, 0.05),   # between g0.5_t30 and current (lighter)
    ("g0.8_t80",  0.8,  80, 0.05),   # between g0.5_t30 and current (user candidate)
    ("g1.0_t100", 1.0, 100, 0.05),   # CURRENT production
]


def build_surface(mask_path: Path, gauss_var: float, taubin_iter: int, pass_band: float):
    """Replicates 03_mesh.mask_to_mesh up to the smoothing, with tunable knobs."""
    img = sitk.ReadImage(str(mask_path))
    p = [PAD_VOXELS] * 3
    img = sitk.ConstantPad(img, p, p, 0)

    img_f = sitk.Cast(img, sitk.sitkFloat32)
    if gauss_var > 0:
        img_f = sitk.DiscreteGaussian(img_f, variance=[gauss_var] * 3)
    arr = sitk.GetArrayFromImage(img_f)
    spacing = img.GetSpacing()
    origin  = img.GetOrigin()

    grid = pv.ImageData(dimensions=np.array(arr.shape[::-1]), spacing=spacing, origin=origin)
    grid.point_data["mask"] = arr.transpose(2, 1, 0).astype(np.float32).ravel(order="F")
    surf = grid.contour([0.5], scalars="mask", method="marching_cubes")
    surf = surf.connectivity(extraction_mode="largest")
    if taubin_iter > 0:
        surf = surf.smooth_taubin(n_iter=taubin_iter, pass_band=pass_band)
    surf = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)
    return surf


if __name__ == "__main__":
    cases = sys.argv[1:] or DEFAULT_CASES
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for case in cases:
        mask_path = LA_DIR / f"{case}_LA.nii.gz"
        if not mask_path.exists():
            print(f"[{case}] mask missing: {mask_path}, skip")
            continue
        for name, gv, ti, pb in CONFIGS:
            out = OUT_DIR / f"{case}_{name}.stl"
            if out.exists():
                print(f"[{case}] {name:>10}: exists, skip")
                continue
            surf = build_surface(mask_path, gv, ti, pb)
            surf.save(str(out))
            print(f"[{case}] {name:>10}: {surf.n_points} pts, {surf.n_cells} tris -> {out.name}")
    print(f"\n[total] {time.perf_counter() - t0:.1f}s   files in {OUT_DIR}")
