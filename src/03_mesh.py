"""LA binary mask -> surface mesh (VTK + STL).

v2 change: pad mask with a background border before marching cubes so masks
that touch the CT FOV border get a closed (flat) cap instead of an open hole.

Per case:
  derivatives/seg_la/<case>_LA.nii.gz  (final refined mask, NOT _LA_seed)
    -> pad -> marching cubes -> largest CC -> Taubin smoothing -> decimate
    -> derivatives/meshes/<case>_LA.{vtk,stl}   (decimated; full-res no longer written)

Decimated is the ONLY output: it is visually indistinguishable from full-res here
and the full-res files were large. So <case>_LA.{vtk,stl} IS the decimated mesh.
"""
import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pyvista as pv

LA_DIR   = Path("derivatives/seg_la")
MESH_DIR = Path("derivatives/meshes")

# === Tunables ===
PAD_VOXELS        = 4       # background border added on every side; closes border-touching caps
GAUSS_VAR         = 0.7     # mask pre-smooth variance (mm^2). VALIDATED g0.7_t60 on the
                            # smooth sweep (854/540/2643): g1.0/t100 was over-smoothed vs
                            # the crisp masks; g0.7/t60 keeps PV-ostia/carina detail. The
                            # Gaussian blur is the detail-eater + fusion risk -- re-check
                            # the 540 carina on the batch; drop toward 0.5 if any fuses.
TAUBIN_ITER       = 60      # smoothing iterations; more = smoother. SAFE lever (volume-
                            # preserving, cannot re-fuse). Lowered 100 -> 60 with GAUSS 0.7.
TAUBIN_PASS_BAND  = 0.05    # 0.01-0.2 range; lower = stronger smoothing
DECIMATE_FRAC     = 0.7     # 0.7 = remove 70% of triangles (keep 30%)
GLOB              = "*_LA.nii.gz"   # final masks only; ignores *_LA_seed.nii.gz
SKIP_EXISTING     = True    # skip cases whose full VTK + STL already exist. Set False to
                            # force a full re-mesh (e.g. after a global 02b threshold change
                            # that overwrites every mask). Passing explicit case args also
                            # forces a re-mesh regardless of this flag.

# === Tiny timing helper ===
class T:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *a): print(f"    [time] {self.label}: {time.perf_counter() - self.t0:.2f}s")

def mask_to_mesh(mask_path: Path, case: str):
    with T("read mask"):
        img = sitk.ReadImage(str(mask_path))

    # Reorient to LPS canonical so the direction matrix becomes identity.
    # pv.ImageData below is axis-aligned (origin + spacing only) and ignores the
    # direction cosines, so any non-identity direction would mirror the mesh.
    # Bern data is already LPS-identity (no-op); public datasets (e.g. ImageCAS,
    # stored L-A-S with a negative-determinant affine) need this or the mesh comes
    # out chirality-flipped (LAA on the wrong side). DICOMOrient keeps world coords.
    img = sitk.DICOMOrient(img, "LPS")

    # Pad with a background (0) border. Any foreground voxel that sat on the
    # volume edge now gets a 0 neighbour, so marching cubes produces a 0.5
    # crossing there and caps the surface. ConstantPad updates the origin, so
    # the pyvista grid below stays in correct world coordinates.
    with T("pad border"):
        p = [PAD_VOXELS] * 3
        img = sitk.ConstantPad(img, p, p, 0)

    # Smooth the binary mask slightly so marching cubes gets sub-voxel detail
    # instead of a staircase. Gaussian variance in mm^2.
    with T("mask smoothing"):
        img_f = sitk.Cast(img, sitk.sitkFloat32)
        img_f = sitk.DiscreteGaussian(img_f, variance=[GAUSS_VAR, GAUSS_VAR, GAUSS_VAR])
    arr = sitk.GetArrayFromImage(img_f)   # shape (z, y, x), float
    spacing = img.GetSpacing()          # (x, y, z) mm
    origin  = img.GetOrigin()           # (x, y, z) world position of first voxel

    # Build pyvista grid in patient/world coords.
    # point_data + dimensions = arr.shape so contour iso=0.5 lands at the voxel boundary.
    with T("build grid"):
        grid = pv.ImageData(
            dimensions=np.array(arr.shape[::-1]),   # (nx, ny, nz)
            spacing=spacing,
            origin=origin,
        )
        # VTK is x-fastest, numpy is z-fastest. Transpose (z,y,x) -> (x,y,z) then ravel F-order.
        grid.point_data["mask"] = arr.transpose(2, 1, 0).astype(np.float32).ravel(order="F")

    # Marching cubes at iso=0.5 (binary mask)
    with T("marching cubes"):
        surf = grid.contour([0.5], scalars="mask", method="marching_cubes")

    # Largest connected component (drops floating speckle)
    with T("largest CC"):
        surf = surf.connectivity(extraction_mode="largest")

    # Taubin smoothing (preserves volume better than Laplacian)
    with T("Taubin smoothing"):
        surf = surf.smooth_taubin(n_iter=TAUBIN_ITER, pass_band=TAUBIN_PASS_BAND)

    # Make sure normals are consistent and outward-pointing
    with T("compute normals"):
        surf = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)

    # Decimate (target_reduction = fraction of triangles to REMOVE) and save as the
    # single output. Full-res is no longer written; <case>_LA.{vtk,stl} IS decimated.
    with T("decimate + save"):
        dec = surf.decimate(DECIMATE_FRAC)
        dec.save(str(MESH_DIR / f"{case}_LA.vtk"))
        dec.save(str(MESH_DIR / f"{case}_LA.stl"))

    # Report (volume from the decimated mesh that ships)
    vol_ml = dec.volume / 1000.0   # pyvista returns mm^3; convert to mL
    print(f"[{case}] mesh (decimated): {dec.n_points} pts, {dec.n_cells} tris  | vol={vol_ml:.1f} mL")

if __name__ == "__main__":
    import sys
    cases_filter = set(sys.argv[1:])   # optional: pass case stems to re-mesh only those
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()
    for mask_path in sorted(LA_DIR.glob(GLOB)):
        if mask_path.name.endswith("_LA_seed.nii.gz"):
            continue
        case = mask_path.name.replace("_LA.nii.gz", "")
        if cases_filter and case not in cases_filter:
            continue
        out_vtk = MESH_DIR / f"{case}_LA.vtk"
        out_stl = MESH_DIR / f"{case}_LA.stl"
        if SKIP_EXISTING and not cases_filter and out_vtk.exists() and out_stl.exists():
            print(f"[{case}] mesh exists, skip")
            continue
        print(f"\n[{case}] meshing...")
        mask_to_mesh(mask_path, case)
    print(f"\n[total] {time.perf_counter() - t_total:.1f}s")