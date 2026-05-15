"""LA binary mask -> surface mesh (VTK + STL).

Per case:
  derivatives/seg_la/<case>_LA.nii.gz  (final refined mask, NOT _LA_seed)
    -> marching cubes -> largest CC -> Taubin smoothing
    -> save full + decimated meshes
    -> derivatives/meshes/<case>_LA.{vtk,stl}, <case>_LA_decimated.stl
"""
import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pyvista as pv

LA_DIR   = Path("derivatives/seg_la")
MESH_DIR = Path("derivatives/meshes")

# === Tunables ===
TAUBIN_ITER       = 50      # smoothing iterations; more = smoother, can over-smooth
TAUBIN_PASS_BAND  = 0.05     # 0.01-0.2 range; lower = stronger smoothing
DECIMATE_FRAC     = 0.7     # 0.7 = remove 70% of triangles (keep 30%)
GLOB              = "*_LA.nii.gz"   # final masks only; ignores *_LA_seed.nii.gz

# === Tiny timing helper ===
class T:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *a): print(f"    [time] {self.label}: {time.perf_counter() - self.t0:.2f}s")

def mask_to_mesh(mask_path: Path, case: str):
    with T("read mask"):
        img = sitk.ReadImage(str(mask_path))
    # Smooth the binary mask slightly so marching cubes gets sub-voxel detail
    # instead of a staircase. Gaussian variance in mm^2.
    with T("mask smoothing"):
        img_f = sitk.Cast(img, sitk.sitkFloat32)
        img_f = sitk.DiscreteGaussian(img_f, variance=[0.6, 0.6, 0.6])
    arr = sitk.GetArrayFromImage(img_f)   # shape (z, y, x), float
    spacing = img.GetSpacing()          # (x, y, z) mm
    origin  = img.GetOrigin()           # (x, y, z) world position of first voxel

    # Build pyvista grid in patient/world coords.
    # Use point_data (not cell_data) so contour iso=0.5 lands at the voxel boundary,
    # and dimensions = arr.shape (NOT shape+1) for point-centered scalars.
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

    # Save full-resolution mesh
    with T("save VTK + STL (full)"):
        surf.save(str(MESH_DIR / f"{case}_LA.vtk"))
        surf.save(str(MESH_DIR / f"{case}_LA.stl"))

    # Decimated copy for sharing / lightweight downstream use
    with T("decimate + save"):
        # decimate uses target_reduction = fraction of triangles to REMOVE
        dec = surf.decimate(DECIMATE_FRAC)
        dec.save(str(MESH_DIR / f"{case}_LA_decimated.stl"))
        dec.save(str(MESH_DIR / f"{case}_LA_decimated.vtk"))

    # Report
    vol_ml = surf.volume / 1000.0   # pyvista returns mm^3; convert to mL
    print(f"[{case}] mesh: {surf.n_points} pts, {surf.n_cells} tris  "
          f"-> decimated {dec.n_points} pts, {dec.n_cells} tris  | vol={vol_ml:.1f} mL")

if __name__ == "__main__":
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()
    for mask_path in sorted(LA_DIR.glob(GLOB)):
        # Skip seed files (we only want the refined _LA.nii.gz, not _LA_seed.nii.gz)
        if mask_path.name.endswith("_LA_seed.nii.gz"):
            continue
        case = mask_path.name.replace("_LA.nii.gz", "")
        print(f"\n[{case}] meshing...")
        mask_to_mesh(mask_path, case)
    print(f"\n[total] {time.perf_counter() - t_total:.1f}s")