"""LA binary mask -> surface mesh (STL + VTK)."""
from pathlib import Path
import pyvista as pv
import SimpleITK as sitk
import numpy as np

LA_DIR   = Path("derivatives/seg_la")
MESH_DIR = Path("derivatives/meshes")
MESH_DIR.mkdir(parents=True, exist_ok=True)

for mask_path in sorted(LA_DIR.glob("*_LA.nii.gz")):
    case = mask_path.name.replace("_LA.nii.gz", "")
    img  = sitk.ReadImage(str(mask_path))
    spacing = img.GetSpacing()        # (x,y,z) in mm
    origin  = img.GetOrigin()
    arr = sitk.GetArrayFromImage(img) # shape: (z,y,x)
    # Build pyvista UniformGrid in patient coords
    grid = pv.ImageData(
        dimensions=np.array(arr.shape[::-1]) + 1,  # cell-centered -> +1 in dims
        spacing=spacing,
        origin=origin,
    )
    # flatten as Fortran order to match VTK's x-fastest
    grid.cell_data["mask"] = arr.transpose(2,1,0).flatten(order="F")
    # Marching cubes at iso=0.5
    surf = grid.contour([0.5], scalars="mask", method="marching_cubes")
    # Keep largest connected component (drops spurious blobs)
    surf = surf.connectivity(extraction_mode="largest")
    # Taubin smoothing preserves volume better than Laplacian
    surf = surf.smooth_taubin(n_iter=30, pass_band=0.1)
    # Save both raw-ish and decimated
    surf.save(str(MESH_DIR / f"{case}_LA.vtk"))
    surf.save(str(MESH_DIR / f"{case}_LA.stl"))
    dec = surf.decimate(0.7)   # keep 30% of triangles
    dec.save(str(MESH_DIR / f"{case}_LA_decimated.stl"))
    print(f"[{case}] mesh: {surf.n_points} pts -> decimated {dec.n_points} pts")