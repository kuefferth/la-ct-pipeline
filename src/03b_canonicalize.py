"""Center each LA on its centroid; preserve scanner orientation.

Patients are positioned consistently on the scanner table, so the scanner
(LPS) frame is already a reasonable canonical orientation. PCA was making
things worse (LA isn't elongated enough for stable axis assignment).

Per case:
  - Compute LA voxel centroid in world (mm) coords.
  - Build 4x4 transform = translation only (centroid -> origin).
  - Save T as .npy sidecar AND apply to a copy of the mesh.
  - Originals untouched (patient-space data preserved for EAM/outcome overlay).

Reads:
  derivatives/seg_la/<case>_LA.nii.gz
  derivatives/meshes/<case>_LA.stl
  derivatives/meshes/<case>_LA_decimated.stl

Writes:
  derivatives/canonical/<case>_LA_T.npy     (4x4 world->canonical)
  derivatives/canonical/<case>_LA_canonical.stl
  derivatives/canonical/<case>_LA_decimated_canonical.stl
"""
import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pyvista as pv

LA_DIR   = Path("derivatives/seg_la")
MESH_DIR = Path("derivatives/meshes")
OUT_DIR  = Path("derivatives/canonical")
GLOB     = "*_LA.nii.gz"

def la_centroid_world(mask_img):
    """World-coord centroid of all positive voxels in the LA mask."""
    arr = sitk.GetArrayFromImage(mask_img)        # (z, y, x)
    zs, ys, xs = np.where(arr > 0)
    if len(zs) == 0:
        return None
    spacing   = np.array(mask_img.GetSpacing())            # (sx, sy, sz)
    origin    = np.array(mask_img.GetOrigin())             # (ox, oy, oz)
    direction = np.array(mask_img.GetDirection()).reshape(3, 3)
    mean_idx  = np.array([xs.mean(), ys.mean(), zs.mean()])
    return origin + direction @ (spacing * mean_idx)

def apply_transform_to_mesh(mesh_path: Path, T44, out_path: Path):
    """Load mesh, apply 4x4 transform, save."""
    mesh = pv.read(str(mesh_path))
    mesh = mesh.transform(T44, inplace=False)
    mesh.save(str(out_path))

def canonicalize_one(case: str):
    la_path  = LA_DIR  / f"{case}_LA.nii.gz"
    stl_path = MESH_DIR / f"{case}_LA.stl"
    dec_path = MESH_DIR / f"{case}_LA_decimated.stl"
    if not la_path.exists() or not stl_path.exists():
        print(f"  [skip] missing inputs for {case}"); return

    la_img = sitk.ReadImage(str(la_path))
    la_c   = la_centroid_world(la_img)
    if la_c is None:
        print(f"  [skip] empty LA mask"); return

    # 4x4 pure translation: x_canon = x_world - centroid
    T44 = np.eye(4)
    T44[:3, 3] = -la_c
    print(f"  LA centroid (mm): ({la_c[0]:.1f}, {la_c[1]:.1f}, {la_c[2]:.1f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{case}_LA_T.npy", T44)
    apply_transform_to_mesh(stl_path, T44,
                            OUT_DIR / f"{case}_LA_canonical.stl")
    if dec_path.exists():
        apply_transform_to_mesh(dec_path, T44,
                                OUT_DIR / f"{case}_LA_decimated_canonical.stl")
    print(f"[{case}] centered")

if __name__ == "__main__":
    t_total = time.perf_counter()
    for mask_path in sorted(LA_DIR.glob(GLOB)):
        if mask_path.name.endswith("_LA_seed.nii.gz"):
            continue
        case = mask_path.name.replace("_LA.nii.gz", "")
        print(f"\n[{case}] centering...")
        canonicalize_one(case)
    print(f"\n[total] {time.perf_counter() - t_total:.1f}s")
