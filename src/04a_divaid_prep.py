"""DIVAID prep (la env): open the LA mesh at the mitral valve so DIVAID needs no
manual MV spline.

DIVAID's stage 1.3 auto-detects an already-open valve: if the INPUT mesh has a
boundary hole wider than its threshold it prints "Manual clipping is not required"
and uses that hole as the MV. Our 03 mesh is closed (the border-pad caps every
face), so we clip it open at the mitral plane here. The plane is derived
deterministically from the LA/LV interface in the heartchambers seg (LA mask vs
LV label), which also finally delivers the long-pending mitral annulus cut.

PV/LAA seeds are NOT generated here: voxel morphology cannot cleanly isolate the
stubby PVs from the lobulated LA body. Instead 04_parcellate.py derives seeds inside
the DIVAID run from DIVAID's own SDF vein criterion (diameter < 25 mm), which is the
geometrically consistent signal.

Per case, reads:
  derivatives/seg_la/<case>_LA.nii.gz          final binary LA mask
  derivatives/seg_full/<case>_chambers.nii.gz  multi-label chambers (for LV)
  derivatives/meshes/<case>_LA.vtk             our decimated LA surface
Writes (into derivatives/parcellation/<case>/):
  <case>_LA_open.vtk     LA surface clipped open at the mitral plane (DIVAID input)
  <case>_prep.json       QC: plane params + open-boundary-edge count
"""
import json
import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pyvista as pv

LA_DIR   = Path("derivatives/seg_la")
SEG_DIR  = Path("derivatives/seg_full")
MESH_DIR = Path("derivatives/meshes")
OUT_ROOT = Path("derivatives/parcellation")

LV_LABEL = 3   # heart_ventricle_left in heartchambers_highres

# === Tunables ===
LV_DILATE_MM     = 3.0   # grow LV to find the LA voxels bordering it (the mitral rim)
MITRAL_OFFSET_MM = 0.0   # shift the cut plane along its normal toward LV (+) or LA (-)


def index_to_world(img, idx_xyz):
    spacing   = np.array(img.GetSpacing())
    origin    = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    return origin + direction @ (spacing * np.asarray(idx_xyz, float))


def centroid_world(img, arr_bool):
    zs, ys, xs = np.where(arr_bool)
    if len(zs) == 0:
        return None
    return index_to_world(img, [xs.mean(), ys.mean(), zs.mean()])


def _radius_vox(img, mm):
    return [max(1, int(round(mm / s))) for s in img.GetSpacing()]


def mitral_plane(la_img, seg_img):
    """Cut plane at the mitral valve from the LA/LV interface.
    origin = centroid of LA voxels touching a dilated LV (the mitral rim),
    normal = LA-centroid -> LV-centroid (so the discarded side is the LV cap)."""
    la = sitk.GetArrayFromImage(la_img).astype(bool)
    lv = sitk.BinaryThreshold(seg_img, LV_LABEL, LV_LABEL, 1, 0)
    lv_d = sitk.GetArrayFromImage(sitk.BinaryDilate(lv, _radius_vox(la_img, LV_DILATE_MM))).astype(bool)

    interface = la & lv_d
    if interface.sum() == 0:
        return None
    origin = centroid_world(la_img, interface)
    la_c = centroid_world(la_img, la)
    lv_c = centroid_world(la_img, sitk.GetArrayFromImage(lv).astype(bool))
    normal = lv_c - la_c
    n = np.linalg.norm(normal)
    if n == 0:
        return None
    normal = normal / n
    return origin + MITRAL_OFFSET_MM * normal, normal


def open_mesh_at_plane(mesh, origin, normal, la_c):
    """Clip with the mitral plane, keep the LA-body side (leaves an open rim)."""
    a = mesh.clip(normal=normal, origin=origin, invert=False)
    b = mesh.clip(normal=normal, origin=origin, invert=True)
    cand = [m for m in (a, b) if m.n_points > 0]
    if not cand:
        return mesh
    keep = min(cand, key=lambda m: np.linalg.norm(np.array(m.center) - la_c))
    # clip yields an UnstructuredGrid; hand DIVAID clean triangle polydata
    return keep.extract_surface().triangulate()


def prep_one(case):
    la_path   = LA_DIR  / f"{case}_LA.nii.gz"
    seg_path  = SEG_DIR / f"{case}_chambers.nii.gz"
    mesh_path = MESH_DIR / f"{case}_LA.vtk"
    if not (la_path.exists() and seg_path.exists() and mesh_path.exists()):
        print(f"[{case}] missing inputs, skip"); return

    out_dir = OUT_ROOT / case
    out_dir.mkdir(parents=True, exist_ok=True)

    la_img  = sitk.ReadImage(str(la_path))
    seg_img = sitk.ReadImage(str(seg_path))
    mesh    = pv.read(str(mesh_path))
    la_c    = centroid_world(la_img, sitk.GetArrayFromImage(la_img).astype(bool))

    plane = mitral_plane(la_img, seg_img)
    if plane is None:
        print(f"[{case}] no LA/LV interface; mesh left closed (DIVAID would force manual MV)")
        open_mesh = mesh; origin = normal = None
    else:
        origin, normal = plane
        open_mesh = open_mesh_at_plane(mesh, origin, normal, la_c)

    # strip the junk arrays pyvista's clip/extract_surface adds (vtkOriginal* Ids):
    # DIVAID re-derives Ids/Normals/SDF itself, and transferring this array onto the
    # remeshed mesh just NaNs the mitral-fill points -> the "no scalar value" warning.
    open_mesh.clear_data()
    open_path = out_dir / f"{case}_LA_open.vtk"
    open_mesh.save(str(open_path))
    n_edges = open_mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False).n_cells

    qc = {
        "case": case,
        "mitral_origin": None if origin is None else [round(float(x), 2) for x in origin],
        "mitral_normal": None if normal is None else [round(float(x), 3) for x in normal],
        "open_boundary_edges": int(n_edges),
        "open_mesh_points": int(open_mesh.n_points),
    }
    with open(out_dir / f"{case}_prep.json", "w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)
    print(f"[{case}] open mesh: {open_mesh.n_points} pts, {n_edges} boundary edges "
          f"(want >0 = MV opened) -> {open_path}")


if __name__ == "__main__":
    import sys
    cases_filter = set(sys.argv[1:])
    t0 = time.perf_counter()
    for la in sorted(LA_DIR.glob("*_LA.nii.gz")):
        if la.name.endswith("_LA_seed.nii.gz"):
            continue
        case = la.name.replace("_LA.nii.gz", "")
        if cases_filter and case not in cases_filter:
            continue
        print(f"\n[{case}] prepping...")
        prep_one(case)
    print(f"\n[total] {time.perf_counter() - t0:.1f}s")
