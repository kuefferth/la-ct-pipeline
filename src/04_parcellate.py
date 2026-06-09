"""Run DIVAID on our LA mesh fully unattended (divaid env), producing the
EHRA/EACVI 15-segment division + PV/LAA orifice clips.

MUST run in the `divaid` conda env (pinned pyvista/vtk/pymeshlab), NOT `la`:
  C:\\ProgramData\\miniconda3\\Scripts\\conda.exe run -n divaid python src/04_parcellate.py <case>

Input is the mitral-OPEN mesh from 04a_divaid_prep.py:
  derivatives/parcellation/<case>/<case>_LA_open.vtk
DIVAID writes its results next to it under <..>_LA_open_division/.

DIVAID's default flow needs three human actions; we remove all three:
  - MV spline   -> the input mesh is already open at the valve (04a), so stage 1.3
                   auto-detects it ("Manual clipping is not required").
  - PV/LAA seeds-> we REPLACE the interactive seed step with auto_seed_ids(): one
                   seed per thin (SDF < 25 mm diameter) connected region. This is
                   DIVAID's own vein criterion, so the seeds match the veins stage
                   1.5 then clips. No voxel guesswork.
  - review_clip -> stage 1.5 pops a manual clip-review window unconditionally;
                   we monkeypatch it to accept the automatic clips as-is.
Plus the stage-3 result viewer (plot_divided_mesh) is patched to a no-op so the
process exits without a window. Both patches live here, leaving third_party/divaid
pristine for the eventual submodule.
"""
import sys
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIVAID_SRC = REPO / "third_party" / "divaid" / "src"
sys.path.insert(0, str(DIVAID_SRC))

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

# DIVAID stage entry points
from stage1_1_remesh import remesh_main
from stage1_2_separate_atria import separate_atria_main
from stage1_3_clip_valves import clip_valves_main
from stage2_annotate_orifices import annotate_orifices_main
import stage1_5_clip_veins
import stage3_divide_atria

# DIVAID helpers reused for the SDF auto-seed
from utils import (add_array_to_mesh, create_directories, get_connected_regions,
                   get_points_faces_normals, get_sdf_main, read_vtk, threshold_filter,
                   write_ids, write_vtk)

# === Tunables ===
# DIVAID's SDF = local DIAMETER (mm). We seed from it, but NOT at its own 25 mm vein cutoff:
# at 25 mm adjacent ostia merge across the wide carina (on case 121 LSPV fused into the LAA
# region -> LSPV missed). We separate at a tighter diameter so each ostium is its own thin
# region, then seed each at a proximal ~6 mm point (Thomas QC: distal tips are too narrow).
SEPARATE_DIAM_MM = 12    # threshold (mm) for splitting ostia into distinct regions. 12 cut
                         # the LSPV/LAA carina and dropped the CS-ostium dimple (>12 mm =
                         # body) on case 121. Lower = more separation but more fragments.
SEED_DIAM_MM     = 6     # seed each region where its diameter is ~this (proximal trunk),
                         # not at the narrow distal tip.
MIN_REGION_PTS   = 80    # drop thin specks smaller than a real ostium
SDF_CHUNK        = 2000
DROP_MPV         = True   # ignore DIVAID's spurious 'middle-PV' orifices. Its vein clip
                          # fragments our remeshed surface into ~180 pinholes, each annotated
                          # as an MPV; stage 3 ingests every MPV and crashes. The 6 CORE
                          # orifices (MV + 4 PV + LAA) are found correctly, so we delete the
                          # MPV files before stage 3 and divide off the core. Loses any genuine
                          # extra PV (acceptable first pass); revisit if real MPVs matter.
REUSE_SDF       = False                   # DANGER: DIVAID's remesh is NON-DETERMINISTIC, so a
                                          # cached SDF mesh can have a different point set than
                                          # the on-disk remeshed/clipped meshes. Clipping then
                                          # applies vein ids across mismatched meshes and shreds
                                          # the surface into ~180 pinholes. Keep False so every
                                          # run is fresh and internally consistent. The guard in
                                          # run_case also refuses a stale cache if this is on.


def auto_seed_ids(mesh, sdf_values, atrium):
    """One seed point-id per vein. Split ostia apart at SEPARATE_DIAM_MM (so the
    LSPV/LAA carina is cut and each ostium is its own thin region), then seed each
    region where its diameter is closest to SEED_DIAM_MM (the proximal trunk, not
    the narrow tip). 'Ids' are positional indices (vtkIdFilter), so a region's Id
    IS the seed index expected downstream."""
    m = add_array_to_mesh(mesh, sdf_values, "sdf")
    thin = threshold_filter(m, "sdf", 0, SEPARATE_DIAM_MM)
    seeds = []
    for reg in get_connected_regions(thin):
        if reg.GetNumberOfPoints() < MIN_REGION_PTS:
            continue
        reg_sdf = vtk_to_numpy(reg.GetPointData().GetArray("sdf"))
        reg_ids = vtk_to_numpy(reg.GetPointData().GetArray("Ids"))
        j = int(np.argmin(np.abs(reg_sdf - SEED_DIAM_MM)))   # ~6 mm proximal trunk
        seeds.append(int(reg_ids[j]))
    return np.array(sorted(seeds), dtype=int)


def auto_sdf_and_seeds_main(args, pipeline=True):
    """Drop-in for DIVAID stage 1.4: compute the SDF once, then auto-seed from it
    (no interactive picking, no nested SDF compute)."""
    sdf_dir = Path(f"{args.mesh}_division") / "stage1_preprocessing"
    mesh_name = Path(args.mesh).stem
    create_directories([sdf_dir.parent, sdf_dir])

    for atrium in args.atrium.split(","):
        sdf_vtk = sdf_dir / f"{mesh_name}_remeshed_{atrium}_sdf.vtk"

        if REUSE_SDF and sdf_vtk.exists():
            # reuse: take the mesh FROM the cached SDF file so points/Ids/sdf stay
            # consistent (remesh is non-deterministic, so the fresh remesh would not
            # match the cached SDF array length).
            mesh = read_vtk(sdf_vtk)
            sdf_values = vtk_to_numpy(mesh.GetPointData().GetArray("sdf"))
            print(f"AUTO-SEEDS [{atrium}]: reusing cached SDF ({sdf_vtk.name})")
        else:
            mesh = read_vtk(sdf_dir / f"{mesh_name}_remeshed_{atrium}.vtk")
            points, faces, normals = get_points_faces_normals(mesh)
            sdf_values, sdf_inter = get_sdf_main(points, faces, normals, SDF_CHUNK)
            sdf_mesh = add_array_to_mesh(mesh, sdf_values, "sdf")
            sdf_mesh = add_array_to_mesh(sdf_mesh, sdf_inter, "sdf_intersection_ids")
            write_vtk(sdf_mesh, sdf_vtk)

        seed_ids = auto_seed_ids(mesh, sdf_values, atrium)
        write_ids(seed_ids, sdf_dir / f"{mesh_name}_{atrium}_seeds.txt")
        print(f"AUTO-SEEDS [{atrium}]: {len(seed_ids)} veins "
              f"(split<{SEPARATE_DIAM_MM}mm, seed@~{SEED_DIAM_MM}mm) -> ids {list(seed_ids)}")


def run_case(case, atrium="LA", cos="x,z"):   # x,z = our LPS body frame (R->L, I->S)
    open_mesh = REPO / "derivatives" / "parcellation" / case / f"{case}_LA_open"
    if not open_mesh.with_suffix(".vtk").exists():
        print(f"[{case}] missing {open_mesh}.vtk -- run 04a_divaid_prep.py first"); return

    args = Namespace(
        mesh=str(open_mesh), atrium=atrium,
        target_mesh_resolution=1, scale=1,
        use_given_orifices=True,     # lenient MV-hole detection (5 mm) on our open mesh
        seed_file="", sdf_threshold=-1,
        use_given_cos=cos,           # hand DIVAID our LPS body frame (x = R->L, z = I->S)
        plot_subdivision=False, start_step=1,
    )

    # neutralize the two interactive windows (keep third_party pristine)
    stage1_5_clip_veins.review_clip = lambda mesh, clipped_mesh, veins: veins
    stage3_divide_atria.plot_divided_mesh = lambda *a, **k: None

    s1 = Path(f"{args.mesh}_division") / "stage1_preprocessing"
    stem = Path(args.mesh).stem
    sdf_vtk = s1 / f"{stem}_remeshed_{atrium}_sdf.vtk"
    remeshed = s1 / f"{stem}_remeshed_{atrium}.vtk"
    reuse = REUSE_SDF and sdf_vtk.exists() and remeshed.exists()
    if reuse:
        # refuse a cache whose point set no longer matches the on-disk remeshed mesh
        import pyvista as _pv
        if _pv.read(str(sdf_vtk)).n_points != _pv.read(str(remeshed)).n_points:
            print("WARN: SDF cache point count != remeshed mesh -> ignoring cache (re-running fresh)")
            reuse = False

    print(f"\n=== [{case}] DIVAID (auto{', reuse cache' if reuse else ''}) ===")
    if not reuse:
        remesh_main(args)                                # 1.1
        separate_atria_main(args, True)                  # 1.2
        clip_valves_main(args, True)                     # 1.3 (MV auto from open mesh)
    auto_sdf_and_seeds_main(args, True)                  # 1.4 (auto seeds, replaces picking)
    stage1_5_clip_veins.clip_veins_main(args, True)      # 1.5 (review patched off)
    annotate_orifices_main(args, True)                   # 2
    if DROP_MPV:
        s2 = Path(f"{args.mesh}_division") / "stage2_orifice_annotation"
        removed = [f for f in s2.glob("*MPV*.txt")]
        for f in removed:
            f.unlink()
        print(f"DROP_MPV: removed {len(removed)} spurious middle-PV orifices, kept core 6")
    stage3_divide_atria.divide_atria_main(args)          # 3 (viewer patched off)

    out = Path(f"{args.mesh}_division") / "stage3_division" / f"{Path(args.mesh).stem}_{atrium}_division.vtk"
    print(f"\n[{case}] DONE -> {out}  (exists={out.exists()})")


if __name__ == "__main__":
    cases = sys.argv[1:]
    if not cases:
        print("usage: python src/04_parcellate.py <case> [<case> ...]"); sys.exit(1)
    for c in cases:
        run_case(c)
