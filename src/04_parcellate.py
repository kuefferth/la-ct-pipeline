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
from stage1_4_compute_sdf_and_place_seeds import compute_sdf_and_place_seeds_main as native_seed_main
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
SEPARATE_DIAM_LADDER = [12, 15, 18, 22]  # threshold (mm) for splitting ostia into distinct
                         # regions, tried in order and LOOSENED only if too few ostia resolve.
                         # 12 is right for high-res pcct (cuts the LSPV/LAA carina, drops the
                         # CS dimple). Low-res legacy meshes have wider PVs, so fewer cross a
                         # tight threshold (case 1161: 2 at 12mm); loosening finds more.
TARGET_MIN_VEINS = 3     # DIVAID's L/R superior/inferior split needs >=3 veins (case 114
                         # passed with 3, 1161 crashed with 2). Stop loosening once reached.
SEED_DIAM_MM     = 6     # seed each region where its diameter is ~this (proximal trunk),
                         # not at the narrow distal tip.
MIN_REGION_PTS   = 80    # drop thin specks smaller than a real ostium
SDF_CHUNK        = 2000
MANUAL_SEEDS     = False  # True (--manual) = use DIVAID's native interactive seed picker for
                          # PV/LAA instead of the SDF auto-seeder (MV is still auto from the
                          # open mesh; review_clip + result viewer stay patched off). Used to
                          # build the ground-truth training set. Seeds saved to ..._LA_seeds.txt.
SEEDS_ONLY       = False  # True (--seeds-only) = stop right after seeding (skip clip/annotate/
                          # divide). For fast ground-truth collection: place seeds -> saved ->
                          # next case, no downstream stage-3 crashes to derail the seeding pass.
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


def _seeds_at(m, thr):
    """Seed ids for one split diameter: one per thin region, at its ~SEED_DIAM_MM
    (proximal trunk) point. 'Ids' are positional indices (vtkIdFilter)."""
    thin = threshold_filter(m, "sdf", 0, thr)
    seeds = []
    for reg in get_connected_regions(thin):
        if reg.GetNumberOfPoints() < MIN_REGION_PTS:
            continue
        reg_sdf = vtk_to_numpy(reg.GetPointData().GetArray("sdf"))
        reg_ids = vtk_to_numpy(reg.GetPointData().GetArray("Ids"))
        j = int(np.argmin(np.abs(reg_sdf - SEED_DIAM_MM)))   # ~6 mm proximal trunk
        seeds.append(int(reg_ids[j]))
    return sorted(seeds)


def auto_seed_ids(mesh, sdf_values, atrium):
    """One seed per vein. Split ostia apart at the tightest ladder diameter that
    resolves >=TARGET_MIN_VEINS (so the LSPV/LAA carina stays cut on pcct but
    low-res legacy meshes loosen until enough PVs separate). Returns (ids, thr)."""
    m = add_array_to_mesh(mesh, sdf_values, "sdf")
    best, best_thr = [], SEPARATE_DIAM_LADDER[0]
    for thr in SEPARATE_DIAM_LADDER:
        s = _seeds_at(m, thr)
        if len(s) > len(best):
            best, best_thr = s, thr
        if len(s) >= TARGET_MIN_VEINS:
            return np.array(s, dtype=int), thr
    return np.array(best, dtype=int), best_thr   # best effort if none reach the target


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

        seed_ids, split_thr = auto_seed_ids(mesh, sdf_values, atrium)
        write_ids(seed_ids, sdf_dir / f"{mesh_name}_{atrium}_seeds.txt")
        print(f"AUTO-SEEDS [{atrium}]: {len(seed_ids)} veins "
              f"(split<{split_thr}mm, seed@~{SEED_DIAM_MM}mm) -> ids {list(seed_ids)}")


def resolve_case(case):
    """Accept a full stem or a short case id (the token before '_0___')."""
    pr = REPO / "derivatives" / "parcellation"
    if (pr / case / f"{case}_LA_open.vtk").exists():
        return case
    cand = [d.name for d in pr.glob(f"{case}_0___*") if (d / f"{d.name}_LA_open.vtk").exists()]
    if len(cand) == 1:
        return cand[0]
    return None


def run_case(case, atrium="LA", cos="x,z"):   # x,z = our LPS body frame (R->L, I->S)
    stem = resolve_case(case)
    if stem is None:
        print(f"[{case}] no open mesh found (run 04a, or give the full stem)")
        return f"skip: no open mesh for '{case}'"
    case = stem
    open_mesh = REPO / "derivatives" / "parcellation" / case / f"{case}_LA_open"

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

    mode = "manual seeds" if MANUAL_SEEDS else "auto seeds"
    if SEEDS_ONLY:
        mode += ", seeds-only"
    print(f"\n=== [{case}] DIVAID ({mode}{', reuse cache' if reuse else ''}) ===")
    if not reuse:
        remesh_main(args)                                # 1.1
        separate_atria_main(args, True)                  # 1.2
        clip_valves_main(args, True)                     # 1.3 (MV auto from open mesh)
    if MANUAL_SEEDS:
        native_seed_main(args, True)                     # 1.4 DIVAID interactive PV/LAA picker
    else:
        auto_sdf_and_seeds_main(args, True)              # 1.4 auto seeds (replaces picking)

    if SEEDS_ONLY:                                       # stop after capturing seeds
        seeds_txt = (Path(f"{args.mesh}_division") / "stage1_preprocessing"
                     / f"{Path(args.mesh).stem}_{atrium}_seeds.txt")
        print(f"[{case}] seeds captured -> {seeds_txt}")
        return "seeds-only"

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
    return "ok" if out.exists() else "FAIL: no division written"


if __name__ == "__main__":
    import traceback
    argv = sys.argv[1:]
    if "--manual" in argv:                 # native DIVAID picker for PV/LAA seeds
        MANUAL_SEEDS = True
    if "--seeds-only" in argv:             # stop after seeding (no clip/annotate/divide)
        SEEDS_ONLY = True
    cases = [a for a in argv if not a.startswith("--")]
    if not cases:
        print("usage: python src/04_parcellate.py [--manual] [--seeds-only] <case> ..."); sys.exit(1)
    results = {}
    for c in cases:
        try:
            results[c] = run_case(c) or "ok"
        except Exception as e:                         # one bad case must not abort the batch
            traceback.print_exc()
            results[c] = f"FAIL: {type(e).__name__}: {e}"
    print("\n=== BATCH SUMMARY ===")
    for c, r in results.items():
        print(f"  {c}: {r}")
    if any(r != "ok" for r in results.values()):
        sys.exit(1)
