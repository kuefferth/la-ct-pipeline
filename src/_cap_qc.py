"""QC harness for the adaptive geodesic cap (02b): run a scrap set through the
REAL 02b -> 03 -> 03b code twice -- fixed MAX_GEODESIC_MM vs ADAPTIVE_CAP -- and
print a before/after table. Production derivatives are left untouched; everything
lands under derivatives/_scrap/cap_qc/<mode>/{seg_la,meshes,canonical}.

Reuses the actual module functions (loaded via importlib because the filenames
start with digits) so this tests the shipped implementation, not a copy.

Open the two canonical STLs per case side by side in Slicer/ParaView to eyeball
PV length; the printed table gives the volume + derived-cap spread.

Run (la env):
  python src/_cap_qc.py                 # default 5-case scrap set
  python src/_cap_qc.py <stem> ...      # explicit case stems
"""
import sys
import time
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parent
SCRAP = Path("derivatives/_scrap/cap_qc")

# Size-varied + legacy/pcct mix to stress the size scaling (3 pcct from the smooth
# sweep: large / PV-heavy / smaller-high-contrast; 2 legacy = lower contrast).
DEFAULT_CASES = [
    "854_0___do92008832__pcct",    # large
    "540_0___do92008590__pcct",    # PV-heavy
    "2643_0___do92008629__pcct",   # smaller, high-contrast
    "1300_0___do92008575__legacy", # legacy
    "2294_0___do92008627__legacy", # legacy
]


def load(stem):
    """Load a digit-prefixed src module by file stem (e.g. '02b_regiongrow')."""
    path = SRC / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("0", "m"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_mode(m02b, m03, m03b, cases, adaptive):
    """Run all cases through 02b->03->03b with the cap mode fixed/adaptive.
    Returns {case: stats dict from grow_one}."""
    mode = "adaptive" if adaptive else "fixed"
    seg_la = SCRAP / mode / "seg_la"
    meshes = SCRAP / mode / "meshes"
    canon  = SCRAP / mode / "canonical"
    for d in (seg_la, meshes, canon):
        d.mkdir(parents=True, exist_ok=True)

    # Point the real modules at the scrap dirs; inputs (nifti/seg_full) stay shared.
    m02b.ADAPTIVE_CAP = adaptive
    m02b.LA_DIR = seg_la
    m03.LA_DIR = seg_la
    m03.MESH_DIR = meshes
    m03.SKIP_EXISTING = False
    m03b.LA_DIR = seg_la
    m03b.MESH_DIR = meshes
    m03b.OUT_DIR = canon

    stats = {}
    for case in cases:
        print(f"\n===== [{mode}] {case} =====")
        s = m02b.grow_one(case)
        if s is None:
            print(f"  [skip] 02b produced nothing for {case}")
            continue
        m03.mask_to_mesh(seg_la / f"{case}_LA.nii.gz", case)
        m03b.canonicalize_one(case)
        stats[case] = s
    return stats


def main():
    cases = sys.argv[1:] or DEFAULT_CASES
    m02b = load("02b_regiongrow")
    m03  = load("03_mesh")
    m03b = load("03b_canonicalize")

    t0 = time.perf_counter()
    fixed    = run_mode(m02b, m03, m03b, cases, adaptive=False)
    adaptive = run_mode(m02b, m03, m03b, cases, adaptive=True)

    # === Comparison table ===
    print("\n" + "=" * 78)
    print("ADAPTIVE GEODESIC CAP -- before/after (mask volume from 02b)")
    print("=" * 78)
    hdr = (f"{'case':32s} {'body_ext':>8s} {'capF':>5s} {'capA':>5s} "
           f"{'volF':>7s} {'volA':>7s} {'dVol%':>7s}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for case in cases:
        f, a = fixed.get(case), adaptive.get(case)
        if f is None or a is None:
            print(f"{case[:32]:32s}  (incomplete -- skipped)")
            continue
        dvol = (a["vol_ml"] - f["vol_ml"]) / f["vol_ml"] * 100.0 if f["vol_ml"] else 0.0
        print(f"{case[:32]:32s} {a['body_extent_mm']:8.1f} {f['cap_mm']:5.0f} "
              f"{a['cap_mm']:5.1f} {f['vol_ml']:7.1f} {a['vol_ml']:7.1f} {dvol:+7.1f}")
        rows.append((case, a, f, dvol))

    # Write a CSV alongside the meshes for the record (UTF-8 BOM per house style).
    csv_path = SCRAP / "cap_qc_summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as fh:
        fh.write("case,body_extent_mm,cap_fixed_mm,cap_adaptive_mm,vol_fixed_ml,vol_adaptive_ml,dvol_pct\n")
        for case, a, f, dvol in rows:
            fh.write(f"{case},{a['body_extent_mm']:.2f},{f['cap_mm']:.1f},{a['cap_mm']:.2f},"
                     f"{f['vol_ml']:.2f},{a['vol_ml']:.2f},{dvol:.2f}\n")

    print("-" * len(hdr))
    print(f"capF = fixed MAX_GEODESIC_MM; capA = adaptive (body_ext P{m02b.CAP_PCTILE:.0f} + "
          f"size-scaled collar [ref {m02b.PV_KEEP_MM:.0f}mm @ body {m02b.BODY_REF_MM:.0f}mm, "
          f"slope {m02b.PV_KEEP_SLOPE}], clamp [{m02b.CAP_MIN_MM:.0f},{m02b.CAP_MAX_MM:.0f}])")
    print(f"CSV  -> {csv_path}")
    print(f"STLs -> {SCRAP}/<fixed|adaptive>/canonical/<case>_LA_canonical.stl")
    print(f"\n[total] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
