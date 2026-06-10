"""PV-length sweep for the adaptive cap (02b): hold ADAPTIVE_CAP on and sweep
PV_KEEP_MM (the constant vein stub kept past the body wall) so we can pick how
much PV to keep. Runs the REAL 02b -> 03 per (case, keep) and writes meshes in
PATIENT SPACE (no canonicalize) so every keep-variant of a case overlays exactly
-> load them together in Slicer and read off vein length directly.

Output: derivatives/_scrap/pvkeep/<case>_keep{NN}_LA.stl   (gitignored)
Run (la env):
  python src/_pvkeep_sweep.py
"""
import sys
import time
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = Path("derivatives/_scrap/pvkeep")

# small / mid / large from the cohort volume ranking (59 / 133 / 200 mL).
CASES = [
    "121_0___do92008547__pcct",
    "1062_0___do92008692__pcct",
    "854_0___do92008832__pcct",
]
KEEP_LADDER = [18.0, 14.0, 10.0, 6.0]   # mm of vein past the body wall. 23 = old default.
CAP_MIN_OVERRIDE = 26.0           # lower the safety floor so small/mid atria aren't
                                  # pinned (45 forces ~20mm vein onto a small LA).
                                  # 26 still catches a degenerate seed (real body
                                  # walls are >= ~25mm). NOTE: keepNN is only
                                  # comparable WITHIN one floor -- the cap is encoded
                                  # in the filename too so cross-run mixups are obvious.

# production fixed-55 mesh volumes (mL), for a Δ reference column.
FIXED55_VOL = {
    "121_0___do92008547__pcct": 59.1,
    "1062_0___do92008692__pcct": 132.7,
    "854_0___do92008832__pcct": 199.5,
}


def load(stem):
    path = SRC / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("0", "m"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m02b = load("02b_regiongrow")
    m03  = load("03_mesh")

    m02b.ADAPTIVE_CAP = True
    m02b.CAP_MIN_MM = CAP_MIN_OVERRIDE
    m02b.LA_DIR = OUT
    m03.LA_DIR = OUT
    m03.MESH_DIR = OUT
    m03.SKIP_EXISTING = False

    t0 = time.perf_counter()
    results = []   # (case, keep, body_ext, cap, vol)
    for case in CASES:
        for keep in KEEP_LADDER:
            m02b.PV_KEEP_MM = keep
            print(f"\n===== {case} keep{int(keep):02d} =====")
            # 02b writes <LA_DIR>/<case>_LA.nii.gz -- rename per keep so they don't clobber
            s = m02b.grow_one(case)
            if s is None:
                print(f"  [skip] no mask for {case}")
                continue
            # Encode BOTH keep and the resulting cap in the name. keepNN only sorts
            # by PV length within one floor; cap is the actual physical cutoff.
            tag = f"{case}_keep{int(keep):02d}_cap{int(round(s['cap_mm'])):02d}"
            mask = OUT / f"{case}_LA.nii.gz"
            keep_mask = OUT / f"{tag}_LA.nii.gz"
            if keep_mask.exists():
                keep_mask.unlink()
            mask.rename(keep_mask)
            m03.mask_to_mesh(keep_mask, f"{tag}")
            results.append((case, keep, s["body_extent_mm"], s["cap_mm"], s["vol_ml"]))

    # === Table ===
    print("\n" + "=" * 76)
    print("PV_KEEP SWEEP -- vein stub vs LA size (mask volume from 02b)")
    print("=" * 76)
    hdr = f"{'case':28s} {'keep':>5s} {'body':>6s} {'cap':>6s} {'vol':>7s} {'vs f55%':>8s}"
    print(hdr); print("-" * len(hdr))
    for case, keep, body, cap, vol in results:
        base = FIXED55_VOL.get(case)
        dv = (vol - base) / base * 100.0 if base else float("nan")
        print(f"{case[:28]:28s} {keep:5.0f} {body:6.1f} {cap:6.1f} {vol:7.1f} {dv:+8.1f}")
    print("-" * len(hdr))
    print("vs f55% = mask volume vs the production fixed-55 mesh (current default 23mm).")
    print(f"STLs (patient space, overlayable) -> {OUT}/<case>_keep{{NN}}_LA.stl")
    print(f"\n[total] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
