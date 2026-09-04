"""Analysis (scratch): compare Thomas's manual PV/LAA seeds against SDF-thin
connected components, to learn how to auto-place seeds that match.

For each manually-seeded case it loads the saved SDF mesh + the manual seed ids,
then for a ladder of split diameters reports how well the thin connected
components (CCs) line up 1:1 with the manual seeds:
  matched   CCs containing >=1 manual seed   (want = #seeds)
  collision CCs containing >=2 manual seeds  (merged ostia -- bad)
  empty     CCs with no manual seed          (spurious regions -- bad)
  missed    manual seeds in no CC            (seed off the thin set -- bad)
Also dumps each manual seed's local diameter (SDF) and the LA bbox size, to see
whether the best threshold scales with atrium size (-> adaptive).

Run (divaid env):
  python src/_seed_analysis.py
"""
import sys, glob, os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "divaid", "src"))
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from utils import read_vtk, threshold_filter, get_connected_regions, add_array_to_mesh

ROOT = os.path.join(REPO_ROOT, "derivatives", "parcellation")
LADDER = [8, 10, 12, 15, 18, 22]
MIN_REGION_PTS = 80


def seed_files():
    for sf in glob.glob(ROOT + r"\*\*_division\stage1_preprocessing\*_LA_seeds.txt"):
        ids = np.loadtxt(sf, skiprows=2, dtype=int, ndmin=1)
        if list(ids) == sorted(ids):          # auto seeds are sorted; skip non-manual
            continue
        yield sf, ids


def analyze(sf, ids):
    d = os.path.dirname(sf)
    stem = os.path.basename(sf).replace("_LA_seeds.txt", "")
    case = stem.replace("_LA_open", "")
    mesh = read_vtk(os.path.join(d, f"{stem}_remeshed_LA_sdf.vtk"))
    pts = vtk_to_numpy(mesh.GetPoints().GetData())
    sdf = vtk_to_numpy(mesh.GetPointData().GetArray("sdf"))
    seed_pts = pts[ids]
    seed_sdf = sdf[ids]
    bbox = pts.max(0) - pts.min(0)
    diag = float(np.linalg.norm(bbox))

    print(f"\n=== {case[:30]:30s}  {len(ids)} seeds | LA bbox {bbox[0]:.0f}x{bbox[1]:.0f}x{bbox[2]:.0f} "
          f"diag={diag:.0f} | seed diam(SDF): {', '.join(f'{v:.0f}' for v in seed_sdf)}")
    best = None
    for thr in LADDER:
        m = add_array_to_mesh(mesh, sdf, "sdf")
        thin = threshold_filter(m, "sdf", 0, thr)
        regs = [r for r in get_connected_regions(thin) if r.GetNumberOfPoints() >= MIN_REGION_PTS]
        # assign each manual seed to the nearest-centroid CC it lies within (<3mm)
        hit = [0] * len(regs)
        missed = 0
        reg_pts = [vtk_to_numpy(r.GetPoints().GetData()) for r in regs]
        for sp in seed_pts:
            best_r, best_d = -1, 1e9
            for ri, rp in enumerate(reg_pts):
                dmin = np.min(np.linalg.norm(rp - sp, axis=1))
                if dmin < best_d:
                    best_d, best_r = dmin, ri
            if best_r >= 0 and best_d < 3.0:
                hit[best_r] += 1
            else:
                missed += 1
        matched = sum(1 for h in hit if h >= 1)
        collision = sum(1 for h in hit if h >= 2)
        empty = sum(1 for h in hit if h == 0)
        score = matched - collision - empty - missed     # higher = cleaner 1:1
        flag = " <-- best" if best is None or score > best[1] else ""
        if best is None or score > best[1]:
            best = (thr, score)
        print(f"   thr<{thr:>2}: {len(regs):>2} CCs | matched={matched} collision={collision} "
              f"empty={empty} missed={missed}{flag}")
    print(f"   BEST split for {case[:20]}: <{best[0]}mm")
    return diag, best[0]


if __name__ == "__main__":
    pairs = []
    for sf, ids in seed_files():
        pairs.append(analyze(sf, ids))
    print("\n=== does best threshold scale with LA size? (diag_mm, best_thr) ===")
    for diag, thr in sorted(pairs):
        print(f"  diag={diag:.0f}mm -> best<{thr}mm")
