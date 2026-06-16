"""LA refinement v7: TotalSeg LA seed -> region grow -> forbidden-neighbour block ->
geodesic distance cap -> thick-trunk-only opening -> closing.

v7 changes vs v6:
  - Distance cap is now GEODESIC (in-mask path distance from seed), not Euclidean.
    Follows the PV tubes instead of slicing a sphere; also drops components not
    connected to the seed (doubles as a CC filter).
  - Thin-branch removal is now a PLAIN morphological opening (erode -> dilate),
    NOT opening-then-reconstruction. Reconstruction regrew every thin tube wired
    to the body, so it was a near no-op. Plain opening actually deletes connected
    tubes thinner than ~2*OPEN_RADIUS_MM diameter while keeping the thick trunk.
    (LAA is not preserved by this; per project scope that is fine, and LAAs that
    are contrast-filled are trunk-sized so they survive anyway.)

Pipeline:
  1. Erode TotalSeg LA mask by ERODE_MM -> seed
  2. Adaptive threshold from CT intensities inside seed (percentile-based)
  3. Region grow from seed under that threshold window
  4. Subtract forbidden neighbours (LV, RA raw; aorta, RV, PA dilated)
  5. Keep largest connected component
  6. Geodesic distance cap from seed centroid (cuts distal PVs along the tube)
  6b. Plain opening: delete thin branches, keep thick trunk + ostia
  7. Closing to fill small dents/holes
"""

import os
os.environ["nnUNet_n_proc_DA"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# === Paths ===
NIFTI_DIR = Path("derivatives/nifti")
SEG_DIR   = Path("derivatives/seg_full")
LA_DIR    = Path("derivatives/seg_la")

# === Label indices in heartchambers_highres ===
LA_LABEL    = 2   # heart_atrium_left
LV_LABEL    = 3   # heart_ventricle_left
RA_LABEL    = 4   # heart_atrium_right
RV_LABEL    = 5   # heart_ventricle_right
AORTA_LABEL = 6   # aorta
PA_LABEL    = 7   # pulmonary_artery

# === Forbidden neighbours: label -> dilation (mm) ===
# Subtracting these seals region-grow leaks across thin shared walls. largest-CC
# then drops whatever the dilated barrier severed.
#   LV  raw : dilation would cut the mitral plane (keep LA-LV interface intact)
#   RA  raw : interatrial septum is thin; dilation would eat into LA
#   aorta   : TotalSeg under-segments the root, pad a bit
#   RV / PA : main leak targets; pad to seal the thin wall the grow jumps
# If PA leaks persist, bump PA to 2.0-2.5, but watch the left superior PV ostium.
FORBIDDEN = {
    LV_LABEL:    0.0,
    RA_LABEL:    0.0,
    AORTA_LABEL: 1.5,
    RV_LABEL:    1.5,
    PA_LABEL:    1.5,
}

# === Tunables ===
ERODE_MM             = 5.0   # seed erosion before sampling intensities + growing
MAX_GEODESIC_MM      = 55.0  # FIXED in-mask path-distance cap from seed centroid.
                             # Tightened 60 -> 55 post-tuning to trim PVs a touch
                             # shorter. Geodesic >= Euclidean. Used when ADAPTIVE_CAP
                             # is False (validated default).

# --- Adaptive geodesic cap (scale the PV cutoff to LA size) -------------------
# The fixed cap does two jobs at once: cover the LA body (centroid -> far body
# wall, which scales with atrium size) AND keep a PV collar past the ostium.
# Adaptive splits it: measure the body extent geodesically, then add a collar
# that itself SCALES with body size (bigger atria keep a slightly longer collar):
#   body_extent = P{CAP_PCTILE} of the geodesic arrival field restricted to the
#                 TotalSeg LA body (label 2, ~no PV) -> "how far the body wall runs".
#                 P90 ~= 0.96 * true wall for a solid body, and is robust to the PV
#                 stub TotalSeg includes in its LA label (P100 would chase the vein).
#   collar      = PV_KEEP_MM + PV_KEEP_SLOPE * (body_extent - BODY_REF_MM)
#   cap         = clamp(body_extent + collar, CAP_MIN_MM, CAP_MAX_MM)
# Tuned on a small/mid/large scrap set (body 24/32/38mm): collar 14/18/21mm ->
# cap 38/50/59mm. SLOPE=0.5 came from fitting small->14, mid->18; hinge at the
# typical body size so PV_KEEP_MM is just "the collar on a normal atrium".
# IMPORTANT: to keep LESS vein, lower BOTH PV_KEEP_MM and CAP_MIN_MM. The floor
# (was 45) silently pins the cap on small/mid atria; 26 still catches a degenerate
# seed (real body walls are >= ~25mm) without forcing vein onto a small LA.
ADAPTIVE_CAP         = True   # True = derive the cap per case (see above).
                              # False = fixed MAX_GEODESIC_MM (old validated default).
CAP_PCTILE           = 90.0   # percentile of the in-body geodesic field = body extent.
                              # P90 not max: guards against a stray ostial spur.
PV_KEEP_MM           = 18.0   # PV collar (mm) at the reference body size BODY_REF_MM.
PV_KEEP_SLOPE        = 0.5    # collar grows this many mm per mm of body beyond ref.
BODY_REF_MM          = 32.0   # reference (typical) body extent; collar = PV_KEEP_MM here.
CAP_MIN_MM           = 26.0   # safety floor: a degenerate/tiny seed can't collapse the
                              # cap into the body. Keep < (smallest real body + collar).
CAP_MAX_MM           = 70.0   # safety ceiling: a leaky body can't blow it open.
REACHED_MIN_FRAC     = 0.05  # REJECT the case if the in-mask geodesic front reaches
                             # fewer than this fraction of the TotalSeg LA body voxels.
                             # Near-zero = the grown region is topologically disconnected
                             # from the LA (TotalSeg mislabel / off seed, e.g. spine on
                             # case 621). A real LA's body is ~fully reachable, so this
                             # only fires on broken cases; it does NOT recover them --
                             # it emits an empty mask so the case self-excludes (03 skips
                             # empties). Hard topological signal, not a tuned shape cutoff.
OPEN_RADIUS_MM       = 2.0   # opening kernel radius. VALIDATED = 2.0 on the 2 legacy
                             # + 2 pcct scrap set, paired with CLOSE_MM=0: the carina
                             # held (the old recon2 fusion was the CLOSING, not the
                             # opening) and r=2.0 strips more small artifacts than 1.0.
                             # With RECON_MM matched, the body floods back so net
                             # removal saturates; this is the chosen cleanup/detail
                             # balance. Radius sets BOTH cutoff (2r) AND rounding (r);
                             # 4.0 ate the LAA + ostia, so do not exceed ~3.0.

# Opening mode -- how the de-potato / branch-cut step works. Opening is anti-
# extensive (open(mask) subset of mask) so it can only REMOVE surface, never add
# it back. That gives two endpoints and one knob between them:
#   "plain"         : erode->dilate. The ONLY mode that severs a connected thin
#                     tube, but rounds the body (potato) because the dilate-back
#                     cannot restore concavities/detail it ate.
#   "recon_full"    : open to a core, then flood it back up under the original
#                     mask (BinaryReconstructionByDilation). Restores the body
#                     PERFECTLY but on a single blob is ~a no-op: it also refloods
#                     every connected branch. Use to get the crisp-body / no-cut
#                     endpoint back.
#   "recon_limited" : open to a core, then flood back toward the mask boundary
#                     ONLY up to RECON_MM. Body concavities within RECON_MM
#                     recover (de-potato); branches longer than RECON_MM regrow
#                     just an RECON_MM stub. RECON_MM=0 == plain; large == recon_full.
#   "none"          : no opening at all (v4 behaviour). Crispest body, no potato,
#                     no opening-dilate bridging the LSPV/LAA carina -- but leaves
#                     thin vessels the geodesic cap did not already trim.
# NOTE: branches running PARALLEL/close to the LA wall have no thin neck to open
# through -- NO morphology mode cuts them. That needs the centerline 90-deg cut.
OPEN_MODE            = "recon_limited"
RECON_MM             = 2.0   # only used in recon_limited: how far (mm) the opened
                             # core floods back toward the mask boundary. Matched to
                             # OPEN_RADIUS_MM=2.0 in the validated config. NOTE: matching
                             # makes net removal saturate (erode r, flood back r). To
                             # remove MORE decisively, DECOUPLE: keep this low (~1.0)
                             # while raising OPEN_RADIUS_MM (costs a little body potato).

CLOSE_MM             = 0.0   # closing DISABLED (validated). A closing dilate BRIDGES
                             # the LSPV/LAA carina and the LPV/LA ridge -- that was the
                             # fusion seen on the scrap set. 0 = skip closing entirely.
                             # If tiny pinholes ever need filling, 1.0 is the ceiling;
                             # do not go back to 2.0.
GLOB                 = "*.nii.gz"

# === Timing helper ===
class T:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *a): print(f"    [time] {self.label}: {time.perf_counter() - self.t0:.2f}s")

# === Morphology helpers ===
def _radius_vox(mask, mm):
    """Per-axis voxel radius approximating `mm` mm, given image spacing."""
    return [max(1, int(round(mm / s))) for s in mask.GetSpacing()]

def erode_mm(mask, mm):
    """Binary erode by ~`mm` mm."""
    return sitk.BinaryErode(mask, _radius_vox(mask, mm))

def dilate_mm(mask, mm):
    """Binary dilate by ~`mm` mm."""
    return sitk.BinaryDilate(mask, _radius_vox(mask, mm))

def open_mm(mask, mm):
    """Plain morphological opening (erode -> dilate) at `mm` radius. Erases
    structures thinner than ~2*mm diameter and does NOT regrow them (no geodesic
    reconstruction). Thick body shrinks then regrows to ~original, losing only
    surface concavities smaller than the kernel."""
    r = _radius_vox(mask, mm)
    return sitk.BinaryDilate(sitk.BinaryErode(mask, r), r)

def close_mm(mask, mm):
    """Binary closing (dilate -> erode) by ~mm. Fills small dents/holes."""
    r = _radius_vox(mask, mm)
    return sitk.BinaryErode(sitk.BinaryDilate(mask, r), r)

def recon_by_dilation_limited(marker, mask, mm):
    """Geodesic (mask-constrained) dilation of `marker`, capped at ~mm distance.
    Iterated 1-voxel dilate-then-AND-mask: the marker grows back toward the mask
    boundary up to mm, never leaving the mask. Restores body concavities the
    opening rounded off (within mm), while a branch longer than mm only regrows
    an mm-long stub. mm large -> approaches full reconstruction; mm 0 -> no-op."""
    min_sp = min(mask.GetSpacing())
    steps  = max(0, int(round(mm / min_sp)))
    cur = marker
    for _ in range(steps):
        cur = sitk.And(sitk.BinaryDilate(cur, [1, 1, 1]), mask)
    return cur

def open_branchcut(mask):
    """Branch-cut / de-potato step, dispatched on OPEN_MODE. See OPEN_MODE docs."""
    if OPEN_MODE == "none":
        return mask                       # v4-style: no opening at all (crispest body)
    if OPEN_MODE == "plain":
        return open_mm(mask, OPEN_RADIUS_MM)
    core = open_mm(mask, OPEN_RADIUS_MM)
    if OPEN_MODE == "recon_full":
        return sitk.BinaryReconstructionByDilation(core, mask)
    if OPEN_MODE == "recon_limited":
        return recon_by_dilation_limited(core, mask, RECON_MM)
    raise ValueError(f"unknown OPEN_MODE: {OPEN_MODE!r}")

# === Threshold from seed intensities ===
def adaptive_threshold(ct, seed_mask):
    """Blood-pool HU window from intensities inside eroded seed. Percentile-based."""
    ct_arr   = sitk.GetArrayFromImage(ct).astype(np.float32)
    seed_arr = sitk.GetArrayFromImage(seed_mask).astype(bool)
    vals = ct_arr[seed_arr]
    p2, p98 = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    lo = max(120.0,  p2  - 130.0)   # VALIDATED default (was 50, then 100): widened in
                                    # steps to reach dimmer blood in distal PV segments
                                    # (PVs still a tad short at -100). v6 forbidden blockers
                                    # (RV/PA padded) keep the wider window from spilling.
                                    # Floor stays 120 HU; if darker PVs persist (legacy
                                    # cases clamp at the floor), lower the floor next.
    hi = min(1500.0, p98 + 150.0)
    print(f"    seed HU: p2={p2:.0f}  p98={p98:.0f}  -> threshold [{lo:.0f}, {hi:.0f}]")
    return lo, hi

# === Connected components ===
def largest_component(mask):
    """Keep only the largest connected component."""
    cc  = sitk.ConnectedComponent(mask)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.BinaryThreshold(rel, 1, 1, 1, 0)

# === Forbidden mask builder ===
def build_forbidden(seg):
    """Union of forbidden-neighbour labels, each dilated per FORBIDDEN dict."""
    forbidden = None
    counts = {}
    for label, dmm in FORBIDDEN.items():
        m = sitk.BinaryThreshold(seg, label, label, 1, 0)
        if dmm > 0:
            m = dilate_mm(m, dmm)
        counts[label] = int(sitk.GetArrayFromImage(m).sum())
        forbidden = m if forbidden is None else sitk.Or(forbidden, m)
    return forbidden, counts

# === Geodesic distance-from-centroid cap (cuts distal PVs along the tube) ===
def derive_adaptive_cap(arrival, body_arr):
    """Scale the cap to LA size. body_extent = P{CAP_PCTILE} of the geodesic
    arrival field restricted to the TotalSeg LA body (no PV); the PV collar grows
    with body size; cap = body_extent + collar, clamped. Returns
    (cap_mm, body_extent_mm, collar_mm)."""
    bvals = arrival[body_arr > 0]
    bvals = bvals[bvals < 1e6]          # drop FastMarching's unreached sentinel
    if bvals.size == 0:                 # body not reached (bad grow) -> fall back
        return MAX_GEODESIC_MM, float("nan"), float("nan")
    body_extent = float(np.percentile(bvals, CAP_PCTILE))
    collar = PV_KEEP_MM + PV_KEEP_SLOPE * (body_extent - BODY_REF_MM)
    cap = min(CAP_MAX_MM, max(CAP_MIN_MM, body_extent + collar))
    return cap, body_extent, collar


def geodesic_centroid_cap(mask, seed, body=None):
    """Cap mask to voxels within a GEODESIC (in-mask path) distance of the LA seed
    centroid. Distance travels up the PV tubes instead of slicing a sphere, so
    body voxels stay ~0 and PV voxels grow with how far they run up the tube.
    Components not reachable from the seed are dropped (CC filter).

    Cap distance: fixed MAX_GEODESIC_MM, or derived per case from `body` (the
    TotalSeg LA body mask) when ADAPTIVE_CAP. Returns (capped_mask, info) where
    info = {cap_mm, body_extent_mm}."""
    arr = sitk.GetArrayFromImage(seed)
    zs, ys, xs = np.where(arr > 0)
    if len(zs) == 0:
        return mask, {"cap_mm": float("nan"), "body_extent_mm": float("nan")}

    # LA-seed centroid, then snap to the nearest voxel of the GROWN mask (the
    # speed=1 region being capped) -- NOT the seed mask. The eroded TotalSeg seed
    # can lie outside the grown body after forbidden-subtraction + largest-CC; a
    # trial point there sits in speed=0, so FastMarching never propagates and the
    # arrival field degenerates (body P90=0 -> cap clamps to the floor and empties
    # the mask, e.g. case 621). Snapping into `mask` guarantees a valid front.
    mask_arr = sitk.GetArrayFromImage(mask)
    mzs, mys, mxs = np.where(mask_arr > 0)
    if len(mzs) == 0:                       # grown mask empty -> nothing to cap
        return mask, {"cap_mm": float("nan"), "body_extent_mm": float("nan")}
    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())
    d2 = (mxs - cx) ** 2 + (mys - cy) ** 2 + (mzs - cz) ** 2
    j = int(np.argmin(d2))
    seed_idx = (int(mxs[j]), int(mys[j]), int(mzs[j]))   # (x, y, z) index, in-mask
    print(f"    geodesic seed index (x,y,z): {seed_idx}")

    # Speed = 1 inside mask, 0 outside. With speed 0 the front cannot leave the
    # mask, so FastMarching arrival time IS the in-mask geodesic distance.
    # FastMarching is spacing-aware, so arrival is in mm. March far enough to
    # cover the largest possible cap (CAP_MAX_MM when adaptive; cap is unknown
    # until the body is measured).
    march_to = CAP_MAX_MM if ADAPTIVE_CAP else MAX_GEODESIC_MM
    speed = sitk.Cast(mask, sitk.sitkFloat32)
    fm = sitk.FastMarchingImageFilter()
    fm.AddTrialPoint(seed_idx)
    fm.SetStoppingValue(march_to * 1.5)   # stop marching past the cap, for speed
    arrival = sitk.GetArrayFromImage(fm.Execute(speed))

    if ADAPTIVE_CAP and body is not None:
        body_arr = sitk.GetArrayFromImage(body)
        # Fraction of the TotalSeg LA body the in-mask geodesic front actually reached.
        # If ~none of it is reachable, the grown region is disconnected from the LA
        # (TotalSeg mislabel / off seed -- e.g. case 621 grew the spine). REJECT the
        # case rather than recover it: a recovered mask here is plausible-by-volume
        # garbage that could slip into training, whereas an empty mask self-excludes
        # (03 skips empties). This is a hard topological test, not a shape heuristic.
        n_body = int((body_arr > 0).sum())
        reached_frac = float((arrival[body_arr > 0] < 1e6).sum()) / max(n_body, 1)
        if reached_frac < REACHED_MIN_FRAC:
            print(f"    [REJECT] grown mask reaches only {reached_frac:.1%} of the LA "
                  f"body (< {REACHED_MIN_FRAC:.0%}) -> not LA (TotalSeg/seed failure); "
                  f"emitting empty mask")
            empty = sitk.Image(mask.GetSize(), mask.GetPixelID())
            empty.CopyInformation(mask)
            return empty, {"cap_mm": float("nan"), "body_extent_mm": float("nan"),
                           "reached_frac": reached_frac, "rejected": True}
        cap_mm, body_extent, collar = derive_adaptive_cap(arrival, body_arr)
        print(f"    adaptive cap: body P{CAP_PCTILE:.0f}={body_extent:.1f}mm, "
              f"size-scaled collar={collar:.1f}mm -> cap {cap_mm:.1f}mm "
              f"(clamp [{CAP_MIN_MM:.0f},{CAP_MAX_MM:.0f}]) | body reached={reached_frac:.0%}")
    else:
        cap_mm, body_extent = MAX_GEODESIC_MM, float("nan")

    # Unreached / far voxels hold FastMarching's large default value -> dropped.
    keep = (arrival <= cap_mm).astype(np.uint8)
    keep_img = sitk.GetImageFromArray(keep)
    keep_img.CopyInformation(mask)
    return sitk.And(mask, keep_img), {"cap_mm": cap_mm, "body_extent_mm": body_extent}

# === Per-case pipeline ===
def grow_one(case):
    ct_path  = NIFTI_DIR / f"{case}.nii.gz"
    seg_path = SEG_DIR   / f"{case}_chambers.nii.gz"
    if not (ct_path.exists() and seg_path.exists()):
        print(f"[{case}] missing inputs, skip"); return

    with T("read CT + seg"):
        ct  = sitk.ReadImage(str(ct_path))
        seg = sitk.ReadImage(str(seg_path))

    # 1) Seed = TotalSeg LA eroded
    with T("build + erode LA seed"):
        la_full = sitk.BinaryThreshold(seg, LA_LABEL, LA_LABEL, 1, 0)
        la_seed = erode_mm(la_full, ERODE_MM)
    n_seed = int(sitk.GetArrayFromImage(la_seed).sum())
    if n_seed == 0:
        print(f"[{case}] seed empty after erosion, skip"); return
    print(f"  seed voxels: {n_seed}")

    # 2) Threshold
    lo, hi = adaptive_threshold(ct, la_seed)

    # 3) Forbidden zone: LV + RA (raw) + aorta + RV + PA (dilated)
    with T("forbidden mask"):
        forbidden, counts = build_forbidden(seg)
    print("  forbidden voxels: " + "  ".join(f"L{k}={v}" for k, v in counts.items()))

    # 4) Seed list for ConnectedThreshold (subsample for speed; result identical)
    with T("seed list"):
        seed_arr = sitk.GetArrayFromImage(la_seed)
        zs, ys, xs = np.where(seed_arr > 0)
        rng = np.random.default_rng(0)
        if len(zs) > 1000:
            idx = rng.choice(len(zs), 1000, replace=False)
            zs, ys, xs = zs[idx], ys[idx], xs[idx]
        seed_list = [(int(x), int(y), int(z)) for x, y, z in zip(xs, ys, zs)]

    # 5) Region grow
    with T("region grow"):
        grown = sitk.ConnectedThreshold(ct, seedList=seed_list,
                                        lower=float(lo), upper=float(hi),
                                        replaceValue=1)
    with T("subtract forbidden"):
        grown = sitk.And(grown, sitk.Not(forbidden))
    with T("largest CC #1"):
        grown = largest_component(grown)
    print(f"  after forbidden-block + largest CC: "
          f"{int(sitk.GetArrayFromImage(grown).sum())} voxels")

    # 6) Geodesic distance cap (cuts distal PVs along the tube; drops stray CCs).
    #    Pass la_full (TotalSeg LA body) so the adaptive cap can measure body extent.
    with T("geodesic centroid cap"):
        pruned, cap_info = geodesic_centroid_cap(grown, la_seed, body=la_full)
    print(f"  after geodesic cap (<{cap_info['cap_mm']:.1f}mm in-mask from centroid): "
          f"{int(sitk.GetArrayFromImage(pruned).sum())} voxels")

    # 6b) Thin-branch removal: opening (mode-dependent) keeps the thick trunk
    with T(f"opening (mode={OPEN_MODE}, r={OPEN_RADIUS_MM}mm, recon={RECON_MM}mm)"):
        pruned = open_branchcut(pruned)
    with T("largest CC after opening"):
        pruned = largest_component(pruned)
    print(f"  after opening (cut <~{2*OPEN_RADIUS_MM:.0f}mm diameter): "
          f"{int(sitk.GetArrayFromImage(pruned).sum())} voxels")

    # 7) Closing to fill small dents/holes (skip entirely when CLOSE_MM <= 0;
    #    a closing dilate can BRIDGE the LSPV/LAA carina or the LPV/LA ridge).
    if CLOSE_MM > 0:
        with T("closing"):
            pruned = close_mm(pruned, CLOSE_MM)
        with T("largest CC #final"):
            pruned = largest_component(pruned)

    # 8) Save
    with T("write output"):
        LA_DIR.mkdir(parents=True, exist_ok=True)
        n_final = int(sitk.GetArrayFromImage(pruned).sum())
        vol_ml  = n_final * np.prod(pruned.GetSpacing()) / 1000.0
        sitk.WriteImage(pruned, str(LA_DIR / f"{case}_LA.nii.gz"))
    print(f"  final: {n_final} voxels = {vol_ml:.1f} mL")
    print(f"[{case}] saved")
    return {"case": case, "voxels": n_final, "vol_ml": vol_ml,
            "cap_mm": cap_info["cap_mm"], "body_extent_mm": cap_info["body_extent_mm"]}

# === Entry point ===
if __name__ == "__main__":
    import sys
    cases_filter = set(sys.argv[1:])   # optional: pass case stems to process only those
    t_total = time.perf_counter()
    for nii in sorted(NIFTI_DIR.glob(GLOB)):
        case = nii.name.replace(".nii.gz", "")
        if cases_filter and case not in cases_filter:
            continue
        print(f"\n[{case}] region-growing...")
        grow_one(case)
    print(f"\n[total] {time.perf_counter() - t_total:.1f}s")