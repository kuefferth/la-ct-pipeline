"""Run TotalSegmentator heartchambers_highres on each NIfTI, extract LA mask.
Uses --force_split to fit 1060 6GB VRAM. Requires academic license (already set).
"""
import os
os.environ["nnUNet_n_proc_DA"] = "0"             # disable nnUNet dataloader workers
os.environ["nnUNet_def_n_proc"] = "1"            # 1 process for inference
os.environ["TORCH_COMPILE_DISABLE"] = "1"        # no torch.compile (Windows-broken)
os.environ["nnUNet_compile"] = "f"

import gc
import time
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from totalsegmentator.python_api import totalsegmentator
from _deadline import time_up

NIFTI_DIR = Path("derivatives/nifti")
SEG_DIR   = Path("derivatives/seg_full")
LA_DIR    = Path("derivatives/seg_la")

# heartchambers_highres label order: 1=myocardium, 2=atrium_left, 3=ventricle_left,
# 4=atrium_right, 5=ventricle_right, 6=aorta, 7=pulmonary_artery
LA_LABEL  = 2
GLOB      = "*.nii.gz"   # change to "*.nii.gz" for all cases later

# A failing case (e.g. MemoryError on the 16 GB work PC) must not kill the whole night:
# it is logged, its partial output removed, and the loop moves on. After MAX_FAILS logged
# failures a case is skipped until its row is removed from FAIL_LOG.
FAIL_LOG  = Path("derivatives/logs/seg_failures.csv")
MAX_FAILS = 3
# Soft RAM guard only: this PC idles at ~3 GB "available" (Chrome/Citrix open) and still
# segments fine via the 26 GB pagefile. A hard 6 GB gate skipped EVERY case on 11-04/05 and
# cost two nights, so after a short wait the case is attempted anyway (try/except below
# catches a real MemoryError).
MIN_FREE_GB = 2.0
RAM_WAIT_S  = 120

def fail_counts():
    if not FAIL_LOG.exists():
        return {}
    counts = {}
    for line in FAIL_LOG.read_text(encoding="utf-8-sig").splitlines()[1:]:
        c = line.split(";")[1] if ";" in line else ""
        counts[c] = counts.get(c, 0) + 1
    return counts

def log_fail(case, err):
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not FAIL_LOG.exists()
    with open(FAIL_LOG, "a", encoding="utf-8-sig" if new else "utf-8") as f:
        if new:
            f.write("time;case;error\n")
        msg = " ".join(str(err).replace(";", ",").split())[:150]
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')};{case};{msg}\n")

def wait_for_ram():
    """True once enough RAM is free (polls up to RAM_WAIT_S); True if psutil is missing."""
    try:
        import psutil
    except ImportError:
        return True
    waited = 0
    while psutil.virtual_memory().available / 1e9 < MIN_FREE_GB:
        if waited >= RAM_WAIT_S or time_up():
            return False
        time.sleep(30)
        waited += 30
    return True

def run():
    LA_DIR.mkdir(parents=True, exist_ok=True)
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    fails = fail_counts()
    for nii in sorted(NIFTI_DIR.glob(GLOB)):
        if time_up():                      # nightly window over: stop between cases
            print("[stop] time window over")
            break
        case = nii.name.replace(".nii.gz", "")
        seg_out = SEG_DIR / f"{case}_chambers.nii.gz"
        seed_out = LA_DIR / f"{case}_LA_seed.nii.gz"
        if seg_out.exists() and seed_out.exists():
            continue                       # done (re-extracting ~700 seeds cost ~5 min a night)
        if fails.get(case, 0) >= MAX_FAILS:
            print(f"[{case}] skipped, failed {fails[case]}x (see {FAIL_LOG})")
            continue
        try:
            if not seg_out.exists():
                if not wait_for_ram():
                    print(f"[{case}] low free RAM after {RAM_WAIT_S}s wait, trying anyway")
                print(f"[{case}] segmenting (heartchambers_highres, force_split)...")
                # force_split = z-chunks, fits 6GB VRAM on GTX 1060
                totalsegmentator(
                                input=str(nii), output=str(seg_out),
                                task="heartchambers_highres", ml=True,
                                device="cpu",
                            )
            # Extract LA
            seg = sitk.ReadImage(str(seg_out))
            arr = sitk.GetArrayFromImage(seg)
            la  = (arr == LA_LABEL).astype(np.uint8)
            if la.sum() == 0:
                print(f"[{case}] WARNING: no voxels for LA label {LA_LABEL}")
            la_img = sitk.GetImageFromArray(la)
            la_img.CopyInformation(seg)
            sitk.WriteImage(la_img, str(seed_out))
            print(f"[{case}] LA mask written ({la.sum()} voxels)")
        except Exception as e:             # MemoryError, nnU-Net worker death, bad file ...
            print(f"[{case}] FAILED: {type(e).__name__}: {e}")
            log_fail(case, f"{type(e).__name__}: {e}")
            seg_out.unlink(missing_ok=True)   # never leave a half-written segmentation
            seed_out.unlink(missing_ok=True)
            gc.collect()
            time.sleep(30)                 # let the OS reclaim memory before the next case

if __name__ == "__main__":
    run()