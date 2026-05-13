"""Run TotalSegmentator on each NIfTI, keep LA label only."""
from pathlib import Path
import subprocess, SimpleITK as sitk, numpy as np

NIFTI_DIR = Path("derivatives/nifti")
SEG_DIR   = Path("derivatives/seg_full")     # full multi-label output
LA_DIR    = Path("derivatives/seg_la")       # binary LA mask only
LA_DIR.mkdir(parents=True, exist_ok=True)
SEG_DIR.mkdir(parents=True, exist_ok=True)

# TotalSegmentator heartchambers_highres label indices (check docs if version differs)
# 1=heart_myocardium, 2=heart_atrium_left, 3=heart_atrium_right, 4=ventricle_left, 5=ventricle_right, ...
LA_LABEL = 2

for nii in sorted(NIFTI_DIR.glob("*.nii.gz")):
    case = nii.name.replace(".nii.gz", "")
    seg_out = SEG_DIR / f"{case}.nii.gz"
    if not seg_out.exists():
        print(f"[{case}] segmenting...")
        # --fast off; highres heart task; CPU is fine for prototyping
        subprocess.run([
            "TotalSegmentator", "-i", str(nii), "-o", str(seg_out),
            "--task", "heartchambers_highres", "--ml"   # --ml writes single multi-label file
        ], check=True)
    # Extract LA
    seg = sitk.ReadImage(str(seg_out))
    arr = sitk.GetArrayFromImage(seg)
    la  = (arr == LA_LABEL).astype(np.uint8)
    la_img = sitk.GetImageFromArray(la)
    la_img.CopyInformation(seg)   # preserve spacing/origin/direction
    sitk.WriteImage(la_img, str(LA_DIR / f"{case}_LA.nii.gz"))
    print(f"[{case}] LA mask written")