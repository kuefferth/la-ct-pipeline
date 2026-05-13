"""Run TotalSegmentator heartchambers_highres on each NIfTI, extract LA mask.
Uses --force_split to fit 1060 6GB VRAM. Requires academic license (already set).
"""
import os
os.environ["nnUNet_n_proc_DA"] = "0"             # disable nnUNet dataloader workers
os.environ["nnUNet_def_n_proc"] = "1"            # 1 process for inference
os.environ["TORCH_COMPILE_DISABLE"] = "1"        # no torch.compile (Windows-broken)
os.environ["nnUNet_compile"] = "f"

from pathlib import Path
import SimpleITK as sitk
import numpy as np
from totalsegmentator.python_api import totalsegmentator

NIFTI_DIR = Path("derivatives/nifti")
SEG_DIR   = Path("derivatives/seg_full")
LA_DIR    = Path("derivatives/seg_la")

# heartchambers_highres label order: 1=myocardium, 2=atrium_left, 3=ventricle_left,
# 4=atrium_right, 5=ventricle_right, 6=aorta, 7=pulmonary_artery
LA_LABEL  = 2
GLOB      = "4733.nii.gz"   # change to "*.nii.gz" for all cases later

def run():
    LA_DIR.mkdir(parents=True, exist_ok=True)
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    for nii in sorted(NIFTI_DIR.glob(GLOB)):
        case = nii.name.replace(".nii.gz", "")
        seg_out = SEG_DIR / f"{case}_chambers.nii.gz"
        if not seg_out.exists():
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
        sitk.WriteImage(la_img, str(LA_DIR / f"{case}_LA.nii.gz"))
        print(f"[{case}] LA mask written ({la.sum()} voxels)")

if __name__ == "__main__":
    run()