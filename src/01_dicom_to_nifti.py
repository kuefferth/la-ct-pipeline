"""DICOM -> NIfTI with smart series selection (Siemens cardiac CT).
Rules: Bv40f kernel, 0.4mm slices, cardiac FOV (<250mm), AcqNum 501 (diastolic 70%+ RR).
"""
import SimpleITK as sitk
import pydicom
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("data")
OUT_ROOT  = Path("derivatives/nifti")

# Selection criteria
KERNEL = "Bv40f"
THK    = 0.4
FOV_MAX = 250    # mm; cardiac recons are ~170-225, thorax ~368
ACQ_NUM = 501    # diastolic (70%+ RR pulse window)

def group_series(case_dir: Path):
    """Group all files under case_dir by SeriesInstanceUID."""
    series = defaultdict(list)
    for f in case_dir.rglob("*"):
        if not f.is_file(): continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid: series[uid].append((f, ds))
    return series

def pick_series(series):
    """Apply selection rules; return (uid, files) of chosen series."""
    candidates = []
    for uid, items in series.items():
        ds = items[0][1]
        kernel = str(getattr(ds, "ConvolutionKernel", ""))
        thk    = getattr(ds, "SliceThickness", None)
        fov    = getattr(ds, "ReconstructionDiameter", None)
        acq    = getattr(ds, "AcquisitionNumber", None)
        ser    = getattr(ds, "SeriesNumber", None)
        try:
            if kernel != KERNEL: continue
            if abs(float(thk) - THK) > 0.05: continue
            if float(fov) > FOV_MAX: continue
            if int(acq) != ACQ_NUM: continue
        except Exception:
            continue
        candidates.append((int(ser), uid, items))
    if not candidates:
        return None, None
    # If multiple matches, take lowest SeriesNumber and warn
    candidates.sort(key=lambda x: x[0])
    if len(candidates) > 1:
        sers = [c[0] for c in candidates]
        print(f"  [warn] multiple candidates: series {sers}, picking {sers[0]}")
    _, uid, items = candidates[0]
    return uid, items

def convert(items, out_path: Path):
    """Convert a list of (path, ds) DICOM files into a NIfTI volume."""
    file_paths = [str(p) for p, _ in items]
    reader = sitk.ImageSeriesReader()
    # Sort by ImagePositionPatient (z) — SimpleITK doesn't do this automatically when given a flat list
    # Use GDCM's series file ordering via SeriesInstanceUID lookup
    folder = str(items[0][0].parent)
    series_id = items[0][1].SeriesInstanceUID
    ordered = reader.GetGDCMSeriesFileNames(folder, series_id)
    if not ordered:  # fallback: trust given order
        ordered = file_paths
    reader.SetFileNames(ordered)
    img = reader.Execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    print(f"  wrote {out_path.name}  shape={img.GetSize()}  spacing={img.GetSpacing()}")

if __name__ == "__main__":
    for case_dir in sorted(DATA_ROOT.iterdir()):
        if not case_dir.is_dir(): continue
        print(f"\n[{case_dir.name}]")
        series = group_series(case_dir)
        uid, items = pick_series(series)
        if uid is None:
            print(f"  [skip] no matching series found")
            continue
        ds = items[0][1]
        print(f"  picked series {ds.SeriesNumber} (n={len(items)}, FOV={ds.ReconstructionDiameter})")
        convert(items, OUT_ROOT / f"{case_dir.name}.nii.gz")