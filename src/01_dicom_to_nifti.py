"""DICOM series -> NIfTI. Expects data/<case_id>/venous/ with .dcm files."""
import SimpleITK as sitk
from pathlib import Path
import sys

def convert(dicom_dir: Path, out_path: Path):
    # Read DICOM series (sorts slices by position automatically)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    # If multiple series in folder, take the largest (usually the main acquisition)
    best = max(series_ids, key=lambda s: len(reader.GetGDCMSeriesFileNames(str(dicom_dir), s)))
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), best)
    reader.SetFileNames(files)
    img = reader.Execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    print(f"  wrote {out_path}  shape={img.GetSize()}  spacing={img.GetSpacing()}")

if __name__ == "__main__":
    data_root = Path("data")
    out_root  = Path("derivatives/nifti")
    # Iterate over each case folder
    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir(): continue
        ven = case_dir / "venous"
        if not ven.exists():
            print(f"skip {case_dir.name}: no venous/ folder"); continue
        print(f"[{case_dir.name}]")
        convert(ven, out_root / f"{case_dir.name}.nii.gz")