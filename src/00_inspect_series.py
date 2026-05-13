"""List all DICOM series in data/ with key metadata, so we can choose."""
from pathlib import Path
import pydicom
from collections import defaultdict

DATA = Path("data")

def summarize_series(case_dir: Path):
    # Group files by SeriesInstanceUID
    series = defaultdict(list)
    for f in case_dir.rglob("*"):
        if not f.is_file(): continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid: series[uid].append((f, ds))

    print(f"\n=== {case_dir.name}: {len(series)} series ===")
    rows = []
    for uid, items in series.items():
        ds = items[0][1]  # first file's header
        rows.append({
            "n":        len(items),
            "series":   getattr(ds, "SeriesNumber", "?"),
            "desc":     str(getattr(ds, "SeriesDescription", ""))[:40],
            "thk":      getattr(ds, "SliceThickness", "?"),
            "px":       getattr(ds, "PixelSpacing", ["?"])[0],
            "kernel":   getattr(ds, "ConvolutionKernel", "?"),
            "uid":      uid[-12:],   # short tail for ID
        })
    # Sort by series number for readability
    rows.sort(key=lambda r: (str(r["series"])))
    # Print as table
    hdr = f"{'n':>4} {'series':>7} {'thk':>5} {'px':>5} {'kernel':>10}  desc"
    print(hdr); print("-"*len(hdr))
    for r in rows:
        print(f"{r['n']:>4} {str(r['series']):>7} {str(r['thk']):>5} {str(r['px']):>5} {str(r['kernel']):>10}  {r['desc']}")

if __name__ == "__main__":
    for case in sorted(DATA.iterdir()):
        if case.is_dir(): summarize_series(case)