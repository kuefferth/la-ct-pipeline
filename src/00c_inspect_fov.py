"""Check FOV/ImageType to distinguish cardiac vs thorax recon."""
from pathlib import Path
import pydicom
from collections import defaultdict

DATA = Path("data")

def dump(case_dir: Path):
    series = defaultdict(list)
    for f in case_dir.rglob("*"):
        if not f.is_file(): continue
        try: ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception: continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid: series[uid].append(ds)

    print(f"\n=== {case_dir.name} ===")
    print(f"{'ser':>4} {'n':>4} {'kernel':>8} {'thk':>5} {'px':>6} {'FOV':>6} {'acq':>4}  imageType")
    rows = []
    for uid, items in series.items():
        ds = items[0]
        rows.append({
            "ser":   getattr(ds, "SeriesNumber", "?"),
            "n":     len(items),
            "kernel": str(getattr(ds, "ConvolutionKernel", ""))[:8],
            "thk":   getattr(ds, "SliceThickness", "?"),
            "px":    getattr(ds, "PixelSpacing", ["?"])[0],
            "fov":   getattr(ds, "ReconstructionDiameter", "?"),
            "acq":   getattr(ds, "AcquisitionNumber", "?"),
            "itype": "/".join(getattr(ds, "ImageType", [])),
        })
    rows.sort(key=lambda r: str(r["ser"]))
    for r in rows:
        # Only show real recons (skip topograms / tiny series)
        if r["n"] < 50: continue
        print(f"{str(r['ser']):>4} {r['n']:>4} {r['kernel']:>8} {str(r['thk']):>5} "
              f"{str(r['px'])[:6]:>6} {str(r['fov']):>6} {str(r['acq']):>4}  {r['itype']}")

if __name__ == "__main__":
    for case in sorted(DATA.iterdir()):
        if case.is_dir(): dump(case)