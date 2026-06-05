"""For cases that 01 skipped, show all Bv40f 0.4mm series with their
FOV + AcquisitionNumber, so we can see which selection rule rejected them."""
import zipfile, tempfile, shutil
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
import pydicom

DATA_ROOT = Path("data")

# Cases that skipped in the last run. Edit this list as needed.
SKIPPED = [
    "1161_0___do92008654", "1300_0___do92008575", "1492_0___do92008759",
    "1638_0___do92008611", "1752_0___do92008644", "1892_0___do92008720",
    "2096_0___do92008778", "2250_0___do92008633", "226_0___do92008722",
    "2294_0___do92008627", "258_0___do92008744", "2781_0___do92008557",
    "43_0___do92008740", "492_0___do92008672",
]

def kernel_str(ds):
    k = getattr(ds, "ConvolutionKernel", "")
    if not isinstance(k, str):
        try: k = k[0]
        except Exception: k = ""
    return str(k)

@contextmanager
def case_source(entry: Path):
    if entry.is_dir():
        yield entry; return
    tmp = tempfile.mkdtemp(prefix=f"{entry.stem}_")
    try:
        with zipfile.ZipFile(entry) as zf: zf.extractall(tmp)
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def dump(case_dir: Path, name: str):
    series = defaultdict(list)
    for f in case_dir.rglob("*"):
        if not f.is_file(): continue
        try: ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception: continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid: series[uid].append(ds)

    print(f"\n=== {name} ===")
    rows = []
    for uid, items in series.items():
        ds = items[0]
        rows.append({
            "ser":  getattr(ds, "SeriesNumber", "?"),
            "n":    len(items),
            "ker":  kernel_str(ds),
            "thk":  getattr(ds, "SliceThickness", "?"),
            "fov":  getattr(ds, "ReconstructionDiameter", "?"),
            "acq":  getattr(ds, "AcquisitionNumber", "?"),
        })
    rows.sort(key=lambda r: str(r["ser"]))
    for r in rows:
        if r["n"] < 50: continue                 # skip topograms / tiny series
        # Flag the ones that are Bv40f 0.4mm (our target kernel/thickness)
        is_target_kt = (r["ker"] == "Bv40f")
        try: is_target_kt = is_target_kt and abs(float(r["thk"]) - 0.4) <= 0.05
        except Exception: is_target_kt = False
        mark = " <-- Bv40f 0.4mm" if is_target_kt else ""
        print(f"  ser {str(r['ser']):>4}  n={r['n']:>4}  {r['ker']:>8}  "
              f"thk={str(r['thk']):>4}  FOV={str(r['fov']):>8}  acq={str(r['acq']):>5}{mark}")

if __name__ == "__main__":
    for name in SKIPPED:
        entry = DATA_ROOT / f"{name}.zip"
        if not entry.exists():
            entry = DATA_ROOT / name            # fallback: unzipped folder
        if not entry.exists():
            print(f"\n=== {name} === [not found in data/]"); continue
        with case_source(entry) as cd:
            dump(cd, name)