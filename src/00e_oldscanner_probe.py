"""Rich metadata for old-scanner (non-PCCT) cardiac candidates: thin-slice,
small-FOV series with ImageType + AcquisitionTime, so we can pin down
arterial-vs-venous and standard-vs-DE-variant before writing selection."""
import zipfile, tempfile, shutil
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
import pydicom

DATA_ROOT = Path("data")

# Old-scanner cases that skipped. Edit as needed.
CASES = [
    "1161_0___do92008654", "1300_0___do92008575", "1492_0___do92008759",
    "1638_0___do92008611", "1752_0___do92008644", "1892_0___do92008720",
    "2096_0___do92008778", "2250_0___do92008633", "226_0___do92008722",
    "2294_0___do92008627", "258_0___do92008744", "2781_0___do92008557",
    "43_0___do92008740", "492_0___do92008672",
]

# Candidate filter: thin slice, cardiac-sized FOV, real recon (not topogram)
THK_MAX = 1.0
FOV_MAX = 250
N_MIN   = 100

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
        n  = len(items)
        thk = getattr(ds, "SliceThickness", None)
        fov = getattr(ds, "ReconstructionDiameter", None)
        try:
            if n < N_MIN: continue
            if float(thk) > THK_MAX + 0.05: continue
            if float(fov) > FOV_MAX: continue
        except Exception:
            continue
        rows.append({
            "ser":   getattr(ds, "SeriesNumber", "?"),
            "n":     n,
            "ker":   kernel_str(ds),
            "thk":   thk,
            "fov":   fov,
            "acq":   getattr(ds, "AcquisitionNumber", "?"),
            "time":  getattr(ds, "AcquisitionTime", "?"),
            "itype": "/".join(getattr(ds, "ImageType", [])),
            "desc":  str(getattr(ds, "SeriesDescription", ""))[:30],
        })
    rows.sort(key=lambda r: (str(r["acq"]), str(r["ser"])))
    for r in rows:
        print(f"  ser {str(r['ser']):>4}  n={r['n']:>4}  {r['ker']:>6}  "
              f"thk={str(r['thk']):>4}  FOV={str(r['fov'])[:6]:>6}  "
              f"acq={str(r['acq']):>4}  t={str(r['time']):>10}  "
              f"{r['itype']}  [{r['desc']}]")

if __name__ == "__main__":
    for name in CASES:
        entry = DATA_ROOT / f"{name}.zip"
        if not entry.exists(): entry = DATA_ROOT / name
        if not entry.exists():
            print(f"\n=== {name} === [not found]"); continue
        with case_source(entry) as cd:
            dump(cd, name)