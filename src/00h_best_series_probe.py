"""Dump ALL substantial recons for given cases (no FOV filter), with TP%,
best-phase token, FOV, kernel, ImageType. Goal: identify the 'best' series
the tech/scanner selected and find a tag that singles it out."""
import re
import zipfile, tempfile, shutil
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
import pydicom

DATA_ROOT = Path("data")
CASES = ["2781_0___do92008557", "1161_0___do92008654", "1638_0___do92008611"]

def kernel_str(ds):
    k = getattr(ds, "ConvolutionKernel", "")
    if not isinstance(k, str):
        try: k = k[0]
        except Exception: k = ""
    return str(k)

def scan_opts_str(ds):
    so = getattr(ds, "ScanOptions", "")
    if isinstance(so, str): return so
    try: return " ".join(str(x) for x in so)
    except Exception: return str(so)

def parse_tp(so):
    m = re.search(r"TP(\d+)PC", so); return int(m.group(1)) if m else None
def parse_bestph(so):
    m = re.search(r"BESTPH_S_P(\d+)PC", so); return int(m.group(1)) if m else None
def has_gate(so):
    return "GATE_" in so

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
    print(f"{'ser':>4} {'n':>5} {'kernel':>6} {'thk':>4} {'FOV':>6} "
          f"{'acq':>4} {'TP':>4} {'best':>4} {'gate':>4}  imageType")
    rows = []
    for uid, items in series.items():
        ds = items[0]; n = len(items)
        if n < 50: continue
        so = scan_opts_str(ds)
        rows.append((getattr(ds, "SeriesNumber", "?"), n, kernel_str(ds),
                     getattr(ds, "SliceThickness", "?"),
                     getattr(ds, "ReconstructionDiameter", "?"),
                     getattr(ds, "AcquisitionNumber", "?"),
                     parse_tp(so), parse_bestph(so), has_gate(so),
                     "/".join(getattr(ds, "ImageType", []))))
    rows.sort(key=lambda r: str(r[0]))
    for ser, n, ker, thk, fov, acq, tp, best, gate, itype in rows:
        print(f"{str(ser):>4} {n:>5} {ker:>6} {str(thk):>4} {str(fov)[:6]:>6} "
              f"{str(acq):>4} {str(tp):>4} {str(best):>4} {str(gate):>4}  {itype}")

if __name__ == "__main__":
    for name in CASES:
        entry = DATA_ROOT / f"{name}.zip"
        if not entry.exists(): entry = DATA_ROOT / name
        if not entry.exists():
            print(f"\n=== {name} === [not found]"); continue
        with case_source(entry) as cd:
            dump(cd, name)