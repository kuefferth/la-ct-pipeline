"""Find the cardiac-phase discriminator on old dual-source studies.
For each I30f 0.75mm series, dump phase-related tags + per-series timing so we
can tell (a) arterial vs venous across acqs and (b) which R-R phase among
same-acq single-phase recons. Reads first file per series (representative for
single-phase recons)."""
import zipfile, tempfile, shutil
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
import pydicom

DATA_ROOT = Path("data")

# One of each sub-pattern: gated-multiseries (1161, 1300) + twin-acq (1492, 43)
CASES = ["1161_0___do92008654", "1300_0___do92008575", "1492_0___do92008759",
    "1638_0___do92008611", "1752_0___do92008644", "1892_0___do92008720",
    "2096_0___do92008778", "2250_0___do92008633", "226_0___do92008722",
    "2294_0___do92008627", "258_0___do92008744", "2781_0___do92008557",
    "43_0___do92008740", "492_0___do92008672"]

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

# Standard + likely-relevant cardiac/timing tags
TAGS = [
    ("SeriesTime",                      "serTime"),
    ("ContentTime",                     "contTime"),
    ("TriggerTime",                     "trigger"),    # (0018,1060) phase within R-R
    ("NominalPercentageOfCardiacPhase", "phase%"),     # (0020,9241)
    ("CardiacRRIntervalSpecified",      "rrInt"),
    ("CardiacNumberOfImages",           "nCardImg"),
    ("ScanOptions",                     "scanOpts"),
]

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
        if kernel_str(ds) != "I30f": continue
        try:
            if float(thk) > 0.75 + 0.05: continue
        except Exception: continue
        rows.append((getattr(ds, "SeriesNumber", "?"), n,
                     getattr(ds, "AcquisitionNumber", "?"), ds))
    rows.sort(key=lambda r: (str(r[2]), str(r[0])))   # by acq, then series

    for ser, n, acq, ds in rows:
        print(f"\n  ser {ser}  n={n}  acq={acq}")
        for tag, label in TAGS:
            val = getattr(ds, tag, None)
            if val not in (None, ""):
                print(f"    {label:>9}: {val}")
        # Siemens private tags mentioning phase / % / RR / PCT
        for elem in ds:
            if elem.tag.is_private:
                try:
                    sval = str(elem.value)
                    if any(k in sval.upper() for k in ("PHASE", "%", "RR", "PCT", "PULS")):
                        print(f"    priv {elem.tag}: {sval[:70]}")
                except Exception: pass

if __name__ == "__main__":
    for name in CASES:
        entry = DATA_ROOT / f"{name}.zip"
        if not entry.exists(): entry = DATA_ROOT / name
        if not entry.exists():
            print(f"\n=== {name} === [not found]"); continue
        with case_source(entry) as cd:
            dump(cd, name)