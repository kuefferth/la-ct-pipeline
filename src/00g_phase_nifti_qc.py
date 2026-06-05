"""Phase QC: for old-scanner (and PCCT, for cross-check) cardiac candidates,
parse the TPxx %R-R token from ScanOptions, print a calibration table, and
export one NIfTI per discrete (non-4D) phase recon for visual QC in Slicer.

TPxxPCyyyy in ScanOptions: xx = cardiac phase as % R-R (proven via GATE_nn_OF_21,
e.g. gate 15 of 21 -> TP70). yyyy is ~R-R interval, ignored here.
Series carrying a GATE_ token are the 4D multiphase stack and are skipped.
"""
import re
import zipfile, tempfile, shutil
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
import pydicom
import SimpleITK as sitk

DATA_ROOT = Path("data")
QC_DIR    = Path("derivatives/qc_phases")

# Cases to export NIfTIs for (keep small; each series is a full volume).
# Mix: a gated phase-sweep case + two twin-acq cases.
EXPORT_CASES = [
    "1161_0___do92008654",   # gated sweep: TP10/20/30
    "1638_0___do92008611",   # twin-acq: TP75 + partner
    "1492_0___do92008759",   # twin-acq: TP8 + TP13
]

# Cases to include in the printed TP table only (no NIfTI). Add PCCT for cross-check.
TABLE_ONLY = ["4733", "4734", "4735", "4736", "4738"]

# Candidate filter for old-scanner thin-slice cardiac recons
def is_candidate(ds):
    n_ok = True  # n checked by caller
    thk = getattr(ds, "SliceThickness", None)
    fov = getattr(ds, "ReconstructionDiameter", None)
    try:
        if float(thk) > 1.0 + 0.05: return False
        if float(fov) > 250: return False
    except Exception:
        return False
    return True

def scan_opts_str(ds):
    """ScanOptions may be str or MultiValue; join to one searchable string."""
    so = getattr(ds, "ScanOptions", "")
    if isinstance(so, str):
        return so
    try:
        return " ".join(str(x) for x in so)
    except Exception:
        return str(so)

def parse_tp(scanopts):
    """Return cardiac phase %R-R from TPxx token, or None."""
    m = re.search(r"TP(\d+)PC", scanopts)
    return int(m.group(1)) if m else None

def has_gate(scanopts):
    """True if this is a 4D multiphase gate recon (exclude)."""
    return "GATE_" in scanopts

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

def group_series(case_dir: Path):
    series = defaultdict(list)
    for f in case_dir.rglob("*"):
        if not f.is_file(): continue
        try: ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception: continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid: series[uid].append((f, ds))
    return series

def collect_candidates(series):
    """Return list of dicts for non-4D thin-slice cardiac series with TP parsed."""
    out = []
    for uid, items in series.items():
        ds = items[0][1]
        n  = len(items)
        if n < 100: continue
        if not is_candidate(ds): continue
        so = scan_opts_str(ds)
        if has_gate(so): continue          # skip 4D stack
        out.append({
            "uid":  uid,
            "ser":  getattr(ds, "SeriesNumber", "?"),
            "n":    n,
            "acq":  getattr(ds, "AcquisitionNumber", "?"),
            "tp":   parse_tp(so),
            "items": items,
            "folder": str(items[0][0].parent),
        })
    out.sort(key=lambda r: (str(r["acq"]), str(r["ser"])))
    return out

def convert_series(cand, out_path: Path):
    reader = sitk.ImageSeriesReader()
    ordered = reader.GetGDCMSeriesFileNames(cand["folder"], cand["uid"])
    if not ordered:
        ordered = [str(p) for p, _ in cand["items"]]
    reader.SetFileNames(ordered)
    img = reader.Execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))

if __name__ == "__main__":
    print("=== TP (%R-R) calibration table ===")
    print(f"{'case':>16} {'ser':>4} {'n':>5} {'acq':>4} {'TP%':>5}")

    # Table-only cases (PCCT cross-check, plus quick scan of export cases)
    for name in TABLE_ONLY + EXPORT_CASES:
        entry = DATA_ROOT / f"{name}.zip"
        if not entry.exists(): entry = DATA_ROOT / name
        if not entry.exists():
            print(f"{name:>16}  [not found]"); continue
        short = name.split("_")[0]
        with case_source(entry) as cd:
            cands = collect_candidates(group_series(cd))
            for c in cands:
                tp = c["tp"] if c["tp"] is not None else "?"
                print(f"{short:>16} {str(c['ser']):>4} {c['n']:>5} "
                      f"{str(c['acq']):>4} {str(tp):>5}")

    print("\n=== exporting QC NIfTIs ===")
    QC_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_CASES:
        entry = DATA_ROOT / f"{name}.zip"
        if not entry.exists(): entry = DATA_ROOT / name
        if not entry.exists(): continue
        short = name.split("_")[0]
        with case_source(entry) as cd:
            cands = collect_candidates(group_series(cd))
            for c in cands:
                tp = c["tp"] if c["tp"] is not None else 99
                out = QC_DIR / f"{short}_ser{c['ser']}_TP{tp:02d}_acq{c['acq']}.nii.gz"
                if out.exists():
                    print(f"  [skip] {out.name}"); continue
                print(f"  writing {out.name} (n={c['n']})...")
                convert_series(c, out)
    print("\ndone. Load derivatives/qc_phases/*.nii.gz in Slicer to compare phases.")