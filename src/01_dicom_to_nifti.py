"""DICOM -> NIfTI with smart series selection (Siemens cardiac CT).

Two scanner profiles, tried in order:
  pcct   : photon-counting. Bv40f, 0.4mm, FOV<250, AcqNum 501 (arterial diastolic).
  legacy : older dual-source (SOMATOM). I30f, 0.75mm, FOV<250, ORIGINAL, no 4D gate.
           Arterial LA phase = lowest AcquisitionNumber; if a phase sweep shares
           that acq, take highest TP%R-R (empirically the largest/cleanest LA).

Accepts unzipped case folders (data/<case>/) and per-case zips (data/<case>.zip).
Output NIfTI is suffixed with the profile that matched: <case>__pcct.nii.gz or
<case>__legacy.nii.gz, so the cohort is visible downstream. A selection_log.csv
(UTF-8 BOM) records what was chosen per case.
"""
import re
import csv
import zipfile
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict

import SimpleITK as sitk
import pydicom

DATA_ROOT = Path("data")
OUT_ROOT  = Path("derivatives/nifti")
LOG_PATH  = Path("derivatives/selection_log.csv")

FOV_MAX = 250    # mm; cardiac recons are well under, thorax ~330-386

# === Tag helpers (robust to MultiValue / anonymization quirks) ===
def kernel_str(ds):
    """ConvolutionKernel: single ('Bv40f') or multi (['Bv40f','3']) -> first elem."""
    k = getattr(ds, "ConvolutionKernel", "")
    if not isinstance(k, str):
        try: k = k[0]
        except Exception: k = ""
    return str(k)

def scan_opts_str(ds):
    """ScanOptions: str or MultiValue -> one searchable string."""
    so = getattr(ds, "ScanOptions", "")
    if isinstance(so, str):
        return so
    try:
        return " ".join(str(x) for x in so)
    except Exception:
        return str(so)

def parse_tp(scanopts):
    """Cardiac phase as %R-R from the TPxxPC token, or None.
    Proven via GATE_nn_OF_21 (gate 15/21 -> TP70)."""
    m = re.search(r"TP(\d+)PC", scanopts)
    return int(m.group(1)) if m else None

def has_gate(scanopts):
    """True if this is a 4D multiphase gate recon (exclude)."""
    return "GATE_" in scanopts

def image_type_str(ds):
    try:
        return "/".join(getattr(ds, "ImageType", []))
    except Exception:
        return ""

def fnum(x):
    """float() that tolerates None / junk by returning None."""
    try: return float(x)
    except Exception: return None

# === Zip / folder source ===
@contextmanager
def case_source(entry: Path):
    """Yield a scannable dir for a data/ entry (folder or .zip). Zips extract to
    a temp dir removed on exit, so peak disk stays ~one study."""
    if entry.is_dir():
        yield entry
        return
    tmp = tempfile.mkdtemp(prefix=f"{entry.stem}_")
    try:
        with zipfile.ZipFile(entry) as zf:
            zf.extractall(tmp)
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

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

# === Selection profiles ===
def pick_pcct(series):
    """Photon-counting: Bv40f, 0.4mm, FOV<250, AcqNum 501. Lowest SeriesNumber
    among matches. Returns dict or None."""
    cands = []
    for uid, items in series.items():
        ds = items[0][1]
        if kernel_str(ds) != "Bv40f": continue
        thk = fnum(getattr(ds, "SliceThickness", None))
        fov = fnum(getattr(ds, "ReconstructionDiameter", None))
        acq = getattr(ds, "AcquisitionNumber", None)
        if thk is None or abs(thk - 0.4) > 0.05: continue
        if fov is None or fov > FOV_MAX: continue
        try:
            if int(acq) != 501: continue
        except Exception:
            continue
        ser = getattr(ds, "SeriesNumber", None)
        try: ser_i = int(ser)
        except Exception: ser_i = 1_000_000
        cands.append({"uid": uid, "items": items, "ser": ser, "ser_i": ser_i,
                      "acq": acq, "tp": parse_tp(scan_opts_str(ds)), "fov": fov})
    if not cands:
        return None
    cands.sort(key=lambda c: c["ser_i"])
    return cands[0]

def pick_legacy(series):
    """Older dual-source: I30f, 0.75mm, FOV<250, ORIGINAL, n>=100, no 4D gate.
    Arterial LA phase = lowest AcquisitionNumber; tie-break highest TP%R-R.
    Returns dict or None."""
    cands = []
    for uid, items in series.items():
        ds = items[0][1]
        if not kernel_str(ds).startswith("I30f"): continue
        thk = fnum(getattr(ds, "SliceThickness", None))
        fov = fnum(getattr(ds, "ReconstructionDiameter", None))
        if thk is None or abs(thk - 0.75) > 0.05: continue
        if fov is None or fov > FOV_MAX: continue
        if len(items) < 100: continue
        if "ORIGINAL" not in image_type_str(ds): continue   # drop DERIVED/MPR
        so = scan_opts_str(ds)
        if has_gate(so): continue                            # drop 4D stack
        acq = getattr(ds, "AcquisitionNumber", None)
        try: acq_i = int(acq)
        except Exception: acq_i = 1_000_000
        tp = parse_tp(so)
        cands.append({"uid": uid, "items": items,
                      "ser": getattr(ds, "SeriesNumber", None),
                      "acq": acq, "acq_i": acq_i,
                      "tp": tp, "tp_sort": tp if tp is not None else -1,
                      "fov": fov})
    if not cands:
        return None
    # lowest acq first, then highest TP within that acq
    cands.sort(key=lambda c: (c["acq_i"], -c["tp_sort"]))
    return cands[0]

PROFILES = [("pcct", pick_pcct), ("legacy", pick_legacy)]

# === Conversion ===
def convert(items, out_path: Path):
    """Convert a list of (path, ds) DICOM files into a NIfTI volume."""
    reader = sitk.ImageSeriesReader()
    folder = str(items[0][0].parent)
    series_id = items[0][1].SeriesInstanceUID
    ordered = reader.GetGDCMSeriesFileNames(folder, series_id)
    if not ordered:                       # fallback: trust given order
        ordered = [str(p) for p, _ in items]
    reader.SetFileNames(ordered)
    img = reader.Execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    print(f"  wrote {out_path.name}  shape={img.GetSize()}  spacing={img.GetSpacing()}")

# === Entry point ===
if __name__ == "__main__":
    # Collect entries: unzipped folders + per-case zips. Case name = folder name
    # or zip stem.
    entries = []
    for p in sorted(DATA_ROOT.iterdir()):
        if p.is_dir():
            entries.append((p.name, p))
        elif p.suffix.lower() == ".zip":
            entries.append((p.stem, p))

    # Open the selection log (UTF-8 BOM so Excel on a German locale reads umlauts).
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(LOG_PATH, "w", newline="", encoding="utf-8-sig")
    log = csv.writer(log_f, delimiter=";")     # ; plays nicer with German Excel
    log.writerow(["case", "profile", "series", "acq", "tp_pct", "fov_mm",
                  "n_files", "status", "out_file"])

    for case_name, entry in entries:
        print(f"\n[{case_name}]")

        # Idempotent: skip if either profile's output already exists. Check before
        # extraction so reruns cost no IO.
        existing = list(OUT_ROOT.glob(f"{case_name}__*.nii.gz"))
        # Back-compat: also honor an old un-suffixed NIfTI (e.g. 4733-4738).
        if (OUT_ROOT / f"{case_name}.nii.gz").exists():
            existing.append(OUT_ROOT / f"{case_name}.nii.gz")
        if existing:
            print(f"  [skip] exists: {existing[0].name}")
            log.writerow([case_name, "", "", "", "", "", "", "skip_exists",
                          existing[0].name])
            continue

        with case_source(entry) as case_dir:
            series = group_series(case_dir)

            chosen = None
            prof_name = None
            for name, fn in PROFILES:
                chosen = fn(series)
                if chosen is not None:
                    prof_name = name
                    break

            if chosen is None:
                print(f"  [skip] no matching series (any profile)")
                log.writerow([case_name, "", "", "", "", "", "",
                              "no_match", ""])
                continue

            n = len(chosen["items"])
            tp = chosen["tp"] if chosen["tp"] is not None else ""
            print(f"  profile={prof_name}  series {chosen['ser']}  "
                  f"acq={chosen['acq']}  TP={tp}  FOV={chosen['fov']:.0f}  n={n}")

            out_path = OUT_ROOT / f"{case_name}__{prof_name}.nii.gz"
            convert(chosen["items"], out_path)
            log.writerow([case_name, prof_name, chosen["ser"], chosen["acq"],
                          tp, f"{chosen['fov']:.1f}", n, "ok", out_path.name])

    log_f.close()
    print(f"\n[log] wrote {LOG_PATH}")