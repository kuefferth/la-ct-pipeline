"""Deep-dive on the Bv40f 0.4mm candidates: phase, time, AcquisitionNumber, etc."""
from pathlib import Path
import pydicom
from collections import defaultdict

DATA = Path("data")
TARGET_KERNEL = "Bv40f"
TARGET_THK = 0.4  # mm

def dump_case(case_dir: Path):
    series = defaultdict(list)
    for f in case_dir.rglob("*"):
        if not f.is_file(): continue
        try: ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception: continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid: series[uid].append(ds)

    print(f"\n=== {case_dir.name} ===")
    for uid, items in series.items():
        ds = items[0]
        kernel = str(getattr(ds, "ConvolutionKernel", ""))
        thk    = getattr(ds, "SliceThickness", None)
        if kernel != TARGET_KERNEL: continue
        try:
            if float(thk) > TARGET_THK + 0.05: continue
        except Exception: continue
        # Print extended metadata
        print(f"\n  Series {ds.SeriesNumber}  ({len(items)} files)")
        for tag, name in [
            ("SeriesDescription", "desc"),
            ("ProtocolName", "protocol"),
            ("AcquisitionTime", "acqTime"),
            ("ContentTime", "contentTime"),
            ("ImageComments", "comments"),
            ("AcquisitionNumber", "acqNum"),
            ("ScanOptions", "scanOpts"),
            ("CardiacRRIntervalSpecified", "rrInt"),
            ("NominalPercentageOfCardiacPhase", "phase%"),
            ("CardiacNumberOfImages", "nCardImg"),
        ]:
            val = getattr(ds, tag, None)
            if val not in (None, ""): print(f"    {name:>12}: {val}")
        # Siemens private — phase often in (0021,1041) or (0029,...) or image comments
        # Dump any private tag with 'phase' or '%' in its description
        for elem in ds:
            if elem.tag.is_private:
                try:
                    val = str(elem.value)
                    if "%" in val or "phase" in val.lower() or "RR" in val:
                        print(f"    priv {elem.tag}: {val[:80]}")
                except Exception: pass

if __name__ == "__main__":
    for case in sorted(DATA.iterdir()):
        if case.is_dir(): dump_case(case)