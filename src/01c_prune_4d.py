"""Drop the multiphase 4D gate stacks from the study zips and list the cases.

Legacy-scanner studies ship one huge series holding ALL cardiac phases
(ScanOptions GATE_nn_OF_21, ~2800-3300 images, 1.5-1.7 GB of a ~1.8 GB zip). It is
the only thing worth removing: everything else (cardiac and thorax FOV, 3 mm
previews, VNC / iodine maps, venous and extra acquisitions) is kept untouched.
Single-phase recons (pcct recons are tagged GATE_01_OF_01) are NOT 4D and stay.

Each affected case is recorded in data/logs/ct_4d_cases.csv so the cohort that has
a 4D stack is known and can be re-exported later if wanted.

Each zip is rewritten IN PLACE (same path, same name) without the 4D series, so
01_dicom_to_nifti.py keeps working unchanged: cases that already have a NIfTI are
skipped there, and a case needing re-conversion still finds its chosen series.

SAFETY
  * Dry-run by default. Nothing is written without --apply.
  * Only cases whose NIfTI already exists are touched (never a not-yet-converted
    zip), and only if that NIfTI is older than MIN_AGE_S (not being written now).
  * New zip is written to <name>.zip.slim.tmp, fully verified (CRC + per-series
    file counts + 01's selection reproduces the same series), only then swapped in
    with an atomic os.replace. The 4D cases CSV row is written after the swap.
  * --archive-dir: copy the untouched original there first (size-checked).
  * Idempotent: a rewritten zip carries a zip comment marker; a zip without any 4D
    stack is recorded in ct_prune_log.csv. Both are skipped on rerun.

Usage (from repo root):
  python src/01c_prune_4d.py                          dry-run, report only
  python src/01c_prune_4d.py --apply                  prune in place
  python src/01c_prune_4d.py --apply --archive-dir D:/ct_full   keep originals there
  python src/01c_prune_4d.py --limit 5 [case ...]     try a few first
Safe to rerun at any time, e.g. after each batch 01 converts.
"""
import argparse
import csv
import importlib.util
import io
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import pydicom

# Reuse 01's helpers and selection functions so "chosen" is guaranteed identical.
_spec = importlib.util.spec_from_file_location(
    "dicom_to_nifti", Path(__file__).parent / "01_dicom_to_nifti.py")
d2n = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d2n)

DATA_ROOT  = d2n.DATA_ROOT
LIST_4D    = Path("data/logs/ct_4d_cases.csv")      # cohort that had a 4D stack
PRUNE_LOG  = Path("data/logs/ct_prune_log.csv")     # every case checked (gitignored)

MARK       = b"LA-CT-PRUNE4D v1"
MIN_AGE_S  = 600      # NIfTI must be at least this old (not mid-write by 01)
ARCHIVE_MARGIN_GB = 50
HEAD_BYTES = 65536    # bytes of each DICOM read for classification

# Only parse the tags the selection logic needs (much faster than a full header).
TAGS = ["SeriesInstanceUID", "SeriesNumber", "ConvolutionKernel", "SliceThickness",
        "ReconstructionDiameter", "AcquisitionNumber", "ScanOptions", "ImageType"]

_UID_TAG = b"\x20\x00\x0e\x00UI"     # (0020,000E) SeriesInstanceUID, explicit VR LE

def quick_uid(head):
    """SeriesInstanceUID found by a raw byte search, or None. A full pydicom parse
    costs ~3 ms/file (x ~4500 files/study); the UID alone is enough to assign a file
    to a series, and only the first file of each series then needs a real parse."""
    i = head.find(_UID_TAG)
    if i < 0:
        return None
    n = int.from_bytes(head[i + 6:i + 8], "little")
    v = head[i + 8:i + 8 + n].rstrip(b"\x00 ")
    if 8 <= len(v) <= 64 and all(c in b"0123456789." for c in v):
        return v.decode()
    return None

def scan_zip(zf):
    """Header-only pass. Returns {uid: {"members": [ZipInfo], "ds": first ds}}."""
    series = {}
    for zi in zf.infolist():
        if zi.is_dir():
            continue
        ds = None
        try:
            # Decompress only the head of each file: every tag we need sits in the
            # first few KB.
            with zf.open(zi) as f:
                head = f.read(HEAD_BYTES)
            uid = quick_uid(head)
            if uid in series:                  # known series: no parse needed
                series[uid]["members"].append(zi)
                continue
            ds = pydicom.dcmread(io.BytesIO(head), stop_before_pixels=True,
                                 force=True, specific_tags=TAGS)
            if not getattr(ds, "SeriesInstanceUID", None) or \
               not hasattr(ds, "ReconstructionDiameter") and len(head) == HEAD_BYTES:
                ds = None                      # head too short: take the slow path
        except Exception:
            ds = None
        if ds is None:
            try:
                with zf.open(zi) as f:
                    ds = pydicom.dcmread(f, stop_before_pixels=True, force=True,
                                         specific_tags=TAGS)
            except Exception:
                continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if not uid:
            continue
        s = series.setdefault(uid, {"members": [], "ds": ds})
        s["members"].append(zi)
    return series

def choose(series):
    """01's own picker on this scan. Returns (profile, uid) or (None, None)."""
    # The pickers only need items[0][1] (the ds) and len(items).
    fake = {uid: [(None, s["ds"])] * len(s["members"]) for uid, s in series.items()}
    for name, fn in d2n.PROFILES:
        c = fn(fake)
        if c is not None:
            return name, c["uid"]
    return None, None

def gate_phases(scanopts):
    """Number of gated phases mm from GATE_nn_OF_mm (0 if untagged)."""
    m = re.search(r"GATE_\d+_OF_(\d+)", scanopts)
    return int(m.group(1)) if m else 0

def is_4d(s, chosen_uid, uid):
    """Multiphase stack: GATE_nn_OF_mm with mm > 1. NOT d2n.has_gate(): the pcct
    recons we keep are tagged GATE_01_OF_01 (single phase). Never the chosen one."""
    return uid != chosen_uid and gate_phases(d2n.scan_opts_str(s["ds"])) > 1

def mb(members, packed=False):
    """Size of members in MB: uncompressed, or as stored in the zip (packed)."""
    return sum(m.compress_size if packed else m.file_size for m in members) / 1e6

def write_pruned(src, tmp, series, drop):
    """Copy every non-dropped member into tmp (deflate; DICOM stays byte-identical)."""
    with zipfile.ZipFile(src) as zf, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as out:
        out.comment = MARK
        for uid, s in series.items():
            if uid in drop:
                continue
            for zi in s["members"]:
                out.writestr(zi.filename, zf.read(zi))

def verify(tmp, series, drop, chosen_uid, prof):
    """Reopen tmp: CRC clean, same per-series file counts, 01 picks the same series."""
    with zipfile.ZipFile(tmp) as zf:
        if zf.testzip() is not None:
            return "crc_fail"
        new = scan_zip(zf)
    want = {u: len(s["members"]) for u, s in series.items() if u not in drop}
    got = {u: len(s["members"]) for u, s in new.items()}
    if want != got:
        return "count_mismatch"
    if choose(new) != (prof, chosen_uid):
        return "selection_changed"
    return "ok"

def process(case, entry, apply, archive_dir):
    """Returns (status, orig_mb, new_mb, info). info = 4D details for the cases list."""
    orig_size = entry.stat().st_size
    with zipfile.ZipFile(entry) as zf:
        if zf.comment == MARK:
            return "already_pruned", orig_size / 1e6, orig_size / 1e6, None
        series = scan_zip(zf)
    prof, chosen_uid = choose(series)
    drop = {u for u, s in series.items() if is_4d(s, chosen_uid, u)}
    if not drop:
        return "no_4d", orig_size / 1e6, orig_size / 1e6, None
    members = [m for u in drop for m in series[u]["members"]]
    info = {"profile": prof or "",
            "n_phases": max(gate_phases(d2n.scan_opts_str(series[u]["ds"])) for u in drop),
            "n_series": len(drop), "n_files": len(members), "mb": f"{mb(members):.0f}"}
    new_mb = orig_size / 1e6 - mb(members, packed=True)   # estimate (dry-run)
    if not apply:
        return "would_prune", orig_size / 1e6, new_mb, info

    tmp = entry.with_name(entry.name + ".slim.tmp")
    try:
        write_pruned(entry, tmp, series, drop)
        v = verify(tmp, series, drop, chosen_uid, prof)
        if v != "ok":
            tmp.unlink(missing_ok=True)
            return f"verify_{v}", orig_size / 1e6, new_mb, info
        info["archive"] = ""
        if archive_dir is not None:
            root = next(r for r in d2n.DATA_ROOTS if r in entry.parents)
            dest = archive_dir / entry.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, dest)
            if dest.stat().st_size != orig_size:
                tmp.unlink(missing_ok=True)
                return "archive_size_mismatch", orig_size / 1e6, new_mb, info
            info["archive"] = str(dest)
        new_mb = tmp.stat().st_size / 1e6
        os.replace(tmp, entry)               # atomic swap; 4D data gone from here
        return "pruned", orig_size / 1e6, new_mb, info
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"error_{type(e).__name__}: {e}", orig_size / 1e6, new_mb, info

def open_csv(path, header, enabled=True):
    """Append-mode CSV, UTF-8 BOM on creation, ';' delimiter (German Excel).
    enabled=False (dry-run) returns a sink that writes nothing and creates no file."""
    if not enabled:
        f = open(os.devnull, "w")
        return f, csv.writer(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8")
    w = csv.writer(f, delimiter=";")
    if new:
        w.writerow(header)
    return f, w

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*", help="only these case names (default: all)")
    ap.add_argument("--apply", action="store_true", help="actually rewrite zips")
    ap.add_argument("--archive-dir", type=Path, help="copy originals here first")
    ap.add_argument("--limit", type=int, help="stop after N pruned zips")
    args = ap.parse_args()

    if args.archive_dir is not None and not args.apply:
        print("[note] --archive-dir has no effect without --apply")
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] data root: {DATA_ROOT}"
          + (f"  archive: {args.archive_dir}" if args.archive_dir else ""))

    # Leftovers from an interrupted run are ours and safe to delete.
    for root in d2n.DATA_ROOTS:
        if root.exists():
            for t in root.rglob("*.slim.tmp"):
                t.unlink(missing_ok=True)

    # Cases already checked (had no 4D stack) need no rescan.
    checked = set()
    if PRUNE_LOG.exists():
        with open(PRUNE_LOG, encoding="utf-8-sig", newline="") as f:
            checked = {r[0] for r in csv.reader(f, delimiter=";")}

    entries = [(c, e) for c, e in d2n.collect_all_entries()    # all raw roots
               if e.suffix.lower() == ".zip"]
    if args.cases:
        entries = [(c, e) for c, e in entries if c in set(args.cases)]

    f4d, w4d = open_csv(LIST_4D, ["case", "profile", "n_phases", "n_series",
                                  "n_files", "mb_removed", "archive"], args.apply)
    fchk, wchk = open_csv(PRUNE_LOG, ["case", "status"], args.apply)

    tot_orig = tot_new = n_done = 0
    skipped = {}
    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    t0 = time.perf_counter()
    for case, entry in entries:
        if args.limit and n_done >= args.limit:
            break
        if case in checked:
            skip("already_checked")
            continue
        # Never touch a zip 01 has not converted yet (or is converting right now).
        nii = list(d2n.OUT_ROOT.glob(f"{case}__*.nii.gz"))
        un = d2n.OUT_ROOT / f"{case}.nii.gz"
        if un.exists(): nii.append(un)
        if not nii:
            skip("no_nifti (not converted, or superseded duplicate)")
            continue
        if time.time() - max(p.stat().st_mtime for p in nii) < MIN_AGE_S:
            skip("nifti_too_new")
            continue
        if args.apply and args.archive_dir is not None:
            free = shutil.disk_usage(args.archive_dir if args.archive_dir.exists()
                                     else args.archive_dir.anchor).free / 1e9
            if free < ARCHIVE_MARGIN_GB + entry.stat().st_size / 1e9:
                print(f"[stop] archive drive nearly full ({free:.0f} GB free)")
                break

        status, o_mb, n_mb, info = process(case, entry, args.apply, args.archive_dir)
        if status in ("would_prune", "pruned"):
            n_done += 1
            tot_orig += o_mb
            tot_new += n_mb
            print(f"{case}: {status}  {o_mb:.0f} -> {n_mb:.0f} MB  "
                  f"4D: {info['n_files']} files, {info['mb']} MB, "
                  f"{info['n_phases']} phases")
            if args.apply:
                w4d.writerow([case, info["profile"], info["n_phases"], info["n_series"],
                              info["n_files"], info["mb"], info.get("archive", "")])
                wchk.writerow([case, status]); f4d.flush(); fchk.flush()
        else:
            skip(status.split(":")[0])
            if status == "no_4d":
                if args.apply:
                    wchk.writerow([case, status]); fchk.flush()
            elif status != "already_pruned":
                print(f"{case}: {status}")

    f4d.close(); fchk.close()
    dt = time.perf_counter() - t0
    print(f"\n[{mode}] {n_done} zips {'pruned' if args.apply else 'would be pruned'}: "
          f"{tot_orig/1e3:.1f} GB -> {tot_new/1e3:.1f} GB "
          f"(frees {(tot_orig-tot_new)/1e3:.1f} GB) in {dt:.0f}s")
    for k, v in sorted(skipped.items()):
        print(f"  skipped {v:5d}  {k}")

if __name__ == "__main__":
    main()
