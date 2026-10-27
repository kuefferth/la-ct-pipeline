"""Move already-converted raw CT zips to the overflow drive to free space here.

01_dicom_to_nifti.py searches every root in DATA_ROOTS (data/ct, then the overflow
root, default D:/la-ct-raw), so a moved study is found exactly as before. Later steps
(02, 02b, 03) read only the NIfTI and never touch the raw zips. The zips are only
needed to re-run 01 (new selection rule, another series such as venous / iodine map).

Moves ONLY studies that already have a NIfTI (older than MIN_AGE_S, so one that 01 is
writing right now is skipped). Unconverted zips stay in data/ct unless you pass
--include-unconverted.

SAFETY
  * Dry-run by default. Nothing is copied or deleted without --apply.
  * Copy to <name>.part, size check, full zip CRC check of the copy (the CRCs come
    from the original zip, so this proves the bytes arrived intact), atomic rename,
    and only then delete the source. --fast skips the CRC pass (size check only).
  * Resumable: a finished destination of the same size is not copied again; leftover
    .part files are removed at start. Folder layout (batch subfolders) is preserved.
  * Stops when the destination drive has less than --margin-gb free.
  * Patient CT: the destination drive should be encrypted like F: (BitLocker).

Usage (from repo root):
  python src/01d_move_raw_ct.py                       dry-run, report only
  python src/01d_move_raw_ct.py --apply               move converted studies
  python src/01d_move_raw_ct.py --apply --max-gb 300  move at most ~300 GB this run
  python src/01d_move_raw_ct.py --apply --include-unconverted   also fill remaining space
  python src/01d_move_raw_ct.py --dest E:/la-ct-raw   other destination (also set
                                                      LA_CT_ROOTS so 01 finds it)
Safe to rerun, e.g. after each batch 01 converts.
"""
import argparse
import csv
import importlib.util
import os
import shutil
import time
import zipfile
from pathlib import Path

# Reuse 01's helpers so "converted" means exactly what 01 means.
_spec = importlib.util.spec_from_file_location(
    "dicom_to_nifti", Path(__file__).parent / "01_dicom_to_nifti.py")
d2n = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d2n)

SRC_ROOT  = d2n.DATA_ROOT
LOG_PATH  = Path("data/logs/ct_move_log.csv")      # gitignored (data/)
MIN_AGE_S = 600                                    # NIfTI must be at least this old
CHUNK     = 16 * 1024 * 1024

def default_dest():
    return d2n.DATA_ROOTS[1] if len(d2n.DATA_ROOTS) > 1 else Path("D:/la-ct-raw")

def copy_verified(src, dst, fast):
    """Copy src -> dst via dst.part; True if size (and zip CRCs) check out."""
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(part, "wb") as fo:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
    if part.stat().st_size != src.stat().st_size:
        part.unlink(missing_ok=True)
        return False, "size_mismatch"
    if not fast:
        try:
            with zipfile.ZipFile(part) as zf:
                bad = zf.testzip()
        except Exception as e:
            part.unlink(missing_ok=True)
            return False, f"unreadable_zip: {type(e).__name__}"
        if bad is not None:
            part.unlink(missing_ok=True)
            return False, "crc_fail"
    os.replace(part, dst)
    return True, "ok"

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*", help="only these case names (default: all)")
    ap.add_argument("--apply", action="store_true", help="actually copy and delete")
    ap.add_argument("--dest", type=Path, default=default_dest(),
                    help="destination root (default: second entry of DATA_ROOTS)")
    ap.add_argument("--max-gb", type=float, help="stop after about this many GB moved")
    ap.add_argument("--margin-gb", type=float, default=50, help="keep this free on dest")
    ap.add_argument("--include-unconverted", action="store_true",
                    help="also move zips that have no NIfTI yet")
    ap.add_argument("--fast", action="store_true", help="size check only, skip CRC pass")
    ap.add_argument("--limit", type=int, help="stop after N zips")
    args = ap.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {SRC_ROOT} -> {args.dest}")
    if args.dest.resolve() == SRC_ROOT.resolve():
        raise SystemExit("destination equals source")
    if not args.dest.anchor or not Path(args.dest.anchor).exists():
        raise SystemExit(f"destination drive not available: {args.dest.anchor}")
    if args.dest.resolve() not in [r.resolve() for r in d2n.DATA_ROOTS if r.exists()] \
            and args.dest not in d2n.DATA_ROOTS:
        print(f"[warn] {args.dest} is not in DATA_ROOTS, so 01 will not find moved "
              f"studies; set env LA_CT_ROOTS={SRC_ROOT}{os.pathsep}{args.dest}")

    if args.apply:                                  # leftovers of an interrupted run
        for t in args.dest.rglob("*.part"):
            t.unlink(missing_ok=True)

    # zips directly under the primary root (loose or one level of batch folders)
    entries = [(c, e) for c, e in d2n.collect_entries(SRC_ROOT)
               if e.suffix.lower() == ".zip"]
    if args.cases:
        entries = [(c, e) for c, e in entries if c in set(args.cases)]

    log_f = log = None
    if args.apply:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        log_f = open(LOG_PATH, "a", newline="", encoding="utf-8-sig" if new else "utf-8")
        log = csv.writer(log_f, delimiter=";")
        if new:
            log.writerow(["case", "mb", "dest", "status"])

    moved_gb = 0.0
    n_done = 0
    skipped = {}
    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    t0 = time.perf_counter()
    for case, src in entries:
        if args.limit and n_done >= args.limit:
            break
        if args.max_gb and moved_gb >= args.max_gb:
            break
        nii = d2n.nifti_paths(case)
        if not nii and not args.include_unconverted:
            skip("not_converted_yet (stays)")
            continue
        if nii and time.time() - max(p.stat().st_mtime for p in nii) < MIN_AGE_S:
            skip("nifti_too_new")
            continue
        size = src.stat().st_size
        dst = args.dest / src.relative_to(SRC_ROOT)
        if args.apply:
            free = shutil.disk_usage(args.dest.anchor).free / 1e9
            if free < args.margin_gb + size / 1e9:
                print(f"[stop] destination nearly full ({free:.0f} GB free)")
                break

        if not args.apply:
            n_done += 1
            moved_gb += size / 1e9
            continue

        if dst.exists() and dst.stat().st_size == size:
            ok, why = True, "already_there"         # earlier run copied it, delete pending
        else:
            ok, why = copy_verified(src, dst, args.fast)
        if not ok:
            print(f"{case}: FAILED ({why}), source kept")
            skip(f"failed_{why.split(':')[0]}")
            continue
        src.unlink()                                # verified copy exists: free the space
        n_done += 1
        moved_gb += size / 1e9
        log.writerow([case, f"{size/1e6:.0f}", str(dst), why]); log_f.flush()
        print(f"{case}: moved {size/1e6:.0f} MB  (total {moved_gb:.0f} GB)")

    if log_f:
        log_f.close()
    dt = time.perf_counter() - t0
    print(f"\n[{mode}] {n_done} zips {'moved' if args.apply else 'would move'}: "
          f"{moved_gb:.1f} GB in {dt:.0f}s")
    if not args.apply:
        free = shutil.disk_usage(args.dest.anchor).free / 1e9
        print(f"  destination has {free:.0f} GB free; F: would gain {moved_gb:.0f} GB")
    for k, v in sorted(skipped.items()):
        print(f"  skipped {v:5d}  {k}")

if __name__ == "__main__":
    main()
