"""04c_eam_match.py — link CARTO EAM exports to CT Study-IDs via procedure date.

*** PHI WARNING ***
This script reads identifying data (PID, name, DOB) and writes a crosswalk
containing it to data/logs/eam_crosswalk.csv. data/ is gitignored, so this
never reaches GitHub. Pipeline DERIVATIVES stay keyed by Study-ID only --
no date, PID, or name appears in any derivatives/ filename.

Matching chain (the CARTO export carries ONLY a procedure date -- no PID/name/DOB):
    CARTO export  --procedure date-->  patients_merged.csv (prc_date1..5 -> PID, name, DOB)
                                            | PID
                                            v
                                DCM-Test-Mapping.xlsx (PID <-> Study-ID)
                                            | Study-ID
                                            v
                       derivatives/meshes/<StudyID>_0___do<num>__{pcct,legacy}_LA.vtk

Match status (column `status` in the crosswalk):
    unique    - exactly one PID on that date that also has a CT mesh -> ready to run 05_eam
    ambiguous - multiple PIDs share the procedure date (manual review; cross-center/lab
                duplicate study labels are possible, so the PiPAF# label cannot disambiguate)
    no_ct     - PID(s) matched the date but none has a CT mesh in the cohort
    no_match  - no patient in patients_merged.csv has that procedure date

Inputs:
    data/eam/Export_*.zip                  CARTO3 export archives (7z, single-zip format)
    data/logs/patients_merged.csv          PID, Vorname, Nachname, Geburtsdatum, prc_date1..5
    data/logs/DCM-Test-Mapping.xlsx        PID <-> Study-ID
    derivatives/meshes/*_LA.vtk            CT meshes (Study-ID prefix gives the de-id case key)

Output:
    data/logs/eam_crosswalk.csv            UTF-8 BOM; one row per EAM export

Usage:
    python src/04c_eam_match.py
    python src/04c_eam_match.py --eam-dir data/eam --review-only   # print flagged rows only
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import py7zr

EAM_DIR   = Path("data/eam")
LOGS_DIR  = Path("data/logs")
MESH_DIR  = Path("derivatives/meshes")

PATIENTS_CSV = LOGS_DIR / "patients_merged.csv"
DCM_XLSX     = LOGS_DIR / "DCM-Test-Mapping.xlsx"
CROSSWALK    = LOGS_DIR / "eam_crosswalk.csv"

PRC_DATE_COLS = ["prc_date1", "prc_date2", "prc_date3", "prc_date4", "prc_date5"]


# ---------------------------------------------------------------------------
# CARTO export: extract procedure date + study label
# ---------------------------------------------------------------------------

def extract_carto_meta(zip_path: Path) -> dict:
    """Read the study label, procedure date, and time from a CARTO export.

    The study root XML element is <Study name="PiPAF8 04_17_2025 10-26-18">.
    We parse that name; format is "<label> MM_DD_YYYY HH-MM-SS".
    Returns {'label', 'date' (YYYY-MM-DD), 'time' (HH:MM:SS)}.
    """
    with py7zr.SevenZipFile(str(zip_path), "r") as z:
        names = z.getnames()
    study_xml = [n for n in names if n.endswith(".xml") and "/" not in n
                 and re.search(r"\d\d_\d\d_\d{4}", n)]
    # The study root XML filename itself encodes the same info; prefer it.
    src = study_xml[0] if study_xml else None

    label, date, time = None, None, None
    if src:
        m = re.search(r"^(.*?)\s+(\d\d)_(\d\d)_(\d{4})\s+(\d\d)-(\d\d)-(\d\d)", src)
        if m:
            label = m.group(1).strip()
            mm, dd, yyyy = m.group(2), m.group(3), m.group(4)
            hh, mi, ss   = m.group(5), m.group(6), m.group(7)
            date = f"{yyyy}-{mm}-{dd}"
            time = f"{hh}:{mi}:{ss}"

    # Fallback: parse from the archive filename (Export_PiPAF8-04_17_2025-10-26-18.zip)
    if date is None:
        m = re.search(r"(\d\d)_(\d\d)_(\d{4})", zip_path.name)
        if m:
            mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
            date = f"{yyyy}-{mm}-{dd}"
        lm = re.search(r"Export_(.+?)-\d\d_", zip_path.name)
        if lm:
            label = lm.group(1)

    return {"label": label, "date": date, "time": time, "source_xml": src}


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

def load_patients(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"PID": str})
    # Normalize PID (strip trailing .0 if read as float-string)
    df["PID"] = df["PID"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def load_dcm_mapping(path: Path) -> pd.DataFrame:
    """PID <-> Study-ID. Returns DataFrame with str columns 'pid', 'study_id'."""
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]])
    # Expected columns: PID, Study-ID
    df = df.rename(columns={"PID": "pid", "Study-ID": "study_id"})
    df["pid"]      = df["pid"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["study_id"] = df["study_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df[["pid", "study_id"]]


def scan_mesh_cases(mesh_dir: Path) -> dict:
    """Map Study-ID (prefix before _0___) -> list of mesh case stems.

    Mesh files are named <StudyID>_0___do<num>__{pcct,legacy}_LA.vtk.
    The case stem (the --case value for 05_eam.py) is the part before _LA.vtk.
    """
    out: dict[str, list[str]] = {}
    for vtk in sorted(mesh_dir.glob("*_LA.vtk")):
        stem = vtk.name[: -len("_LA.vtk")]
        m = re.match(r"^(\d+)_0___", stem)
        study_id = m.group(1) if m else stem.split("_")[0]
        out.setdefault(study_id, []).append(stem)
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_export(meta: dict, patients: pd.DataFrame, dcm: pd.DataFrame,
                 mesh_cases: dict) -> list[dict]:
    """Return one or more candidate match rows for a single EAM export."""
    date = meta["date"]
    base = {
        "carto_label":  meta["label"],
        "carto_date":   date,
        "carto_time":   meta["time"],
    }

    if date is None:
        return [{**base, "status": "no_date", "pid": None, "name": None,
                 "study_id": None, "ct_case": None, "n_candidates": 0}]

    # Find all patients with this procedure date in any prc_date column
    date_mask = patients[PRC_DATE_COLS].apply(
        lambda r: date in set(r.dropna().astype(str)), axis=1
    )
    cands = patients[date_mask]

    if len(cands) == 0:
        return [{**base, "status": "no_match", "pid": None, "name": None,
                 "study_id": None, "ct_case": None, "n_candidates": 0}]

    rows = []
    n = len(cands)
    for _, p in cands.iterrows():
        pid  = str(p["PID"])
        name = f"{p.get('Vorname','')} {p.get('Nachname','')}".strip()
        dcm_match = dcm[dcm["pid"] == pid]
        study_id  = dcm_match.iloc[0]["study_id"] if len(dcm_match) else None
        cases     = mesh_cases.get(str(study_id), []) if study_id else []
        ct_case   = cases[0] if cases else None

        if study_id is None or not cases:
            status = "no_ct"
        elif n == 1:
            status = "unique"
        else:
            status = "ambiguous"

        rows.append({
            **base,
            "status":        status,
            "pid":           pid,
            "name":          name,
            "dob":           p.get("Geburtsdatum"),
            "study_id":      study_id,
            "ct_case":       ct_case,
            "n_ct_meshes":   len(cases),
            "n_candidates":  n,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Match CARTO EAM exports to CT Study-IDs")
    ap.add_argument("--eam-dir", default=str(EAM_DIR))
    ap.add_argument("--review-only", action="store_true",
                    help="Print only flagged rows (ambiguous / no_ct / no_match)")
    args = ap.parse_args()

    eam_dir = Path(args.eam_dir)
    if not PATIENTS_CSV.exists():
        raise SystemExit(f"Missing {PATIENTS_CSV}")
    if not DCM_XLSX.exists():
        raise SystemExit(f"Missing {DCM_XLSX}")

    patients   = load_patients(PATIENTS_CSV)
    dcm        = load_dcm_mapping(DCM_XLSX)
    mesh_cases = scan_mesh_cases(MESH_DIR)
    print(f"Loaded {len(patients)} patients, {len(dcm)} PID<->StudyID rows, "
          f"{len(mesh_cases)} Study-IDs with CT meshes.")

    exports = sorted(eam_dir.glob("Export_*.zip"))
    print(f"Found {len(exports)} CARTO export(s) in {eam_dir}.\n")

    all_rows = []
    for zp in exports:
        meta = extract_carto_meta(zp)
        rows = match_export(meta, patients, dcm, mesh_cases)
        for r in rows:
            r["eam_archive"] = zp.name
        all_rows.extend(rows)

        # Console summary per export
        statuses = {r["status"] for r in rows}
        tag = ("OK" if statuses == {"unique"} else "REVIEW")
        print(f"[{tag}] {zp.name}")
        print(f"       label={meta['label']} date={meta['date']} time={meta['time']}")
        for r in rows:
            print(f"       -> {r['status']:9s} PID={r['pid']} "
                  f"StudyID={r['study_id']} case={r['ct_case']}")
        print()

    if not all_rows:
        print("No exports found; nothing written.")
        return

    cw = pd.DataFrame(all_rows)
    # Column order: de-id fields first, PHI grouped at the end
    col_order = ["eam_archive", "carto_label", "carto_date", "carto_time",
                 "status", "study_id", "ct_case", "n_ct_meshes", "n_candidates",
                 "pid", "name", "dob"]
    cw = cw[[c for c in col_order if c in cw.columns]]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cw.to_csv(CROSSWALK, index=False, encoding="utf-8-sig")
    print(f"Wrote crosswalk: {CROSSWALK}  ({len(cw)} rows)")
    print("  (PHI-containing; data/ is gitignored. Derivatives stay keyed by Study-ID.)")

    # Summary
    n_unique = (cw["status"] == "unique").sum()
    n_review = cw["status"].isin(["ambiguous", "no_ct", "no_match", "no_date"]).sum()
    print(f"\nSummary: {n_unique} ready to run, {n_review} need review.")

    if args.review_only or n_review:
        flagged = cw[cw["status"] != "unique"]
        if len(flagged):
            print("\nRows needing review:")
            print(flagged[["eam_archive", "carto_date", "status",
                           "study_id", "n_candidates"]].to_string(index=False))


if __name__ == "__main__":
    main()
