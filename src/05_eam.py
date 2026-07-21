"""05_eam.py — CARTO3 EAM -> voltage-colored CT LA mesh

Reads a CARTO3 v8 study export (7z archive mislabeled .zip), registers the
CARTO LA mesh to the CT-derived LA mesh via ICP, interpolates bipolar/unipolar
voltages to CT mesh vertices, and writes a per-case VTK + stats CSV.

Input:
    data/eam/<name>.zip                   CARTO3 7z export archive
    derivatives/meshes/<case>_LA.vtk      CT LA mesh (from 03_mesh.py)

Output (per case, in derivatives/eam/):
    <case>_LA_eam.vtk        CT mesh geometry + EAM voltage scalars
    <case>_eam_points.csv    raw accepted EAM point table (UTF-8 BOM)
    <case>_eam_stats.csv     per-map summary stats (UTF-8 BOM)

Usage:
    python src/05_eam.py --case 4734 --eam data/eam/Export_PiPAF8-*.zip
    python src/05_eam.py --case 4734 --eam data/eam/Export_PiPAF8-*.zip --map "1-1-ReMap"

Car.txt column layout (VERSION_6_0, confirmed on CARTO3 v8.1.1.944):
    0:'P'  1:GroupID  2:PointID  3:SubID
    4-6:   X Y Z  (mm, CARTO electromagnetic frame)
    7-9:   NormalX NormalY NormalZ
    10:    Unipolar voltage (mV)
    11:    Bipolar voltage (mV)
    12:    LAT (ms)  -10000 = invalid
    17:    Category  'A'=accepted, other=rejected

TPI note: this cohort (CARTO v8.1.1, no force catheter) has Impedance=10000
on all mesh vertices and zero impedance samples in per-point XMLs.
Quality filter uses CARTO's acceptance flag + LAT validity only.
If a force catheter is used in future, the 'Force' column in the .mesh
[VerticesColorsSection] and the ContactForceData.txt will carry TPI.

CARTO bipolar color thresholds (AF substrate, standard clinical):
    < 0.5 mV  -> scar (dense)
    0.5-1.5   -> border zone
    > 1.5 mV  -> healthy

Registration: centroid-align + vtkIterativeClosestPointTransform (rigid).
Voltage interpolation: inverse-distance weighted from accepted EAM points
to CT mesh vertices (radius 10 mm, k=10 nearest neighbors).
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

try:
    import py7zr
except ImportError:
    sys.exit("py7zr not installed: pip install py7zr")

EAM_DIR  = Path("derivatives/eam")
MESH_DIR = Path("derivatives/meshes")

# === Tunables ===
INTERP_RADIUS_MM  = 10.0   # max distance for voltage interpolation
INTERP_K          = 10     # number of nearest EAM points to blend
INTERP_POWER      = 2.0    # IDW exponent (higher = sharper)
ICP_MAX_ITER      = 200    # ICP iterations
ICP_MAX_LANDMARKS = 5000   # subsample mesh for ICP speed

SCAR_BIPOLAR_MV   = 0.5    # dense scar threshold (mV)
BORDER_BIPOLAR_MV = 1.5    # border zone upper limit (mV)
SCAR_UNIPOLAR_MV  = 1.0    # unipolar scar threshold (mV)

INVALID_SENTINEL  = -10000  # CARTO uses this for missing data


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def list_maps(zip_path: Path) -> list[str]:
    """Return sorted map base names present in the archive, e.g. ['1-Map', '1-1-ReMap']."""
    with py7zr.SevenZipFile(str(zip_path), "r") as z:
        names = z.getnames()
    car_files = [n for n in names if n.endswith("_car.txt")]
    maps = sorted(set(n[: -len("_car.txt")] for n in car_files))
    return maps


def extract_map_files(zip_path: Path, map_name: str, tmpdir: Path) -> dict[str, Path]:
    """Extract car.txt and .mesh for a map to tmpdir in one archive pass.

    Returns dict: {'car': Path, 'mesh': Path}
    """
    targets = [f"{map_name}_car.txt", f"{map_name}.mesh"]
    with py7zr.SevenZipFile(str(zip_path), "r") as z:
        z.extract(path=str(tmpdir), targets=targets)
    return {
        "car":  tmpdir / f"{map_name}_car.txt",
        "mesh": tmpdir / f"{map_name}.mesh",
    }


# ---------------------------------------------------------------------------
# Car.txt parser
# ---------------------------------------------------------------------------

def parse_car_txt(car_path: Path) -> pd.DataFrame:
    """Parse a CARTO3 _car.txt file.

    Returns DataFrame with columns:
        point_id, group_id, x, y, z, nx, ny, nz,
        unipolar_mv, bipolar_mv, lat_ms, accepted (bool)
    """
    with open(car_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    rows = []
    for line in text.splitlines():
        if not (line.startswith("P\t") or line.startswith("P ")):
            continue
        tok = line.split()
        if len(tok) < 18:
            continue
        try:
            group_id   = int(tok[1])
            point_id   = int(tok[2])
            x, y, z    = float(tok[4]), float(tok[5]), float(tok[6])
            nx, ny, nz = float(tok[7]), float(tok[8]), float(tok[9])
            uni_mv     = float(tok[10])
            bi_mv      = float(tok[11])
            lat_ms     = float(tok[12])
            category   = tok[17]
        except (ValueError, IndexError):
            continue

        accepted = (category == "A")
        rows.append(
            (point_id, group_id, x, y, z, nx, ny, nz, uni_mv, bi_mv, lat_ms, accepted)
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "point_id", "group_id",
            "x", "y", "z", "nx", "ny", "nz",
            "unipolar_mv", "bipolar_mv", "lat_ms", "accepted",
        ],
    )
    return df


# ---------------------------------------------------------------------------
# CARTO .mesh parser
# ---------------------------------------------------------------------------

def parse_carto_mesh(mesh_path: Path) -> pv.PolyData:
    """Parse a CARTO3 .mesh file.

    Returns pyvista PolyData with point arrays:
        unipolar, bipolar, lat, impedance
        (values == 10000 are CARTO's invalid sentinel; stored as NaN)
    """
    with open(mesh_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    # Locate sections
    sec = {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "[VerticesSection]":
            sec["verts"] = i
        elif s == "[TrianglesSection]":
            sec["tris"] = i
        elif s == "[VerticesColorsSection]":
            sec["colors"] = i
        elif s == "[VerticesAttributesSection]":
            sec["attrs"] = i

    # Parse scalar column names from VerticesColorsSection header
    def _safe_scalar_name(s: str) -> str:
        # Normalize Unicode (e.g. Delta -> 'delta'), strip non-ASCII, lowercase
        import unicodedata
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.replace("-", "_")
        return s.strip("_").lower() if s.strip("_") else "unknown"

    color_names = []
    if "colors" in sec:
        for ln in lines[sec["colors"] : sec["colors"] + 5]:
            if ln.strip().startswith(";") and "Unipolar" in ln:
                raw_names = ln.lstrip(";").split()
                color_names = [_safe_scalar_name(n) for n in raw_names]
                break

    def parse_indexed_section(start, end):
        result = []
        for ln in lines[start + 1 : end]:
            if "=" not in ln or ln.strip().startswith(";"):
                continue
            vals = ln.split("=")[1].split()
            result.append([float(v) for v in vals])
        return result

    verts_end = sec.get("tris", len(lines))
    tris_end  = sec.get("colors", len(lines))
    cols_end  = sec.get("attrs", len(lines))

    verts_data = parse_indexed_section(sec["verts"], verts_end)
    tris_data  = parse_indexed_section(sec["tris"],  tris_end)
    cols_data  = parse_indexed_section(sec.get("colors", len(lines)), cols_end) if "colors" in sec else []

    if not verts_data or not tris_data:
        raise ValueError(f"Empty mesh in {mesh_path}")

    verts = np.array(verts_data)
    points = verts[:, :3]  # X Y Z
    tris   = np.array(tris_data, dtype=int)
    faces  = np.column_stack([np.full(len(tris), 3, dtype=int), tris[:, :3]])

    mesh = pv.PolyData(points, faces.ravel())

    if cols_data and color_names:
        cols_arr = np.array(cols_data, dtype=float)
        # CARTO uses 10000 as invalid sentinel; convert to NaN
        cols_arr[cols_arr == 10000.0] = np.nan
        cols_arr[cols_arr == INVALID_SENTINEL] = np.nan
        for i, cname in enumerate(color_names):
            if i < cols_arr.shape[1]:
                mesh.point_data[cname.lower()] = cols_arr[:, i]

    return mesh


# ---------------------------------------------------------------------------
# ICP registration: CARTO mesh -> CT mesh
# ---------------------------------------------------------------------------

def _subsample_mesh(mesh: pv.PolyData, n: int) -> pv.PolyData:
    """Uniform vertex subsample for ICP speed."""
    pts = mesh.points
    if len(pts) <= n:
        return mesh
    idx = np.random.default_rng(0).choice(len(pts), size=n, replace=False)
    return pv.PolyData(pts[idx])


def register_carto_to_ct(
    carto_mesh: pv.PolyData,
    ct_mesh:    pv.PolyData,
    max_iter:   int = ICP_MAX_ITER,
    n_pts:      int = ICP_MAX_LANDMARKS,
) -> np.ndarray:
    """Rigid ICP registration: CARTO mesh -> CT LA mesh.

    Returns 4x4 numpy transform (apply to CARTO coordinates to get CT space).

    Strategy:
      1. Centroid-align (translation) as initialization.
      2. vtkIterativeClosestPointTransform (rigid body).

    Assumes both meshes represent the same LA anatomy in different
    coordinate frames. The CARTO internal frame is typically within
    ~45 degrees of anatomical orientation (SI along Z), so centroid
    alignment + ICP should converge without PCA pre-alignment.
    If results are poor (check mean_distance in verbose output), try
    adding PCA rotation init: align eigenvectors of both meshes before ICP.
    """
    carto_sub = _subsample_mesh(carto_mesh, n_pts)
    ct_sub    = _subsample_mesh(ct_mesh,    n_pts)

    # Step 1: centroid translate CARTO toward CT
    carto_cen = carto_sub.points.mean(axis=0)
    ct_cen    = ct_sub.points.mean(axis=0)
    T_init    = np.eye(4)
    T_init[:3, 3] = ct_cen - carto_cen

    carto_aligned = pv.PolyData(carto_sub.points + T_init[:3, 3])

    # Step 2: VTK ICP
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(carto_aligned)
    icp.SetTarget(ct_sub)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(max_iter)
    icp.SetMaximumMeanDistance(0.001)
    icp.SetCheckMeanDistance(True)
    icp.Update()

    vtk_mat = icp.GetMatrix()
    T_icp = np.array(
        [[vtk_mat.GetElement(i, j) for j in range(4)] for i in range(4)]
    )

    # T_full: first apply T_init (translation), then T_icp (rotation + refinement)
    T_full = T_icp @ T_init
    return T_full


def apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 rigid transform to (N,3) point array."""
    h = np.column_stack([pts, np.ones(len(pts))])
    return (T @ h.T).T[:, :3]


# ---------------------------------------------------------------------------
# Voltage interpolation: EAM points -> CT mesh vertices
# ---------------------------------------------------------------------------

def interpolate_voltage_to_mesh(
    eam_pts:    np.ndarray,   # (N, 3) registered EAM positions
    eam_values: dict,         # {name: (N,) array}  e.g. {'bipolar_mv': ..., 'unipolar_mv': ...}
    ct_mesh:    pv.PolyData,
    radius:     float = INTERP_RADIUS_MM,
    k:          int   = INTERP_K,
    power:      float = INTERP_POWER,
) -> pv.PolyData:
    """Inverse-distance weighted interpolation from EAM point cloud to CT mesh.

    For each CT vertex, blend the k nearest EAM points within `radius` mm
    using weights = 1/d^power. Vertices with no EAM point within `radius`
    receive NaN (unmapped region).
    """
    tree  = cKDTree(eam_pts)
    dists, idxs = tree.query(ct_mesh.points, k=k, workers=-1)

    # dists/idxs shape: (n_verts, k) or (n_verts,) if k==1
    if k == 1:
        dists = dists[:, np.newaxis]
        idxs  = idxs[:, np.newaxis]

    mesh_out = ct_mesh.copy()

    for name, values in eam_values.items():
        out = np.full(len(ct_mesh.points), np.nan)
        for vi in range(len(ct_mesh.points)):
            d = dists[vi]
            idx = idxs[vi]
            in_range = d <= radius
            if not in_range.any():
                continue
            d_r = d[in_range]
            v_r = values[idx[in_range]]
            # avoid division by zero for exact hits
            if d_r.min() < 1e-6:
                out[vi] = v_r[d_r.argmin()]
            else:
                w = 1.0 / d_r ** power
                out[vi] = np.sum(w * v_r) / np.sum(w)
        mesh_out.point_data[name] = out

    return mesh_out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(
    df:        pd.DataFrame,   # accepted EAM points
    map_name:  str,
    case_id:   str,
    ct_mesh:   pv.PolyData,    # CT mesh WITH voltage scalars added
) -> dict:
    """Compute per-map summary statistics."""
    acc  = df[df.accepted]
    bi   = acc.bipolar_mv.values
    uni  = acc.unipolar_mv.values
    lat  = acc.lat_ms.values
    lat_valid = lat[lat != INVALID_SENTINEL]

    stats: dict = {
        "case_id":            case_id,
        "map_name":           map_name,
        "n_points_total":     len(df),
        "n_points_accepted":  len(acc),
        "bipolar_mean_mv":    float(np.mean(bi)),
        "bipolar_median_mv":  float(np.median(bi)),
        "bipolar_p5_mv":      float(np.percentile(bi, 5)),
        "bipolar_p25_mv":     float(np.percentile(bi, 25)),
        "scar_frac_bipolar":  float(np.mean(bi < SCAR_BIPOLAR_MV)),
        "border_frac_bipolar":float(np.mean((bi >= SCAR_BIPOLAR_MV) & (bi < BORDER_BIPOLAR_MV))),
        "unipolar_mean_mv":   float(np.mean(uni)),
        "unipolar_median_mv": float(np.median(uni)),
        "scar_frac_unipolar": float(np.mean(uni < SCAR_UNIPOLAR_MV)),
        "lat_range_ms":       float(np.ptp(lat_valid)) if len(lat_valid) else np.nan,
        "lat_mean_ms":        float(np.mean(lat_valid)) if len(lat_valid) else np.nan,
    }

    # Mesh-based scar area (if voltage was interpolated)
    if "bipolar_mv" in ct_mesh.point_data:
        bi_v = ct_mesh.point_data["bipolar_mv"]
        valid = ~np.isnan(bi_v)
        if valid.sum() > 0:
            # Compute approximate surface area per vertex via cell area
            mesh_scar = ct_mesh.compute_cell_sizes(area=True)
            # Distribute cell area to vertices via cell connectivity
            n_v = ct_mesh.n_points
            vert_area = np.zeros(n_v)
            cells = ct_mesh.faces.reshape(-1, 4)[:, 1:]
            cell_areas = mesh_scar.cell_data["Area"]
            for ci, tri in enumerate(cells):
                vert_area[tri] += cell_areas[ci] / 3.0
            # Total LA surface area
            total_area_cm2 = vert_area.sum() / 100.0  # mm2 -> cm2
            scar_area_cm2  = vert_area[valid & (bi_v < SCAR_BIPOLAR_MV)].sum() / 100.0
            stats["la_area_cm2"]        = float(total_area_cm2)
            stats["scar_area_cm2"]      = float(scar_area_cm2)
            stats["scar_frac_mesh"]     = float(scar_area_cm2 / total_area_cm2) if total_area_cm2 > 0 else np.nan
            stats["coverage_frac_mesh"] = float(valid.mean())

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_case(
    case_id:   str,
    zip_path:  Path,
    map_name:  str | None = None,
    verbose:   bool = True,
) -> None:
    ct_vtk  = MESH_DIR / f"{case_id}_LA.vtk"
    out_dir = EAM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ct_vtk.exists():
        print(f"[{case_id}] CT mesh not found: {ct_vtk}")
        return
    if not zip_path.exists():
        print(f"[{case_id}] EAM archive not found: {zip_path}")
        return

    if verbose:
        print(f"\n[{case_id}] Loading CT mesh: {ct_vtk}")
    ct_mesh = pv.read(str(ct_vtk))

    # Discover maps
    available_maps = list_maps(zip_path)
    if verbose:
        print(f"[{case_id}] Maps in archive: {available_maps}")

    if not available_maps:
        print(f"[{case_id}] No maps found in archive.")
        return

    # Default: prefer the ReMap if present; otherwise last map alphabetically
    if map_name is None:
        remaps = [m for m in available_maps if "ReMap" in m]
        map_name = remaps[-1] if remaps else available_maps[-1]
    if map_name not in available_maps:
        print(f"[{case_id}] Map '{map_name}' not in archive. Available: {available_maps}")
        return

    if verbose:
        print(f"[{case_id}] Using map: {map_name}")

    # Extract map files to temp dir (single archive pass)
    tmpdir = Path(tempfile.mkdtemp(prefix="carto_"))
    try:
        if verbose:
            print(f"[{case_id}] Extracting map files ...")
        paths = extract_map_files(zip_path, map_name, tmpdir)

        # Parse EAM points
        df = parse_car_txt(paths["car"])
        if verbose:
            acc = df.accepted.sum()
            print(f"[{case_id}] EAM points: {len(df)} total, {acc} accepted")

        # Parse CARTO mesh
        if verbose:
            print(f"[{case_id}] Parsing CARTO mesh ...")
        carto_mesh = parse_carto_mesh(paths["mesh"])
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)
    if verbose:
        print(f"[{case_id}] CARTO mesh: {carto_mesh.n_points} vertices, "
              f"{carto_mesh.n_cells} triangles")

    # ICP registration
    if verbose:
        print(f"[{case_id}] Running ICP registration ...")
    T = register_carto_to_ct(carto_mesh, ct_mesh, max_iter=ICP_MAX_ITER)
    if verbose:
        print(f"[{case_id}] ICP transform:\n{np.round(T, 4)}")

    # Apply transform to EAM points
    acc_df   = df[df.accepted].copy()
    eam_xyz  = apply_transform(acc_df[["x", "y", "z"]].values, T)
    eam_vals = {
        "bipolar_mv":  acc_df.bipolar_mv.values,
        "unipolar_mv": acc_df.unipolar_mv.values,
        "lat_ms":      np.where(acc_df.lat_ms == INVALID_SENTINEL,
                                np.nan, acc_df.lat_ms.values),
    }
    # Binary scar scalars (point-level)
    eam_vals["scar_bipolar"]  = (acc_df.bipolar_mv.values  < SCAR_BIPOLAR_MV).astype(float)
    eam_vals["scar_unipolar"] = (acc_df.unipolar_mv.values < SCAR_UNIPOLAR_MV).astype(float)

    # Interpolate to CT mesh
    if verbose:
        print(f"[{case_id}] Interpolating voltages to CT mesh ...")
    ct_eam = interpolate_voltage_to_mesh(eam_xyz, eam_vals, ct_mesh)

    # Also transfer CARTO's own pre-interpolated mesh scalars as a secondary reference
    carto_pts_reg = apply_transform(carto_mesh.points, T)
    carto_scalars: dict = {}
    for sname in ["unipolar", "bipolar", "lat", "impedance"]:
        if sname in carto_mesh.point_data:
            arr = carto_mesh.point_data[sname].copy().astype(float)
            valid = ~np.isnan(arr)
            if valid.any():
                # Only pass vertices that have real (non-NaN) scalar values
                carto_scalars[f"carto_{sname}"] = (carto_pts_reg[valid], arr[valid])
    if carto_scalars:
        # Interpolate each scalar separately from its valid-only point cloud
        for k, (pts, vals) in carto_scalars.items():
            ct_eam_k = interpolate_voltage_to_mesh(
                pts, {k: vals}, ct_mesh, radius=INTERP_RADIUS_MM
            )
            ct_eam.point_data[k] = ct_eam_k.point_data[k]

    # Compute stats
    stats = compute_stats(acc_df, map_name, case_id, ct_eam)
    if verbose:
        print(f"[{case_id}] Scar fraction (bipolar <0.5mV): "
              f"{stats.get('scar_frac_mesh', stats.get('scar_frac_bipolar')):.1%}")

    # Save VTK
    vtk_out = out_dir / f"{case_id}_LA_eam.vtk"
    ct_eam.save(str(vtk_out))
    if verbose:
        print(f"[{case_id}] Saved: {vtk_out}")

    # Save registered EAM point cloud (for ML/stats use)
    acc_df_out = acc_df.copy()
    acc_df_out[["x_ct", "y_ct", "z_ct"]] = eam_xyz
    csv_pts = out_dir / f"{case_id}_eam_points.csv"
    acc_df_out.to_csv(str(csv_pts), index=False, encoding="utf-8-sig")
    if verbose:
        print(f"[{case_id}] Saved: {csv_pts}")

    # Save stats
    stats_df = pd.DataFrame([stats])
    csv_stats = out_dir / f"{case_id}_eam_stats.csv"
    stats_df.to_csv(str(csv_stats), index=False, encoding="utf-8-sig")
    if verbose:
        print(f"[{case_id}] Saved: {csv_stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CARTO3 EAM -> CT mesh voltage mapping")
    ap.add_argument("--case",  required=True,
                    help="Patient case ID (must match derivatives/meshes/<id>_LA.vtk)")
    ap.add_argument("--eam",   required=True,
                    help="Path to CARTO3 export archive (.zip / .7z)")
    ap.add_argument("--map",   default=None,
                    help="Map name to use (e.g. '1-1-ReMap'); default = last ReMap found")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    zip_path = Path(args.eam)
    if "*" in str(zip_path):
        import glob as _glob
        matches = _glob.glob(str(zip_path))
        if not matches:
            sys.exit(f"No file matches: {zip_path}")
        zip_path = Path(sorted(matches)[-1])

    process_case(
        case_id  = args.case,
        zip_path = zip_path,
        map_name = args.map,
        verbose  = not args.quiet,
    )


if __name__ == "__main__":
    main()
