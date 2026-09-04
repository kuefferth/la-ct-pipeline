# LA-CT Pipeline — Project History & Handoff

Condensed record of the Claude chats for this project, for Claude Code to use as
context. Covers what the pipeline does, the current code state, and the decision
log (what was tried, what was rejected, and why). Drop this in the repo root;
Claude Code auto-loads `CLAUDE.md` from there. Rename if you prefer.

---

## Who / collaboration

- Thomas Kuffer, biomedical engineer, Inselspital Bern. Collab with an Oxford
  cardiac imaging lab. Bern produces clean LA geometry from CT; Oxford does the
  deep learning on the geometry.
- Cohort: AF ablation patients. EAM points + post-ablation outcomes exist for
  some special cohorts (to be mapped into the LA frame later).
- End goal: robust, reproducible pipeline producing LA surface meshes
  (LA body + LAA + PV cuffs as a single structure) for DL training, scaling to
  ~2000 cases. Also EAM/outcome data mapped into a canonical frame.
- Comms: terse, commented code, full files for big changes / line edits for
  small ones, no em dashes. Windows + German locale. CSVs always UTF-8 with BOM.
- Strong clinical/imaging knowledge; newer to Python ecosystem, GitHub, conda.

## Hardware / environment

- Work desktop: Win 10, 16 GB RAM, GTX 1060 6 GB. GPU is BROKEN specifically for
  TotalSegmentator `heartchambers_highres` (silent CUDA crash, fully debugged ->
  CPU only). Other models run on GPU fine.
- Home: RTX 5060 Ti 16 GB (sufficient for inference). Office laptop (no GPU) backup.
- Target purchase identified: mobile workstation (RTX 4080/4090 mobile, 64 GB RAM,
  2 TB NVMe, Windows) for portability + Oxford visits.
- Python 3.11, Miniconda env `la`. VS Code, Git for Windows, cmd.exe with
  `conda init cmd.exe`. 3D Slicer / ParaView for QC.
- Repo: github.com/kuefferth/la-ct-pipeline (private). Now lives on a new
  BitLocker-encrypted HD at `F:\oxford\la-ct-pipeline` (was `D:\la-ct-pipeline`,
  before that `C:\Users\cit\Documents\...`). This move did NOT throw "dubious
  ownership" (unlike the D: move), so no new `safe.directory` entry was needed;
  the stale `D:/la-ct-pipeline` entry in `C:\Users\cit\.gitconfig` is harmless
  and was left in place.
- `data/` and `derivatives/` are gitignored. Patient CT must never hit GitHub.
- Stack: totalsegmentator, SimpleITK, pydicom, nibabel, numpy, scipy, vtk,
  pyvista, tqdm. `torch==2.5.1+cu121` + `torchvision==0.20.1+cu121` (matched pair,
  install together). torchvision DLL warning is benign.

## Data

- Siemens dual-source cardiac CT, ~30 series/study, anonymized
  (SeriesDescription empty).
- Initial 5 sample cases: 4733, 4734, 4735, 4736, 4738.
- Production anonymization chain validated on case 1111 (see kernel fix below).
- Two cardiac acquisitions per case: AcqNum 501 (diastolic 70%+ RR, arterial,
  CHOSEN) and AcqNum 601 (systolic 30-60% RR, venous twin ~10-30s later).

---

## Current pipeline

```
data/ct/<case>.zip                            DICOMs (gitignored; also accepts one
data/ct/<batch dir>/<case>.zip                 level of batch subfolders, e.g. "1000 to 1498")
  -> 01_dicom_to_nifti.py                      series selection + per-patient dedup
derivatives/nifti/<case>__{pcct,legacy}.nii.gz  one per patient (pcct preferred)
  -> 02_segment.py                             TotalSeg heartchambers_highres (CPU)
derivatives/seg_full/<case>_chambers.nii.gz    multi-label (myo,LA,LV,RA,RV,aorta,PA)
derivatives/seg_la/<case>_LA_seed.nii.gz       TotalSeg LA only (seed reference)
  -> 02b_regiongrow.py                         refinement (see below)
derivatives/seg_la/<case>_LA.nii.gz            binary LA+PV-cuffs (final)
  -> 03_mesh.py                                mask -> marching cubes -> Taubin -> decimate
derivatives/meshes/<case>_LA.{vtk,stl}         decimated mesh (30% tris; ONLY output)
  -> 03b_canonicalize.py  (optional, not in run_all)  translation recenter + 4x4 sidecar
derivatives/canonical/<case>_LA_T.npy          4x4 world->canonical
derivatives/canonical/<case>_LA_canonical.stl
```

Orchestrator: `run_all.py` (runs 01,02,02b,03 in order; idempotent; skips existing
outputs). Diagnostics 00*: `00_inspect_series`, `00b_inspect_bv40`, `00c_inspect_fov`,
plus probes `00d_why_skipped`, `00e_oldscanner_probe`, `00f_phase_probe`,
`00g_phase_nifti_qc`, `00h_best_series_probe`.

### Series selection (01) — current rules

- ConvolutionKernel == "Bv40f" (vascular soft kernel)
- SliceThickness == 0.4 mm
- ReconstructionDiameter < 250 mm (cardiac FOV; thorax recons ~368)
- AcquisitionNumber == 501 (diastolic arterial)
- Multiple matches -> lowest SeriesNumber + warn.
- PER-PATIENT DEDUP: some patients ship two studies (two zips, different do-numbers)
  -- a 0.4mm pcct recon AND a 0.75mm legacy recon. Rule: ONE series/patient, prefer
  pcct (high-res). Implemented in 01: patient_id = token before first `_0___`;
  PROFILE_PRIORITY pcct<legacy; post-loop removes the superseded legacy nifti; a
  pcct-sibling skip avoids re-extracting on reruns. Dedup is at the nifti stage, so
  02/02b/03 (which iterate over existing nifti) see one-per-patient automatically.
  NOTE: this does NOT catch cross-ID DATA duplicates (same scan under two IDs, e.g.
  114==1111, 1348==4736 were byte-identical) -- only content-hashing finds those.
- IMPORTANT kernel fix (from case 1111 / production anonymization): the
  anonymizer exports ConvolutionKernel as a MULTI-VALUED element
  (`['Bv40f','3']`), not a plain string. `str(...)` of that never equals
  "Bv40f", so naive matching silently skips every series. `01` now has a
  `kernel_str()` helper that returns the first element in both cases. All ~3k
  production cases will have the multi-valued form, so this fix is load-bearing.

### Region-grow refinement (02b) — current = v8

1. Seed = TotalSeg LA (label 2) eroded by 5 mm.
2. Adaptive HU threshold = [p2-130, p98+150] of intensities inside eroded seed
   (floor 120 HU). Lower margin widened in steps 50 -> 100 -> 130 to reach dim blood
   in distal PV segments (PVs came up short at -50 and -100). v6 forbidden blockers
   keep the wider window from spilling. NEXT lever if still short: lower the 120 floor
   (legacy cases clamp there), not the margin.
3. SimpleITK ConnectedThreshold from ~1000 sub-sampled seed voxels.
4. Forbidden zone subtracted (seals region-grow leaks across thin shared walls):
   - LV (label 3) raw, RA (label 4) raw  -- dilation would cut the mitral plane /
     eat the interatrial septum.
   - aorta (6), RV (5), PA (7) each +1.5 mm  -- padded to seal the wall the grow
     jumps. RV + PA were ADDED IN v6 to stop PA/RV leaks (the big bug from the
     ~40-case batch: leaks into PA/RV in many cases).
5. Largest connected component.
6. GEODESIC distance cap: drop voxels > 55 mm IN-MASK path distance from the LA
   seed centroid (FastMarching, speed=1 in mask / 0 outside; centroid snapped to
   nearest seed voxel). Follows the PV tubes instead of slicing a sphere, and
   drops components not connected to the seed. (v5/v6 used Euclidean distance.)
   THIS STEP DOES THE BODY/PV SHAPING -- confirmed in tuning, morphology barely
   moves the result (<2% volume across all opening/closing settings).
7. Thin-branch removal: opening dispatched on OPEN_MODE (see file). Validated mode
   = "recon_limited": open to a core at OPEN_RADIUS_MM, then flood the core back
   toward the mask boundary only up to RECON_MM (geodesic conditional dilation).
   Matched OPEN_RADIUS_MM=RECON_MM=2.0 -> strips small free-standing artifacts
   while the body floods back (net removal saturates; raising r self-limits). Then
   largest CC.
8. Closing DISABLED (CLOSE_MM=0). The closing dilate was BRIDGING the LSPV/LAA
   carina and the LPV/LA ridge (the fusion seen on the batch); skipping it fixes
   that. Save.

Tunables at top of file: ERODE_MM=5, MAX_GEODESIC_MM=55, OPEN_MODE="recon_limited",
OPEN_RADIUS_MM=2.0, RECON_MM=2.0, CLOSE_MM=0. New OPEN_MODE options: "plain",
"recon_full", "recon_limited", "none".

### Meshing (03)

- Pad mask with a 4-voxel background border before contouring (ConstantPad,
  updates origin). This closes caps where the mask touches the CT FOV edge, so
  border-touching masks become watertight instead of open. ADDED to fix open
  meshes seen on PA-leak cases and on PVs running vertically to the scan border.
- Gaussian pre-smooth (GAUSS_VAR variance mm^2) for sub-voxel marching-cubes detail.
  Set to 0.7 post-sweep (g1.0 was over-smoothed vs the crisp masks). WARNING: GAUSS_VAR
  is the fusion-risky knob + the detail-eater -- too high re-closes the carina the
  CLOSE_MM=0 mask kept open. Drop toward 0.5 if any tight carina fuses.
- Marching cubes iso=0.5 (point_data, dims = arr.shape).
- Largest CC, Taubin smoothing (TAUBIN_ITER=60, pass_band 0.05, volume-preserving),
  consistent outward normals. Taubin is the SAFE smoothness lever (cannot re-fuse).
- Decimate (30% tris) and save as the SINGLE output `<case>_LA.{vtk,stl}`. Full-res
  is no longer written (large, no visible quality gain). SKIP_EXISTING skips cases
  whose VTK+STL already exist (set False / pass case args to force a re-mesh).
- Smoothing was tuned via `src/_smooth_sweep.py` (renders a g/t ladder to
  `_scrap/smooth_test/` for visual QC). Validated = g0.7/t60.

### Canonicalization (03b)

- Translation-ONLY recenter (centroid -> origin). Scale intentionally NOT
  normalized (LA size is a clinically meaningful predictor).
- Saves 4x4 transform as `.npy` sidecar so EAM/ablation points in patient space
  can be inverse-mapped via `np.linalg.inv(T)`. Originals untouched.
- Output coords: patient/world, LPS convention (SimpleITK + PyVista).

### 3D-EAM integration (04c + 05) -- NEW, in progress

Goal: map CARTO3 voltage onto the CT LA mesh, color by the standard scale, per-region
stats (later via DIVAID), then predict voltage/scar from CT and link to outcomes
(loop-recorder recurrence, ~400 pts).

CARTO3 EXPORT FORMAT (validated on PiPAF8 sample, SW 8.1.1.944 / project 8.1.0.325):
- The "single zip" (Export_*.zip) is a 7z archive mislabeled .zip. py7zr opens it
  directly -- NO password, NOT encrypted. (The S###### folder with Study.zip.0NN +
  StudyEncMetadata.xml is the OTHER, split/encrypted format; we do NOT use it. Both
  start with the 7z magic 37 7A BC AF 27 1C.)
- Contents: one .mesh + one _car.txt + thousands of per-point/ECG/MCC files per map.
  Two maps here: 1-Map (6350 pts) and 1-1-ReMap (6059 pts, post-ablation; default).
- _car.txt (VERSION_6_0): col4-6 = XYZ (mm, CARTO EM frame), col10 = Unipolar mV,
  col11 = Bipolar mV, col12 = LAT ms (-10000 invalid), col17 = Category ('A'=accepted).
  Confirmed against per-point XMLs (<Voltages Unipolar=.. Bipolar=..>).
- .mesh = Biosense TriangulatedMeshVersion2.0: [VerticesSection]/[TrianglesSection]/
  [VerticesColorsSection] (per-vertex Unipolar/Bipolar/LAT/Impedance/Force/... already
  interpolated by CARTO; 10000 = invalid sentinel) + [VerticesAttributesSection].
- TPI / contact NOT usable in this cohort: no force catheter -> Impedance=10000 on all
  vertices, zero impedance samples in per-point XMLs, Force column empty. Quality filter
  falls back to CARTO's 'A' acceptance flag only. If a force catheter is used later, the
  .mesh Force column + VisiTagExport/ContactForceData.txt carry it.

05_eam.py: extract car.txt+mesh (one 7z pass) -> parse points -> ICP-register CARTO mesh
to CT mesh (centroid-align + vtkIterativeClosestPointTransform, rigid) -> apply T to EAM
points -> IDW interpolate (k=10, r=10mm) bipolar/unipolar/LAT + binary scar to CT vertices
-> also transfer CARTO's own mesh scalars as carto_* reference -> write
derivatives/eam/<case>_LA_eam.vtk + _eam_points.csv + _eam_stats.csv (scar %, area, voltage
percentiles). Bipolar scale: <0.5 mV scar, 0.5-1.5 border, >1.5 healthy. NOTE: ICP rotation
relies on PV/LAA asymmetry; if a case's transform has large off-diagonals, add PCA init.

PATIENT MATCHING (04c_eam_match.py): the CARTO export carries ONLY a procedure date
(+ time, + lab study label like "PiPAF8") -- no PID/name/DOB. Chain: procedure date ->
patients_merged.csv (prc_date1..5 -> PID, name, DOB) -> DCM-Test-Mapping.xlsx (PID<->
Study-ID) -> derivatives/meshes/<StudyID>_0___do..._LA.vtk. Study label is NOT a reliable
key (cross-center/lab duplicate labels possible). Date alone is specific (~3/5031) but
cannot resolve same-day collisions (CSV has no time-of-day, export has no DOB) -> those
are flagged `ambiguous` for manual review. Status per export: unique / ambiguous / no_ct /
no_match. DE-ID: PHI crosswalk written to data/logs/eam_crosswalk.csv (gitignored);
ALL derivatives stay keyed by Study-ID only (no date/PID/name in any derivatives/ name).
Mapping files live in data/logs/ (patients_merged.csv, DCM-Test-Mapping.xlsx).

OPEN: the PiPAF8 sample's 3 date-candidates are NOT in the 37-row DCM mapping (no_ct) --
likely a demo export or its CT not yet added/mapped. Infra built + validated; needs a
real EAM/CT pair to run end-to-end. Long-term: per-region stats via DIVAID; CT->scar
prediction (anterior/posterior scar y/n); EAM<->outcome (Cox on recurrence).

---

## Decision log / key learnings (what was tried and rejected, and why)

- TotalSeg default `total` task v2 does NOT separate chambers (single `heart`
  blob, label 51). `heartchambers_highres` is REQUIRED. Academic license set via
  `totalseg_set_license`. `heartchambers_highres` does not support `--fast`.
- GPU inference on GTX 1060 6 GB silently crashes for this task (nvidia-smi shows
  0% util). Fix: `device="cpu"`, `if __name__ == "__main__":` guard, and env vars
  BEFORE imports: `nnUNet_n_proc_DA=0`, `nnUNet_def_n_proc=1`,
  `TORCH_COMPILE_DISABLE=1`, `nnUNet_compile=f` (Windows multiprocessing safety).
- DICOM SliceThickness != z-spacing. True z-spacing from ImagePositionPatient
  deltas (these recons: 0.4 mm thick, 0.3 mm interval).
- Series identity relies on ConvolutionKernel/SliceThickness/ReconstructionDiameter/
  AcquisitionNumber (SeriesDescription is empty). Phase also decodable from Siemens
  ScanOptions (PULSTART_P####PC).
- PV PRUNING: pure radius pruning fails (LA body max radius ~9 mm, comparable to
  PV trunks). Euclidean centroid cap at 60 mm worked as a first pass; now replaced
  by geodesic cap (better: measures distance along the tube, drops stray CCs).
- THIN-VESSEL REMOVAL: opening + geodesic reconstruction was a NEAR NO-OP -
  reconstruction floods every thin tube back from the LA body, so connected PVs
  regrew. Replaced with PLAIN opening (no reconstruction), which actually deletes
  connected thin tubes. KEY LESSON: a single opening's radius sets BOTH the
  diameter cutoff (2r) AND the body-rounding (r); they cannot be tuned apart.
  radius=4 mm (8 mm cutoff) DESTROYED the LA (ate LAA + PV ostia + surface detail).
  radius=2 mm (4 mm cutoff) removes the 2-3 mm speckle while keeping the body.
- Opening and closing are DUALS, not inverses. Matching closing radius to opening
  does NOT regrow what opening removed (opening removes convex protrusions, closing
  fills concave dents). A large closing can BRIDGE adjacent PV stumps or fuse the
  wall across thin gaps - keep it small.
- LAA: not preserved by the opening step, and that is fine per scope.
  Contrast-filled LAAs are trunk-sized and survive anyway; unfilled LAAs are not
  in the segmentation at all.
- PCA canonicalization ABANDONED: LA is not elongated enough, eigenvalues too
  similar -> random axis swaps / sign flips. Scanner (LPS) frame is already well
  aligned due to consistent patient positioning. Translation-only is the right
  pragmatic choice.
- LAA VENOUS-PHASE FUSION fully implemented then fully REVERTED (uncommitted):
  AcqNum 601 venous twin exists for all cases but lacks sufficient contrast for
  adaptive HU thresholding - region-grow ran away to a ~60 mm sphere with no real
  boundary. No true late-enhancement cardiac-res scan exists in this dataset.
- DIVAID and AugmentA operate on already-segmented MESHES, not voxels. The real
  bottleneck is CT-voxels -> closed LA surface, which is what this pipeline solves.
  UPDATE (2026-06): DIVAID is open-source (gitlab.kit.edu/kit/ibt-public/divaid,
  Python; Loewe/KIT-IBT) and does MORE than parcellation -- it auto-CLIPS PVs (81%
  correct) + annotates orifices (100%) and divides the LA into the EHRA/EACVI
  15-segment model (Dice 0.98 vs experts, validated on 140 geometries incl EAM). So
  it is a strong candidate to (a) replace the planned vmtk PV centerline cuts and
  (b) give the canonical regional frame for EAM/outcome mapping. Consumes the surface
  mesh this pipeline produces. EVALUATE NEXT (next major step). Confirm its input
  needs (mesh format, single-LA vs bi-atrial, watertight/orientation) from the repo.
- Pipeline reverts cleanly; commit working state after every meaningful change;
  verify git state before destructive actions.
- POST-TUNING (2 legacy + 2 pcct scrap set, derivatives/_scrap, gitignored):
  - The GEODESIC CAP does the body/PV shaping. Morphology (opening/closing at any
    tested radius) moves volume <2%. The "potato" was never the opening.
  - The CLOSING was the LSPV/LAA + LPV/LA RIDGE FUSION. Disabling it (CLOSE_MM=0)
    fixes the fusion. The old recon2 fusion was closing, NOT the r=2.0 opening:
    r=2.0 with CLOSE=0 keeps the carina separated.
  - recon_limited opening with RECON_MM matched to OPEN_RADIUS_MM makes net removal
    SATURATE (erode r, flood back r). To remove MORE, decouple: low RECON, high r.
  - Smoothness: Taubin iters = safe (volume-preserving, cannot change topology /
    re-fuse). Gaussian mask blur = risky (re-closes carina). Add smoothness via
    Taubin first.
- SMOOTHING SWEEP (src/_smooth_sweep.py, 854/540/2643 across a g/t ladder): g1.0/t100
  was over-smoothed vs the crisp masks. Mechanism confirmed: Gaussian blur MOVES the
  iso-surface (vertex count drops, real detail lost) while Taubin only relaxes vertex
  positions (count preserved). Picked g0.7/t60 (crisp but not staircased).
- DECIMATED-ONLY OUTPUT: decimated (30% tris) is visually indistinguishable from
  full-res here, and full-res files were large -> 03 now writes ONLY the decimated
  mesh as `<case>_LA.{vtk,stl}`; 03b reads it directly.
- DATA-DUP CHECK: content-hash the nifti when curating a cohort. Patient-ID dedup
  (01) misses same-scan-under-two-IDs duplicates (114==1111, 1348==4736 were found
  byte-identical; 1111 + the 47xx samples are being removed by the user).
- SCALE-UP DATA LAYOUT CHANGE (2026-07): the bulk drop for the ~750-case scale-up
  reorganized `data/` into `data/{ct,eam,logs}` subfolders, with CT zips further
  split into batch subfolders (`data/ct/1000 to 1498/`, `.../1500 to 2000/`). 01's
  `DATA_ROOT` still pointed at bare `data/`, so it silently treated `data/ct` itself
  as a single "case" and rglob+dcmread'd every file under it (all ~750 zips) --
  looked like a hang, not a crash (CPU/RAM churning on garbage DICOM parses of zip
  bytes). Fixed: `DATA_ROOT = data/ct`; entry collection now descends one level
  into any subfolder that directly contains zips (a "batch dir") instead of
  treating it as one case. Verified no dup case IDs are silently dropped (9 cases
  exist as both a loose top-level zip AND inside a batch dir -- same file,
  idempotent skip handles it fine).
- BATCH RUN NOW CATCHES PER-CASE ERRORS: 01 previously let any exception (e.g. a
  corrupt zip) crash the whole run -- lost a 12h batch on file #600-ish when
  `1653_0___do92390238.zip` (truncated 447 bytes short of the 4 GiB boundary,
  likely a FAT32/export truncation) raised `BadZipFile`. 01 now wraps the per-case
  body in try/except, logs `status=error` + exception text to selection_log.csv,
  flushes the log after every case, and continues. Scanned all 765 zips for other
  near-4GiB files: only that one is suspiciously short of the boundary (two others
  are legitimately >4GiB and already converted fine) -- looks like an isolated
  export truncation, not a systemic issue. That case still needs a re-export if
  the CT is wanted.

## Status & open issues / roadmap

- TUNING DONE (2026-06). Steps 01-03 working and tuned; 39-patient cohort validated.
  Final config: 01 per-patient dedup; 02b threshold p2-130 / geodesic 55 / forbidden
  RV+PA+aorta +1.5 / recon_limited r2 recon2 / CLOSE=0; 03 mesh pad 4 / g0.7 / t60 /
  decimated-only. Resolved across the iterations: PA/RV leaks (v6 forbidden), open
  meshes (4-voxel pad), carina fusion (CLOSE=0), short distal PVs (threshold -50->130),
  over-smoothing (g1.0/t100 -> g0.7/t60), duplicate patients (dedup) + duplicate data
  (content-hash; user removing 1111 + 47xx samples -> clean ~33-case production set).
- IN PROGRESS (2026-07): scale-up run underway on the full data/ct drop (~756 unique
  cases, ~723 new beyond the 33 already converted). run_all.py running unattended
  through 01-03; 01 hardened to log-and-skip per-case errors instead of dying (see
  decision log). Watch for: series-selection no_match, PA/RV leaks, carina fusion,
  and 02_segment CPU throughput at this volume (~6 min/case estimate from the
  39-case cohort -- confirm it holds at ~750).
- NEXT MAJOR STEP after this run lands: evaluate DIVAID (open source, KIT-IBT) for PV
  clipping + EHRA/EACVI 15-segment regional parcellation -- likely replaces the planned
  vmtk PV centerline cuts AND provides the canonical regional frame for EAM/outcome.
- Residual medium PV stumps remain after r=2 opening (acceptable for a first geometry
  handoff; DIVAID's clipping is the intended fix). Mitral annulus plane cut still TBD.
- 03b_canonicalize NOT in run_all; run separately when canonical outputs are needed
  (derivatives/canonical/ goes stale after a re-mesh).
- Scaling to ~2000: CPU ~6 min/case ~= ~8 days. Options: cloud GPU (RunPod/Colab T4
  ~$0.30-0.50/hr) or local/HPC; consider Docker for portable deploy.
- Data governance: moving patient CT to home machines or across borders to Oxford
  needs explicit DTA coverage. Verify with the DPO before the first Oxford visit.

## Open questions for Oxford

- Mesh format: VTK, STL, or both? Target vertex count (drives decimation)?
- Coordinate / orientation convention (LPS vs RAS)?
- Regional model: is the EHRA/EACVI 15-segment scheme (DIVAID) the parcellation they want?
- Per-case quality metrics they want logged?
- Any GPU/HPC on their side for the 2000-case run, or cloud?
