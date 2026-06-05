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
- Repo: github.com/kuefferth/la-ct-pipeline (private). Now lives on external SSD
  at `D:\la-ct-pipeline` (was `C:\Users\cit\Documents\...`). After the move, Git
  threw "dubious ownership"; fixed with
  `git config --global --add safe.directory D:/la-ct-pipeline`.
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
data/<case>/                                  DICOMs (gitignored)
  -> 01_dicom_to_nifti.py                      smart series selection
derivatives/nifti/<case>.nii.gz
  -> 02_segment.py                             TotalSeg heartchambers_highres (CPU)
derivatives/seg_full/<case>_chambers.nii.gz    multi-label (myo,LA,LV,RA,RV,aorta,PA)
derivatives/seg_la/<case>_LA_seed.nii.gz       TotalSeg LA only (seed reference)
  -> 02b_regiongrow.py                         refinement (see below)
derivatives/seg_la/<case>_LA.nii.gz            binary LA+PV-cuffs (final)
  -> 03_mesh.py                                mask -> marching cubes -> Taubin
derivatives/meshes/<case>_LA.{vtk,stl}         full-res
derivatives/meshes/<case>_LA_decimated.{vtk,stl}  decimated (30% tris)
  -> 03b_canonicalize.py                       translation-only recenter + 4x4 sidecar
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
- IMPORTANT kernel fix (from case 1111 / production anonymization): the
  anonymizer exports ConvolutionKernel as a MULTI-VALUED element
  (`['Bv40f','3']`), not a plain string. `str(...)` of that never equals
  "Bv40f", so naive matching silently skips every series. `01` now has a
  `kernel_str()` helper that returns the first element in both cases. All ~3k
  production cases will have the multi-valued form, so this fix is load-bearing.

### Region-grow refinement (02b) — current = v7

1. Seed = TotalSeg LA (label 2) eroded by 5 mm.
2. Adaptive HU threshold = [p2-50, p98+150] of intensities inside eroded seed.
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
  Set to 1.0 post-tuning (smoothest of the scrap set). WARNING: GAUSS_VAR is the
  fusion-risky knob -- too high re-closes the carina the CLOSE_MM=0 mask kept open.
  Re-check carinas on the full batch; drop toward 0.6 if any tight carina fuses.
- Marching cubes iso=0.5 (point_data, dims = arr.shape).
- Largest CC, Taubin smoothing (TAUBIN_ITER=100, pass_band 0.05, volume-preserving),
  consistent outward normals. Taubin is the SAFE smoothness lever (cannot re-fuse).
- Save full + decimated (30% tris) as VTK + STL. SKIP_EXISTING skips cases whose
  full VTK + STL already exist.

### Canonicalization (03b)

- Translation-ONLY recenter (centroid -> origin). Scale intentionally NOT
  normalized (LA size is a clinically meaningful predictor).
- Saves 4x4 transform as `.npy` sidecar so EAM/ablation points in patient space
  can be inverse-mapped via `np.linalg.inv(T)`. Originals untouched.
- Output coords: patient/world, LPS convention (SimpleITK + PyVista).

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

## Status & open issues / roadmap

- Steps 0-3 + 03b working. A ~40-case batch run completed on the PRIOR version and
  surfaced two real bugs: PA/RV leaks (11/16 legacy, 6/20 PCCT) and open meshes on
  border-touching cases.
- v6 (RV+PA forbidden), v7 (geodesic cap), the mesh 4-voxel pad, and the post-tuning
  config (recon_limited r=2/recon=2, CLOSE=0, geodesic 55, mesh g1.0/t100) address
  those bugs + the carina fusion. RUNNING the full 41-case batch now to confirm on
  all cases (re-run 02b then 03; 01/02 are idempotent and reused). Spot-check
  carinas (Gaussian 1.0 risk) and PA/RV leaks on the output.
- Residual medium PV stumps remain after r=2 opening (acceptable for a first
  geometry handoff). The CLEAN fix for "keep LA body + LAA crisp, cut each PV
  perpendicular at a set distance from its ostium" is vmtk centerline cut-planes -
  the planned next milestone. Morphology cannot do this (radius couples cutoff and
  rounding). Mitral annulus plane cut also planned.
- PV ostia refinement candidate: DTU LAA-Net weights (arXiv 2510.06090).
- Scaling to ~2000: CPU ~5-6 min/case ~= 7 days. Options: cloud GPU
  (Colab Pro / RunPod T4 ~$0.30-0.50/hr, est ~$15 total) or better local/HPC.
  Consider containerizing (Docker) for portable deploy.
- Data governance: moving patient CT to home machines or across borders to Oxford
  needs explicit DTA coverage. Verify with the DPO before the first Oxford visit.

## Open questions for Oxford

- Mesh format: VTK, STL, or both? Target vertex count (drives decimation)?
- Coordinate / orientation convention (LPS vs RAS)?
- Per-case quality metrics they want logged?
- Any GPU/HPC on their side for the 2000-case run, or cloud?
