# LA-CT Pipeline

Automated left-atrium segmentation from cardiac CT for downstream DL training on raw geometry (LA body + LAA + PV cuffs as a single structure).

## Context
- **Thomas Küffer**, Inselspital Bern, collab with Oxford cardiac imaging lab
- Cohort: AF ablation patients; some w/ EAM + outcomes
- Comm preference: terse, commented code

## Hardware
- Desktop: Win 10, 16 GB RAM, GTX 1060 6 GB (driver 581.57)
- GPU broken for `heartchambers_highres` on this setup (silent CUDA crash, fully debugged → CPU only); other tasks GPU-fine
- 16 GB office laptop (no GPU) as backup

## Stack
- Python 3.11, Miniconda env `la`
- `totalsegmentator`, `SimpleITK`, `pydicom`, `nibabel`, `numpy`, `scipy`, `vtk`, `pyvista`, `tqdm`
- `torch==2.5.1+cu121` + `torchvision==0.20.1+cu121` (matched pair, install together)
- VS Code, Git for Windows, Command Prompt terminal w/ `conda init cmd.exe`
- GitHub: `kuefferth/la-ct-pipeline` (private)
- 3D Slicer / ParaView for QC

## Data
- 5 sample cases under `data/<case_id>/` (gitignored): 4733, 4734, 4735, 4736, 4738
- Siemens dual-source, ~30 series per study, anonymized (empty SeriesDescription)
- Two cardiac acquisitions per case: AcqNum 501 (diastolic 70%+ RR) **chosen**, 601 (systolic 30–60% RR)

## Series selection rules (Siemens cardiac CT)
- `ConvolutionKernel == "Bv40f"` (vascular soft kernel)
- `SliceThickness == 0.4` mm
- `ReconstructionDiameter < 250` mm (cardiac FOV)
- `AcquisitionNumber == 501`
- Phase decoded from `ScanOptions` field (`PULSTART_P####PC`)

## Pipeline
data/<case>/                                            DICOMs (gitignored)
→ 01_dicom_to_nifti.py                                smart series selection
derivatives/nifti/<case>.nii.gz
→ 02_segment.py                                       TotalSegmentator heartchambers_highres on CPU
derivatives/seg_full/<case>_chambers.nii.gz             multi-label (myo, LA, LV, RA, RV, aorta, PA)
derivatives/seg_la/<case>_LA_seed.nii.gz                LA-only from TotalSeg (kept as seed reference)
→ 02b_regiongrow.py                                   refinement (see below)
derivatives/seg_la/<case>_LA.nii.gz                     binary LA+LAA+PV-cuffs (final)
→ 03_mesh.py                                          smooth mask → marching cubes → Taubin → decimate
derivatives/meshes/<case>_LA.{vtk,stl}                  full-res mesh
derivatives/meshes/<case>_LA_decimated.{vtk,stl}        decimated (30% triangles)

Diagnostics: `00_inspect_series.py`, `00b_inspect_bv40.py`, `00c_inspect_fov.py`.

## Segmentation strategy
**TotalSegmentator `heartchambers_highres`** — academic license registered (`totalseg_set_license`).
Produces LA chamber blob (label 2). Limitations: PVs cut at ostia, LAA undershot, boundaries undershoot wall.
Used as **seed** for region-grow refinement, not as final output.

**Region-grow refinement v5** (`02b_regiongrow.py`):
1. Seed = TotalSeg LA eroded by 5 mm
2. Adaptive threshold = `[p2−50, p98+150]` HU of intensities inside eroded seed
3. SimpleITK `ConnectedThreshold` from ~1000 sub-sampled seed voxels
4. Forbidden zone = LV (raw, label 3) + aorta (raw, label 6); subtracted
5. Largest connected component
6. Distance cap: drop voxels >60 mm from LA seed centroid (cuts distal PVs)
7. Thin-vessel removal: morphological opening + geodesic reconstruction (kernel 2.5 mm)
8. Closing (1.5 mm) to fill small wall dents

## Meshing (`03_mesh.py`)
- Gaussian pre-smooth (variance 0.6 mm²) → sub-voxel marching-cubes detail
- Marching cubes (iso=0.5)
- Largest CC, Taubin smoothing (50 iters, pass_band 0.05), consistent outward normals
- Save full-res VTK+STL, decimated 30% VTK+STL

## Known Windows + nnUNet workarounds (in scripts)
1. Env vars BEFORE imports: `nnUNet_n_proc_DA=0`, `nnUNet_def_n_proc=1`, `TORCH_COMPILE_DISABLE=1`, `nnUNet_compile=f`
2. `if __name__ == "__main__":` guard (Windows multiprocessing safety)
3. `device="cpu"` for `heartchambers_highres` — GPU silently crashes on this setup
4. `torchvision` DLL warning is benign — ignore
5. `heartchambers_highres` doesn't support `--fast`

## Status
- ✅ Steps 0–3 all working on 5 sample cases
- ✅ Visual QC done; meshes look clean (minor PV remnants accepted)
- ⬜ Discuss mesh format/conventions w/ Oxford
- ⬜ Scale: 2000-case run will need GPU (cloud / better hardware / HPC)

## Open issues / future work
- **Scaling:** CPU at ~5 min/case × 2000 ≈ 7 days. Cloud GPU (Colab Pro / RunPod T4 ~ $15 total) or RTX A4000-class card recommended.
- **PV pruning:** distance-from-centroid is crude; for production, fit cut-planes at PV ostia
- **Valve replacements / LAA plugs:** currently produce small artifacts; ignored for now
- **Aorta under-segmentation:** if region grows into aortic root, may need to add `pulmonary_artery` (label 7) to forbidden mask too
- **Per-case parameter tuning:** all tunables at top of each script; values were set on normal anatomy

## Coordination notes for assistants resuming this project
- User is biomed engineer: strong clinical/imaging knowledge, comfortable with concepts; learning Python ecosystem, GitHub, conda
- Terse responses, commented code, full files for big changes, line-edits for small ones
- Commit working state after every meaningful change