# LA-CT Pipeline

Automated left-atrium segmentation from cardiac CT for downstream DL training on raw geometry (LA body + LAA + PVs as single structure).

## Context
- **Thomas Küffer**, Inselspital Bern, collab with Oxford cardiac imaging lab
- Cohort: AF ablation patients; some w/ EAM + outcomes
- Comm preference: terse, commented code

## Hardware
- Desktop: Win 10, 16 GB RAM, **GTX 1060 6 GB** (driver 581.57)
- 16 GB office laptop (no GPU) as backup

## Stack
- Python 3.11, Miniconda env `la`
- `totalsegmentator`, `SimpleITK`, `pydicom`, `nibabel`, `numpy`, `scipy`, `vtk`, `pyvista`, `tqdm`
- `torch==2.5.1+cu121` + `torchvision==0.20.1+cu121` (matched pair, install together)
- VS Code, Git for Windows, Command Prompt terminal w/ `conda init cmd.exe`
- GitHub: `kuefferth/la-ct-pipeline` (private)
- 3D Slicer for QC

## Data
- 5 sample cases under `data/<case_id>/` (gitignored): 4733, 4734, 4735, 4736, 4738
- Siemens dual-source, ~30 series per study, anonymized (empty SeriesDescription)
- Two cardiac acquisitions per case: AcqNum 501 (diastolic 70%+ RR) **chosen**, 601 (systolic 30–60% RR)

## Series selection rules
- `ConvolutionKernel == "Bv40f"` (vascular soft kernel)
- `SliceThickness == 0.4` mm
- `ReconstructionDiameter < 250` mm (cardiac FOV)
- `AcquisitionNumber == 501`
- Phase decoded from `ScanOptions` field (`PULSTART_P####PC`)

## Pipeline
data/<case>/                                            DICOMs (gitignored)
→ 01_dicom_to_nifti.py                                smart series selection
derivatives/nifti/<case>.nii.gz
→ 02_segment.py                                       TotalSegmentator heartchambers_highres (LA label = 2)
derivatives/seg_full/<case>_chambers.nii.gz             multi-label (myo, LA, LV, RA, RV, aorta, PA)
→ 02b_regiongrow.py                                   refine: erode TotalSeg LA by 5mm → adaptive threshold → region grow → largest CC
derivatives/seg_la/<case>_LA.nii.gz                     binary LA+LAA+PVs
→ 03_mesh.py                                          marching cubes → Taubin smooth → decimate
derivatives/meshes/<case>_LA.{vtk,stl,_decimated.stl}

Diagnostics: `00_inspect_series.py`, `00b_inspect_bv40.py`, `00c_inspect_fov.py`.

## Segmentation
- **TotalSegmentator `heartchambers_highres`** — academic license registered (`totalseg_set_license`). Produces LA chamber blob only; **PVs cut at ostia, LAA undershot**. Used as seed for region-grow refinement.
- **NOT** the default `total` task — v2 dropped chamber separation, only single `heart` blob (label 51).

## Region grow refinement (v1, current)
- Seed = TotalSeg LA eroded by 5 mm
- Threshold = adaptive: `mean ± 3·SD` of CT intensities inside eroded seed, clipped to [80, 800] HU
- Grow via `SimpleITK.ConnectedThreshold`
- Keep largest CC
- **No forbidden mask yet** (LV/aorta/etc) — add if leaks observed in v2
- **No PV distance cap yet** — add if PVs overshoot

## Known Windows + nnUNet workarounds (in scripts)
1. Set env vars BEFORE imports: `nnUNet_n_proc_DA=0`, `nnUNet_def_n_proc=1`, `TORCH_COMPILE_DISABLE=1`, `nnUNet_compile=f`
2. Wrap entry point in `if __name__ == "__main__":` (avoids fork-bomb)
3. `force_split=True` for highres on 6 GB VRAM
4. `torchvision` DLL warning is benign — ignore
5. `heartchambers` task name doesn't exist (only `heartchambers_highres`); doesn't support `--fast`

## Status
- ✅ Step 0/1: inspection + DICOM→NIfTI for all 5 cases
- ✅ Step 2: TotalSegmentator `heartchambers_highres` ran on 4733 (~5 min GPU). Output: LA blob, PVs cut, LAA undershot — confirmed insufficient.
- 🟡 Step 2b: region-grow refinement script written (`02b_regiongrow.py`), **not yet run/QC'd**
- ⬜ Step 3: meshing not tested
- ⬜ Run all 5 cases, QC

## Next actions
1. Run `02b_regiongrow.py` on case 4733, QC in Slicer
2. If leaks (LV/aorta) → add forbidden-mask constraint using other TotalSeg labels (3,4,5,6,7,1) dilated 1–2 mm
3. If PVs overshoot → add distance-from-seed cap (~40–50 mm)
4. Switch `GLOB = "*.nii.gz"`, run on all 5
5. Run meshing
6. Discuss mesh output format/conventions with Oxford collaborators

## Coordination notes for assistants resuming this project
- User is biomed engineer: strong clinical/imaging knowledge, comfortable with concepts; learning Python ecosystem, GitHub, conda
- Terse responses, commented code, full files for big changes, line-edits for small ones
- Commit working state after every meaningful change