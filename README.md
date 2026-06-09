LA-CT segmentation pipeline (Bern → Oxford)

INPUT
- Siemens cardiac CT, DICOM (photon-counting "pcct" and older dual-source "legacy")
- ECG-gated, ~30 series/study (multiple recons)
- Auto-selected per scanner profile:
    pcct   : Bv40f kernel, 0.4mm slice, cardiac FOV (<250mm), AcqNum 501 (diastolic 70%+ RR)
    legacy : I30f kernel, 0.75mm slice, cardiac FOV, ORIGINAL, arterial LA phase
- Per-patient dedup: when a patient ships both a pcct and a legacy study, keep the
  high-res pcct and drop the legacy (one geometry per patient).

PIPELINE (Python, fully automated; orchestrated by run_all.py = steps 01,02,02b,03)

1. DICOM → NIfTI  (01_dicom_to_nifti.py)
   - SimpleITK series reader; smart series selection by DICOM tags
   - Patient-level dedup (pcct preferred over legacy)
   - Output suffixed with profile: <case>__pcct.nii.gz / <case>__legacy.nii.gz

2. Coarse LA segmentation  (02_segment.py)
   - TotalSegmentator heartchambers_highres (nnU-Net v2, CPU)
   - Output: multi-label volume (myo, LA, LV, RA, RV, aorta, PA)
   - LA label = 2; used as SEED only (PVs cut, walls undershot on their own)

3. Refinement via region growing  (02b_regiongrow.py)
   - Seed = TotalSeg LA eroded by 5 mm
   - Adaptive threshold: [p2−130, p98+150] HU of intensities inside seed
     (floor 120 HU; widened in steps from −50 to reach dim distal-PV blood)
   - SimpleITK ConnectedThreshold from ~1000 sub-sampled seed voxels
   - Forbidden zone (subtracted to seal region-grow leaks):
       LV (raw), RA (raw)       — thin shared walls; dilating would cut the
                                  mitral plane / eat into interatrial septum
       aorta, RV, PA (+1.5 mm)  — padded to seal the wall the grow jumps
   - Largest connected component
   - Geodesic distance cap: drop voxels >55 mm IN-MASK path distance from the LA
     seed centroid (follows the PV tubes; drops components not wired to the seed).
     THIS step does the body/PV shaping; morphology moves volume <2%.
   - Thin-branch removal: opening, mode "recon_limited" (open to a 2 mm core, flood
     back up to 2 mm) — strips small free-standing artifacts, body floods back.
   - Closing DISABLED (it bridged the LSPV/LAA carina + LPV/LA ridge = fusion).

4. Meshing  (03_mesh.py)
   - Pad mask with 4-voxel background border before contouring (closes caps where
     the mask touches the CT FOV edge → watertight)
   - Gaussian pre-smooth (variance 0.7 mm²) for sub-voxel detail
   - Marching cubes (iso=0.5) → largest connected component
   - Taubin smoothing (60 iters, pass_band 0.05) — volume-preserving
   - Consistent outward-pointing normals
   - Decimate (30% triangles) and save as the SINGLE output (VTK + STL).
     Full-res is no longer written (large, no visible quality gain vs decimated).
   - Smoothing tuned on a sweep (src/_smooth_sweep.py): g0.7/t60. Gaussian is the
     detail-eater + carina-fusion risk; Taubin is the safe (volume-preserving) lever.

(03b_canonicalize.py — optional, not in run_all) translation-only recenter to the LA
   centroid + 4×4 sidecar (.npy) for inverse-mapping EAM/outcome points. Scale kept
   (LA size is clinically meaningful). Output: derivatives/canonical/.

OUTPUT
- Binary LA+PV-cuffs mask (NIfTI, voxel-accurate)
- Triangulated surface mesh, decimated (VTK + STL) — single connected structure
- TotalSeg seed kept as *_LA_seed.nii.gz for reference

RUNTIME (per case)
- DICOM → NIfTI:     ~10 s
- TotalSeg (CPU):    ~5 min  (GPU silently crashes on GTX 1060 + Win 10)
- Region grow:       ~50 s
- Meshing:           ~10 s
- Total:             ~6 min / case

KNOWN LIMITATIONS
- PV cut is diameter + length based, not anatomical. Candidate production fix =
  DIVAID (KIT-IBT, open source) which auto-clips PVs + annotates orifices and divides
  the LA into the EHRA/EACVI 15-segment model — operates on the surface mesh this
  pipeline produces. Evaluate next. (Was: vmtk centerline cut-planes.)
- LAA not preserved by the opening step (thin appendage tips erased). Acceptable per
  scope: contrast-filled LAAs are trunk-sized and survive; unfilled LAAs are not in
  the segmentation anyway.
- Valve replacements / LAA plugs cause local artifacts (ignored for now).

STACK
- Python 3.11, SimpleITK, TotalSegmentator (nnU-Net v2), pyvista, numpy
- Code: github.com/kuefferth/la-ct-pipeline

STATUS
- Steps 01–03 working and tuned. 39-patient cohort validated (p2−130 threshold,
  g0.7/t60 smoothing, decimated-only output, per-patient dedup).
- Next: scale-up trial run on ~200 cases; evaluate DIVAID for PV clipping + regional
  segmentation; GPU/cloud for the full ~2000-case scale.

OPEN QUESTIONS FOR OXFORD
- Mesh format preference: VTK, STL, or both? Target vertex count (drives decimation)?
- Coordinate system / orientation conventions (LPS vs RAS, etc.)?
- Quality metrics they want logged per case?
- Regional model: is the EHRA/EACVI 15-segment (DIVAID) the parcellation they want?
- For scaling to ~2000 cases: GPU/HPC on their side, or cloud (RunPod/Colab T4)?
