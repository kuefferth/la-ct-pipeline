LA-CT segmentation pipeline (Bern → Oxford)

INPUT
- Siemens dual-source cardiac CT, DICOM
- ECG-gated, ~30 series/study (multiple recons)
- Auto-selected: Bv40f kernel, 0.4mm slice, cardiac FOV (<250mm),
  AcqNum 501 = diastolic 70%+ RR

PIPELINE (Python, fully automated)

1. DICOM → NIfTI  (01_dicom_to_nifti.py)
   - SimpleITK series reader
   - Smart series selection by DICOM tags (kernel, thickness, FOV, acq#)

2. Coarse LA segmentation  (02_segment.py)
   - TotalSegmentator heartchambers_highres (nnU-Net v2, CPU)
   - Output: multi-label volume (myo, LA, LV, RA, RV, aorta, PA)
   - LA label = 2; used as seed only (PVs cut, LAA undershot, walls
     undershot — boundaries not reliable on their own)

3. Refinement via region growing  (02b_regiongrow.py, v7)
   - Seed = TotalSeg LA eroded by 5 mm
   - Adaptive threshold: [p2−50, p98+150] HU of intensities inside seed
   - SimpleITK ConnectedThreshold from ~1000 sub-sampled seed voxels
   - Forbidden zone (subtracted to seal region-grow leaks):
       LV (raw), RA (raw)       — thin shared walls; dilating would cut the
                                  mitral plane / eat into interatrial septum
       aorta, RV, PA (+1.5 mm)  — padded to seal the wall the grow jumps;
                                  RV + PA added in v6 to stop PA/RV leaks
   - Largest connected component
   - Geodesic distance cap: drop voxels >60 mm IN-MASK path distance from the
     LA seed centroid. Follows the PV tubes instead of slicing a sphere, and
     drops any component not connected to the seed. (Was Euclidean in v5/v6.)
   - Thin-branch removal: plain morphological opening (4 mm radius, NO
     reconstruction). Deletes connected tubes thinner than ~8 mm diameter,
     keeps thick PV trunk + ostia. (v5/v6 used opening + geodesic
     reconstruction, which regrew every connected tube → near no-op; dropped.)
   - Closing (1.5 mm): fills small wall dents

4. Meshing  (03_mesh.py)
   - Pad mask with 4-voxel background border before contouring
     (closes caps where the mask touches the CT FOV edge → watertight)
   - Gaussian pre-smooth (variance 0.6 mm²) for sub-voxel detail
   - Marching cubes (iso=0.5)
   - Largest connected component
   - Taubin smoothing (50 iters, pass_band 0.05) — volume-preserving
   - Consistent outward-pointing normals
   - Save full-resolution + decimated (30% triangles) as VTK + STL

OUTPUT
- Binary LA+PV-cuffs mask (NIfTI, voxel-accurate)
- Triangulated surface mesh (VTK + STL, full + decimated)
- Single connected structure, no manual annotation
- TotalSeg seed kept as *_LA_seed.nii.gz for reference

RUNTIME (per case)
- DICOM → NIfTI:     ~10 s
- TotalSeg (CPU):    ~5 min  (GPU silently crashes on GTX 1060 + Win 10)
- Region grow:       ~30 s
- Meshing:           ~10 s
- Total:             ~6 min / case

KNOWN LIMITATIONS
- PV cut is diameter + length based, not anatomical. Production fix =
  vmtk centerline cut-planes at PV ostia (planned).
- LAA not preserved by the opening step (thin appendage tips erased).
  Acceptable per scope: contrast-filled LAAs are trunk-sized and survive;
  unfilled LAAs are not in the segmentation anyway.
- Valve replacements / LAA plugs cause local artifacts (ignored for now)

STACK
- Python 3.11, SimpleITK, TotalSegmentator (nnU-Net v2), pyvista, numpy
- Code: github.com/kuefferth/la-ct-pipeline

STATUS
- Steps 0–3 working; ~40-case batch run completed on prior version
- v6 (PA/RV leak fix) + v7 (geodesic cap, thick-trunk opening) + mesh
  border-pad implemented; pending re-validation on the 40-case batch
- Next: vmtk centerline PV cuts; GPU/cloud for the 2000-case scale

OPEN QUESTIONS FOR OXFORD
- Mesh format preference: VTK, STL, or both?
- Target vertex count for DL training (drives decimation level)?
- Coordinate system / orientation conventions (LPS vs RAS, etc.)?
- Quality metrics they want logged per case?
- For scaling to ~2000 cases: do they have GPU/HPC we could leverage,
  or should we use cloud (Colab Pro / RunPod ~ $0.30-0.50/hr T4)?