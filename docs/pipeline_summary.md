LA-CT segmentation pipeline (Bern → Oxford)

INPUT
- Siemens dual-source cardiac CT, DICOM
- ECG-gated, ~30 series/study (multiple recons)
- Auto-selected: Bv40f kernel, 0.4mm slice, cardiac FOV (<250mm),
  AcqNum 501 = diastolic 70%+ RR

PIPELINE (Python, fully automated)

1. DICOM → NIfTI
   - SimpleITK series reader
   - Smart series selection by DICOM tags (kernel, thickness, FOV, acq#)

2. Coarse LA segmentation
   - TotalSegmentator heartchambers_highres (nnU-Net v2, CPU)
   - Output: multi-label volume (myo, LA, LV, RA, RV, aorta, PA)
   - LA label = 2; used as seed only (PVs cut, LAA undershot, walls
     undershot — boundaries not reliable on their own)

3. Refinement via region growing
   - Seed = TotalSeg LA eroded by 5 mm
   - Adaptive threshold: [p2−50, p98+150] HU of intensities inside seed
   - SimpleITK ConnectedThreshold from ~1000 sub-sampled seed voxels
   - Forbidden zone: LV + aorta (raw, no dilation — preserves mitral plane
     and avoids aortic-root encroachment)
   - Largest connected component
   - Distance cap: drop voxels >60 mm from LA seed centroid (cuts distal PVs)
   - Thin-vessel removal: morphological opening (2.5 mm kernel) + geodesic
     reconstruction (drops branches thinner than ~5 mm diameter,
     preserves LA body + thick PV trunks)
   - Closing (1.5 mm): fills small wall dents

4. Meshing
   - Gaussian pre-smooth (variance 0.6 mm²) for sub-voxel detail
   - Marching cubes (iso=0.5)
   - Largest connected component
   - Taubin smoothing (50 iters, pass_band 0.05) — volume-preserving
   - Consistent outward-pointing normals
   - Save full-resolution + decimated (30% triangles) as VTK + STL

OUTPUT
- Binary LA+LAA+PV-cuffs mask (NIfTI, voxel-accurate)
- Triangulated surface mesh (VTK + STL, full + decimated)
- Single connected structure, no manual annotation
- TotalSeg seed kept as *_LA_seed.nii.gz for reference

RUNTIME (per case)
- DICOM → NIfTI:     ~10 s
- TotalSeg (CPU):    ~5 min  (GPU silently crashes on GTX 1060 + Win 10)
- Region grow:       ~30 s
- Meshing:           ~10 s
- Total:             ~6 min / case

VOLUMES OBSERVED (sample n=5)
- 111–241 mL final LA; consistent with normal-to-dilated AF cohort

KNOWN LIMITATIONS
- Minor PV branch remnants survive thin-vessel removal (cosmetic)
- LAA tip undersegmented in cases with thin/trabeculated appendages
- Valve replacements / LAA plugs cause local artifacts (ignored for now)

STACK
- Python 3.11, SimpleITK, TotalSegmentator (nnU-Netv2), pyvista, numpy
- Code: github.com/kuefferth/la-ct-pipeline

OPEN QUESTIONS FOR OXFORD
- Mesh format preference: VTK, STL, or both?
- Target vertex count for DL training (drives decimation level)?
- Coordinate system / orientation conventions (LPS vs RAS, etc.)?
- Quality metrics they want logged per case?
- For scaling to ~2000 cases: do they have GPU/HPC we could leverage,
  or should we use cloud (Colab Pro / RunPod ~ $0.30-0.50/hr T4)?