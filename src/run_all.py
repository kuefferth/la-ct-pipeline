"""End-to-end pipeline orchestrator.

Runs (in order) the four pipeline steps on whatever cases are present under data/.
Each step is idempotent: outputs that already exist are reused.

Usage:
    python src/run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

# Steps in execution order. Diagnostics (00*) are intentionally skipped here;
# run them manually when first inspecting a new dataset.
STEPS = [
    "01_dicom_to_nifti.py",   # DICOM -> NIfTI w/ smart series selection
    "02_segment.py",          # TotalSegmentator heartchambers_highres (CPU)
    "02b_regiongrow.py",      # Region-grow refinement (LV+aorta block, centroid cap, etc.)
    "03_mesh.py",             # Mask -> smoothed surface mesh (VTK + STL)
]

SRC = Path(__file__).parent

def main():
    t_total = time.perf_counter()
    for step in STEPS:
        path = SRC / step
        if not path.exists():
            print(f"!! missing script: {path}")
            sys.exit(1)
        print(f"\n{'='*60}\n=== {step}\n{'='*60}")
        t0 = time.perf_counter()
        # Use sys.executable to ensure the active conda env is used
        result = subprocess.run([sys.executable, str(path)])
        dt = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"\n!! {step} failed (returncode={result.returncode}) after {dt:.1f}s")
            sys.exit(result.returncode)
        print(f"\n--- {step} done in {dt:.1f}s")
    print(f"\n[run_all] total: {time.perf_counter() - t_total:.1f}s")

if __name__ == "__main__":
    main()