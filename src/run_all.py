"""Run full pipeline."""
import subprocess, sys
from pathlib import Path
SCRIPTS = ["01_dicom_to_nifti.py", "02_segment.py", "03_mesh.py"]
for s in SCRIPTS:
    print(f"\n=== {s} ===")
    subprocess.run([sys.executable, str(Path("src") / s)], check=True)