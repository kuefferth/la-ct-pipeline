"""End-to-end pipeline orchestrator.
Runs the four pipeline steps on whatever cases are present under data/.
Each step is idempotent: outputs that already exist are reused.

Keeps the system (and display) awake for the whole run via SetThreadExecutionState,
so locking the screen or idle timeouts won't suspend long CPU segmentation.
"""
import subprocess
import sys
import time
import ctypes
from pathlib import Path

STEPS = [
    "01_dicom_to_nifti.py",
    "02_segment.py",
    "02b_regiongrow.py",
    "03_mesh.py",
]

SRC = Path(__file__).parent

# Windows execution-state flags to block idle sleep for the duration of the run.
ES_CONTINUOUS        = 0x80000000   # keep the state until we clear it
ES_SYSTEM_REQUIRED   = 0x00000001   # don't let the system sleep
ES_DISPLAY_REQUIRED  = 0x00000002   # also keep the display on (optional)

def keep_awake(enable: bool, keep_display: bool = False):
    """Toggle the OS idle-sleep blocker. enable=False restores normal behavior."""
    if not sys.platform.startswith("win"):
        return
    flags = ES_CONTINUOUS
    if enable:
        flags |= ES_SYSTEM_REQUIRED
        if keep_display:
            flags |= ES_DISPLAY_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(flags)

def main():
    keep_awake(True)            # block idle sleep; display may still turn off (fine)
    try:
        t_total = time.perf_counter()
        for step in STEPS:
            path = SRC / step
            if not path.exists():
                print(f"!! missing script: {path}")
                sys.exit(1)
            print(f"\n{'='*60}\n=== {step}\n{'='*60}")
            t0 = time.perf_counter()
            result = subprocess.run([sys.executable, str(path)])
            dt = time.perf_counter() - t0
            if result.returncode != 0:
                print(f"\n!! {step} failed (returncode={result.returncode}) after {dt:.1f}s")
                sys.exit(result.returncode)
            print(f"\n--- {step} done in {dt:.1f}s")
        print(f"\n[run_all] total: {time.perf_counter() - t_total:.1f}s")
    finally:
        keep_awake(False)       # always restore, even on error/exit