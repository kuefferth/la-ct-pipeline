"""Cooperative stop time for the nightly window.

nightly.py sets LA_STOP_AT (epoch seconds). Each step checks time_up() between cases
and exits cleanly, so a stop never lands in the middle of writing a NIfTI / mask.
Unset (manual runs) = never time up.
"""
import os
import time


def time_up() -> bool:
    stop = os.environ.get("LA_STOP_AT")
    return bool(stop) and time.time() >= float(stop)
