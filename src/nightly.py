"""Nightly window runner: runs run_all.py between START_H and STOP_H (default 19:00-07:00).

Started by Windows Task Scheduler at 19:00 (task "LA-CT nightly"). Sets LA_STOP_AT so the
steps stop cleanly BETWEEN cases at the stop hour (the last case may overrun by a few
minutes). If the pipeline is still alive GRACE_MIN after that, it is killed as a backstop.
Does nothing when started outside the window (e.g. a late catch-up after boot at noon).
All steps are idempotent, so each night simply continues where the last one stopped.
"""
import os
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path

START_H, STOP_H = 19, 7
GRACE_MIN = 30
BELOW_NORMAL = 0x00004000          # child processes inherit this priority class

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "derivatives" / "logs"


def stop_time(now: dt.datetime):
    """Next STOP_H after 'now' if we are inside the window, else None."""
    if now.hour >= START_H:
        day = now.date() + dt.timedelta(days=1)
    elif now.hour < STOP_H:
        day = now.date()
    else:
        return None
    return dt.datetime.combine(day, dt.time(STOP_H))


def main():
    now = dt.datetime.now()
    stop = stop_time(now)
    if stop is None:
        print(f"{now:%H:%M} is outside the {START_H}:00-{STOP_H}:00 window, not starting")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"nightly_{now:%Y%m%d}.log"
    env = dict(os.environ, LA_STOP_AT=str(stop.timestamp()),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n##### nightly start {now:%Y-%m-%d %H:%M}, stop at {stop:%Y-%m-%d %H:%M}\n")
        lf.flush()
        proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "src" / "run_all.py")],
                                cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                creationflags=BELOW_NORMAL)
        hard = stop.timestamp() + GRACE_MIN * 60
        while proc.poll() is None:
            if time.time() > hard:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
                lf.write(f"\n##### backstop kill at {dt.datetime.now():%H:%M}\n")
                break
            time.sleep(30)
        lf.write(f"\n##### nightly end {dt.datetime.now():%H:%M}, rc={proc.returncode}\n")


if __name__ == "__main__":
    main()
