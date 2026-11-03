"""Nightly window runner: runs run_all.py 19:00-07:00 on Tue-Fri and non-stop from Fri 19:00
to Tue 07:00 (weekend + Monday, the PC is free then); see stop_time().

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
NO_WINDOW = 0x08000000             # hidden console for run_all + its steps
# This script itself runs under pythonw.exe (Task Scheduler, no window), but pythonw
# children lose stdout/stderr (the 2026-10-27 night log was empty and 02 died without a
# traceback). run_all and the steps therefore run under python.exe with a hidden console.
PYTHON = Path(sys.executable).with_name("python.exe")

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "derivatives" / "logs"


BUSY_DAYS = {1, 2, 3, 4}           # Tue-Fri: the PC is in use 07:00-19:00. Sat/Sun/Mon: free.


def stop_time(now: dt.datetime):
    """When to stop if running is allowed at 'now', else None.

    Not allowed: Tue-Fri 07:00-19:00. Otherwise run until the next 07:00 that starts a
    busy day, i.e. nightly on weekdays and straight through Fri 19:00 -> Tue 07:00."""
    if now.weekday() in BUSY_DAYS and STOP_H <= now.hour < START_H:
        return None
    for add in range(8):
        cand = dt.datetime.combine(now.date() + dt.timedelta(days=add), dt.time(STOP_H))
        if cand > now and cand.weekday() in BUSY_DAYS:
            return cand


def main():
    now = dt.datetime.now()
    stop = stop_time(now)
    if stop is None:
        print(f"{now:%a %H:%M} is a busy-hours slot (Tue-Fri {STOP_H}:00-{START_H}:00), not starting")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"nightly_{now:%Y%m%d}.log"
    env = dict(os.environ, LA_STOP_AT=str(stop.timestamp()),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n##### nightly start {now:%Y-%m-%d %H:%M}, stop at {stop:%Y-%m-%d %H:%M}\n")
        lf.flush()
        proc = subprocess.Popen([str(PYTHON), "-u", str(ROOT / "src" / "run_all.py")],
                                cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                creationflags=BELOW_NORMAL | NO_WINDOW)
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
