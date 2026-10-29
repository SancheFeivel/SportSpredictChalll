"""One command for the live step: refresh -> injuries -> run_today.

Usage:
  python live.py --date 2026-10-29              # full run (first run of the day)
  python live.py --date 2026-10-29 --quick      # skip refresh (reruns before tip-off)
  python live.py --date 2026-10-29 --no-injuries   # keep your hand-edited out.txt

A failed step is reported but does not stop the next one: stale data or an old
out.txt is better than submitting nothing.
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--date", help="US Eastern game date, YYYY-MM-DD (default: today US Eastern)")
ap.add_argument("--quick", action="store_true", help="skip refresh.py")
ap.add_argument("--no-injuries", action="store_true", help="skip injuries.py (keep out.txt as is)")
a = ap.parse_args()

dflag = ["--date", a.date] if a.date else []
steps = []
if not a.quick:
    steps.append(("refresh.py", []))
if not a.no_injuries:
    steps.append(("injuries.py", dflag))
steps.append(("run_today.py", dflag))

failed = []
for script, args in steps:
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    if subprocess.run([sys.executable, script, *args]).returncode != 0:
        failed.append(script)
        print(f"!! {script} failed, continuing", flush=True)

print("\nlive done" + (f" (failed: {', '.join(failed)})" if failed else ""))
