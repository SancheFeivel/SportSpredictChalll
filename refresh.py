"""Re-pull data and rebuild everything. Run before each live day (or when you want fresh form).
Optional steps (play-by-play) may fail without stopping the rest, so models still retrain."""
import subprocess
import sys
import time

STEPS = [
    "pull_data.py",
    "pull_pbp.py",
    "build_labels.py",
    "build_features.py",
    "build_player_features.py",
    "train_all.py",
    "plot_track.py",
]
OPTIONAL = {"pull_pbp.py"}  # rate limits / slow; old cached data is fine

failed = []
for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, step])
    if r.returncode != 0:
        if step in OPTIONAL:
            print(f"WARNING: {step} failed (exit {r.returncode}); continuing with cached data", flush=True)
            failed.append(step)
        else:
            sys.exit(f"{step} failed (exit {r.returncode}); stopping")
    print(f"({time.time() - t0:.0f}s)", flush=True)

print("\nrefresh done" + (f" (optional steps failed: {', '.join(failed)})" if failed else ""))
