"""Re-pull data and rebuild everything. Run before each live day (or when you want fresh form)."""
import subprocess
import sys

STEPS = [
    "pull_data.py",
    "pull_pbp.py",
    "build_labels.py",
    "build_features.py",
    "build_player_features.py",
    "train_all.py",
]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, step], check=True)
print("\nrefresh done")
