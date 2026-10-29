"""Read tracking/weights_history.jsonl, draw tracking/progress.png.
Usage: python plot_progress.py [--q home_win]   (--q picks question for the weight-drift panel)
Needs: pip install matplotlib
"""
import argparse

import matplotlib.pyplot as plt

import track

ap = argparse.ArgumentParser()
ap.add_argument("--q", default="home_win")
a = ap.parse_args()

hist = track.load()
if len(hist) < 2:
    raise SystemExit(f"need >=2 runs, have {len(hist)}. run train_all.py again after new data.")

runs = [h["run"] for h in hist]
qnames = sorted({q for h in hist for q in h["questions"]})
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

# 1. edge per question
for q in qnames:
    ax[0][0].plot(runs, [h["questions"].get(q, {}).get("edge_pts") for h in hist], marker="o", label=q)
ax[0][0].axhline(0, color="k", lw=0.8)
ax[0][0].set(title="Edge vs baseline (pts/game, test season)", xlabel="run")
ax[0][0].legend(fontsize=7, ncol=2)

# 2. total edge over time
tot = [sum(v["edge_pts"] for v in h["questions"].values() if v["use"] == "model") for h in hist]
ax[0][1].plot(runs, tot, marker="o", color="tab:green")
ax[0][1].set(title="Summed edge, questions using model", xlabel="run")

# 3. coefficient paths for one question
feats = sorted({f for h in hist for f in ((h["questions"].get(a.q) or {}).get("coef") or {})})
top = sorted(feats, key=lambda f: -max(abs(((h["questions"].get(a.q) or {}).get("coef") or {}).get(f, 0)) for h in hist))[:8]
for f in top:
    ax[1][0].plot(runs, [((h["questions"].get(a.q) or {}).get("coef") or {}).get(f) for h in hist], marker=".", label=f)
ax[1][0].set(title=f"{a.q}: top weights over runs", xlabel="run")
ax[1][0].legend(fontsize=7)

# 4. drift per run
for q in qnames:
    ax[1][1].plot(runs, [h["questions"].get(q, {}).get("drift") for h in hist], marker=".", label=q)
ax[1][1].set(title="Weight drift vs previous run (L2)", xlabel="run")

plt.tight_layout()
plt.savefig("tracking/progress.png", dpi=130)
print("saved tracking/progress.png")
first, last = hist[0], hist[-1]
print(f"summed model edge: run1 {tot[0]:+.2f} -> run{last['run']} {tot[-1]:+.2f}  ({last['n_games'] - first['n_games']} new games)")
