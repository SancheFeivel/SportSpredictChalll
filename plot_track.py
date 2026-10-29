"""Draw charts from tracking/weights_history.jsonl (written by track.log_run).

Run:  python plot_track.py
Makes:
  tracking/edge_by_question.png  3x3 grid, edge vs baseline per training run (pts/game, + = better)
  tracking/drift.png             how much each question's weights moved per run (L2)
Filled dot = model used live. Hollow dot = baseline used live (model not trusted yet).
"""
import matplotlib
import matplotlib.pyplot as plt

import track

matplotlib.use("Agg")

LABEL = {
    "home_win": "Q1 home win", "total_230": "Q2 total 230+", "threes_25": "Q3 threes >25",
    "pts_30": "Q4 30+ pts", "ast_10": "Q5 10+ ast", "reb_12": "Q6 12+ reb",
    "team_120": "Q7 team 120+", "margin_10": "Q8 margin 10+", "threes_5": "Q10 5+ threes",
}
GOOD, BAD = "#1D9E75", "#D85A30"

hist = track.load()
if not hist:
    raise SystemExit("no runs yet: call track.log_run(...) in train_all.py, then run it")

runs = [h["run"] for h in hist]
games = [h["n_games"] for h in hist]

# ---- edge per question ----
fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
fig.suptitle(f"Edge vs baseline per question (pts/game, + = better)   {len(runs)} runs", fontsize=14)
for ax, q in zip(axes.flatten(), LABEL):
    ys = [h["questions"].get(q, {}).get("edge_pts") for h in hist]
    use = [h["questions"].get(q, {}).get("use") for h in hist]
    pts = [(r, y, u) for r, y, u in zip(runs, ys, use) if y is not None]
    if not pts:
        continue
    x, y, u = zip(*pts)
    color = GOOD if y[-1] > 0 else BAD
    ax.plot(x, y, "-", color=color, lw=2)
    for xi, yi, ui in pts:
        ax.plot(xi, yi, "o", color=color, mfc=color if ui == "model" else "white", ms=7)
    ax.axhline(0, color="black", lw=0.8)
    change = y[-1] - y[0]
    ax.set_title(f"{LABEL[q]}   now {y[-1]:+.2f}  ({change:+.2f} since run {x[0]})", fontsize=9)
    ax.set_xlabel("training run #", fontsize=8)
    ax.set_ylabel("edge pts/game", fontsize=8)
    if len(x) <= 12:
        ax.set_xticks(x)
fig.savefig("tracking/edge_by_question.png", dpi=130)
plt.close(fig)

# ---- weight drift ----
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
for q in LABEL:
    d = [(h["run"], h["questions"].get(q, {}).get("drift")) for h in hist]
    d = [(r, v) for r, v in d if v is not None]
    if d:
        ax.plot(*zip(*d), marker="o", label=LABEL[q])
ax.set_title("Weight drift per run (L2 change in coefficients). Falling = model settling")
ax.set_xlabel("training run #")
ax.set_ylabel("drift")
ax.legend(fontsize=7, ncol=3)
fig.savefig("tracking/drift.png", dpi=130)
plt.close(fig)

# ---- text summary ----
last = hist[-1]
print(f"run {last['run']}  games {last['n_games']}  ({games[0]} -> {games[-1]} over {len(runs)} runs)\n")
for q in LABEL:
    r = last["questions"].get(q)
    if r:
        print(f"{LABEL[q]:<16} edge {r['edge_pts']:+6.2f}  use {r['use']:<8} drift {r['drift'] if r['drift'] is None else round(r['drift'], 3)}")
print("\nsaved tracking/edge_by_question.png, tracking/drift.png")
