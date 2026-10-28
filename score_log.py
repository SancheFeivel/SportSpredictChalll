"""Score logged forecasts (data/log.csv) against real results (data/labels.csv).

Run after games finish AND after `python build_labels.py` (or refresh.py) so results are in labels.csv.
Usage: python score_log.py            # scores the FIRST logged forecast per game (safest)
       python score_log.py --which last
Q9 is skipped (no labels yet).
"""
import argparse

import numpy as np
import pandas as pd

MAP = {
    "q1": "home_win", "q2": "total_230", "q3": "threes_25", "q4": "pts_30", "q5": "ast_10",
    "q6": "reb_12", "q7": "team_120", "q8": "margin_10", "q10": "threes_5",
}

ap = argparse.ArgumentParser()
ap.add_argument("--which", choices=["first", "last"], default="first")
a = ap.parse_args()

log = pd.read_csv("data/log.csv", dtype={"game_id": str}).sort_values("logged_at")
log = log[log["error"].isna() | (log["error"] == "")]
log = log.groupby("game_id").first() if a.which == "first" else log.groupby("game_id").last()

labels = pd.read_csv("data/labels.csv", dtype={"GAME_ID": str}, index_col=0)
j = log.join(labels[list(MAP.values())], how="inner")
print(f"games logged: {len(log)}   with results: {len(j)}   (using {a.which} forecast per game)")
if j.empty:
    raise SystemExit("no finished games yet: rerun after games end and build_labels.py")

# baseline: flat rate from the two most recent full seasons
recent = labels[labels["season"].isin(sorted(labels["season"].unique())[-3:-1])]
base = {col: recent[col].mean() for col in MAP.values()}

rows, total_pts = [], 0.0
for q, col in MAP.items():
    y = j[col].values
    p = j[q].values
    b_bot = np.mean((p - y) ** 2)
    b_base = np.mean((base[col] - y) ** 2)
    total_pts += (b_base - b_bot) * 100 * len(y)
    rows.append({
        "question": f"{q} {col}", "n": len(y),
        "bot": round(b_bot, 4), "baseline": round(b_base, 4),
        "edge_pts": round((b_base - b_bot) * 100, 2),
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
print(f"\nmean bot Brier      {res['bot'].mean():.4f}")
print(f"mean baseline Brier {res['baseline'].mean():.4f}")
print(f"summed edge vs flat baseline: {total_pts:+.1f} points over {len(j) * len(MAP)} graded questions")
print("\nBaseline is NOT the crowd. Under ~100 games, treat results as noise.")