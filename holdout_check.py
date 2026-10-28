"""Walk-forward check: for each test season, train only on earlier seasons.
Edge = baseline Brier - model Brier, in points (x100). Stable positive edge across seasons = real skill."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLIP = (0.02, 0.98)
TARGETS = [
    "home_win", "total_230", "threes_25", "pts_30", "ast_10",
    "reb_12", "team_120", "margin_10", "threes_5",
]

df = pd.read_csv("data/features.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
feats = [c for c in df.columns if c.startswith(("h_r_", "a_r_", "d_", "sum_", "pf_"))] + [
    "h_rest", "a_rest", "h_b2b", "a_b2b",
]

seasons = sorted(df.season.unique())
test_seasons = seasons[1:]  # need at least one prior season to train on

rows = []
for ts in test_seasons:
    train = df[df.season < ts]
    test = df[df.season == ts]
    recent = sorted(train.season.unique())[-2:]
    for t in TARGETS:
        base_p = train[train.season.isin(recent)][t].mean()
        b_base = brier_score_loss(test[t], np.full(len(test), base_p))
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        m.fit(train[feats], train[t])
        p = np.clip(m.predict_proba(test[feats])[:, 1], *CLIP)
        rows.append({
            "question": t, "season": ts,
            "edge_pts": (b_base - brier_score_loss(test[t], p)) * 100,
        })

res = pd.DataFrame(rows).pivot(index="question", columns="season", values="edge_pts")
res = res.loc[TARGETS]
res["mean"] = res.mean(axis=1)
print("edge vs recent-rate baseline, points per game (positive = model better)\n")
print(res.round(2).to_string())
print("\nrows with all seasons positive:", list(res.index[(res.drop(columns='mean') > 0).all(axis=1)]))
