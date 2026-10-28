"""Walk-forward check: for each test season, train only on earlier seasons.
Edge = baseline Brier - model Brier, in points (x100). Stable positive edge across seasons = real skill.
Seasons with <100 graded games are shown as NaN (too small to mean anything)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLIP = (0.02, 0.98)
MIN_TEST, MIN_TRAIN = 100, 300
TARGETS = [
    "home_win", "total_230", "threes_25", "pts_30", "ast_10",
    "reb_12", "team_120", "margin_10", "lead_20", "threes_5",
]

df = pd.read_csv("data/features.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
feats = [c for c in df.columns if c.startswith(("h_r_", "a_r_", "d_", "sum_", "pf_"))] + [
    "h_rest", "a_rest", "h_b2b", "a_b2b",
]

seasons = sorted(df.season.unique())
rows = []
for ts in seasons[1:]:
    for t in TARGETS:
        if t not in df.columns:
            continue
        d = df.dropna(subset=[t])
        train, test = d[d.season < ts], d[d.season == ts]
        edge = np.nan
        if len(train) >= MIN_TRAIN and len(test) >= MIN_TEST:
            recent = sorted(train.season.unique())[-2:]
            base_p = train[train.season.isin(recent)][t].mean()
            b_base = brier_score_loss(test[t], np.full(len(test), base_p))
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            m.fit(train[feats], train[t])
            p = np.clip(m.predict_proba(test[feats])[:, 1], *CLIP)
            edge = (b_base - brier_score_loss(test[t], p)) * 100
        rows.append({"question": t, "season": ts, "edge_pts": edge})

res = pd.DataFrame(rows).pivot(index="question", columns="season", values="edge_pts")
res = res.reindex([t for t in TARGETS if t in res.index])
res["mean"] = res.mean(axis=1)
print("edge vs recent-rate baseline, points per game (positive = model better)\n")
print(res.round(2).to_string())
ok = res.drop(columns="mean").apply(lambda r: r.dropna().gt(0).all() and r.notna().any(), axis=1)
print("\nrows with all graded seasons positive:", list(res.index[ok]))
