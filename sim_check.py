"""Walk-forward check of the joint simulator vs recent-rate baseline (same rule as holdout_check.py).
Writes data/sim_ok.json (questions positive in EVERY test season) and data/sim.joblib (fit on all data)."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import sim

CLIP = (0.02, 0.98)
df = pd.read_csv("data/features.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
feats = sim.feat_cols(df)
seasons = sorted(df.season.unique())

rows = []
for ts in seasons[1:]:
    train, test = df[df.season < ts], df[df.season == ts]
    recent = sorted(train.season.unique())[-2:]
    P = sim.probs(sim.fit(train, feats), test).clip(*CLIP)
    for t in sim.TARGETS:
        base = train[train.season.isin(recent)][t].mean()
        rows.append({"question": t, "season": ts,
                     "edge_pts": (brier_score_loss(test[t], np.full(len(test), base))
                                  - brier_score_loss(test[t], P[t])) * 100})

res = pd.DataFrame(rows).pivot(index="question", columns="season", values="edge_pts")
res["mean"] = res.mean(axis=1)
print("simulator edge vs recent-rate baseline, pts/game\n")
print(res.round(2).to_string())
ok = list(res.index[(res.drop(columns="mean") > 0).all(axis=1)])
print("\nuse sim for:", ok)
json.dump(ok, open("data/sim_ok.json", "w"))
joblib.dump(sim.fit(df, feats), "data/sim.joblib")
