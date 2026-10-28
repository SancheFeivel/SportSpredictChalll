"""Train one model per question. Keep model only if it beats the recent-rate baseline on the test season."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_SEASON = "2025-26"
CLIP = (0.02, 0.98)
# positive edge in every walk-forward season (holdout_check.py); extend after player features
MODEL_OK = {"home_win", "total_230", "threes_25", "team_120", "pts_30", "ast_10", "reb_12"}
PLAYER_Q = {"pts_30", "ast_10", "reb_12", "threes_5"}

TARGETS = [
    "home_win", "total_230", "threes_25", "pts_30", "ast_10",
    "reb_12", "team_120", "margin_10", "threes_5",
]  # Q9 (20-pt lead) needs play-by-play, added later

df = pd.read_csv("data/features.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
feats = [c for c in df.columns if c.startswith(("h_r_", "a_r_", "d_", "sum_"))] + [
    "h_rest", "a_rest", "h_b2b", "a_b2b",
]
pf_feats = [c for c in df.columns if c.startswith("pf_")]

train = df[df.season < TEST_SEASON]
test = df[df.season == TEST_SEASON]
recent_seasons_train = sorted(train.season.unique())[-2:]
recent_all = sorted(df.season.unique())[-2:]


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


rows, saved = [], {}
for t in TARGETS:
    base_rate = train[train.season.isin(recent_seasons_train)][t].mean()
    b_base = brier_score_loss(test[t], np.full(len(test), base_rate))

    tf = feats + pf_feats if t in PLAYER_Q else feats
    m = new_model().fit(train[tf], train[t])
    p = np.clip(m.predict_proba(test[tf])[:, 1], *CLIP)
    b_model = brier_score_loss(test[t], p)

    use_model = b_model < b_base and t in MODEL_OK
    rows.append({
        "question": t,
        "base_brier": round(b_base, 4),
        "model_brier": round(b_model, 4),
        "edge_pts": round((b_base - b_model) * 100, 2),
        "use": "model" if use_model else "baseline",
    })

    # refit on all data for live use
    if use_model:
        final = new_model().fit(df[tf], df[t])
        saved[t] = {
            "type": "model", "model": final, "features": tf,
            "p": float(df[df.season.isin(recent_all)][t].mean()),  # fallback if features fail
        }
    else:
        saved[t] = {"type": "const", "p": float(df[df.season.isin(recent_all)][t].mean())}

print(pd.DataFrame(rows).to_string(index=False))
joblib.dump(saved, "data/models.joblib")
print("\nsaved data/models.joblib")
