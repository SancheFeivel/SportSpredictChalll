"""Train one model per question. Keep model only if it is in MODEL_OK and beats the recent-rate baseline.

SHRINK pulls unstable questions toward the base rate: p = w*model + (1-w)*rate.
Base rate (const + fallback) = current season blended with last 2 seasons, weight grows with games played."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RATE_K, prev_season, season_of

TEST_SEASON = "2025-26"
CLIP = (0.02, 0.98)
# positive in most walk-forward seasons (holdout_check.py). Last three are shrunk: 2026-27 went negative on small n.
MODEL_OK = {"home_win", "threes_25", "reb_12", "team_120", "total_230", "pts_30", "ast_10"}
SHRINK = {"total_230": 0.5, "pts_30": 0.5, "ast_10": 0.5}
PLAYER_Q = {"pts_30", "ast_10", "reb_12", "threes_5"}
MIN_TEST, MIN_TRAIN = 100, 300

TARGETS = [
    "home_win", "total_230", "threes_25", "pts_30", "ast_10",
    "reb_12", "team_120", "margin_10", "lead_20", "threes_5",
]

df = pd.read_csv("data/features.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
feats = [c for c in df.columns if c.startswith(("h_r_", "a_r_", "d_", "sum_"))] + [
    "h_rest", "a_rest", "h_b2b", "a_b2b",
]
pf_feats = [c for c in df.columns if c.startswith("pf_")]

CUR = season_of(pd.Timestamp.now())
PRIOR_SEASONS = [prev_season(CUR), prev_season(prev_season(CUR))]


def live_rate(d, t):
    """Current-season rate blended with last 2 seasons; weight = n/(n+RATE_K)."""
    pr = d[d.season.isin(PRIOR_SEASONS)][t].mean()
    if np.isnan(pr):  # e.g. lead_20 labels cover few seasons
        pr = d[t].mean()
    c = d[d.season == CUR][t]
    if len(c) == 0:
        return float(pr)
    w = len(c) / (len(c) + RATE_K)
    return float(w * c.mean() + (1 - w) * pr)


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


rows, saved = [], {}
for t in TARGETS:
    if t not in df.columns or df[t].notna().sum() == 0:
        print(f"skip {t}: no labels (run pull_pbp.py then refresh)")
        continue
    d = df.dropna(subset=[t])
    train, test = d[d.season < TEST_SEASON], d[d.season == TEST_SEASON]
    rate_all = live_rate(d, t)

    if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
        saved[t] = {"type": "const", "p": rate_all}
        rows.append({"question": t, "base_brier": None, "model_brier": None, "edge_pts": None, "use": "baseline (too little data)"})
        continue

    recent_tr = sorted(train.season.unique())[-2:]
    base_rate = train[train.season.isin(recent_tr)][t].mean()
    b_base = brier_score_loss(test[t], np.full(len(test), base_rate))

    tf = feats + pf_feats if t in PLAYER_Q else feats
    m = new_model().fit(train[tf], train[t])
    p = np.clip(m.predict_proba(test[tf])[:, 1], *CLIP)
    b_model = brier_score_loss(test[t], p)

    use_model = b_model < b_base and t in MODEL_OK
    rows.append({
        "question": t, "base_brier": round(b_base, 4), "model_brier": round(b_model, 4),
        "edge_pts": round((b_base - b_model) * 100, 2),
        "use": ("model" + (f" (shrink {SHRINK[t]})" if t in SHRINK else "")) if use_model else "baseline",
    })

    if use_model:
        final = new_model().fit(d[tf], d[t])
        saved[t] = {"type": "model", "model": final, "features": tf, "p": rate_all, "w": SHRINK.get(t, 1.0)}
    else:
        saved[t] = {"type": "const", "p": rate_all}

print(pd.DataFrame(rows).to_string(index=False))
import track
track.log_run(saved, rows, n_games=len(df))
joblib.dump(saved, "data/models.joblib")
print("\nsaved data/models.joblib")