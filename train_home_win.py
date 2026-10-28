"""Q1: will home team win. Time split: train on older seasons, test on newest."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "home_win"
TEST_SEASON = "2025-26"

SETS = {
    "margin_only": ["d_margin"],
    "form": ["d_margin", "d_win", "h_r_margin", "a_r_margin"],
    "form+rest": ["d_margin", "d_win", "h_r_margin", "a_r_margin", "d_rest", "h_b2b", "a_b2b"],
}

df = pd.read_csv("data/features.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
train = df[df.season < TEST_SEASON]
test = df[df.season == TEST_SEASON]
y_tr, y_te = train[TARGET], test[TARGET]
print(f"train {len(train)}  test {len(test)}")

base_p = np.full(len(test), y_tr.mean())
print(f"\n{'always 0.5':<14} brier {brier_score_loss(y_te, np.full(len(test), 0.5)):.4f}")
print(f"{'train rate':<14} brier {brier_score_loss(y_te, base_p):.4f}  (p={y_tr.mean():.3f})")

best = None
for name, feats in SETS.items():
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(train[feats], y_tr)
    p = np.clip(model.predict_proba(test[feats])[:, 1], 0.02, 0.98)
    b = brier_score_loss(y_te, p)
    print(f"{name:<14} brier {b:.4f}  logloss {log_loss(y_te, p):.4f}")
    if best is None or b < best[0]:
        best = (b, name, feats, model, p)

b, name, feats, model, p = best
print(f"\nbest: {name}  brier {b:.4f}  vs baseline {brier_score_loss(y_te, base_p):.4f}")
print(f"edge vs baseline: {(brier_score_loss(y_te, base_p) - b) * 100:+.2f} pts per game")

# calibration: do 60-70% calls happen ~65%?
bins = pd.cut(p, [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
cal = pd.DataFrame({"pred": p, "actual": y_te.values}).groupby(bins, observed=True).agg(
    n=("actual", "size"), mean_pred=("pred", "mean"), actual=("actual", "mean")
)
print("\ncalibration:\n", cal.round(3).to_string())

# refit best on all data for live use
final = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
final.fit(df[feats], df[TARGET])
joblib.dump({"model": final, "features": feats}, "data/home_win.joblib")
print("\nsaved data/home_win.joblib")
