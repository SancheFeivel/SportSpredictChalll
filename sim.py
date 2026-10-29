"""Joint game simulator. One (margin, total) distribution -> Q1, Q2, Q7, Q8 answers, all consistent.
Margin/total = ridge on pre-game features + correlated normal residual. Continuity-corrected thresholds."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_DRAWS = 8000
TARGETS = ["home_win", "total_230", "team_120", "margin_10"]


def feat_cols(df):
    return [c for c in df.columns if c.startswith(("h_r_", "a_r_", "d_", "sum_"))] + [
        "h_rest", "a_rest", "h_b2b", "a_b2b",
    ]


def fit(train, feats):
    ym = train["home_pts"] - train["away_pts"]
    yt = train["home_pts"] + train["away_pts"]
    mk = lambda: make_pipeline(StandardScaler(), Ridge(alpha=200.0))
    mm, mt = mk().fit(train[feats], ym), mk().fit(train[feats], yt)
    res = np.c_[ym - mm.predict(train[feats]), yt - mt.predict(train[feats])]
    return {"mm": mm, "mt": mt, "cov": np.cov(res.T), "feats": feats}


def probs(spec, X, n=N_DRAWS, seed=0):
    """X: DataFrame of features. Returns DataFrame (rows = games) with TARGETS columns."""
    z = np.random.default_rng(seed).multivariate_normal([0, 0], spec["cov"], n)  # shared draws
    mu_m = spec["mm"].predict(X[spec["feats"]])[:, None]
    mu_t = spec["mt"].predict(X[spec["feats"]])[:, None]
    M, T = mu_m + z[:, 0][None, :], mu_t + z[:, 1][None, :]
    H, A = (T + M) / 2, (T - M) / 2
    return pd.DataFrame({
        "home_win": (M > 0).mean(axis=1),
        "total_230": (T >= 229.5).mean(axis=1),
        "team_120": ((H >= 119.5) | (A >= 119.5)).mean(axis=1),
        "margin_10": (np.abs(M) >= 9.5).mean(axis=1),
    }, index=X.index)
