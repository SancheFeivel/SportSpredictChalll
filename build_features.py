"""Pre-game features per game. Every rolling stat uses shift(1): current game excluded.
Early season: rolling form is shrunk toward last season's prior (form.py)."""
import glob
import os

import pandas as pd

from form import K_SEASON, attach_prior, season_year

WINDOW = 10
MIN_GAMES = 5


def load_teams():
    frames = []
    for f in sorted(glob.glob("data/T_*.csv")):
        d = pd.read_csv(f, dtype={"GAME_ID": str})
        d["season"] = os.path.basename(f)[2:-4]
        frames.append(d)
    t = pd.concat(frames, ignore_index=True)
    t["date"] = pd.to_datetime(t["GAME_DATE"])
    t["is_home"] = t["MATCHUP"].str.contains("vs.", regex=False)
    return t


t = load_teams()

t["opp_pts"] = t["PTS"] - t["PLUS_MINUS"]
t["poss"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
t["win"] = (t["WL"] == "W").astype(int)
t["margin"] = t["PLUS_MINUS"]

opp = t[["GAME_ID", "TEAM_ID", "FG3M", "poss"]]
m = t.merge(opp, on="GAME_ID", suffixes=("", "_opp"))
m = m[m["TEAM_ID"] != m["TEAM_ID_opp"]].copy()
m["tot_pts"] = m["PTS"] + m["opp_pts"]
m["tot_3"] = m["FG3M"] + m["FG3M_opp"]
m["pace"] = (m["poss"] + m["poss_opp"]) / 2

m = m.sort_values(["TEAM_ID", "date"]).reset_index(drop=True)
m["sy"] = season_year(m["date"])

ROLL = ["PTS", "opp_pts", "FG3M", "FG3M_opp", "pace", "win", "margin", "tot_pts", "tot_3"]
g = m.groupby("TEAM_ID")
for c in ROLL:
    m[f"r_{c}"] = g[c].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
    )

# shrink toward prior: weight on current form = gs / (gs + K), gs = games played this season
pri = attach_prior(m, ROLL)
gs = m.groupby(["TEAM_ID", "sy"]).cumcount()
w = gs / (gs + K_SEASON)
for c in ROLL:
    raw = m[f"r_{c}"]
    m[f"r_{c}"] = (w * raw + (1 - w) * pri[c]).where(pri[c].notna(), raw)

m["rest"] = g["date"].diff().dt.days.fillna(7).clip(upper=7)
m["b2b"] = (m["rest"] <= 1).astype(int)
m["games_before"] = g.cumcount()

FEATS = [f"r_{c}" for c in ROLL] + ["rest", "b2b", "games_before"]

home = m[m["is_home"]].set_index("GAME_ID")[FEATS].add_prefix("h_")
away = m[~m["is_home"]].set_index("GAME_ID")[FEATS].add_prefix("a_")
X = home.join(away, how="inner")

X["d_win"] = X["h_r_win"] - X["a_r_win"]
X["d_margin"] = X["h_r_margin"] - X["a_r_margin"]
X["d_rest"] = X["h_rest"] - X["a_rest"]
X["sum_pace"] = X["h_r_pace"] + X["a_r_pace"]
X["sum_pts"] = X["h_r_PTS"] + X["a_r_PTS"]
X["sum_3"] = X["h_r_FG3M"] + X["a_r_FG3M"]
X["d_absmargin"] = X["d_margin"].abs()

labels = pd.read_csv("data/labels.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
df = labels.join(X, how="inner")
df = df.sort_values("date")

before = len(df)
df = df.dropna(subset=[c for c in X.columns if c.startswith(("h_r_", "a_r_"))])
print(f"games: {before} -> {len(df)} after dropping early-window NaNs")

df.to_csv("data/features.csv")
print(df.groupby("season").size().to_string())
print("\ncorr with home_win (top):")
print(df.select_dtypes("number").corr()["home_win"].drop("home_win").abs().sort_values(ascending=False).head(8).round(3).to_string())