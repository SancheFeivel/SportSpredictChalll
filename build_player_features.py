"""Add pf_* columns to data/features.csv for the player questions (Q4, Q5, Q6, Q10).

Per player: rolling last-10 means and hit rates (shift(1), no leakage).
Per game: take each team's top-8 players by recent minutes, then
  pf_any_X = 1 - prod(1 - player hit rate)   (chance at least one player hits, assuming independence)
  pf_max_X = best rolling average among those players.
Assumes everyone in the usual rotation plays. Injuries come later.
"""
import glob

import numpy as np
import pandas as pd

WINDOW, MIN_GAMES, TOP_N = 10, 3, 8

p = pd.concat(
    [pd.read_csv(f, dtype={"GAME_ID": str}) for f in sorted(glob.glob("data/P_*.csv"))],
    ignore_index=True,
)
p["date"] = pd.to_datetime(p["GAME_DATE"])
p = p.sort_values(["PLAYER_ID", "date"]).reset_index(drop=True)

p["h30"] = (p["PTS"] >= 30).astype(int)
p["h_ast10"] = (p["AST"] >= 10).astype(int)
p["h_reb12"] = (p["REB"] >= 12).astype(int)
p["h_3x5"] = (p["FG3M"] >= 5).astype(int)

COLS = ["MIN", "PTS", "AST", "REB", "FG3M", "h30", "h_ast10", "h_reb12", "h_3x5"]
g = p.groupby("PLAYER_ID")
for c in COLS:
    p[f"r_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())

p = p.dropna(subset=[f"r_{c}" for c in COLS])
# top-N by recent minutes within each team-game
p["rank"] = p.groupby(["GAME_ID", "TEAM_ID"])["r_MIN"].rank(ascending=False, method="first")
top = p[p["rank"] <= TOP_N]


def any_hit(s):
    return 1.0 - np.prod(1.0 - s.values)


gg = top.groupby("GAME_ID")
pf = pd.DataFrame({
    "pf_any_pts30": gg["r_h30"].apply(any_hit),
    "pf_any_ast10": gg["r_h_ast10"].apply(any_hit),
    "pf_any_reb12": gg["r_h_reb12"].apply(any_hit),
    "pf_any_3x5": gg["r_h_3x5"].apply(any_hit),
    "pf_max_pts": gg["r_PTS"].max(),
    "pf_max_ast": gg["r_AST"].max(),
    "pf_max_reb": gg["r_REB"].max(),
    "pf_max_3": gg["r_FG3M"].max(),
})

df = pd.read_csv("data/features.csv", dtype={"GAME_ID": str}, index_col=0, parse_dates=["date"])
df = df.drop(columns=[c for c in df.columns if c.startswith("pf_")])
df = df.join(pf, how="left")
before = len(df)
df = df.dropna(subset=list(pf.columns))
print(f"games {before} -> {len(df)} with player features")
df.to_csv("data/features.csv")
print(df[list(pf.columns)].describe().loc[["mean", "min", "max"]].round(3).to_string())
