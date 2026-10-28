"""Build the 0/1 answers for Q1-8 and Q10 from box scores. Q9 needs play-by-play."""
import glob
import os

import pandas as pd


def load(kind):
    frames = []
    for f in sorted(glob.glob(f"data/{kind}_*.csv")):
        d = pd.read_csv(f, dtype={"GAME_ID": str})
        d["season"] = os.path.basename(f)[2:-4]  # e.g. 2025-26
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


teams = load("T")
players = load("P")

teams["is_home"] = teams["MATCHUP"].str.contains("vs.", regex=False)
home = teams[teams.is_home].set_index("GAME_ID")
away = teams[~teams.is_home].set_index("GAME_ID")
home, away = home.align(away, join="inner", axis=0)  # drop any unmatched games

g = pd.DataFrame(index=home.index)
g["date"] = pd.to_datetime(home["GAME_DATE"])
g["season"] = home["season"]
g["home_team"] = home["TEAM_ABBREVIATION"]
g["away_team"] = away["TEAM_ABBREVIATION"]

# Q1-Q3, Q7, Q8
g["home_win"] = (home["WL"] == "W").astype(int)
g["total_230"] = ((home.PTS + away.PTS) >= 230).astype(int)
g["threes_25"] = ((home.FG3M + away.FG3M) > 25).astype(int)
g["team_120"] = ((home.PTS >= 120) | (away.PTS >= 120)).astype(int)
g["margin_10"] = ((home.PTS - away.PTS).abs() >= 10).astype(int)

# Q4, Q5, Q6, Q10
p = players.groupby("GAME_ID").agg(
    mp=("PTS", "max"), ma=("AST", "max"), mr=("REB", "max"), m3=("FG3M", "max")
)
g["pts_30"] = (p.mp.reindex(g.index) >= 30).astype(int)
g["ast_10"] = (p.ma.reindex(g.index) >= 10).astype(int)
g["reb_12"] = (p.mr.reindex(g.index) >= 12).astype(int)
g["threes_5"] = (p.m3.reindex(g.index) >= 5).astype(int)

g = g.sort_values("date")
g.to_csv("data/labels.csv")

print(f"games: {len(g)}")
print(g.groupby("season").size().to_string())
print("\nbase rates (all seasons):")
print(g.drop(columns=["date"]).mean(numeric_only=True).round(3).to_string())
print("\nbase rates by season:")
print(g.groupby("season").mean(numeric_only=True).round(3).T.to_string())
