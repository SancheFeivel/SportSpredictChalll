"""predict(home, away, date) -> 10 probabilities in contest order (Q1..Q10).

Usage: python bot.py --home BOS --away NYK --date 2026-11-02
Fallback chain per question: model -> constant rate from models.joblib -> hard default.
Never returns nothing, never returns 0 or 100.
"""
import argparse
import glob
import traceback

import joblib
import pandas as pd

WINDOW = 10
CLIP = (0.02, 0.98)
Q9_RATE = 0.50  # TODO: replace with measured rate once play-by-play labels exist
HARD_DEFAULT = 0.50

# contest order Q1..Q10 -> target name (None = no model yet)
QUESTIONS = [
    "home_win", "total_230", "threes_25", "pts_30", "ast_10",
    "reb_12", "team_120", "margin_10", None, "threes_5",
]


def load_team_logs():
    t = pd.concat([pd.read_csv(f, dtype={"GAME_ID": str}) for f in sorted(glob.glob("data/T_*.csv"))])
    t["date"] = pd.to_datetime(t["GAME_DATE"])
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
    return m.sort_values("date")


ROLL = ["PTS", "opp_pts", "FG3M", "FG3M_opp", "pace", "win", "margin", "tot_pts", "tot_3"]


def team_state(logs, abbr, game_date):
    """Form from the last WINDOW games strictly before game_date."""
    g = logs[(logs["TEAM_ABBREVIATION"] == abbr) & (logs["date"] < game_date)].tail(WINDOW)
    if len(g) < 5:
        raise ValueError(f"not enough games for {abbr}")
    s = {f"r_{c}": g[c].mean() for c in ROLL}
    rest = min((game_date - g["date"].iloc[-1]).days, 7)
    s["rest"] = rest
    s["b2b"] = int(rest <= 1)
    return s


def build_row(home, away, game_date):
    logs = load_team_logs()
    d = pd.Timestamp(game_date)
    h, a = team_state(logs, home, d), team_state(logs, away, d)
    row = {f"h_{k}": v for k, v in h.items()}
    row.update({f"a_{k}": v for k, v in a.items()})
    row["d_win"] = row["h_r_win"] - row["a_r_win"]
    row["d_margin"] = row["h_r_margin"] - row["a_r_margin"]
    row["d_rest"] = row["h_rest"] - row["a_rest"]
    row["sum_pace"] = row["h_r_pace"] + row["a_r_pace"]
    row["sum_pts"] = row["h_r_PTS"] + row["a_r_PTS"]
    row["sum_3"] = row["h_r_FG3M"] + row["a_r_FG3M"]
    return pd.DataFrame([row])


TOP_N = 8
_PLAYER_CACHE = {}


def load_player_logs():
    if "pl" not in _PLAYER_CACHE:
        pl = pd.concat([pd.read_csv(f, dtype={"GAME_ID": str}) for f in sorted(glob.glob("data/P_*.csv"))])
        pl["date"] = pd.to_datetime(pl["GAME_DATE"])
        _PLAYER_CACHE["pl"] = pl.sort_values("date")
    return _PLAYER_CACHE["pl"]


def live_pf(home, away, game_date, out_players=()):
    """Same pf_* features as training. Roster = each player's latest team; out_players are dropped.
    Caveat: early season uses last season's rosters until new games are pulled."""
    d = pd.Timestamp(game_date)
    pl = load_player_logs()
    prior = pl[pl["date"] < d]
    last10 = prior.groupby("PLAYER_ID").tail(WINDOW)
    agg = last10.groupby("PLAYER_ID").agg(
        n=("PTS", "size"), team=("TEAM_ABBREVIATION", "last"), name=("PLAYER_NAME", "last"),
        MIN=("MIN", "mean"), PTS=("PTS", "mean"), AST=("AST", "mean"),
        REB=("REB", "mean"), FG3M=("FG3M", "mean"),
        h30=("PTS", lambda s: (s >= 30).mean()), h_ast10=("AST", lambda s: (s >= 10).mean()),
        h_reb12=("REB", lambda s: (s >= 12).mean()), h_3x5=("FG3M", lambda s: (s >= 5).mean()),
    )
    out_l = {n.strip().lower() for n in out_players}
    agg = agg[(agg["n"] >= 3) & agg["team"].isin([home, away]) & ~agg["name"].str.lower().isin(out_l)]
    top = agg.sort_values("MIN", ascending=False).groupby("team").head(TOP_N)
    if top["team"].nunique() < 2:
        raise ValueError("roster missing for one team")

    def any_hit(c):
        return 1.0 - float((1.0 - top[c]).prod())

    return pd.DataFrame([{
        "pf_any_pts30": any_hit("h30"), "pf_any_ast10": any_hit("h_ast10"),
        "pf_any_reb12": any_hit("h_reb12"), "pf_any_3x5": any_hit("h_3x5"),
        "pf_max_pts": top["PTS"].max(), "pf_max_ast": top["AST"].max(),
        "pf_max_reb": top["REB"].max(), "pf_max_3": top["FG3M"].max(),
    }])


def clip(p):
    return float(min(max(p, CLIP[0]), CLIP[1]))


def predict(home, away, game_date, out_players=(), models=None):
    models = models or joblib.load("data/models.joblib")
    try:
        X = build_row(home, away, game_date)
    except Exception:
        traceback.print_exc()
        X = None  # features failed: constants only
    Xp = None
    if X is not None:
        try:
            Xp = pd.concat([X, live_pf(home, away, game_date, out_players)], axis=1)
        except Exception:
            traceback.print_exc()  # player questions fall back to constants

    out = []
    for target in QUESTIONS:
        if target is None:
            out.append(clip(Q9_RATE))
            continue
        spec = models.get(target)
        try:
            if spec is None:
                p = HARD_DEFAULT
            elif spec["type"] == "model" and (Xp if any(f.startswith("pf_") for f in spec["features"]) else X) is not None:
                Xuse = Xp if any(f.startswith("pf_") for f in spec["features"]) else X
                p = spec["model"].predict_proba(Xuse[spec["features"]])[:, 1][0]
            else:  # const spec, or model spec with failed features
                p = spec.get("p", HARD_DEFAULT)
        except Exception:
            traceback.print_exc()
            p = spec.get("p", HARD_DEFAULT) if spec else HARD_DEFAULT
        out.append(clip(p))
    return out


def submit(game_id, probs):
    """STUB: wire to the contest API once docs are available."""
    raise NotImplementedError("read API docs: auth, submit, edit, rate limits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="BOS")
    ap.add_argument("--away", default="NYK")
    ap.add_argument("--date", default="2026-11-02")
    ap.add_argument("--out", default="", help="comma-separated injured player names, e.g. \"Jayson Tatum,Jalen Brunson\"")
    a = ap.parse_args()
    probs = predict(a.home, a.away, a.date, out_players=[n for n in a.out.split(",") if n.strip()])
    for i, p in enumerate(probs, 1):
        print(f"Q{i:<2} {p * 100:5.1f}")
