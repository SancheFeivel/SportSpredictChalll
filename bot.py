"""predict(home, away, date) -> 10 probabilities in contest order (Q1..Q10).

Usage: python bot.py --home BOS --away NYK --date 2026-11-02
Fallback chain per question: sim (if validated) -> blended model -> constant rate -> hard default.
Model output is blended with the recent base rate (BLEND) to cut early-season overconfidence.
Early-season form is shrunk toward last season's prior (form.py), same as training.
Injured names are matched through names.norm (accents, Jr./III, punctuation ignored).
Never returns nothing, never returns 0 or 100.
"""
import argparse
import glob
import json
import traceback

import joblib
import pandas as pd

import form
import names

WINDOW = 10
CLIP = (0.02, 0.98)
Q9_RATE = 0.50  # TODO: replace with measured rate once play-by-play labels exist
HARD_DEFAULT = 0.50

# weight on model; rest goes to the recent base rate (spec["p"]).
# 12-game log: model beat baseline on home_win only, ran 0.2-0.3 too high on total/pts_30/team_120.
# Raise weights toward 1.0 once score_log.py shows stable edge over 100+ games.
BLEND_DEFAULT = 0.5
BLEND = {"home_win": 1.0}

# player must have played within this many days of their team's latest game to count as rotation
ROSTER_RECENCY_DAYS = 21

QUESTIONS = [
    "home_win", "total_230", "threes_25", "pts_30", "ast_10",
    "reb_12", "team_120", "margin_10", "lead_20", "threes_5",
]

ROLL = ["PTS", "opp_pts", "FG3M", "FG3M_opp", "pace", "win", "margin", "tot_pts", "tot_3"]
_CACHE = {}  # team logs + prior are loaded once per process, not once per game


def load_team_logs():
    if "tl" in _CACHE:
        return _CACHE["tl"]
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
    m["sy"] = form.season_year(m["date"])
    _CACHE["tl"] = m.sort_values("date")
    return _CACHE["tl"]


def team_state(logs, abbr, game_date, pri):
    """Form from last WINDOW games strictly before game_date, shrunk early in the season."""
    allg = logs[(logs["TEAM_ABBREVIATION"] == abbr) & (logs["date"] < game_date)]
    g = allg.tail(WINDOW)
    if len(g) < 5:
        raise ValueError(f"not enough games for {abbr}")
    s = {f"r_{c}": g[c].mean() for c in ROLL}
    sy = form.season_year_of(game_date)
    gs = int((allg["sy"] == sy).sum())
    p = form.prior_for(pri, int(allg["TEAM_ID"].iloc[-1]), sy, ROLL)
    if p is not None:
        w = gs / (gs + form.K_SEASON)
        s = {k: w * v + (1 - w) * float(p[k[2:]]) for k, v in s.items()}
    rest = min((game_date - g["date"].iloc[-1]).days, 7)
    s["rest"] = rest
    s["b2b"] = int(rest <= 1)
    return s


def build_row(home, away, game_date):
    logs = load_team_logs()
    if "pri" not in _CACHE:
        _CACHE["pri"] = form.prior_frame(logs, ROLL)
    pri = _CACHE["pri"]
    d = pd.Timestamp(game_date)
    h, a = team_state(logs, home, d, pri), team_state(logs, away, d, pri)
    row = {f"h_{k}": v for k, v in h.items()}
    row.update({f"a_{k}": v for k, v in a.items()})
    row["d_win"] = row["h_r_win"] - row["a_r_win"]
    row["d_margin"] = row["h_r_margin"] - row["a_r_margin"]
    row["d_rest"] = row["h_rest"] - row["a_rest"]
    row["sum_pace"] = row["h_r_pace"] + row["a_r_pace"]
    row["sum_pts"] = row["h_r_PTS"] + row["a_r_PTS"]
    row["sum_3"] = row["h_r_FG3M"] + row["a_r_FG3M"]
    row["d_absmargin"] = abs(row["d_margin"])
    return pd.DataFrame([row])


TOP_N = 8


def load_player_logs():
    if "pl" not in _CACHE:
        pl = pd.concat([pd.read_csv(f, dtype={"GAME_ID": str}) for f in sorted(glob.glob("data/P_*.csv"))])
        pl["date"] = pd.to_datetime(pl["GAME_DATE"])
        pl["sy"] = form.season_year(pl["date"])
        pl["name_key"] = pl["PLAYER_NAME"].map(names.norm)
        _CACHE["pl"] = pl.sort_values("date")
    return _CACHE["pl"]


def live_pf(home, away, game_date, out_players=()):
    """Same pf_* features as training. Roster = each player's latest team; out_players are dropped.
    Stale-roster guard: a player counts only if he played within ROSTER_RECENCY_DAYS of his team's
    latest game, and, once the team has played this season, only if he has played this season too.
    Caveat: traded players who have not yet played for the new team still map to the old team."""
    d = pd.Timestamp(game_date)
    sy = form.season_year_of(d)
    pl = load_player_logs()
    prior = pl[pl["date"] < d]
    last10 = prior.groupby("PLAYER_ID").tail(WINDOW)
    agg = last10.groupby("PLAYER_ID").agg(
        n=("PTS", "size"), team=("TEAM_ABBREVIATION", "last"), name=("name_key", "last"),
        last_date=("date", "max"), last_sy=("sy", "max"),
        MIN=("MIN", "mean"), PTS=("PTS", "mean"), AST=("AST", "mean"),
        REB=("REB", "mean"), FG3M=("FG3M", "mean"),
        h30=("PTS", lambda s: (s >= 30).mean()), h_ast10=("AST", lambda s: (s >= 10).mean()),
        h_reb12=("REB", lambda s: (s >= 12).mean()), h_3x5=("FG3M", lambda s: (s >= 5).mean()),
    )
    agg = agg[agg["team"].isin([home, away])]

    keep = pd.Series(True, index=agg.index)
    for team, grp in agg.groupby("team"):
        team_last = grp["last_date"].max()
        recent = grp["last_date"] >= team_last - pd.Timedelta(days=ROSTER_RECENCY_DAYS)
        if (grp["last_sy"] == sy).any():  # team already has games this season
            recent &= grp["last_sy"] == sy
        keep[grp.index] = recent
    agg = agg[keep]

    out_l = {names.norm(n) for n in out_players}
    agg = agg[(agg["n"] >= 3) & ~agg["name"].isin(out_l)]
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


def blend(target, p_model, p_base):
    w = BLEND.get(target, BLEND_DEFAULT)
    return w * p_model + (1 - w) * p_base


def sim_override(X):
    """Joint-simulator probs for questions that passed sim_check.py. {} if none/unavailable."""
    try:
        with open("data/sim_ok.json") as f:
            ok = json.load(f)
        if not ok or X is None:
            return {}
        import sim
        row = sim.probs(joblib.load("data/sim.joblib"), X).iloc[0]
        return {t: float(row[t]) for t in ok}
    except FileNotFoundError:
        return {}
    except Exception:
        traceback.print_exc()
        return {}


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

    sim_p = sim_override(X)
    out = []
    for target in QUESTIONS:
        spec = models.get(target)
        if target in sim_p:
            base = spec.get("p", HARD_DEFAULT) if spec else HARD_DEFAULT
            out.append(clip(blend(target, sim_p[target], base)))
            continue
        try:
            if spec is None:
                p = Q9_RATE if target == "lead_20" else HARD_DEFAULT
            elif spec["type"] == "model":
                Xuse = Xp if any(f.startswith("pf_") for f in spec["features"]) else X
                if Xuse is None:
                    p = spec.get("p", HARD_DEFAULT)
                else:
                    p_model = spec["model"].predict_proba(Xuse[spec["features"]])[:, 1][0]
                    p = blend(target, p_model, spec.get("p", HARD_DEFAULT))
            else:
                p = spec.get("p", HARD_DEFAULT)
        except Exception:
            traceback.print_exc()
            p = spec.get("p", HARD_DEFAULT) if spec else HARD_DEFAULT
        out.append(clip(p))
    # consistency: P(either team >= 120) must exceed P(total >= 230) by a margin
    i_t120, i_tot = QUESTIONS.index("team_120"), QUESTIONS.index("total_230")
    out[i_t120] = clip(max(out[i_t120], out[i_tot] + 0.05))
    return out


def fallback_probs(models=None):
    """Per-question base rates from models.joblib. Used when predict() blows up. Better than flat 0.5."""
    try:
        models = models or joblib.load("data/models.joblib")
    except Exception:
        return [HARD_DEFAULT] * len(QUESTIONS)
    out = []
    for t in QUESTIONS:
        spec = models.get(t)
        p = spec.get("p", HARD_DEFAULT) if spec else (Q9_RATE if t == "lead_20" else HARD_DEFAULT)
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
