"""Predict every game on a date, print a table, append to data/log.csv.

Usage:
  python run_today.py                      # today's NBA date (US Eastern)
  python run_today.py --date 2026-11-03
  python run_today.py --include-started    # also forecast games already live/final
Injuries: out.txt, written by injuries.py. Ignored if its header date != the game date
(override with --use-stale-out). Submit step is a stub in bot.py until the contest API docs are known.
Tip: late evening in Manila can already be the NEXT NBA date, so pass --date when in doubt.
"""
import argparse
import csv
import os
import re
import sys
import traceback
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, scoreboardv3
from nba_api.stats.static import teams as nba_teams

import bot
import names

LOG = "data/log.csv"
OUT_FILE = "out.txt"
HEADER = ["logged_at", "game_id", "game_date", "home", "away", "out_players"] + [f"q{i}" for i in range(1, 11)] + ["error"]


def nba_today():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def manila_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Manila")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_out(game_date, allow_stale=False):
    if not os.path.exists(OUT_FILE):
        return []
    with open(OUT_FILE, encoding="utf-8") as f:
        lines = f.read().splitlines()
    m = re.search(r"for game date (\d{4}-\d{2}-\d{2})", lines[0]) if lines else None
    if m and m.group(1) != game_date and not allow_stale:
        print(f"WARNING: out.txt is for {m.group(1)}, not {game_date}: IGNORING it. "
              f"Run: python injuries.py --date {game_date}  (or pass --use-stale-out)\n")
        return []
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def _games_v3(date):
    ds = scoreboardv3.ScoreboardV3(game_date=date, timeout=60).get_data_frames()
    games_df, teams_df = ds[1], ds[2]
    games = []
    for _, g in games_df.iterrows():
        code = str(g["gameCode"]).split("/")[-1]  # "ATLMIA" = away ATL @ home MIA
        away, home = code[:3], code[3:]
        tri = set(teams_df.loc[teams_df["gameId"] == g["gameId"], "teamTricode"])
        if tri != {away, home}:
            raise ValueError(f"gameCode {code} does not match teams {tri}")
        games.append((g["gameId"], home, away, int(g["gameStatus"])))  # 1 scheduled, 2 live, 3 final
    return games


def _games_v2(date):
    d = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
    gh = scoreboardv2.ScoreboardV2(game_date=d, timeout=60).get_data_frames()[0]
    ids = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}
    return [
        (r["GAME_ID"], ids[r["HOME_TEAM_ID"]], ids[r["VISITOR_TEAM_ID"]], int(r.get("GAME_STATUS_ID", 1)))
        for _, r in gh.iterrows()
    ]


def games_on(date):
    """[(game_id, home, away, status)]. Tries V3, falls back to V2."""
    try:
        games = _games_v3(date)
        if games:
            return games
        print("V3 returned no games, trying V2")
    except Exception as e:
        print("V3 failed, trying V2:", e)
    return _games_v2(date)


def verify_home_away(games):
    """A swapped home/away flips every Q1 forecast. Check against finished games in labels.csv."""
    if not os.path.exists("data/labels.csv"):
        print("home/away check skipped (no labels.csv)")
        return
    lab = pd.read_csv("data/labels.csv", dtype={"GAME_ID": str}, index_col=0)
    ok = bad = 0
    for gid, home, away, _ in games:
        if gid in lab.index:
            if lab.at[gid, "home_team"] == home and lab.at[gid, "away_team"] == away:
                ok += 1
            else:
                bad += 1
    if bad:
        sys.exit(f"ABORT: home/away disagrees with labels.csv for {bad} game(s). Check games_on().")
    if ok:
        print(f"home/away verified against {ok} finished game(s)")
    else:
        print("home/away NOT verifiable (no finished games). First live day: spot-check one game against the NBA schedule.")


def append_log(row):
    os.makedirs("data", exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=nba_today())
    ap.add_argument("--include-started", action="store_true", help="also forecast live/final games")
    ap.add_argument("--use-stale-out", action="store_true", help="use out.txt even if its date is wrong")
    a = ap.parse_args()

    out_players = read_out(a.date, a.use_stale_out)
    try:
        games = games_on(a.date)
    except Exception as e:
        sys.exit(f"could not fetch games for {a.date}: {e}")
    print(f"{a.date}: {len(games)} games, {len(out_players)} players marked out")
    if not games:
        sys.exit("no games found; nothing forecast")

    verify_home_away(games)
    try:
        bad = names.unmatched(out_players)
        if bad:
            print(f"out.txt names with no match in player logs ({len(bad)}): {', '.join(bad[:15])}"
                  + (" ..." if len(bad) > 15 else ""))
    except Exception as e:
        print(f"(name-match check skipped: {e})")
    print()

    for gid, home, away, status in games:
        if status != 1 and not a.include_started:
            print(f"{away} @ {home}  skipped (already started or final)")
            continue
        err = ""
        try:
            probs = bot.predict(home, away, a.date, out_players=out_players)
        except Exception:
            err = traceback.format_exc().strip().splitlines()[-1]
            probs = bot.fallback_probs()  # base rates, never submit nothing
        print(f"{away} @ {home}  " + " ".join(f"{p * 100:4.0f}" for p in probs) + (f"  ERR {err}" if err else ""))
        append_log([manila_now(), gid, a.date, home, away, "|".join(out_players)] + [round(p, 4) for p in probs] + [err])
        # bot.submit(gid, probs)  # enable once API docs known
