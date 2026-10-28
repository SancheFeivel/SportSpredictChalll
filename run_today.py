"""Predict every game on a date, print a table, append to data/log.csv.

Usage:
  python run_today.py                      # today's NBA date (US Eastern)
  python run_today.py --date 2026-11-03
Injuries: put one player name per line in out.txt (lines starting with # ignored).
Submit step is a stub in bot.py until the contest API docs are known.
"""
import argparse
import csv
import os
import traceback
from datetime import datetime

from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as nba_teams

import bot

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


def read_out():
    if not os.path.exists(OUT_FILE):
        return []
    with open(OUT_FILE, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def games_on(date):
    d = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
    gh = scoreboardv2.ScoreboardV2(game_date=d, timeout=60).get_data_frames()[0]
    ids = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}
    return [
        (r["GAME_ID"], ids[r["HOME_TEAM_ID"]], ids[r["VISITOR_TEAM_ID"]])
        for _, r in gh.iterrows()
    ]


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
    a = ap.parse_args()

    out_players = read_out()
    games = games_on(a.date)
    print(f"{a.date}: {len(games)} games, {len(out_players)} players marked out\n")

    for gid, home, away in games:
        err = ""
        try:
            probs = bot.predict(home, away, a.date, out_players=out_players)
        except Exception:
            err = traceback.format_exc().strip().splitlines()[-1]
            probs = [0.5] * 10  # last resort, never submit nothing
        print(f"{away} @ {home}  " + " ".join(f"{p * 100:4.0f}" for p in probs) + (f"  ERR {err}" if err else ""))
        append_log([manila_now(), gid, a.date, home, away, "|".join(out_players)] + [round(p, 4) for p in probs] + [err])
        # bot.submit(gid, probs)  # enable once API docs known
