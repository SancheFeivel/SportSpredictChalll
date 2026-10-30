"""Scrape ESPN's NBA injury page (data by Rotowire) and write out.txt for run_today.py.

Usage:
  python injuries.py                    # for today's NBA date (US Eastern)
  python injuries.py --date 2026-10-30

Needs: pip install requests beautifulsoup4 lxml

Rules (the Status column is unreliable, so we read dates and comments):
  OUT        estimated return date is AFTER the game date, or a fresh comment says doubtful
  QUESTIONABLE  fresh comment (within 1 day of game date) says questionable -> written as "# Q:" comment
  else       treated as available
Never overwrites out.txt if the scrape fails or looks empty: the old file stays.
The header line records the game date; run_today.py ignores the file if that date is wrong.
After writing, lists names that match no PLAYER_NAME in data/P_*.csv (silent mismatches).
"""
import argparse
import os
import re
import sys
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

import names

URL = "https://www.espn.ph/nba/injuries"
OUT_FILE = "out.txt"
FRESH_DAYS = 1


def nba_today():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return date.today()


def parse_dm(text, on_or_after):
    """'28 Oct' -> first date with that day/month on or after `on_or_after`."""
    try:
        d = datetime.strptime(f"{text.strip()} {on_or_after.year}", "%d %b %Y").date()
    except ValueError:
        return None
    if d < on_or_after:
        d = d.replace(year=d.year + 1)
    return d


def classify(ret_txt, comment, game_date):
    """Return 'out', 'questionable' or 'ok'."""
    m = re.match(r"\s*(\d{1,2} [A-Za-z]{3}):\s*(.*)", comment)
    c_date, c_text = (None, comment.lower())
    if m:
        c_date = parse_dm(m.group(1), game_date - timedelta(days=330))
        c_text = m.group(2).lower()

    ret = parse_dm(ret_txt, c_date or game_date) if ret_txt else None

    fresh = c_date is not None and abs((game_date - c_date).days) <= FRESH_DAYS
    if fresh:
        if "doubtful" in c_text:
            return "out"
        if "questionable" in c_text or "game-time decision" in c_text or "day-to-day" in c_text:
            return "questionable"
        if "ruled out" in c_text or "will not play" in c_text or "won't play" in c_text:
            return "out"

    if ret and ret > game_date + timedelta(days=2):
        return "out"
    if ret and ret > game_date:
        return "out"
    return "ok"


def scrape():
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows = []
    for tbl in soup.find_all("table"):
        title = tbl.find_previous(class_="Table__Title")
        team = title.get_text(strip=True) if title else "?"
        for tr in tbl.find_all("tr"):
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) >= 5:
                rows.append((team, tds[0], tds[2], tds[3], tds[4]))  # team, name, return, status, comment
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=nba_today().isoformat())
    a = ap.parse_args()
    game_date = datetime.strptime(a.date, "%Y-%m-%d").date()

    try:
        rows = scrape()
    except Exception as e:
        sys.exit(f"scrape failed ({e}); {OUT_FILE} left unchanged")
    if len(rows) < 20:
        sys.exit(f"only {len(rows)} rows parsed, page layout may have changed; {OUT_FILE} left unchanged")

    out, quest = [], []
    for team, name, ret, status, comment in rows:
        k = classify(ret, comment, game_date)
        if k == "out":
            out.append((team, name))
        elif k == "questionable":
            quest.append((team, name))
    out = sorted(set(out))
    quest = sorted(set(quest))

    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"# generated {datetime.now():%Y-%m-%d %H:%M} for game date {game_date}\n")
        for team, name in out:
            f.write(f"{name}\n")
        for team, name in quest:
            f.write(f"# Q: {name} ({team})  <- move up to drop him\n")
    os.replace(tmp, OUT_FILE)  # atomic: never leaves a half-written file

    print(f"{len(rows)} rows scraped -> {len(out)} out, {len(quest)} questionable (commented) for {game_date}")
    print("\nOUT:", ", ".join(n for _, n in out))
    print("\nQUESTIONABLE:", ", ".join(n for _, n in quest))

    try:
        bad = names.unmatched([n for _, n in out])
        if bad:
            print(f"\nNO MATCH in data/P_*.csv ({len(bad)}): " + ", ".join(bad))
            print("Rookies with no games yet are fine. Real stars here need an entry in names.ALIASES.")
        else:
            print("\nall OUT names match player logs")
    except Exception as e:
        print(f"\n(name-match check skipped: {e})")
