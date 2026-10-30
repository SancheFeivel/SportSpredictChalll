"""Unattended scheduler for the live bot. Leave running (Task Scheduler starts it at logon).

What it does, per US Eastern game date:
  1. FULL_AT (ET): one full `live.py` (refresh + injuries + run_today)
  2. For each game tip-off: `live.py --quick` at 3h, 1h, 15min before (runs within MERGE_MIN merge)
  3. If out.txt was hand-edited since injuries.py wrote it, adds --no-injuries (keeps your edits)
  4. If live.py prints "failed:", sends a Discord alert (DISCORD_WEBHOOK in .env)

Needs: pip install requests nba_api tzdata
Usage:  python auto.py            # run forever
        python auto.py --once     # print today's plan and exit (no runs)
Stop:   Ctrl+C, or end the task in Task Scheduler.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from nba_api.stats.endpoints import scoreboardv2

ET = ZoneInfo("America/New_York")
MNL = ZoneInfo("Asia/Manila")

FULL_AT = "09:00"                 # ET. Previous night's games are final by then.
OFFSETS_MIN = [180, 60, 15]       # minutes before tip-off
MERGE_MIN = 10                    # runs closer than this become one
FALLBACK_ET = ["12:00", "16:00", "18:30"]  # used if tip times can't be read
STATE = "data/auto_state.json"
LOCK = "data/live.lock"
LOCK_STALE_S = 20 * 60
LOG = "data/auto.log"


# ---------- helpers ----------
def log(msg):
    line = f"[{datetime.now(MNL):%Y-%m-%d %H:%M:%S} MNL] {msg}"
    print(line, flush=True)
    os.makedirs("data", exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def env(key):
    if os.environ.get(key):
        return os.environ[key]
    if os.path.exists(".env"):
        for l in open(".env", encoding="utf-8"):
            if l.strip().startswith(key + "="):
                return l.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def alert(msg):
    log("ALERT: " + msg)
    url = env("DISCORD_WEBHOOK")
    if not url:
        return
    try:
        requests.post(url, json={"content": f"forecast bot: {msg}"[:1900]}, timeout=15)
    except Exception as e:
        log(f"alert send failed: {e}")


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {}


def save_state(s):
    os.makedirs("data", exist_ok=True)
    json.dump(s, open(STATE, "w", encoding="utf-8"))


def et_now():
    return datetime.now(ET)


def at_et(date_str, hhmm):
    h, m = map(int, hhmm.split(":"))
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET)


def game_date_now():
    """US Eastern date. Before FULL_AT we still treat it as today's slate."""
    return et_now().strftime("%Y-%m-%d")


# ---------- tip-off times ----------
def tipoffs(date_str):
    """Return sorted list of tz-aware tip times (ET) for games not yet started."""
    d = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m/%d/%Y")
    gh = scoreboardv2.ScoreboardV2(game_date=d, timeout=60).get_data_frames()[0]
    out = []
    for _, r in gh.iterrows():
        txt = str(r.get("GAME_STATUS_TEXT", ""))  # e.g. "7:30 pm ET"
        m = re.match(r"\s*(\d{1,2}):(\d{2})\s*(am|pm)", txt, re.I)
        if not m:
            continue  # live / final / unparseable
        hh, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        hh = hh % 12 + (12 if ap == "pm" else 0)
        out.append(at_et(date_str, f"{hh:02d}:{mm:02d}"))
    return sorted(out), len(gh)


def plan_for(date_str):
    """Run times (tz-aware) for the quick runs."""
    try:
        tips, n_games = tipoffs(date_str)
    except Exception as e:
        alert(f"scoreboard failed ({e}); using fallback times")
        tips, n_games = [], 0
    if tips:
        times = sorted({t - timedelta(minutes=o) for t in tips for o in OFFSETS_MIN})
    else:
        times = [at_et(date_str, x) for x in FALLBACK_ET]
    merged = []
    for t in times:
        if not merged or (t - merged[-1]).total_seconds() > MERGE_MIN * 60:
            merged.append(t)
    return merged, tips, n_games


# ---------- running live.py ----------
def out_hand_edited():
    """True if out.txt changed >2 min after the time injuries.py stamped in its header."""
    try:
        first = open("out.txt", encoding="utf-8").readline()
        m = re.search(r"generated (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", first)
        if not m:
            return True  # no stamp: someone wrote it by hand
        stamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
        mtime = datetime.fromtimestamp(os.path.getmtime("out.txt"))
        return (mtime - stamp).total_seconds() > 120
    except FileNotFoundError:
        return False


def acquire_lock():
    os.makedirs("data", exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - os.path.getmtime(LOCK) > LOCK_STALE_S:
                os.remove(LOCK)  # stale lock from a crash
                continue
            return False
    return False


def run_live(date_str, quick):
    if not acquire_lock():
        log("lock held by another run, skipping this slot")
        return
    try:
        cmd = [sys.executable, "live.py", "--date", date_str]
        if quick:
            cmd.append("--quick")
        if out_hand_edited():
            cmd.append("--no-injuries")
            log("out.txt hand-edited: keeping it (--no-injuries)")
        log("run: " + " ".join(cmd[1:]))
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15 * 60)
        tail = "\n".join(p.stdout.strip().splitlines()[-25:])
        log("output tail:\n" + tail)
        if p.returncode != 0:
            alert(f"live.py exit {p.returncode}: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else '?'}")
        m = re.search(r"live done \(failed: (.*?)\)", p.stdout)
        if m:
            alert(f"step failed: {m.group(1)} (forecasts used old data/out.txt)")
    except subprocess.TimeoutExpired:
        alert("live.py timed out after 15 min")
    except Exception as e:
        alert(f"auto crashed on run: {e}")
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


# ---------- main loop ----------
def day_loop(date_str, state):
    """Handle one game date. Returns when all its runs are done."""
    full_t = at_et(date_str, FULL_AT)
    quick_times, tips, n_games = plan_for(date_str)
    log(f"{date_str}: {n_games} games, {len(tips)} tip times, {len(quick_times)} quick runs")
    for t in quick_times:
        log(f"  quick @ {t:%H:%M} ET = {t.astimezone(MNL):%m-%d %H:%M} MNL")

    # full run
    if state.get("full_date") != date_str:
        now = et_now()
        if now < full_t:
            time.sleep(max(0, (full_t - now).total_seconds()))
        run_live(date_str, quick=False)
        state["full_date"] = date_str
        save_state(state)

    # quick runs, only future ones; skip if less than 1 min away is fine
    for t in quick_times:
        now = et_now()
        if t <= now:
            continue
        time.sleep(max(0, (t - now).total_seconds()))
        run_live(date_str, quick=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print plan for today and exit")
    a = ap.parse_args()

    if a.once:
        d = game_date_now()
        quick_times, tips, n = plan_for(d)
        print(f"{d}: {n} games, {len(tips)} tip times")
        for t in quick_times:
            print(f"  quick @ {t:%H:%M} ET = {t.astimezone(MNL):%m-%d %H:%M} MNL")
        return

    log("auto.py started")
    state = load_state()
    while True:
        d = game_date_now()
        try:
            day_loop(d, state)
        except Exception as e:
            alert(f"day loop error: {e}")
            time.sleep(300)
            continue
        # wait for next ET date (+5 min)
        nxt = at_et(d, "00:05") + timedelta(days=1)
        wait = max(60, (nxt - et_now()).total_seconds())
        log(f"{d} done. sleeping {wait / 3600:.1f}h")
        time.sleep(wait)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
