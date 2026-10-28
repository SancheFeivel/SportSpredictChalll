"""Q9 labels: biggest lead either team held in each game, from play-by-play.
Resumable: appends to data/q9.csv, skips games already done. Run on home PC, takes ~1h per 2 seasons.
Usage: python pull_pbp.py                       # 2024-25 and 2025-26
       python pull_pbp.py --seasons 2023-24 2024-25 2025-26
"""
import argparse
import csv
import os
import time

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

OUT = "data/q9.csv"
ap = argparse.ArgumentParser()
ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
ap.add_argument("--sleep", type=float, default=0.8)
a = ap.parse_args()


def max_lead(pbp):
    h = pd.to_numeric(pbp["scoreHome"], errors="coerce").ffill().fillna(0)
    v = pd.to_numeric(pbp["scoreAway"], errors="coerce").ffill().fillna(0)
    return int((h - v).abs().max())


if __name__ == "__main__":
    labels = pd.read_csv("data/labels.csv", dtype={"GAME_ID": str}, index_col=0)
    ids = labels[labels["season"].isin(a.seasons)].index.tolist()
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT, dtype={"GAME_ID": str})["GAME_ID"])
    todo = [g for g in ids if g not in done]
    print(f"{len(ids)} games, {len(done)} done, {len(todo)} to pull")

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["GAME_ID", "max_lead", "lead_20"])
        failed, streak = [], 0
        for i, gid in enumerate(todo, 1):
            ok = False
            for attempt in range(5):
                try:
                    pbp = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=120).get_data_frames()[0]
                    ml = max_lead(pbp)
                    w.writerow([gid, ml, int(ml >= 20)])
                    f.flush()
                    ok = True
                    break
                except Exception as e:
                    print(f"  {gid} try {attempt + 1}: {type(e).__name__}", flush=True)
                    time.sleep(10 * (attempt + 1))
            if ok:
                streak = 0
            else:
                failed.append(gid)
                streak += 1
                if streak >= 3:  # likely throttled: cool down
                    print("  3 fails in a row, cooling down 120s", flush=True)
                    time.sleep(120)
                    streak = 0
            if i % 50 == 0:
                print(f"{i}/{len(todo)}  failed so far {len(failed)}", flush=True)
            time.sleep(a.sleep)
    print(f"\nfailed games: {len(failed)} (rerun the script to retry them)")

    q = pd.read_csv(OUT)
    print(f"\nq9 rows {len(q)}  lead_20 rate {q['lead_20'].mean():.3f}")