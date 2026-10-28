"""Pull team (T) and player (P) game logs. Skips files already on disk."""
import os
import time

from nba_api.stats.endpoints import leaguegamelog

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
CURRENT = "2026-27"  # always re-pulled; older seasons are cached
os.makedirs("data", exist_ok=True)

for season in SEASONS:
    for kind in ["T", "P"]:
        path = f"data/{kind}_{season}.csv"
        if os.path.exists(path) and season != CURRENT:
            print("skip", path)
            continue
        df = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation=kind,
            timeout=60,
        ).get_data_frames()[0]
        df.to_csv(path, index=False)
        print("saved", path, df.shape)
        time.sleep(2)
