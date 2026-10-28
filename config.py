"""Shared knobs for build (training) and live (bot). Keep both sides reading from here."""

WINDOW = 10           # games in rolling form
MIN_GAMES = 5         # min games for form when no prior exists
SHRINK_K = 15         # games of belief in last season's prior
PRIOR_REGRESS = 0.4   # pull last season's team mean toward league mean
RATE_K = 300          # games of belief in last 2 seasons' base rate


def season_of(d):
    """Season string like '2026-27' for a date (season starts ~Oct)."""
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def prev_season(s):
    y = int(s[:4]) - 1
    return f"{y}-{str(y + 1)[2:]}"
