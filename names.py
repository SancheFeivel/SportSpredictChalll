"""Shared player-name normalizer so ESPN names match nba_api PLAYER_NAME.
Strips accents, punctuation and Jr./Sr./II/III so 'Jaren Jackson Jr.' == 'Jaren Jackson'.
"""
import glob
import re
import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# normalized ESPN name -> normalized nba_api name. Fill from the NO MATCH report, e.g.
# ALIASES = {"some espn name": "some nba api name"}
ALIASES = {}


def norm(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    parts = s.split()
    if len(parts) > 2 and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    s = " ".join(parts)
    return ALIASES.get(s, s)


def known_names(pattern="data/P_*.csv"):
    import pandas as pd
    out = set()
    for f in glob.glob(pattern):
        col = pd.read_csv(f, usecols=["PLAYER_NAME"])["PLAYER_NAME"].dropna().unique()
        out |= {norm(n) for n in col}
    return out


def unmatched(names):
    """Names that match no PLAYER_NAME in data/P_*.csv (rookies with no games yet will show up too)."""
    known = known_names()
    return [n for n in names if norm(n) not in known]
