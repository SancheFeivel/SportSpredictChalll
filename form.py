"""Early-season shrinkage helpers, shared by build_features.py and bot.py.
Blend rolling form toward a prior (last season's team mean, 70% team / 30% league).
Weight on current form = g / (g + K_SEASON), g = games already played this season."""
import pandas as pd

K_SEASON = 8
PRIOR_KEEP = 0.7


def season_year(d):
    """Series of dates -> NBA season start year (Aug+ = new season)."""
    return d.dt.year.where(d.dt.month >= 8, d.dt.year - 1)


def season_year_of(ts):
    return ts.year if ts.month >= 8 else ts.year - 1


def prior_frame(m, roll):
    """Returns (prior indexed by (TEAM_ID, sy), league prior indexed by sy), both for season sy using sy-1 data."""
    tm = m.groupby(["TEAM_ID", "sy"])[roll].mean()
    lg = m.groupby("sy")[roll].mean()
    lg_next = lg.copy()
    lg_next.index = lg.index + 1
    tm_next = tm.copy()
    tm_next.index = pd.MultiIndex.from_tuples([(t, s + 1) for t, s in tm.index], names=["TEAM_ID", "sy"])
    lg_for = lg_next.reindex(tm_next.index.get_level_values("sy"))
    lg_for.index = tm_next.index
    return PRIOR_KEEP * tm_next + (1 - PRIOR_KEEP) * lg_for, lg_next


def attach_prior(m, roll):
    """Training: per-row prior aligned to m.index (team prior, else league prior, else NaN)."""
    prior, lg_next = prior_frame(m, roll)
    p = m[["TEAM_ID", "sy"]].join(prior, on=["TEAM_ID", "sy"])
    l = m[["sy"]].join(lg_next, on="sy")
    return p[roll].fillna(l[roll])


def prior_for(pri, team_id, sy, roll):
    """Live: prior Series for one team-season, or None if no previous season exists."""
    prior, lg_next = pri
    if (team_id, sy) in prior.index:
        return prior.loc[(team_id, sy)][roll]
    if sy in lg_next.index:
        return lg_next.loc[sy][roll]
    return None
