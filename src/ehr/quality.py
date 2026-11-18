"""Input data checks and anomaly detection for the market and fuel series.

Checks per series: hours per local day (23, 24 or 25 with daylight saving), duplicated or missing
timestamps, values outside physical or market bounds, flat runs (stuck feed), and point anomalies
by a robust z-score against a rolling median and median absolute deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOUNDS = {"price": (-500.0, 4000.0), "load": (20_000.0, 100_000.0), "wind_onshore": (0.0, 70_000.0),
          "wind_offshore": (0.0, 15_000.0), "solar": (0.0, 90_000.0)}


def hours_per_day(s: pd.Series) -> pd.Series:
    loc = s.index.tz_convert("Europe/Berlin")
    return s.groupby(loc.date).size()


def expected_hours(days) -> pd.Series:
    out = {}
    for d in days:
        t0 = pd.Timestamp(d).tz_localize("Europe/Berlin")
        t1 = (pd.Timestamp(d) + pd.Timedelta(1, unit="D")).tz_localize("Europe/Berlin")
        out[d] = int((t1 - t0) / pd.Timedelta(1, unit="h"))
    return pd.Series(out)


def flat_runs(s: pd.Series, min_len: int = 6) -> int:
    """Number of runs of at least min_len identical consecutive values above 1 in magnitude
    (SMARD fills some night hours of solar with a constant 1 MWh; those are not a stuck feed)."""
    v = s.to_numpy()
    change = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(change)
    lengths = np.bincount(run_id)
    first = np.flatnonzero(change)
    vals = v[first]
    return int(((lengths[1:] >= min_len) & (np.abs(vals) > 1)).sum())


def robust_anomalies(s: pd.Series, days: int = 14, z: float = 8.0) -> pd.DataFrame:
    """Points more than z robust standard deviations from the rolling median of the same local
    hour over the surrounding days (comparing like with like keeps the daily cycle of load and
    solar out of the score). Hours whose median absolute deviation is zero (night-time solar)
    are not scored."""
    loc_hour = s.index.tz_convert("Europe/Berlin").hour
    parts = []
    for _, g in s.groupby(loc_hour):
        med = g.rolling(days, center=True, min_periods=days // 2).median()
        mad = (g - med).abs().rolling(days, center=True, min_periods=days // 2).median()
        parts.append(pd.DataFrame({"value": g, "median": med,
                                   "score": (g - med) / (1.4826 * mad.replace(0, np.nan))}))
    df = pd.concat(parts).sort_index()
    return df[df["score"].abs() > z]


def check_series(name: str, s: pd.Series) -> dict:
    hpd = hours_per_day(s)
    exp = expected_hours(hpd.index)
    lo, hi = BOUNDS.get(name, (-np.inf, np.inf))
    an = robust_anomalies(s)
    return {
        "series": name, "start": s.index.min(), "end": s.index.max(), "n": len(s),
        "duplicates": int(s.index.duplicated().sum()),
        "missing_values": int(s.isna().sum()),
        "days_wrong_hour_count": int((hpd != exp).sum()),
        "out_of_bounds": int(((s < lo) | (s > hi)).sum()),
        "flat_runs_6h": flat_runs(s),
        "robust_anomalies": len(an),
    }, an


def check_daily(name: str, s: pd.Series, max_gap_days: int = 5) -> dict:
    gaps = s.index.to_series().diff().dt.days
    r = np.log(s).diff()
    return {"series": name, "start": s.index.min(), "end": s.index.max(), "n": len(s),
            "duplicates": int(s.index.duplicated().sum()), "missing_values": int(s.isna().sum()),
            "gaps_over_days": int((gaps > max_gap_days).sum()),
            "out_of_bounds": int((s <= 0).sum()),
            "robust_anomalies": int((r.abs() > 8 * 1.4826 * (r - r.median()).abs().median()).sum())}
