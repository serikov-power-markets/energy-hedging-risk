"""Retail customer book of a German electricity supplier: expected and weather-driven volumes.

Expected volume: BDEW standard load profiles H0 (households, dynamised) and G0 (commerce) from
demandlib, scaled to the annual volumes of the book. Actual volume: the expected profile scaled each
day by a weather factor exp(b_hdd * HDD anomaly + b_cdd * CDD anomaly), with the sensitivities
estimated from German grid load (SMARD) against the population-weighted temperature of ten cities.
Grid load is a proxy: it includes industry and electric heating, so the sensitivity of a
household book can differ; the factor is applied with the national estimate.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from demandlib import bdew

from . import data

HDD_BASE = 15.0   # deg C, heating degree days below
CDD_BASE = 20.0   # deg C, cooling degree days above


def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def holidays_de(year: int) -> list[date]:
    """Nationwide German public holidays."""
    e = _easter(year)
    fixed = [date(year, 1, 1), date(year, 5, 1), date(year, 10, 3), date(year, 12, 25), date(year, 12, 26)]
    moving = [e - timedelta(days=2), e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50)]
    return sorted(fixed + moving)


def expected_profile(year: int, h0_mwh: float, g0_mwh: float) -> pd.Series:
    """Hourly expected volume (MWh per hour, UTC index) of the book for one calendar year."""
    slp = bdew.ElecSlp(year, holidays=holidays_de(year))
    q = slp.get_scaled_profiles({"h0_dyn": h0_mwh, "g0": g0_mwh}).sum(axis=1)
    # demandlib's index is local wall-clock time without DST shifts; read it as German standard
    # time (UTC+1), which keeps every quarter-hour exactly once
    q.index = q.index.tz_localize("Etc/GMT-1").tz_convert("UTC")
    return q.resample("h").sum().rename("expected")


def degree_days(temp: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"hdd": (HDD_BASE - temp).clip(lower=0), "cdd": (temp - CDD_BASE).clip(lower=0)})


def climatology(temp: pd.Series, years: tuple[int, int] = (1991, 2020)) -> pd.DataFrame:
    """Expected HDD and CDD by day of year (normal period, 15-day smoothing)."""
    dd = degree_days(temp.loc[str(years[0]):str(years[1])])
    doy = dd.index.dayofyear.where(dd.index.dayofyear < 366, 365)
    clim = dd.groupby(doy).mean()
    return clim.rolling(15, center=True, min_periods=1).mean()


def anomalies(temp: pd.Series, clim: pd.DataFrame) -> pd.DataFrame:
    dd = degree_days(temp)
    doy = dd.index.dayofyear.where(dd.index.dayofyear < 366, 365)
    return dd - clim.reindex(doy).to_numpy()


def weather_sensitivity(start: str = "2019-01-01", end: str = "2025-12-31") -> dict:
    """OLS of log daily grid load on HDD and CDD anomalies with year-month and weekday effects.

    Year-month fixed effects absorb seasonality, Covid in 2020 and the demand reduction of 2022, so
    the coefficients measure the response to weather surprises within a month.
    """
    load = data.smard_series("load", start, end).tz_convert("Europe/Berlin")
    daily = load.groupby(load.index.date).sum()
    daily.index = pd.to_datetime(daily.index)
    temp = data.temperature_daily()
    an = anomalies(temp, climatology(temp)).reindex(daily.index)
    hol = {d for y in range(daily.index.year.min(), daily.index.year.max() + 1) for d in holidays_de(y)}
    df = pd.DataFrame({"y": np.log(daily), "hdd": an["hdd"], "cdd": an["cdd"]})
    df = df[[d.date() not in hol for d in df.index]]
    fe = pd.get_dummies(df.index.to_period("M").astype(str), prefix="ym", drop_first=True, dtype=float)
    wd = pd.get_dummies(df.index.dayofweek, prefix="wd", drop_first=True, dtype=float)
    X = pd.concat([df[["hdd", "cdd"]].reset_index(drop=True), fe, wd], axis=1)
    X = sm.add_constant(X)
    res = sm.OLS(df["y"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return {"b_hdd": float(res.params["hdd"]), "b_cdd": float(res.params["cdd"]),
            "t_hdd": float(res.tvalues["hdd"]), "t_cdd": float(res.tvalues["cdd"]),
            "r2": float(res.rsquared), "n": int(res.nobs)}


def actual_profile(expected: pd.Series, temp: pd.Series, clim: pd.DataFrame, sens: dict) -> pd.Series:
    """Expected profile scaled each local day by the weather factor of that day's temperature."""
    an = anomalies(temp, clim)
    factor = np.exp(sens["b_hdd"] * an["hdd"] + sens["b_cdd"] * an["cdd"])
    loc_day = expected.index.tz_convert("Europe/Berlin").normalize().tz_localize(None)
    f = factor.reindex(loc_day).fillna(1.0).to_numpy()
    return (expected * f).rename("actual")
