"""Procurement (hedging) strategies of a fixed-price retail book and their settlement.

A strategy is a list of trades: on a trading date, buy (or sell) a constant MW in the base or peak
hours of a delivery period at that date's modelled forward price. The hourly hedge position is the
sum of all trades covering the hour. The open position, actual load minus hedge, is settled at the
day-ahead price. Peak is the EEX definition (Monday to Friday 08:00-20:00).

Base/peak split of an hourly profile over a period (standard block hedge): base MW = mean load in
off-peak hours, peak MW = mean load in peak hours minus base MW.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import CurveModel
from .data import is_peak


@dataclass(frozen=True)
class Trade:
    date: pd.Timestamp      # trading date (key into the curve cache)
    start: pd.Timestamp     # first delivery day (local)
    end: pd.Timestamp       # last delivery day (local, inclusive)
    kind: str               # "base" or "peak"
    mw: float               # positive = buy


def period_hours(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    h = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), freq="h",
                      tz="Europe/Berlin", inclusive="left")
    return h.tz_convert("UTC")


def block_split(profile: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    """(base MW, peak MW) that replicate the profile's off-peak and peak means over the period."""
    hrs = period_hours(start, end)
    p = profile.reindex(hrs)
    pk = is_peak(hrs)
    base = float(p[~pk].mean())
    peak = float(p[pk].mean()) - base
    return base, max(peak, 0.0)


def first_bday(month_start: pd.Timestamp) -> pd.Timestamp:
    return pd.bdate_range(month_start, periods=1)[0]


def month_starts(year: int) -> list[pd.Timestamp]:
    return list(pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS"))


# ---------------------------------------------------------------- strategies for delivery year Y
def strategy(name: str, year: int, expected: pd.Series) -> list[Trade]:
    """Trades for delivery year `year` under a named strategy, sized on the expected profile."""
    y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    base, peak = block_split(expected, y0, y1)
    prev = month_starts(year - 1)
    trades: list[Trade] = []
    if name == "spot":
        return trades
    if name == "cal_once":                     # one Cal base + peak purchase at the sales date
        d = first_bday(pd.Timestamp(f"{year - 1}-10-01"))
        return [Trade(d, y0, y1, "base", base), Trade(d, y0, y1, "peak", peak)]
    if name in ("ladder", "ladder_base_only", "ladder_shaped"):
        share = 0.8 if name == "ladder_shaped" else 1.0
        for m in prev:                         # 12 equal tranches through the year before delivery
            d = first_bday(m)
            if name == "ladder_base_only":
                # same energy as the profile but all as base: average MW over the year
                avg = float(expected.reindex(period_hours(y0, y1)).mean())
                trades.append(Trade(d, y0, y1, "base", avg / 12))
            else:
                trades.append(Trade(d, y0, y1, "base", share * base / 12))
                trades.append(Trade(d, y0, y1, "peak", share * peak / 12))
        if name == "ladder_shaped":            # month products bought one month ahead: top up to 100%
            for m in month_starts(year):
                d = first_bday(m - pd.offsets.MonthBegin(1))
                me = m + pd.offsets.MonthEnd(0)
                mb, mp = block_split(expected, m, me)
                trades.append(Trade(d, m, me, "base", mb - 0.8 * base))
                trades.append(Trade(d, m, me, "peak", mp - 0.8 * peak))
        return trades
    raise ValueError(name)


STRATEGIES = ["spot", "cal_once", "ladder", "ladder_base_only", "ladder_shaped"]


# ---------------------------------------------------------------- settlement
def hedge_position(trades: list[Trade], hours: pd.DatetimeIndex) -> pd.Series:
    pos = pd.Series(0.0, index=hours)
    pk = pd.Series(is_peak(hours), index=hours)
    for t in trades:
        h = period_hours(t.start, t.end)
        h = h[(h >= hours[0]) & (h <= hours[-1])]
        mask = h if t.kind == "base" else h[pk.reindex(h).to_numpy()]
        pos.loc[mask] += t.mw
    return pos


def trade_prices(trades: list[Trade], curves: dict[pd.Timestamp, CurveModel]) -> np.ndarray:
    """Modelled forward price of each trade on its trading date (memoised per product)."""
    memo: dict[tuple, float] = {}
    out = []
    for t in trades:
        key = (t.date, t.start, t.end, t.kind)
        if key not in memo:
            memo[key] = curves[t.date].product(t.start, t.end, t.kind)
        out.append(memo[key])
    return np.array(out)


def settle(trades: list[Trade], prices_fwd: np.ndarray, actual: pd.Series, spot: pd.Series,
           sales_price: float) -> dict:
    """Cost and P&L of one delivery year.

    cost = sum over trades (MW x hours x forward) + sum over hours (actual - hedge) x spot
    P&L  = sales_price x actual volume - cost
    """
    hours = actual.index
    s = spot.reindex(hours)
    if s.isna().any():
        raise ValueError("spot prices missing in delivery period")
    pos = hedge_position(trades, hours)
    pk = is_peak(hours)
    hedge_cost = 0.0
    hedge_mtm = 0.0                                # forward vs realised spot on the hedged hours
    for t, f in zip(trades, prices_fwd):
        h = period_hours(t.start, t.end)
        if t.kind == "peak":
            h = h[is_peak(h)]
        n = len(h)
        hedge_cost += t.mw * n * f
        hedge_mtm += t.mw * (float(s.reindex(h).sum()) - n * f)
    open_cost = float(((actual - pos) * s).sum())
    vol = float(actual.sum())
    cost = hedge_cost + open_cost
    return {
        "volume_mwh": vol,
        "cost_eur_mwh": cost / vol,
        "spot_cost_eur_mwh": float((actual * s).sum()) / vol,   # load-weighted spot (no hedge)
        "base_spot_eur_mwh": float(s.mean()),
        "hedge_gain_eur_m": hedge_mtm / 1e6,
        "open_volume_share": float((actual - pos).abs().sum()) / vol,
        "peak_hedge_share": float(pos[pk].sum()) / max(float(actual[pk].sum()), 1e-9),
        "sales_price": sales_price,
        "pnl_eur_m": (sales_price * vol - cost) / 1e6,
        "pnl_eur_mwh": sales_price - cost / vol,
    }


def sales_price(curve: CurveModel, expected: pd.Series, year: int, margin: float = 0.0) -> float:
    """Fixed retail energy price set on the curve's date: load-weighted HPFC plus margin."""
    hpfc = curve.hpfc(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
    e = expected.reindex(hpfc.index).fillna(0.0)
    return float((e * hpfc).sum() / e.sum()) + margin
