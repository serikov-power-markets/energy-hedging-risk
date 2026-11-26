"""Balancing group (Bilanzkreis) of a supplier with metered industrial customers: what the day-ahead load
forecast error costs at the German imbalance price reBAP, and what an intraday correction changes.

The supplier buys its day-ahead forecast F (MWh per quarter hour) at the day-ahead price p. The load is
A = F (1 + sigma e). The deviation d = A - F is settled at reBAP: a short group (d > 0) pays the "short"
price, a long group (d < 0) is paid the "long" price. Against a perfect forecast the cost is

    cost = (reBAP_short - p) d+  +  (p - reBAP_long) (-d)+

which is negative where the imbalance price happens to favour the position.

The relative error e has unit variance and consists of two parts,

    e = s z_system + sqrt(1 - s^2) z_own,

with z_system the standardised error of the published national load forecast (SMARD), bias removed with
information available two days ahead, and z_own an independent AR(1) error. s is the correlation of the
supplier's error with the national one. Both the size sigma and s are assumptions; no customer data is used.

Intraday correction: five quarter hours before delivery the supplier trades the expected deviation
phi d(t - 5) at the intraday price index ID-AEP plus a half spread, phi estimated on the 28 days before
the delivery day. This assumes that meter data arrive within a quarter hour (own telemetry). ID-AEP is the
volume-weighted average of the trades of a quarter hour, not a price that was available at one moment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PER_DAY = 96
LEAD = 5                      # quarter hours between the last known deviation and delivery
WINDOW_DAYS = 28
HOURLY_PERSISTENCE = 0.9      # autocorrelation of the own error from hour to hour (assumption)
SIGMA = 0.03                  # standard deviation of the relative forecast error (assumption)


def system_error(load: pd.Series, forecast: pd.Series, window: int = WINDOW_DAYS) -> pd.Series:
    """Standardised national load forecast error (load / forecast - 1), bias removed.

    The bias of each quarter hour of the day is the mean of the same local quarter hour over the last
    `window` days ending two days before delivery, so it only uses data known at the day-ahead stage.
    """
    e = (load / forecast - 1.0).dropna()
    loc = e.index.tz_convert("Europe/Berlin")
    qod = loc.hour * 4 + loc.minute // 15
    bias = e.groupby(qod).transform(lambda x: x.shift(2).rolling(window, min_periods=window // 2).mean())
    z = (e - bias).dropna()
    return (z / z.std()).rename("z_system")


def own_error(n: int, rng: np.random.Generator, hourly_persistence: float = HOURLY_PERSISTENCE) -> np.ndarray:
    """Stationary AR(1) with unit variance on a quarter-hour grid."""
    rho = hourly_persistence ** 0.25
    shock = rng.standard_normal(n) * np.sqrt(1.0 - rho ** 2)
    z = np.empty(n)
    z[0] = rng.standard_normal()
    for i in range(1, n):
        z[i] = rho * z[i - 1] + shock[i]
    return z


def deviation(forecast_mwh: np.ndarray, z_system: np.ndarray, z_own: np.ndarray, share: float,
              sigma: float = SIGMA) -> np.ndarray:
    """Deviation load - schedule in MWh per quarter hour."""
    if not 0.0 <= share <= 1.0:
        raise ValueError("share is a correlation between 0 and 1")
    e = share * z_system + np.sqrt(1.0 - share ** 2) * z_own
    return forecast_mwh * sigma * e


def imbalance_cost(d: np.ndarray, da: np.ndarray, short: np.ndarray, long_: np.ndarray) -> np.ndarray:
    """Cost in EUR per quarter hour against a perfect forecast."""
    return (short - da) * np.clip(d, 0.0, None) + (da - long_) * np.clip(-d, 0.0, None)


def persistence_coefficient(d: np.ndarray, day: np.ndarray, lead: int = LEAD, window: int = WINDOW_DAYS) -> np.ndarray:
    """phi of d(t) = phi d(t - lead) for each delivery day, least squares on the `window` days before it."""
    x = np.concatenate([np.zeros(lead), d[:-lead]])
    xy = np.bincount(day, weights=x * d)
    xx = np.bincount(day, weights=x * x)
    cxy = np.concatenate([[0.0], np.cumsum(xy)])
    cxx = np.concatenate([[0.0], np.cumsum(xx)])
    k = np.arange(len(xy))
    lo = np.maximum(k - window, 0)
    num, den = cxy[k] - cxy[lo], cxx[k] - cxx[lo]
    phi = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.clip(phi, 0.0, 1.0)[day]


def intraday_trade(d: np.ndarray, day: np.ndarray, lead: int = LEAD) -> np.ndarray:
    """Quantity bought (positive) intraday for each quarter hour: the expected deviation."""
    x = np.concatenate([np.zeros(lead), d[:-lead]])
    return persistence_coefficient(d, day, lead) * x


def settle(d: np.ndarray, q: np.ndarray, da: np.ndarray, idaep: np.ndarray, short: np.ndarray, long_: np.ndarray,
           half_spread: float) -> tuple[np.ndarray, np.ndarray]:
    """(intraday cost, imbalance cost) per quarter hour of buying q intraday and leaving d - q to reBAP."""
    intraday = (idaep - da) * q + half_spread * np.abs(q)
    return intraday, imbalance_cost(d - q, da, short, long_)


def covariance_split(d: np.ndarray, da: np.ndarray, rebap: np.ndarray) -> dict:
    """With one imbalance price the total cost splits into n cov(reBAP - p, d) and n mean(reBAP - p) mean(d)."""
    spread = rebap - da
    n = len(d)
    cov = float(((spread - spread.mean()) * (d - d.mean())).sum())
    return {"covariance_eur": cov, "bias_eur": float(n * spread.mean() * d.mean())}


def metrics(cost: np.ndarray, day: np.ndarray, energy_mwh: float, years: float) -> dict:
    """Annual cost, cost per MWh and daily risk figures of a cost series per quarter hour."""
    daily = np.bincount(day, weights=cost)
    k = max(1, int(round(0.05 * len(daily))))
    worst = np.sort(daily)[::-1]
    return {"cost_eur_mwh": float(cost.sum() / energy_mwh), "cost_k_eur_per_year": float(cost.sum() / years / 1e3),
            "daily_std_k_eur": float(daily.std(ddof=1) / 1e3), "cvar95_daily_k_eur": float(worst[:k].mean() / 1e3),
            "worst_day_k_eur": float(worst[0] / 1e3)}
