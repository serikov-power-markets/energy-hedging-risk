"""Valuation of structured energy products.

- Spread option (Kirk 1995) and the tolling agreement of a gas-fired plant as a strip of daily
  spark-spread options: the holder may run the plant each day and earns
  max(P_power - HR * P_gas - EF * P_co2 - VOM, 0) per MWh. CO2 is treated as part of the strike
  at its forward (its volatility is folded into the gas leg through the cost combination).
- Swing option on gas: daily right to take between q_min and q_max at a fixed strike, with a
  total number of exercise days. Spot follows a one-factor mean-reverting log model
  (Schwartz 1997) on a trinomial tree (Hull-White construction); dynamic programming over
  (time, price node, rights left).
- Pay-as-produced PPA: fair fixed price and the buyer's P&L distribution when the profile of each
  historical weather year is priced at that year's realised prices (volume and shape risk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- spread options
def kirk(f1: float, f2: float, k: float, s1: float, s2: float, rho: float, t: float,
         r: float = 0.0) -> float:
    """Call on F1 - F2 - K (Kirk's approximation); vols annual, t in years."""
    if t <= 0:
        return max(f1 - f2 - k, 0.0)
    a = f2 / (f2 + k)
    sig = np.sqrt(s1 ** 2 - 2 * rho * s1 * s2 * a + (s2 * a) ** 2)
    d1 = (np.log(f1 / (f2 + k)) + 0.5 * sig ** 2 * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return float(np.exp(-r * t) * (f1 * stats.norm.cdf(d1) - (f2 + k) * stats.norm.cdf(d2)))


def spread_mc(f1, f2, k, s1, s2, rho, t, n=400_000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    p1 = f1 * np.exp(-0.5 * s1 ** 2 * t + s1 * np.sqrt(t) * z1)
    p2 = f2 * np.exp(-0.5 * s2 ** 2 * t + s2 * np.sqrt(t) * z2)
    return float(np.maximum(p1 - p2 - k, 0).mean())


def tolling_value(power_fwd: pd.Series, gas_fwd: pd.Series, co2: float, asof: pd.Timestamp,
                  heat_rate: float, ef: float, vom: float, capacity_mw: float,
                  vol_power, vol_gas: float, rho: float) -> pd.DataFrame:
    """Strip of daily spark-spread options of a tolling contract.

    power_fwd: forward per delivery day (EUR/MWh, block the plant runs in, e.g. peak);
    gas_fwd: gas forward per delivery day (EUR/MWh thermal); vol_power: function of time to
    delivery in years returning the annualised vol of that day's power price.
    Returns per day: intrinsic and option value in EUR for capacity x hours of the block.
    """
    out = []
    for d, fp in power_fwd.items():
        t = (d - asof).days / 365.0
        fg = heat_rate * gas_fwd.loc[d]
        k = ef * co2 + vom
        intr = max(fp - fg - k, 0.0)
        val = kirk(fp, fg, k, vol_power(t), vol_gas, rho, t)
        out.append((d, intr, val))
    df = pd.DataFrame(out, columns=["day", "intrinsic", "option"]).set_index("day")
    return df * capacity_mw


def power_vol_term(params, horizon_years: float) -> float:
    """Annualised vol of the daily spot price seen from today, two-factor model (per-day params).

    Variance of log price at horizon h days: short factor saturates, long factor grows with h;
    divided by the horizon to annualise (vol of the delivery-day price, not of a futures return).
    """
    h = max(horizon_years * 365.0, 1.0)
    k = params.kappa
    v_chi = params.sigma_chi ** 2 * (1 - np.exp(-2 * k * h)) / (1 - np.exp(-2 * k))
    v = v_chi + params.sigma_xi ** 2 * h
    return float(np.sqrt(v / max(horizon_years, 1 / 365)))


# ---------------------------------------------------------------- swing option
def trinomial_ou(x0: float, kappa: float, sigma: float, theta: np.ndarray, dt: float):
    """Hull-White trinomial tree for log price x with dx = kappa (theta(t) - x) dt + sigma dW.

    theta has one value per step. Returns node values per step and branch probabilities.
    """
    n = len(theta)
    dx = sigma * np.sqrt(3 * dt)
    jmax = int(np.ceil(0.184 / (kappa * dt))) if kappa * dt > 0 else n
    js = np.arange(-jmax, jmax + 1)
    M = -kappa * dt
    probs = {}
    for j in js:
        if j == jmax:
            pu = 7 / 6 + (j * j * M * M + 3 * j * M) / 2; pm = -1 / 3 - j * j * M * M - 2 * j * M
            pd_ = 1 / 6 + (j * j * M * M + j * M) / 2; shift = -1
        elif j == -jmax:
            pu = 1 / 6 + (j * j * M * M - j * M) / 2; pm = -1 / 3 - j * j * M * M + 2 * j * M
            pd_ = 7 / 6 + (j * j * M * M - 3 * j * M) / 2; shift = 1
        else:
            pu = 1 / 6 + (j * j * M * M + j * M) / 2; pm = 2 / 3 - j * j * M * M
            pd_ = 1 / 6 + (j * j * M * M - j * M) / 2; shift = 0
        probs[j] = (pu, pm, pd_, shift)
    # deterministic mean path of x
    mean = np.empty(n + 1)
    mean[0] = x0
    for i in range(n):
        mean[i + 1] = mean[i] + kappa * (theta[i] - mean[i]) * dt
    return js, dx, probs, mean


def swing_value(spot0: float, strike: float, kappa: float, sigma: float, fwd: np.ndarray,
                q_max: float, n_rights: int, dt: float = 1 / 365) -> float:
    """Swing contract: on each of len(fwd) days take q_max (or nothing) at the strike, at most
    n_rights times. fwd: the forward per day, used to set the tree's drift so that the expected
    spot matches the curve. Value in EUR per unit of q_max.
    """
    n = len(fwd)
    x0 = np.log(spot0)
    # centre the tree on log forward minus the variance correction of the OU stationary law
    var = sigma ** 2 * (1 - np.exp(-2 * kappa * dt * np.arange(1, n + 1))) / (2 * kappa)
    target = np.log(fwd) - 0.5 * var
    js, dx, probs, _ = trinomial_ou(0.0, kappa, sigma, np.zeros(n), dt)
    J = len(js)
    jmax = js[-1]
    price = np.exp(target[:, None] + js[None, :] * dx)          # (n, J): price at node, day i
    # value[r, j]: value with r rights left at the end; backward induction
    V = np.zeros((n_rights + 1, J))
    for i in range(n - 1, -1, -1):
        cont = np.zeros_like(V)
        if i < n - 1:
            for idx, j in enumerate(js):
                pu, pm, pd_, s = probs[j]
                k = j + s + jmax
                cont[:, idx] = pu * V[:, k + 1] + pm * V[:, k] + pd_ * V[:, k - 1]
        payoff = q_max * (price[i] - strike)
        newV = cont.copy()
        newV[1:] = np.maximum(cont[1:], cont[:-1] + payoff[None, :])
        V = newV
    # first day's distribution: the tree starts at node 0 one step before day 0
    pu, pm, pd_, s = probs[0]
    return float(pu * V[n_rights, jmax + 1] + pm * V[n_rights, jmax] + pd_ * V[n_rights, jmax - 1])


# ---------------------------------------------------------------- PPA with volume risk
def ppa_pap(gen: pd.Series, spot: pd.Series) -> dict:
    """Pay-as-produced PPA on one year: capture price, volume, and base price."""
    s = spot.reindex(gen.index)
    vol = float(gen.sum())
    return {"volume": vol, "capture": float((gen * s).sum() / vol), "base": float(s.mean())}


def ppa_weather_risk(gen_by_year: dict[int, pd.Series], spot_by_year: dict[int, pd.Series],
                     strike: float, expected_volume: float) -> pd.DataFrame:
    """Buyer's P&L per year of a PaP PPA resold at spot: (capture - strike) x volume.

    Each year combines that year's own generation and prices, so the price-volume correlation
    (cannibalisation in sunny or windy years) is kept. Volumes are scaled to the expected annual
    volume of the asset's capacity, so differences are weather, not fleet growth.
    """
    rows = []
    for y, g in gen_by_year.items():
        r = ppa_pap(g, spot_by_year[y])
        rows.append({"year": y, **r, "pnl_eur": (r["capture"] - strike) * r["volume"],
                     "volume_dev": r["volume"] / expected_volume - 1})
    return pd.DataFrame(rows).set_index("year")
