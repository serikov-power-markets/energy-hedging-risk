"""v0.2: retail tariff build-up, demand and churn response, industrial procurement, limits and counterparty credit.

Everything here is an extension of the modelled-forward framework of v0.1. Inputs that are assumptions (grid fees, demand
elasticity, switching sensitivity, default probabilities) are parameters with stated values, not data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import data, hedging


# ------------------------------------------------------------------ retail tariff build-up
@dataclass(frozen=True)
class TariffInputs:
    """Components in ct/kWh before VAT. Levies of 2026 follow the transmission system operators; grid fee, supply costs and
    margin are assumptions."""
    kwkg: float = 0.446
    offshore: float = 0.941
    besondere_netznutzung: float = 1.559        # formerly Paragraph 19 StromNEV levy
    stromsteuer: float = 2.05
    konzessionsabgabe: float = 1.32             # lowest rate for tariff customers (municipalities up to 25,000 residents)
    netzentgelt: float = 9.5                    # assumption: grid fee incl. metering, region-specific
    beschaffungsnebenkosten: float = 0.30       # assumption: balancing, forecasting, guarantees of origin
    vertrieb: float = 2.2                       # assumption: billing, customer service, acquisition
    marge: float = 1.0                          # assumption
    vat: float = 0.19


def tariff(energy_eur_mwh: float, t: TariffInputs = TariffInputs()) -> dict:
    """Gross retail price in ct/kWh and its components for a given energy procurement cost in EUR/MWh."""
    energy = energy_eur_mwh / 10.0
    parts = {"energy procurement": energy, "procurement overhead": t.beschaffungsnebenkosten, "sales cost": t.vertrieb,
             "margin": t.marge, "grid fee": t.netzentgelt, "KWKG levy": t.kwkg, "offshore levy": t.offshore,
             "special grid use levy": t.besondere_netznutzung, "electricity tax": t.stromsteuer,
             "concession fee": t.konzessionsabgabe}
    net = sum(parts.values())
    parts["VAT"] = net * t.vat
    parts["gross price"] = net + parts["VAT"]
    return parts


def churn_probability(price: float, market_price: float, base: float = 0.10, sensitivity: float = 0.35) -> float:
    """Annual probability that a customer leaves, logistic in the price gap to the market (ct/kWh); `base` at zero gap."""
    a = np.log(base / (1 - base))
    return float(1 / (1 + np.exp(-(a + sensitivity * (price - market_price)))))


def expected_margin(price: float, cost: float, market_price: float, volume0: float = 3500.0, elasticity: float = -0.25,
                    base_churn: float = 0.10, sensitivity: float = 0.35, p0: float | None = None) -> float:
    """Expected annual margin per customer (EUR): price minus unit cost times demand, times the chance to keep the customer."""
    p0 = market_price if p0 is None else p0
    vol = volume0 * (price / p0) ** elasticity
    return float((price - cost) / 100.0 * vol * (1 - churn_probability(price, market_price, base_churn, sensitivity)))


def optimal_price(cost: float, market_price: float, **kw) -> tuple[float, float]:
    grid = np.linspace(cost, cost + 15.0, 1501)
    m = np.array([expected_margin(p, cost, market_price, **kw) for p in grid])
    i = int(m.argmax())
    return float(grid[i]), float(m[i])


# ------------------------------------------------------------------ industrial procurement
def industrial_expected(year: int, annual_mwh: float, weekend_factor: float = 0.85) -> pd.Series:
    """Hourly profile (MW) of a three-shift site: flat on weekdays, lower on weekends, scaled to the annual energy."""
    hrs = hedging.period_hours(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
    local = hrs.tz_convert("Europe/Berlin")
    shape = np.where(local.dayofweek >= 5, weekend_factor, 1.0)
    return pd.Series(shape / shape.sum() * annual_mwh, index=hrs)


def industrial_actual(expected: pd.Series, seed: int, sd: float = 0.03, rho: float = 0.9) -> pd.Series:
    rng = np.random.default_rng(seed)
    e = np.empty(len(expected))
    z = 0.0
    for i in range(len(e)):
        z = rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal()
        e[i] = z
    return expected * (1 + sd * e)


def solar_profile(year: int) -> pd.Series:
    s = data.smard_series("solar")
    s = s[(s.index >= pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin").tz_convert("UTC")) &
          (s.index < pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin").tz_convert("UTC"))]
    return s / s.sum()


def industrial_cost(name: str, year: int, curves: dict, spot: pd.Series, annual_mwh: float = 100_000.0, seed: int = 0) -> dict:
    exp = industrial_expected(year, annual_mwh)
    act = industrial_actual(exp, seed + year)
    sale_date = hedging.first_bday(pd.Timestamp(f"{year - 1}-10-01"))
    budget = hedging.sales_price(curves[sale_date], exp, year)
    ppa_extra, ppa_share = 0.0, 0.0
    if name == "index":
        trades = []
    elif name in ("cal_once", "ladder"):
        trades = hedging.strategy(name, year, exp)
    elif name == "ladder_70":
        trades = [hedging.Trade(t.date, t.start, t.end, t.kind, 0.7 * t.mw) for t in hedging.strategy("ladder", year, exp)]
    elif name == "ppa25_ladder":
        ppa_share = 0.25
        rest = exp * (1 - ppa_share)
        trades = hedging.strategy("ladder", year, rest)
        sp = solar_profile(year).reindex(exp.index).fillna(0.0)
        vol = ppa_share * annual_mwh * sp
        hp = curves[sale_date].hpfc(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
        strike = float((hp.reindex(vol.index).fillna(hp.mean()) * vol).sum() / vol.sum())
        ppa_extra = float((vol * (strike - spot.reindex(vol.index))).sum())          # PPA energy costs the strike, saves the spot
    else:
        raise ValueError(name)
    fwd = hedging.trade_prices(trades, curves)
    r = hedging.settle(trades, fwd, act, spot, budget, expected=exp)
    total = r["cost_eur_mwh"] * r["volume_mwh"] + ppa_extra
    # a PPA with pay-as-produced volume also covers part of the load, so the energy to settle stays the total load
    cost = total / r["volume_mwh"]
    return {"year": year, "strategy": name, "cost_eur_mwh": cost, "budget_eur_mwh": budget, "overspend_eur_mwh": cost - budget,
            "overspend_eur_m": (cost - budget) * r["volume_mwh"] / 1e6}


# ------------------------------------------------------------------ limits
def limit_report(var: pd.Series, limit: float, amber: float = 0.8) -> pd.DataFrame:
    """Daily utilisation of a VaR limit with the status green, amber (from `amber` of the limit) or red (above)."""
    u = var / limit
    status = np.where(u > 1.0, "red", np.where(u >= amber, "amber", "green"))
    return pd.DataFrame({"var": var, "utilisation": u, "status": status})


def limit_from_history(var: pd.Series, quantile: float = 0.9, multiple: float = 1.25) -> float:
    """A limit set from a calm reference: a multiple of a quantile of the VaR history."""
    return float(multiple * var.quantile(quantile))


# ------------------------------------------------------------------ counterparty credit
RATING_PD = {"AA": 0.0003, "A": 0.0007, "BBB": 0.0020, "BB": 0.0090, "B": 0.0350}      # assumptions: illustrative one-year PDs


def forward_exposure(f0: float, sigma: float, years: float, steps_per_year: int = 12, n_sim: int = 20_000,
                     seed: int = 0, threshold: float = 0.0, mw: float = 1.0, mu_decay: bool = True) -> pd.DataFrame:
    """Expected positive exposure (EUR) of a long baseload forward at its own price with remaining delivery, per time step.

    Forward price F_t = F0 exp(sigma_t W_t - 0.5 sigma_t^2 t) with a vol that decays towards maturity (Samuelson) when
    mu_decay; value = remaining MWh x (F_t - F0); collateral reduces exposure above the threshold."""
    rng = np.random.default_rng(seed)
    n = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    t = np.arange(1, n + 1) * dt
    z = rng.standard_normal((n_sim, n))
    w = np.cumsum(np.sqrt(dt) * z, axis=1)
    remaining_mwh = mw * 8760 * (years - t)
    vol = sigma * (np.exp(-0.5 * (years - t)) if mu_decay else 1.0)
    f = f0 * np.exp(vol * w - 0.5 * vol ** 2 * t)
    v = remaining_mwh * (f - f0)
    ee = np.maximum(v - threshold, 0.0).mean(axis=0)
    return pd.DataFrame({"t": t, "ee_eur": ee, "pfe95_eur": np.quantile(np.maximum(v - threshold, 0.0), 0.95, axis=0)})


def cva(exposure: pd.DataFrame, pd_annual: float, recovery: float = 0.4) -> float:
    """Unilateral CVA = (1 - R) sum EE(t_i) x marginal default probability of the interval, flat hazard rate."""
    lam = -np.log(1 - pd_annual)
    t = exposure["t"].to_numpy()
    prev = np.concatenate([[0.0], t[:-1]])
    dpd = np.exp(-lam * prev) - np.exp(-lam * t)
    return float((1 - recovery) * (exposure["ee_eur"].to_numpy() * dpd).sum())
