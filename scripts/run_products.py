"""Structured products valued on 1 December 2025 for delivery in 2026.

1. Tolling: a 400 MW CCGT (heat rate 1.82, 0.365 t/MWh, variable O&M 3 EUR/MWh) as a strip of
   daily spark-spread options, base (24 h) and peak block, against the hourly intrinsic value of
   the HPFC. Power vol from the two-factor model, gas vol and power-gas correlation from history.
2. Swing: gas delivery January to March 2026, 1 MWh per exercised day at a strike equal to the
   TTF forward, with 30 to 90 exercise rights; value of flexibility against the strip of options.
3. Pay-as-produced solar PPA: fair price from the HPFC and the average solar profile, and the
   buyer's capture-price risk replaying the price and generation shapes of 2019-2025 at the 2026
   forward level; green bundle with the French GoO auction price.
Output: results/products_*.csv, figures/products_*.png
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ehr import curves, data, goo, products

ASOF = pd.Timestamp("2025-12-01")
Y = 2026
RES = data.ROOT / "results"
FIG = data.ROOT / "figures"


def gas_and_corr(spot: pd.Series, ttf: pd.Series) -> dict:
    """TTF vol (trailing year, annualised from daily log returns), mean reversion of the log
    front-month (AR(1) on three years) and the correlation of monthly log changes of baseload
    spot and TTF (three years)."""
    t = ttf[ttf.index < ASOF]
    r = np.log(t).diff().dropna()
    vol = float(r.iloc[-250:].std() * np.sqrt(250))
    x = np.log(t.iloc[-750:])
    phi = float(np.polyfit(x.iloc[:-1].to_numpy(), x.iloc[1:].to_numpy(), 1)[0])
    kappa = float(-np.log(min(phi, 0.9999)) * 250)
    b = data.daily_base(spot)
    m = pd.DataFrame({"p": np.log(b.clip(lower=5)), "g": np.log(t)}).dropna()
    m = m[m.index >= ASOF - pd.Timedelta(3 * 365, unit="D")].resample("MS").mean().diff().dropna()
    return {"ttf": float(t.iloc[-1]), "vol_gas": vol, "kappa_gas": kappa,
            "rho_power_gas": float(m["p"].corr(m["g"]))}


def tolling(curve: curves.CurveModel, g: dict, eua: float) -> pd.DataFrame:
    hr, ef, vom, cap = 1.82, 0.365, 3.0, 400.0
    days = pd.date_range(f"{Y}-01-01", f"{Y}-12-31")
    hpfc = curve.hpfc(days[0], days[-1])
    pk = data.is_peak(hpfc.index)
    loc = hpfc.index.tz_convert("Europe/Berlin").normalize().tz_localize(None)
    base_day = hpfc.groupby(loc).mean()
    peak_day = hpfc[pk].groupby(loc[pk]).mean()
    gas = pd.Series(g["ttf"], index=days)
    vol = lambda t: products.power_vol_term(curve.params, t)
    base = products.tolling_value(base_day, gas, eua, ASOF, hr, ef, vom, cap, vol, g["vol_gas"],
                                  g["rho_power_gas"]) * 24
    peak = products.tolling_value(peak_day, gas, eua, ASOF, hr, ef, vom, cap, vol, g["vol_gas"],
                                  g["rho_power_gas"]) * 12
    cost = hr * g["ttf"] + ef * eua + vom
    hourly_intr = float(np.maximum(hpfc - cost, 0).sum() * cap)
    out = pd.DataFrame({
        "base_intrinsic": base["intrinsic"], "base_option": base["option"],
        "peak_intrinsic": peak["intrinsic"], "peak_option": peak["option"]})
    summary = pd.Series({
        "fuel_cost_eur_mwh": cost, "base_fwd": float(base_day.mean()), "peak_fwd": float(peak_day.mean()),
        "base_intrinsic_eur_m": out["base_intrinsic"].sum() / 1e6,
        "base_option_eur_m": out["base_option"].sum() / 1e6,
        "peak_intrinsic_eur_m": out["peak_intrinsic"].sum() / 1e6,
        "peak_option_eur_m": out["peak_option"].sum() / 1e6,
        "hourly_intrinsic_eur_m": hourly_intr / 1e6,
        "base_option_eur_per_kw_yr": out["base_option"].sum() / (cap * 1000),
    })
    out.resample("MS").sum().div(1e6).to_csv(RES / "products_tolling_monthly.csv", float_format="%.3f")
    return summary


def swing(g: dict) -> pd.DataFrame:
    days = pd.date_range(f"{Y}-01-01", f"{Y}-03-31")
    fwd = np.full(len(days), g["ttf"])
    kappa = max(g["kappa_gas"], 1.0)     # front-month reversion is slow; floor for the daily spot
    rows = []
    for n in (1, 15, 30, 45, 60, 75, 90):
        v = products.swing_value(g["ttf"], g["ttf"], kappa, g["vol_gas"], fwd, 1.0, n)
        rows.append({"rights": n, "value_eur": v, "value_per_right": v / n})
    return pd.DataFrame(rows)


def solar_ppa(curve: curves.CurveModel, spot: pd.Series, go_price: float) -> tuple[pd.Series, pd.DataFrame]:
    sol = data.smard_series("solar", "2019-01-01", "2025-12-31")
    sl = sol.tz_convert("Europe/Berlin")
    # average shape 2023-2025 by (month, hour), share of the year's energy per hour of that kind
    recent = sl.loc["2023":"2025"]
    norm = recent / recent.groupby(recent.index.year).transform("sum")
    shape = norm.groupby([norm.index.month, norm.index.hour]).mean()
    hpfc = curve.hpfc(pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y}-12-31"))
    hl = hpfc.index.tz_convert("Europe/Berlin")
    prof = shape.reindex(pd.MultiIndex.from_arrays([hl.month, hl.hour])).to_numpy()
    prof = prof / prof.sum()
    fair = float((prof * hpfc.to_numpy()).sum())
    base = float(hpfc.mean())
    rows = []
    for y in range(2019, 2026):
        g = sl.loc[str(y)]
        p = spot.reindex(g.index.tz_convert("UTC")).to_numpy()
        p = p * base / np.nanmean(p)                   # that year's price shape at the 2026 level
        w = g.to_numpy() / g.sum()
        cap = float(np.nansum(w * p))
        rows.append({"shape_year": y, "capture": cap, "capture_rate": cap / base,
                     "buyer_pnl_eur_mwh": cap - fair})
    risk = pd.DataFrame(rows).set_index("shape_year")
    summary = pd.Series({"cal_base_fwd": base, "pap_fair_price": fair, "capture_rate": fair / base,
                         "go_price": go_price, "green_bundle": fair + go_price,
                         "worst_buyer_pnl_eur_mwh": float(risk["buyer_pnl_eur_mwh"].min())})
    return summary, risk


def main() -> None:
    cache = pickle.loads((data.ROOT / "data" / "cache" / "curves.pkl").read_bytes())
    curve = cache[ASOF]
    spot = data.smard_series("price")
    g = gas_and_corr(spot, data.ttf_daily())
    eua = float(data.eua_daily().loc[:ASOF].iloc[-1])
    print("inputs", {k: round(v, 3) for k, v in g.items()}, "eua", eua)

    tol = tolling(curve, g, eua)
    sw = swing(g)
    gp = goo.monthly_price()
    go_price = float(gp.loc[:ASOF, "all"].iloc[-1])
    ppa, ppa_risk = solar_ppa(curve, spot, go_price)

    inputs = pd.Series({**g, "eua": eua, "kappa_power": curve.params.kappa,
                        "sigma_chi": curve.params.sigma_chi, "sigma_xi": curve.params.sigma_xi,
                        "fuel_b": curve.anchor.b, "fuel_kappa_m": curve.anchor.kappa_m})
    inputs.to_csv(RES / "products_inputs.csv", header=["value"], float_format="%.4f")
    tol.to_csv(RES / "products_tolling.csv", header=["value"], float_format="%.3f")
    sw.to_csv(RES / "products_swing.csv", index=False, float_format="%.3f")
    ppa.to_csv(RES / "products_ppa.csv", header=["value"], float_format="%.3f")
    ppa_risk.to_csv(RES / "products_ppa_risk.csv", float_format="%.3f")
    gp.to_csv(RES / "goo_french_auctions.csv", float_format="%.3f")
    print(tol.round(2)); print(sw.round(2)); print(ppa.round(2)); print(ppa_risk.round(2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(sw["rights"], sw["value_per_right"], marker="o")
    ax[0].set_xlabel("exercise rights in 90 days")
    ax[0].set_ylabel("value per right, EUR/MWh")
    ax[0].set_title("Gas swing Q1 2026: value of flexibility")
    gp["all"].plot(ax=ax[1])
    ax[1].set_ylabel("EUR/MWh")
    ax[1].set_title("French GoO auctions (EEX), weighted price")
    fig.tight_layout()
    fig.savefig(FIG / "products_swing_goo.png", dpi=150)


if __name__ == "__main__":
    main()
