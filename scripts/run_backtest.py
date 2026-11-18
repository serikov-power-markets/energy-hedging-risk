"""Historical backtest of supplier procurement strategies, delivery years 2021-2025.

Book: 300 GWh households (H0) + 200 GWh commerce (G0) per year, fixed retail energy price set on
the first trading day of October before delivery (load-weighted HPFC, no margin, so P&L shows the
procurement outcome only). Forwards are modelled (see curves.py); spot is the realised DE-LU
day-ahead price. Output: results/backtest.csv, figures/backtest_pnl.png.
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ehr import book, data, hedging

H0, G0 = 300_000.0, 200_000.0
YEARS = range(2021, 2026)


def main() -> None:
    curves = pickle.loads((data.ROOT / "data" / "cache" / "curves.pkl").read_bytes())
    spot = data.smard_series("price")
    temp = data.temperature_daily()
    clim = book.climatology(temp)
    sens = book.weather_sensitivity()
    rows = []
    for y in YEARS:
        exp = book.expected_profile(y, H0, G0)
        act = book.actual_profile(exp, temp, clim, sens)
        sale_date = hedging.first_bday(pd.Timestamp(f"{y - 1}-10-01"))
        price = hedging.sales_price(curves[sale_date], exp, y)
        for name in hedging.STRATEGIES:
            trades = hedging.strategy(name, y, exp)
            fwd = hedging.trade_prices(trades, curves)
            r = hedging.settle(trades, fwd, act, spot, price)
            rows.append({"year": y, "strategy": name, **r})
            print(y, name, f"cost {r['cost_eur_mwh']:.1f}  pnl {r['pnl_eur_m']:.1f} m", flush=True)
    df = pd.DataFrame(rows)
    out = data.ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "backtest.csv", index=False, float_format="%.4f")

    piv = df.pivot(index="year", columns="strategy", values="pnl_eur_m")[hedging.STRATEGIES]
    summary = pd.DataFrame({
        "mean_pnl_eur_m": piv.mean(), "std_pnl_eur_m": piv.std(), "worst_pnl_eur_m": piv.min(),
        "mean_cost_eur_mwh": df.pivot(index="year", columns="strategy", values="cost_eur_mwh").mean(),
        # tariff stability under cost pass-through: largest year-on-year jump of the unit cost
        "max_yoy_cost_jump_eur_mwh": df.pivot(index="year", columns="strategy", values="cost_eur_mwh")
        .diff().abs().max(),
    })
    summary.to_csv(out / "backtest_summary.csv", float_format="%.3f")
    print(summary.round(2))

    fig, ax = plt.subplots(figsize=(8, 4))
    piv.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("P&L of the fixed-price book, EUR m")
    ax.set_xlabel("delivery year")
    ax.set_title("Procurement strategies, 500 GWh book, modelled forwards")
    fig.tight_layout()
    (data.ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(data.ROOT / "figures" / "backtest_pnl.png", dpi=150)


if __name__ == "__main__":
    main()
