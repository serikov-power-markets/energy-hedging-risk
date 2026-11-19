"""v0.2 results: tariff, churn optimum, industrial procurement 2021-2025, limit report, CVA. Writes results/retail_*.csv."""
import json
import pickle

import numpy as np
import pandas as pd

from ehr import data, hedging, retail

OUT = data.ROOT / "results"
STRATS = ["index", "cal_once", "ladder", "ladder_70", "ppa25_ladder"]


def main() -> dict:
    curves = pickle.loads((data.ROOT / "data" / "cache" / "curves.pkl").read_bytes())
    spot = data.smard_series("price")
    summary = {}

    # industrial buyer, 100 GWh per year, 3-shift profile
    rows = [retail.industrial_cost(s, y, curves, spot) for y in range(2021, 2026) for s in STRATS]
    ind = pd.DataFrame(rows)
    ind.to_csv(OUT / "retail_industrial.csv", index=False, float_format="%.3f")
    sm = ind.groupby("strategy").agg(mean_cost=("cost_eur_mwh", "mean"), std_cost=("cost_eur_mwh", "std"),
                                     mean_overspend=("overspend_eur_mwh", "mean"), worst_overspend_m=("overspend_eur_m", "max"),
                                     best_overspend_m=("overspend_eur_m", "min")).reindex(STRATS)
    sm.to_csv(OUT / "retail_industrial_summary.csv", float_format="%.2f")
    print(sm.round(2))

    # tariff for a household, energy cost from the load-weighted forward of the first October 2025 trading day for 2026
    from ehr import book
    exp = book.expected_profile(2026, 300_000.0, 200_000.0)
    d = hedging.first_bday(pd.Timestamp("2025-10-01"))
    energy = hedging.sales_price(curves[d], exp, 2026) if 2026 else 0.0
    t = retail.tariff(energy)
    pd.Series(t).to_csv(OUT / "retail_tariff.csv", header=["ct_per_kwh"], float_format="%.3f")
    summary["energy_eur_mwh_2026"] = energy
    summary["gross_price_ct_kwh"] = t["gross price"]
    print(energy, {k: round(v, 2) for k, v in t.items()})

    # churn: optimum energy-and-margin price against the market price for several switching sensitivities
    cost = (energy / 10 + 0.30 + 2.2) * 1.0                      # variable supply cost ct/kWh
    market = cost + 1.5
    rows = []
    for sens in (0.15, 0.35, 0.8):
        for el in (-0.1, -0.25, -0.4):
            p, m = retail.optimal_price(cost, market, sensitivity=sens, elasticity=el)
            rows.append({"switching_sensitivity": sens, "elasticity": el, "optimal_price_ct": p, "margin_over_cost_ct": p - cost,
                         "expected_margin_eur": m, "churn": retail.churn_probability(p, market, sensitivity=sens)})
    pd.DataFrame(rows).to_csv(OUT / "retail_churn.csv", index=False, float_format="%.3f")
    print(pd.DataFrame(rows).round(3))

    # limit report on the supplier book VaR (filtered historical simulation)
    var = pd.read_csv(OUT / "risk_daily_power_fhs.csv", index_col="date", parse_dates=True)["var"]
    calm = var[var.index < "2022-01-01"]
    limit = retail.limit_from_history(calm, 0.9, 1.25)
    rep = retail.limit_report(var, limit)
    by_year = rep.groupby(rep.index.year)["status"].value_counts().unstack(fill_value=0)
    by_year.to_csv(OUT / "retail_limit_days_by_year.csv")
    summary["limit_eur"] = limit
    print(limit, by_year)

    # CVA of a 3-year baseload forward, 1 MW, price 80 EUR/MWh, vol 35%
    rows = []
    for thr in (0.0, 50_000.0):
        e = retail.forward_exposure(80.0, 0.35, 3, threshold=thr, n_sim=40_000)
        for rating, pdv in retail.RATING_PD.items():
            rows.append({"threshold_eur": thr, "rating": rating, "pd_annual": pdv, "cva_eur": retail.cva(e, pdv),
                         "cva_eur_per_mwh": retail.cva(e, pdv) / (3 * 8760), "peak_ee_eur": float(e["ee_eur"].max())})
    pd.DataFrame(rows).to_csv(OUT / "retail_cva.csv", index=False, float_format="%.3f")
    print(pd.DataFrame(rows).round(3))
    (OUT / "retail_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
