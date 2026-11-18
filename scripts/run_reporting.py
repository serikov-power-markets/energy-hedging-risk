"""Data checks, DuckDB position store, SQL KPIs, reconciliation and report exports.

1. Checks every input series (quality.py) -> results/data_quality.csv, results/anomalies_price.csv
2. Loads market data, the customer book, the trade blotter of all strategies and the hourly
   positions for delivery 2021-2025 into data/ehr.duckdb
3. Reconciles the hourly position rebuilt in SQL from the blotter against the Python position
   (sql/reconcile.sql must return no rows)
4. Monthly KPIs in SQL (sql/kpi.sql) -> results/kpi_monthly.csv, an Excel report and CSV tables
   for a Power BI star schema (results/powerbi/)
"""
from __future__ import annotations

import pickle

import duckdb
import numpy as np
import pandas as pd

from ehr import book, data, hedging, quality

ROOT = data.ROOT
RES = ROOT / "results"
H0, G0 = 300_000.0, 200_000.0
YEARS = range(2021, 2026)


def run_checks(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["price", "load", "wind_onshore", "wind_offshore", "solar"]:
        r, an = quality.check_series(name, hourly[name])
        rows.append(r)
        if name == "price":
            an.to_csv(RES / "anomalies_price.csv", float_format="%.2f")
    rows.append(quality.check_daily("ttf", data.ttf_daily().loc["2019":]))
    rows.append(quality.check_daily("eua", data.eua_daily().loc["2019":]))
    return pd.DataFrame(rows)


def main() -> None:
    hourly = data.hourly_frame()
    dq = run_checks(hourly)
    dq.to_csv(RES / "data_quality.csv", index=False)
    print(dq.to_string(index=False))

    curves = pickle.loads((ROOT / "data" / "cache" / "curves.pkl").read_bytes())
    temp = data.temperature_daily()
    clim = book.climatology(temp)
    sens = book.weather_sensitivity()

    market = hourly.reset_index(names="ts_utc")
    market["ts_local"] = market["ts_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    market["is_peak"] = data.is_peak(pd.DatetimeIndex(market["ts_utc"]))
    market["ts_utc"] = market["ts_utc"].dt.tz_localize(None)

    books, trades, tmonths, positions = [], [], [], []
    tid = 0
    for y in YEARS:
        exp = book.expected_profile(y, H0, G0)
        act = book.actual_profile(exp, temp, clim, sens)
        b = pd.DataFrame({"ts_utc": exp.index.tz_localize(None),
                          "ts_local": exp.index.tz_convert("Europe/Berlin").tz_localize(None),
                          "expected_mwh": exp.to_numpy(), "actual_mwh": act.to_numpy()})
        books.append(b)
        for strat in hedging.STRATEGIES:
            tr = hedging.strategy(strat, y, exp)
            px = hedging.trade_prices(tr, curves)
            for t, p in zip(tr, px):
                tid += 1
                trades.append({"trade_id": tid, "strategy": strat, "trade_date": t.date,
                               "delivery_start": t.start, "delivery_end": t.end, "kind": t.kind,
                               "mw": t.mw, "price": p})
                for ms in pd.date_range(t.start, t.end, freq="MS"):
                    me = min(ms + pd.offsets.MonthEnd(0), t.end)
                    h = hedging.period_hours(ms, me)
                    n = int(data.is_peak(h).sum()) if t.kind == "peak" else len(h)
                    tmonths.append({"strategy": strat, "trade_id": tid, "delivery_month": ms,
                                    "mwh": t.mw * n, "price": p})
            pos = hedging.hedge_position(tr, act.index)
            positions.append(pd.DataFrame({"strategy": strat, "ts_utc": pos.index.tz_localize(None),
                                           "hedge_mw": pos.to_numpy()}))

    db_path = ROOT / "data" / "ehr.duckdb"
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    for name, df in {"market_hourly": market, "book_hourly": pd.concat(books),
                     "trades": pd.DataFrame(trades), "trade_months": pd.DataFrame(tmonths),
                     "position_hourly": pd.concat(positions), "data_quality": dq.astype(str)}.items():
        con.register("tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")

    bad = con.execute((ROOT / "sql" / "reconcile.sql").read_text()).df()
    print("reconciliation: rows that differ =", len(bad))
    if len(bad):
        raise SystemExit("position reconciliation failed")

    kpi = con.execute((ROOT / "sql" / "kpi.sql").read_text()).df()
    kpi.to_csv(RES / "kpi_monthly.csv", index=False, float_format="%.4f")
    yearly = (kpi.assign(year=kpi["delivery_month"].dt.year)
              .assign(cost=lambda d: d["unit_cost"] * d["actual_mwh"])
              .groupby(["strategy", "year"]).agg(actual_mwh=("actual_mwh", "sum"), cost=("cost", "sum"),
                                                 open_share=("open_share", "mean")))
    yearly["unit_cost"] = yearly["cost"] / yearly["actual_mwh"]
    # SQL result must match the Python settlement of the backtest
    bt = pd.read_csv(RES / "backtest.csv").set_index(["strategy", "year"])["cost_eur_mwh"]
    diff = (yearly["unit_cost"] - bt.reindex(yearly.index)).abs().max()
    print(f"SQL vs Python unit cost, max abs difference: {diff:.6f} EUR/MWh")

    with pd.ExcelWriter(RES / "kpi_report.xlsx") as xw:
        yearly.reset_index().to_excel(xw, sheet_name="yearly", index=False)
        kpi.to_excel(xw, sheet_name="monthly", index=False)
        dq.astype(str).to_excel(xw, sheet_name="data_quality", index=False)

    pbi = RES / "powerbi"
    pbi.mkdir(exist_ok=True)
    kpi.to_csv(pbi / "fact_kpi_monthly.csv", index=False, float_format="%.4f")
    pd.DataFrame(trades).to_csv(pbi / "fact_trades.csv", index=False, float_format="%.4f")
    months = pd.DataFrame({"delivery_month": pd.date_range("2021-01-01", "2025-12-01", freq="MS")})
    months["year"] = months["delivery_month"].dt.year
    months["quarter"] = "Q" + months["delivery_month"].dt.quarter.astype(str)
    months.to_csv(pbi / "dim_month.csv", index=False)
    pd.DataFrame({"strategy": hedging.STRATEGIES,
                  "description": ["no hedge, all volume at day-ahead",
                                  "Cal base and peak bought on the sales date",
                                  "12 monthly tranches of Cal base and peak in the year before",
                                  "12 tranches of Cal base only (no shape hedge)",
                                  "80% Cal ladder plus month products one month ahead"]}
                 ).to_csv(pbi / "dim_strategy.csv", index=False)
    con.close()


if __name__ == "__main__":
    main()
