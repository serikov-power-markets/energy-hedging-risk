"""Market risk of two linear books, one-day VaR and ES with backtests, 2021-2025.

Risk factors (daily changes on business days):
- power_cal1: modelled next-calendar-year German baseload forward (curves.daily_forward_series);
  the day the front calendar year rolls (first business day of a year) is dropped
- ttf: TTF front-month future (Yahoo Finance TTF=F), forward-filled over missing days
- eua: EEX primary auction clearing price of EU allowances, forward-filled between auctions

Books (one year of delivery, constant positions):
- power: long 10 MW Cal baseload (87,600 MWh)
- spark: long 10 MW Cal baseload power, short the gas and allowances a 55% efficient CCGT needs
  (heat rate 1.82, 0.365 t CO2 per MWh electric): a clean spark spread position
Output: results/risk_backtest.csv, results/risk_daily_<book>_<method>.csv, figures/var_<book>.png
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ehr import curves, data, risk

HOURS = 8760
MW = 10.0
HR, EF = 1.82, 0.365
BOOKS = {
    "power": np.array([MW * HOURS, 0.0, 0.0]),
    "spark": np.array([MW * HOURS, -MW * HOURS * HR, -MW * HOURS * EF]),
}
METHODS = ["hs", "ewma", "fhs", "mct"]


def factors() -> pd.DataFrame:
    cache = pickle.loads((data.ROOT / "data" / "cache" / "curves.pkl").read_bytes())
    out = data.ROOT / "data" / "cache" / "daily_forwards.parquet"
    if out.exists():
        fw = pd.read_parquet(out)
    else:
        fw = curves.daily_forward_series(data.smard_series("price"), cache)
        fw.to_parquet(out)
    days = fw.index
    ttf = data.ttf_daily().reindex(days, method="ffill")
    eua = data.eua_daily().reindex(days, method="ffill")
    lv = pd.DataFrame({"power_cal1": fw["power_cal1"], "ttf": ttf, "eua": eua}).dropna()
    dx = lv.diff().dropna()
    roll = [pd.bdate_range(f"{y}-01-01", periods=1)[0] for y in range(2020, 2027)]
    return lv, dx[~dx.index.isin(roll)]


def main() -> None:
    lv, dx = factors()
    res_dir = data.ROOT / "results"
    print("factor levels", lv.index[0].date(), lv.index[-1].date(), len(dx), "daily changes")
    rows = []
    for book, pos in BOOKS.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        for method in METHODS:
            for alpha in (0.99, 0.975):
                f = risk.rolling_var(dx, pos, alpha, 250, method)
                f = f.loc["2021-01-01":]
                bt = risk.backtest(f, alpha)
                rows.append({"book": book, "method": method, "alpha": alpha, **bt,
                             "traffic_light": risk.traffic_light(bt["exceptions"], bt["n"])
                             if alpha == 0.99 else ""})
                if alpha == 0.99:
                    f.to_csv(res_dir / f"risk_daily_{book}_{method}.csv", float_format="%.1f")
                    ax.plot(f.index, -f["var"] / 1e6, lw=0.8, label=f"-VaR99 {method}")
        ax.plot(f.index, f["pnl"] / 1e6, lw=0.4, color="grey", label="daily P&L")
        ax.set_ylabel("EUR m")
        ax.set_title(f"{book} book: daily P&L and 99% one-day VaR")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(data.ROOT / "figures" / f"var_{book}.png", dpi=150)
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "risk_backtest.csv", index=False, float_format="%.4f")
    cols = ["book", "method", "alpha", "n", "exceptions", "expected", "p_pof", "p_ind", "p_cc", "z2",
            "mean_var", "traffic_light"]
    with pd.option_context("display.width", 200):
        print(df[cols].round(3).to_string(index=False))
    corr = dx.loc["2021":].corr().round(2)
    corr.to_csv(res_dir / "factor_correlation.csv")
    print(corr)


if __name__ == "__main__":
    main()
