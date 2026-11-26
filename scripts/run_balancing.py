"""v0.3: balancing group of 100 GWh of metered industrial customers, January 2025 to September 2026.

Writes results/balancing_*.csv, results/balancing_summary.json and figures/balancing.png.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ehr import balancing as b
from ehr import data, imbalance_data, retail

OUT = data.ROOT / "results"
FIG = data.ROOT / "figures"
ANNUAL_MWH = 100_000.0
START, END, WARMUP = "2025-01-01", "2026-09-30", "2024-11-01"
SEEDS = 200
SHARES = (0.0, 0.25, 0.5, 0.75)
HALF_SPREAD = 1.0             # EUR/MWh paid on every intraday trade (assumption)


def load_inputs() -> pd.DataFrame:
    load = imbalance_data.smard_quarter("load", WARMUP, END)
    forecast = imbalance_data.smard_quarter("load_forecast", WARMUP, END)
    price = imbalance_data.smard_quarter("price", START, END)
    rebap = imbalance_data.ntp_series("rebap", START, END)
    idaep = imbalance_data.ntp_series("idaep", START, END)
    z = b.system_error(load, forecast)
    df = pd.concat([price.rename("da"), rebap, idaep, z], axis=1)
    t0 = pd.Timestamp(START, tz="Europe/Berlin").tz_convert("UTC")
    t1 = (pd.Timestamp(END) + pd.Timedelta(days=1)).tz_localize("Europe/Berlin").tz_convert("UTC")
    df = df[(df.index >= t0) & (df.index < t1)]
    n_all = len(df)
    df = df.dropna()
    # schedule: the three-shift industrial profile of the v0.2 buyer, spread evenly over the quarter hours of an hour
    hourly = pd.concat([retail.industrial_expected(y, ANNUAL_MWH) for y in (2025, 2026)])
    df["f_mwh"] = hourly.reindex(df.index.floor("h")).to_numpy() / 4.0
    df.attrs["dropped"] = n_all - len(df)
    return df


def run_case(df, day, years, share, hourly_persistence, sigma, half_spread, seed, z_system=None):
    rng = np.random.default_rng(seed)
    z_own = b.own_error(len(df), rng, hourly_persistence)
    zs = df["z_system"].to_numpy() if z_system is None else z_system
    d = b.deviation(df["f_mwh"].to_numpy(), zs, z_own, share, sigma)
    da, short, long_, ida = (df[c].to_numpy() for c in ("da", "short", "long", "idaep"))
    energy = float(df["f_mwh"].sum() + d.sum())
    rows = {}
    imb = b.imbalance_cost(d, da, short, long_)
    rows["reBAP only"] = {**b.metrics(imb, day, energy, years), **b.covariance_split(d, da, (short + long_) / 2),
                          "imbalance_share": float(np.abs(d).sum() / energy)}
    q = b.intraday_trade(d, day)
    c_id, c_imb = b.settle(d, q, da, ida, short, long_, half_spread)
    rows["intraday, then reBAP"] = {**b.metrics(c_id + c_imb, day, energy, years),
                                    "intraday_cost_k_eur_per_year": float(c_id.sum() / years / 1e3),
                                    "imbalance_cost_k_eur_per_year": float(c_imb.sum() / years / 1e3),
                                    "imbalance_share": float(np.abs(d - q).sum() / energy)}
    # bound: the whole deviation known in time and traded at ID-AEP, nothing left to reBAP
    c_all, _ = b.settle(d, d, da, ida, short, long_, half_spread)
    rows["all at ID-AEP (known in time)"] = {**b.metrics(c_all, day, energy, years), "imbalance_share": 0.0}
    return rows, imb, c_id + c_imb


def monte_carlo(df, day, years, share, hourly_persistence=b.HOURLY_PERSISTENCE, sigma=b.SIGMA,
                half_spread=HALF_SPREAD, seeds=SEEDS) -> list[dict]:
    runs = {}
    for seed in range(seeds):
        rows, _, _ = run_case(df, day, years, share, hourly_persistence, sigma, half_spread, seed)
        for name, m in rows.items():
            runs.setdefault(name, []).append(m)
    out = []
    for name, ms in runs.items():
        t = pd.DataFrame(ms)
        row = {"share": share, "hourly_persistence": hourly_persistence, "sigma": sigma, "half_spread": half_spread,
               "settlement": name}
        row.update(t.mean().to_dict())
        row["cost_eur_mwh_p05"] = float(t["cost_eur_mwh"].quantile(0.05))
        row["cost_eur_mwh_p95"] = float(t["cost_eur_mwh"].quantile(0.95))
        out.append(row)
    return out


def figure_de(table: pd.DataFrame, by_decile: pd.DataFrame) -> None:
    """German version of the figure for the two-page summary in docs/."""
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for name, label, mk in (("reBAP only", "nur reBAP", "o"),
                            ("intraday, then reBAP", "mit Intraday-Korrektur", "s")):
        t = table[table.settlement == name]
        ax[0].plot(t.share, t.cost_eur_mwh, marker=mk, label=label)
        ax[1].plot(t.share, t.cvar95_daily_k_eur, marker=mk, label=label)
    ax[0].set(xlabel="Korrelation s", ylabel="EUR je MWh Verbrauch", title="Mittlere Kosten")
    ax[1].set(xlabel="Korrelation s", ylabel="Tausend EUR", title="Mittel der teuersten 5 % der Tage")
    ax[0].legend(frameon=False, fontsize=8)
    ax[2].bar(by_decile.index, by_decile["spread"], color="tab:grey")
    ax[2].set(xlabel="Dezil des nationalen Lastprognosefehlers", ylabel="EUR/MWh",
              title="reBAP minus Day-Ahead-Preis")
    ax[2].set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(FIG / "balancing_de.png", dpi=200)


def main() -> dict:
    df = load_inputs()
    local = df.index.tz_convert("Europe/Berlin")
    day = pd.factorize(local.date)[0]
    years = len(df) / (365.25 * b.PER_DAY)

    rows = []
    for share in SHARES:
        rows += monte_carlo(df, day, years, share)
    rows += monte_carlo(df, day, years, 1.0, seeds=1)          # fully system-driven: no random part
    main_table = pd.DataFrame(rows)
    main_table.to_csv(OUT / "balancing_scenarios.csv", index=False, float_format="%.4f")
    print(main_table[["share", "settlement", "cost_eur_mwh", "cost_eur_mwh_p05", "cost_eur_mwh_p95",
                      "cost_k_eur_per_year", "daily_std_k_eur", "cvar95_daily_k_eur", "imbalance_share"]].round(3).to_string())

    # sensitivities of the intraday correction at s = 0.5
    sens = []
    for hp in (0.3, 0.6, 0.9):
        for hs in (0.0, 1.0, 5.0):
            sens += monte_carlo(df, day, years, 0.5, hourly_persistence=hp, half_spread=hs, seeds=50)
    pd.DataFrame(sens).to_csv(OUT / "balancing_sensitivity.csv", index=False, float_format="%.4f")

    # placebo: the same system error moved by whole weeks keeps its distribution and loses the timing link
    z = df["z_system"].to_numpy()
    _, real, _ = run_case(df, day, years, 1.0, b.HOURLY_PERSISTENCE, b.SIGMA, HALF_SPREAD, 0)
    placebo = []
    n_weeks = len(df) // (7 * b.PER_DAY)
    for w in range(4, n_weeks - 3):
        rows_w, _, _ = run_case(df, day, years, 1.0, b.HOURLY_PERSISTENCE, b.SIGMA, HALF_SPREAD, 0,
                                z_system=np.roll(z, w * 7 * b.PER_DAY))
        placebo.append(rows_w["reBAP only"]["cost_eur_mwh"])
    placebo = np.array(placebo)
    energy = float(df["f_mwh"].sum() * 1.0)
    d_real = b.deviation(df["f_mwh"].to_numpy(), z, np.zeros(len(df)), 1.0)
    real_unit = float(real.sum() / (energy + d_real.sum()))
    pd.Series(placebo, name="cost_eur_mwh").to_csv(OUT / "balancing_placebo.csv", index=False, float_format="%.4f")

    # concentration of the cost in few days and the price link of the system error
    daily = pd.Series(real).groupby(day).sum()
    spread = (df["short"] - df["da"]).to_numpy()
    deciles = pd.qcut(z, 10, labels=False)
    by_decile = pd.DataFrame({"decile": deciles + 1, "z": z, "spread": spread}).groupby("decile").mean()
    by_decile.to_csv(OUT / "balancing_spread_by_error_decile.csv", float_format="%.3f")
    monthly = pd.Series(real, index=local).groupby(local.strftime("%Y-%m")).sum() / 1e3
    monthly.rename("cost_k_eur").to_csv(OUT / "balancing_monthly_system.csv", index_label="month", float_format="%.2f")

    summary = {
        "quarter_hours": int(len(df)), "dropped_quarter_hours": int(df.attrs["dropped"]), "days": int(day.max() + 1),
        "annual_mwh": ANNUAL_MWH, "mean_da": float(df["da"].mean()), "mean_rebap": float(df["short"].mean()),
        "std_rebap": float(df["short"].std()), "mean_idaep": float(df["idaep"].mean()),
        "std_spread_rebap": float((df["short"] - df["da"]).std()),
        "std_spread_idaep": float((df["idaep"] - df["da"]).std()),
        "corr_z_spread": float(np.corrcoef(z, spread)[0, 1]),
        "system_case_eur_mwh": real_unit, "placebo_mean": float(placebo.mean()),
        "placebo_p05": float(np.quantile(placebo, 0.05)), "placebo_p95": float(np.quantile(placebo, 0.95)),
        "placebo_share_at_or_above": float((placebo >= real_unit).mean()), "placebo_runs": int(len(placebo)),
        "acf_system_error_at_lead": float(pd.Series(z).autocorr(b.LEAD)),
        "top10_days_share_of_system_cost": float(daily.sort_values(ascending=False).head(10).sum() / daily.sum()),
    }
    (OUT / "balancing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(by_decile.round(2))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, mk in (("reBAP only", "o"), ("intraday, then reBAP", "s")):
        t = main_table[main_table.settlement == name]
        ax[0].plot(t.share, t.cost_eur_mwh, marker=mk, label=name)
        ax[1].plot(t.share, t.cvar95_daily_k_eur, marker=mk, label=name)
    ax[0].set(xlabel="correlation with the national forecast error", ylabel="EUR per MWh consumed",
              title="Mean cost of the forecast error")
    ax[1].set(xlabel="correlation with the national forecast error", ylabel="thousand EUR",
              title="Mean of the worst 5% of days")
    ax[0].legend(frameon=False)
    ax[2].bar(by_decile.index, by_decile["spread"], color="tab:grey")
    ax[2].set(xlabel="decile of the national load forecast error", ylabel="EUR/MWh",
              title="reBAP minus day-ahead price")
    fig.tight_layout()
    fig.savefig(FIG / "balancing.png", dpi=150)
    figure_de(main_table, by_decile)
    return summary


if __name__ == "__main__":
    main()
