import numpy as np
import pandas as pd
import pytest

from ehr import curves, hedging
from ehr.data import is_peak


def _hours(year=2023):
    return hedging.period_hours(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))


def test_period_hours_handles_dst():
    assert len(_hours(2023)) == 8760
    assert len(_hours(2024)) == 8784


def test_block_split_reproduces_block_profile():
    h = _hours()
    prof = pd.Series(np.where(is_peak(h), 30.0, 10.0), index=h)
    base, peak = hedging.block_split(prof, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
    assert base == pytest.approx(10.0)
    assert peak == pytest.approx(20.0)


def test_hedge_position_sums_trades():
    h = _hours()
    d = pd.Timestamp("2022-10-03")
    trades = [hedging.Trade(d, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), "base", 10),
              hedging.Trade(d, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), "peak", 5)]
    pos = hedging.hedge_position(trades, h)
    pk = is_peak(h)
    assert (pos[pk] == 15).all() and (pos[~pk] == 10).all()


def test_settle_full_hedge_at_constant_price_is_flat():
    """Fully hedged block profile at forward = spot = constant: cost equals that price, P&L zero."""
    h = _hours()
    prof = pd.Series(np.where(is_peak(h), 30.0, 10.0), index=h)
    spot = pd.Series(80.0, index=h)
    d = pd.Timestamp("2022-10-03")
    y0, y1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    trades = [hedging.Trade(d, y0, y1, "base", 10), hedging.Trade(d, y0, y1, "peak", 20)]
    r = hedging.settle(trades, np.array([80.0, 80.0]), prof, spot, sales_price=80.0)
    assert r["cost_eur_mwh"] == pytest.approx(80.0)
    assert r["pnl_eur_m"] == pytest.approx(0.0, abs=1e-9)
    assert r["open_volume_share"] == pytest.approx(0.0, abs=1e-12)


def test_settle_hedge_gain_offsets_spot_move():
    """With a full hedge, a spot move changes the hedge gain but not the cost."""
    h = _hours()
    prof = pd.Series(20.0, index=h)
    d = pd.Timestamp("2022-10-03")
    y0, y1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    trades = [hedging.Trade(d, y0, y1, "base", 20)]
    lo = hedging.settle(trades, np.array([100.0]), prof, pd.Series(50.0, index=h), 100.0)
    hi = hedging.settle(trades, np.array([100.0]), prof, pd.Series(300.0, index=h), 100.0)
    assert lo["cost_eur_mwh"] == pytest.approx(100.0) and hi["cost_eur_mwh"] == pytest.approx(100.0)
    assert hi["hedge_gain_eur_m"] > 0 > lo["hedge_gain_eur_m"]


def test_seasonality_is_centred():
    idx = pd.date_range("2019-01-01", "2023-12-31")
    rng = np.random.default_rng(0)
    logp = pd.Series(4 + 0.2 * np.cos(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 0.05, len(idx)),
                     index=idx)
    beta = curves.fit_seasonality(logp)
    year = pd.date_range("2021-01-01", "2021-12-31")
    assert curves.seasonal_value(beta, year).mean() == pytest.approx(0.0, abs=1e-12)
    jan = curves.seasonal_value(beta, pd.DatetimeIndex(["2021-01-15"]))[0]
    jul = curves.seasonal_value(beta, pd.DatetimeIndex(["2021-07-15"]))[0]
    assert jan - jul == pytest.approx(0.4, abs=0.05)


def test_kalman_recovers_mean_reversion():
    rng = np.random.default_rng(1)
    n, kappa = 1500, 0.3
    chi = np.zeros(n); xi = np.zeros(n)
    for t in range(1, n):
        chi[t] = np.exp(-kappa) * chi[t - 1] + rng.normal(0, 0.2)
        xi[t] = xi[t - 1] + rng.normal(0, 0.01)
    y = 4 + chi + xi + rng.normal(0, 0.02, n)
    p = curves.fit_two_factor(y)
    assert p.kappa == pytest.approx(kappa, rel=0.35)
    assert p.sigma_chi == pytest.approx(0.2, rel=0.25)


def test_pnl_explain_adds_up():
    h = _hours()
    rng = np.random.default_rng(3)
    exp = pd.Series(np.where(is_peak(h), 30.0, 10.0), index=h)
    act = exp * (1 + rng.normal(0, 0.05, len(h)))
    spot = pd.Series(rng.lognormal(4.4, 0.4, len(h)), index=h)
    d = pd.Timestamp("2022-10-03")
    y0, y1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    trades = [hedging.Trade(d, y0, y1, "base", 8), hedging.Trade(d, y0, y1, "peak", 15)]
    r = hedging.settle(trades, np.array([85.0, 95.0]), act, spot, 90.0, expected=exp)
    parts = r["pnl_level_eur_m"] + r["pnl_shape_eur_m"] + r["pnl_volume_eur_m"] + r["pnl_hedge_eur_m"]
    assert parts == pytest.approx(r["pnl_eur_m"], abs=1e-9)
