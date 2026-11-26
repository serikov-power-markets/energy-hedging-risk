import numpy as np
import pandas as pd
import pytest

from ehr import balancing as b
from ehr import imbalance_data


def test_cost_signs_short_and_long():
    d = np.array([2.0, -2.0, 2.0, -2.0])
    da = np.full(4, 100.0)
    rebap = np.array([150.0, 150.0, 60.0, 60.0])
    cost = b.imbalance_cost(d, da, rebap, rebap)
    # short when reBAP is above day-ahead costs money, long then earns it, and the other way round
    assert cost.tolist() == [100.0, -100.0, -80.0, 80.0]


def test_covariance_split_adds_up_to_total_cost():
    rng = np.random.default_rng(1)
    d = rng.normal(0.3, 1.0, 5000)
    da = rng.normal(90.0, 20.0, 5000)
    rebap = da + 5.0 + 30.0 * d + rng.normal(0.0, 10.0, 5000)
    split = b.covariance_split(d, da, rebap)
    total = b.imbalance_cost(d, da, rebap, rebap).sum()
    assert split["covariance_eur"] + split["bias_eur"] == pytest.approx(total)


def test_deviation_correlation_equals_share():
    rng = np.random.default_rng(2)
    z_sys = b.own_error(200_000, rng, 0.5)
    z_own = b.own_error(200_000, rng, 0.5)
    d = b.deviation(np.ones(200_000), z_sys, z_own, 0.6, 0.03)
    assert np.corrcoef(d, z_sys)[0, 1] == pytest.approx(0.6, abs=0.02)
    assert d.std() == pytest.approx(0.03, abs=0.001)
    with pytest.raises(ValueError):
        b.deviation(np.ones(3), np.ones(3), np.ones(3), 1.5)


def test_own_error_persistence():
    z = b.own_error(200_000, np.random.default_rng(3), 0.9)
    assert np.corrcoef(z[:-4], z[4:])[0, 1] == pytest.approx(0.9, abs=0.01)


def test_persistence_coefficient_uses_past_days_only():
    z = b.own_error(96 * 120, np.random.default_rng(4), 0.9)
    day = np.repeat(np.arange(120), 96)
    phi = b.persistence_coefficient(z, day)
    assert phi[0] == 0.0                                   # first day: nothing known
    assert phi[-1] == pytest.approx(0.9 ** (b.LEAD / 4), abs=0.03)
    changed = z.copy()
    changed[96 * 100:] *= -3.0
    assert np.allclose(b.persistence_coefficient(changed, day)[:96 * 101], phi[:96 * 101])


def test_intraday_trade_with_zero_spread_and_equal_prices_changes_nothing():
    rng = np.random.default_rng(5)
    d, q = rng.normal(size=100), rng.normal(size=100)
    p = rng.normal(80.0, 5.0, 100)
    c_id, c_imb = b.settle(d, q, p, p + 20.0, p + 20.0, p + 20.0, 0.0)
    assert (c_id + c_imb).sum() == pytest.approx(b.imbalance_cost(d, p, p + 20.0, p + 20.0).sum())


def test_system_error_has_no_look_ahead():
    idx = pd.date_range("2025-01-01", periods=96 * 60, freq="15min", tz="UTC")
    rng = np.random.default_rng(6)
    forecast = pd.Series(100.0, index=idx)
    load = forecast * (1.02 + 0.01 * rng.standard_normal(len(idx)))
    z = b.system_error(load, forecast)
    assert abs(z.mean()) < 0.2                             # the constant bias of 2% is removed
    assert z.index[0] >= idx[0] + pd.Timedelta(days=15)


def test_parse_netztransparenz_files():
    rebap = ("Datum;Zeitzone;von;bis;Datenkategorie;Datentyp;Einheit;reBAP unterdeckt;reBAP ueberdeckt\n"
             "01.01.2025;UTC;00:15;00:30;reBAP;Q;EUR/MWh;-1.234,50;-1.234,50\n")
    r = imbalance_data.parse_rebap(rebap)
    assert r.iloc[0].tolist() == [-1234.5, -1234.5]
    assert r.index[0] == pd.Timestamp("2025-01-01 00:15", tz="UTC")
    idaep = ("Datum von;(Uhrzeit) von;Zeitzone;(Uhrzeit) bis;Zeitzone;ID AEP in EUR/MWh\n"
             "2025-01-01;00:00;UTC;00:15;UTC;141,30\n")
    assert imbalance_data.parse_idaep(idaep).iloc[0] == 141.3
