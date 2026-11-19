import numpy as np
import pandas as pd
import pytest

from ehr import retail


def test_tariff_components_add_up_and_vat_is_applied():
    p = retail.tariff(100.0)
    parts = {k: v for k, v in p.items() if k not in ("VAT", "gross price")}
    assert p["energy procurement"] == pytest.approx(10.0)
    assert p["VAT"] == pytest.approx(0.19 * sum(parts.values()))
    assert p["gross price"] == pytest.approx(1.19 * sum(parts.values()))
    assert retail.tariff(150.0)["gross price"] - p["gross price"] == pytest.approx(1.19 * 5.0)


def test_churn_is_base_at_zero_gap_and_rises_with_price():
    assert retail.churn_probability(30, 30, base=0.1) == pytest.approx(0.1)
    assert retail.churn_probability(32, 30) > retail.churn_probability(30, 30) > retail.churn_probability(28, 30)


def test_optimal_price_responds_to_switching_sensitivity_and_cost():
    p_lo, _ = retail.optimal_price(20.0, 30.0, sensitivity=0.15)
    p_hi, _ = retail.optimal_price(20.0, 30.0, sensitivity=0.8)
    assert p_hi < p_lo                                    # customers that react to price push the optimum down
    assert retail.optimal_price(24.0, 30.0)[0] > retail.optimal_price(20.0, 30.0)[0]
    p, m = retail.optimal_price(20.0, 30.0)
    assert m >= retail.expected_margin(p + 1.0, 20.0, 30.0) and m >= retail.expected_margin(p - 1.0, 20.0, 30.0)


def test_limit_report_statuses_and_limit_design():
    var = pd.Series([50.0, 85.0, 100.0, 120.0])
    r = retail.limit_report(var, 100.0)
    assert r["status"].tolist() == ["green", "amber", "amber", "red"]
    assert retail.limit_from_history(pd.Series(np.arange(1, 101.0)), 0.9, 1.25) == pytest.approx(1.25 * 90.1)


def test_forward_exposure_is_zero_without_volatility_and_cva_scales_with_pd_and_recovery():
    flat = retail.forward_exposure(60.0, 0.0, 3)
    assert (flat["ee_eur"] == 0).all()
    e = retail.forward_exposure(60.0, 0.4, 3, n_sim=4000, seed=1)
    assert (e["ee_eur"].iloc[:-1] > 0).all() and e["ee_eur"].iloc[-1] == 0 and e["pfe95_eur"].iloc[0] >= e["ee_eur"].iloc[0]
    c1, c2 = retail.cva(e, 0.002), retail.cva(e, 0.02)
    assert 5 < c2 / c1 < 12                               # roughly proportional to the default probability
    assert retail.cva(e, 0.002, recovery=0.0) == pytest.approx(c1 / 0.6)
    assert retail.cva(retail.forward_exposure(60.0, 0.4, 3, n_sim=4000, seed=1, threshold=1e6), 0.002) < c1


def test_cva_matches_a_closed_form_for_one_period_exposure():
    ex = pd.DataFrame({"t": [1.0], "ee_eur": [1000.0]})
    assert retail.cva(ex, 0.01, 0.4) == pytest.approx(0.6 * 1000 * 0.01)
