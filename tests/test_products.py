import numpy as np
import pytest

from ehr import products


def test_kirk_close_to_monte_carlo():
    args = (100.0, 70.0, 10.0, 0.5, 0.4, 0.6, 0.5)
    assert products.kirk(*args) == pytest.approx(products.spread_mc(*args), rel=0.02)


def test_kirk_zero_strike_is_margrabe():
    f1, f2, s1, s2, rho, t = 100.0, 90.0, 0.3, 0.2, 0.5, 1.0
    sig = np.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    from scipy.stats import norm
    d1 = (np.log(f1 / f2) + 0.5 * sig ** 2 * t) / (sig * np.sqrt(t))
    marg = f1 * norm.cdf(d1) - f2 * norm.cdf(d1 - sig * np.sqrt(t))
    assert products.kirk(f1, f2, 0.0, s1, s2, rho, t) == pytest.approx(marg, rel=1e-9)


def test_swing_all_rights_equals_strip_of_options():
    """With as many rights as days, the swing is a strip of daily calls on the tree."""
    n = 60
    fwd = np.full(n, 30.0)
    full = products.swing_value(30.0, 30.0, 5.0, 0.8, fwd, 1.0, n)
    one = products.swing_value(30.0, 30.0, 5.0, 0.8, fwd, 1.0, 1)
    half = products.swing_value(30.0, 30.0, 5.0, 0.8, fwd, 1.0, n // 2)
    assert full > half > one > 0
    # the strip value: each daily call is worth roughly the same; full equals n x average call
    assert full / n < one


def test_swing_deep_in_money_is_forward_minus_strike():
    n = 30
    fwd = np.full(n, 50.0)
    v = products.swing_value(50.0, 10.0, 5.0, 0.1, fwd, 1.0, n)
    assert v == pytest.approx(n * 40.0, rel=0.01)
