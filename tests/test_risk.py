import numpy as np
import pandas as pd
import pytest

from ehr import risk


def test_kupiec_at_expected_rate_does_not_reject():
    hits = np.zeros(1000, dtype=bool)
    hits[::100] = True                      # exactly 1%
    r = risk.kupiec(hits, 0.99)
    assert r["exceptions"] == 10 and r["p_pof"] > 0.9


def test_kupiec_rejects_too_many():
    hits = np.zeros(500, dtype=bool)
    hits[:25] = True
    assert risk.kupiec(hits, 0.99)["p_pof"] < 0.001


def test_christoffersen_detects_clustering():
    hits = np.zeros(1000, dtype=bool)
    hits[100:110] = True                    # ten exceptions in a row
    assert risk.christoffersen(hits, 0.99)["p_ind"] < 0.001


def test_normal_var_matches_theory():
    rng = np.random.default_rng(0)
    dx = pd.DataFrame(rng.normal(0, 1, (3000, 1)), index=pd.RangeIndex(3000))
    out = risk.rolling_var(dx, np.array([1.0]), 0.99, 250, "ewma")
    assert out["var"].mean() == pytest.approx(2.326, rel=0.08)
    assert out["es"].mean() == pytest.approx(2.665, rel=0.08)
    bt = risk.backtest(out, 0.99)
    assert bt["p_pof"] > 0.01 and abs(bt["z2"]) < 0.5


def test_hs_var_es_ordering():
    rng = np.random.default_rng(1)
    v, e = risk.var_es(rng.standard_t(4, 100000), 0.99)
    assert e > v > 0
