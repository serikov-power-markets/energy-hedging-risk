"""Value at Risk and Expected Shortfall of a linear commodity portfolio, with backtests.

VaR and ES are reported as positive loss numbers for a one-day horizon. Methods:
- hs:   historical simulation over a rolling window of past P&L of today's positions
- ewma: parametric normal with RiskMetrics EWMA covariance (lambda 0.94)
- fhs:  filtered historical simulation: past factor returns rescaled by EWMA vol ratio
- mct:  Monte Carlo with a multivariate Student t (EWMA covariance, dof fitted on the window)
Backtests: Kupiec proportion of failures, Christoffersen independence and conditional coverage,
Acerbi-Szekely Z2 for ES (simulated critical value under the null of the model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LAMBDA = 0.94


def ewma_cov(x: np.ndarray, lam: float = LAMBDA, init: int = 30) -> np.ndarray:
    """EWMA covariance after each row of x (n x k); entry i uses rows up to and including i."""
    n, k = x.shape
    out = np.empty((n, k, k))
    s = np.cov(x[:init].T).reshape(k, k) if n > init else np.eye(k) * np.var(x)
    for i in range(n):
        s = lam * s + (1 - lam) * np.outer(x[i], x[i])
        out[i] = s
    return out


def var_es(pnl: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR and ES (positive losses) at confidence alpha of a P&L sample."""
    q = np.quantile(pnl, 1 - alpha)
    tail = pnl[pnl <= q]
    return float(-q), float(-tail.mean())


def rolling_var(dx: pd.DataFrame, pos: np.ndarray, alpha: float = 0.99, window: int = 250,
                method: str = "hs", n_sim: int = 20000, seed: int = 0) -> pd.DataFrame:
    """One-day VaR and ES forecasts for each day after the first window, from data before that day.

    dx: daily changes of the risk factors (price units), pos: positions (units per price unit),
    so the portfolio P&L of day t is dx_t @ pos.
    """
    rng = np.random.default_rng(seed)
    x = dx.to_numpy()
    cov = ewma_cov(x)
    rows = []
    for t in range(window, len(x)):
        hist = x[t - window:t]
        if method == "hs":
            v, e = var_es(hist @ pos, alpha)
        elif method == "ewma":
            sd = float(np.sqrt(pos @ cov[t - 1] @ pos))
            z = stats.norm.ppf(alpha)
            v, e = z * sd, sd * stats.norm.pdf(z) / (1 - alpha)
        elif method == "fhs":
            prev = np.maximum(np.arange(t - window, t) - 1, 0)   # vol forecast for each past day
            vol_hist = np.sqrt(np.einsum("ijj->ij", cov[prev]))
            vol_now = np.sqrt(np.diag(cov[t - 1]))
            scaled = hist / np.where(vol_hist > 0, vol_hist, 1) * vol_now
            v, e = var_es(scaled @ pos, alpha)
        elif method == "mct":
            p_hist = hist @ pos
            nu = float(np.clip(stats.t.fit(p_hist / p_hist.std(), floc=0)[0], 2.5, 30))
            c = cov[t - 1]
            g = rng.multivariate_normal(np.zeros(len(pos)), c, size=n_sim)
            w = rng.chisquare(nu, size=(n_sim, 1)) / (nu - 2)   # unit-variance t
            v, e = var_es((g / np.sqrt(w)) @ pos, alpha)
        else:
            raise ValueError(method)
        rows.append((dx.index[t], v, e, float(x[t] @ pos)))
    return pd.DataFrame(rows, columns=["date", "var", "es", "pnl"]).set_index("date")


# ---------------------------------------------------------------- backtests
def kupiec(hits: np.ndarray, alpha: float) -> dict:
    """Proportion-of-failures likelihood-ratio test; hits = 1 when loss exceeds VaR."""
    n, x = len(hits), int(hits.sum())
    p = 1 - alpha
    phat = x / n
    def ll(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)
    lr = -2 * (ll(p) - ll(phat))
    return {"n": n, "exceptions": x, "expected": n * p, "rate": phat, "lr_pof": lr,
            "p_pof": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen(hits: np.ndarray, alpha: float) -> dict:
    """Independence test of exceptions (first-order Markov) and conditional coverage."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1); pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def lg(p, k0, k1):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return k0 * np.log(1 - p) + k1 * np.log(p)
    lr_ind = -2 * (lg(pi, n00 + n10, n01 + n11) - lg(pi0, n00, n01) - lg(pi1, n10, n11))
    lr_cc = kupiec(hits, alpha)["lr_pof"] + lr_ind
    return {"n11": n11, "lr_ind": lr_ind, "p_ind": float(1 - stats.chi2.cdf(lr_ind, 1)),
            "lr_cc": lr_cc, "p_cc": float(1 - stats.chi2.cdf(lr_cc, 2))}


def acerbi_z2(df: pd.DataFrame, alpha: float) -> float:
    """Acerbi-Szekely Z2 statistic: 0 under a correct ES, negative when ES is underestimated."""
    hit = (df["pnl"] < -df["var"]).astype(float)
    return float((df["pnl"] * hit / ((1 - alpha) * len(df) * df["es"])).sum() + 1)


def backtest(df: pd.DataFrame, alpha: float) -> dict:
    hits = (df["pnl"] < -df["var"]).to_numpy()
    return {**kupiec(hits, alpha), **christoffersen(hits, alpha), "z2": acerbi_z2(df, alpha),
            "mean_var": float(df["var"].mean()), "mean_es": float(df["es"].mean())}


def traffic_light(exceptions: int, n: int = 250) -> str:
    """Basel traffic light for 99% VaR over 250 days (green 0-4, yellow 5-9, red 10+), scaled to n."""
    x = exceptions * 250 / n
    return "green" if x < 5 else ("yellow" if x < 10 else "red")
