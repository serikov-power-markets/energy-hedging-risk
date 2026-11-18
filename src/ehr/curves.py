"""Modelled forward curves for German power.

Exchange settlement history (EEX Phelix DE futures) is not openly available, so forward prices on
each past trading date are built from the information available on that date only:

1. Two-factor model of the log daily baseload spot price (Schwartz and Smith 2000):
       log P_d = s(d) + chi_d + xi_d + eps_d
   chi: short-term deviation, AR(1) with mean reversion kappa per day,
   xi:  long-term level, random walk with drift mu,
   s:   deterministic seasonality (month of year, day of week), estimated on the trailing window.
   Parameters by maximum likelihood with a Kalman filter on the data up to the trading date.
2. Forward for a delivery day T seen from t:
       F(t, T) = exp(s(T) + e^{-kappa (T-t)} chi_t + xi_t + 0.5 V_chi(t, T))
   with the long-term level a martingale in price terms (no extrapolated drift)
   averaged over the delivery days of a product (month, quarter, calendar year).
3. Peak and off-peak forwards and the hourly price forward curve (HPFC) apply hourly shape factors
   (month x day type x hour, relative to the daily base) from the trailing two years.

The curves contain no risk premium and no information beyond past spot prices. They track the
direction of the market, not exchange levels: in 2022 the exchange priced a crisis premium that a
spot-based model cannot reproduce. Results that depend on forward levels are illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FLOOR = 5.0  # EUR/MWh, floor before taking logs (a few days with near-zero or negative baseload)


# ---------------------------------------------------------------- seasonality
def seasonal_design(dates: pd.DatetimeIndex) -> np.ndarray:
    """Month-of-year (11 dummies) and day-of-week (6 dummies) design, no intercept."""
    m = np.eye(12)[dates.month - 1][:, 1:]
    w = np.eye(7)[dates.dayofweek][:, 1:]
    return np.hstack([m, w])


def fit_seasonality(logp: pd.Series) -> np.ndarray:
    """Seasonal coefficients of log price relative to its centred one-year moving average.

    Detrending by the moving average keeps a level shock (such as the 2022 gas crisis, which peaked
    in late summer) out of the month-of-year pattern; a linear trend would not. Coefficients are
    medians of the detrended log price by month and day of week, centred to average zero.
    """
    trend = logp.rolling(365, center=True, min_periods=120).mean()
    rel = (logp - trend).dropna()
    m = rel.groupby(rel.index.month).median().reindex(range(1, 13)).fillna(0.0)
    resid = rel - m.reindex(rel.index.month).to_numpy()
    w = resid.groupby(resid.index.dayofweek).median().reindex(range(7)).fillna(0.0)
    beta = np.concatenate([(m - m.iloc[0]).to_numpy()[1:], (w - w.iloc[0]).to_numpy()[1:]])
    offset = (seasonal_design(pd.date_range("2021-01-01", "2021-12-31")) @ beta).mean()
    return np.append(beta, offset)   # last element: offset that centres the yearly mean on zero


def seasonal_value(beta: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Seasonal log component; beta holds 17 dummy coefficients and the centring offset."""
    return seasonal_design(dates) @ beta[:-1] - beta[-1]


# ---------------------------------------------------------------- two-factor Kalman filter
@dataclass
class TwoFactor:
    kappa: float      # mean reversion of chi, per day
    sigma_chi: float  # daily vol of chi
    mu: float         # drift of xi, per day
    sigma_xi: float   # daily vol of xi
    rho: float        # correlation of the two shocks
    sigma_eps: float  # measurement noise

    @property
    def phi(self) -> float:
        return float(np.exp(-self.kappa))

    def Q(self) -> np.ndarray:
        q12 = self.rho * self.sigma_chi * self.sigma_xi
        return np.array([[self.sigma_chi ** 2, q12], [q12, self.sigma_xi ** 2]])


def _unpack(theta: np.ndarray) -> TwoFactor:
    return TwoFactor(kappa=float(np.exp(theta[0])), sigma_chi=float(np.exp(theta[1])), mu=float(theta[2]),
                     sigma_xi=float(np.exp(theta[3])), rho=float(np.tanh(theta[4])),
                     sigma_eps=float(np.exp(theta[5])))


def kalman(y: np.ndarray, p: TwoFactor, x0: np.ndarray | None = None
           ) -> tuple[float, np.ndarray, np.ndarray]:
    """Kalman filter for y_t = chi_t + xi_t + eps_t (seasonality already removed).

    Returns the log-likelihood, the filtered states (n x 2) and the last state covariance.
    """
    F = np.array([[p.phi, 0.0], [0.0, 1.0]])
    c = np.array([0.0, p.mu])
    H = np.array([1.0, 1.0])
    Q = p.Q()
    R = p.sigma_eps ** 2
    x = np.array([0.0, y[:30].mean()]) if x0 is None else x0
    P = np.diag([1.0, 1.0])
    ll = 0.0
    out = np.empty((len(y), 2))
    for i, yi in enumerate(y):
        x = F @ x + c
        P = F @ P @ F.T + Q
        S = H @ P @ H + R
        v = yi - H @ x
        K = P @ H / S
        x = x + K * v
        P = P - np.outer(K, H @ P)
        ll += -0.5 * (np.log(2 * np.pi * S) + v * v / S)
        out[i] = x
    return ll, out, P


def fit_two_factor(y: np.ndarray, start: TwoFactor | None = None) -> TwoFactor:
    """Maximum-likelihood parameters of the two-factor model on deseasonalised log prices."""
    if start is None:
        theta0 = np.array([np.log(0.05), np.log(0.15), 0.0, np.log(0.02), 0.0, np.log(0.05)])
    else:
        theta0 = np.array([np.log(start.kappa), np.log(start.sigma_chi), start.mu,
                           np.log(start.sigma_xi), np.arctanh(np.clip(start.rho, -0.95, 0.95)),
                           np.log(start.sigma_eps)])

    def nll(theta: np.ndarray) -> float:
        if not (np.log(1e-3) < theta[0] < np.log(2.0)):
            return 1e10
        return -kalman(y, _unpack(theta))[0]

    res = minimize(nll, theta0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-3})
    return _unpack(res.x)


# ---------------------------------------------------------------- curve on a trading date
@dataclass
class CurveModel:
    asof: pd.Timestamp
    params: TwoFactor
    beta: np.ndarray
    state: np.ndarray       # (chi, xi) on the trading date
    shape: pd.DataFrame     # hourly shape factors: index (month, daytype), columns hour 0..23

    def daily_forward(self, days: pd.DatetimeIndex) -> pd.Series:
        """Expected daily baseload price for future delivery days."""
        p = self.params
        h = np.asarray((days - self.asof).days, dtype=float)
        if (h <= 0).any():
            raise ValueError("delivery days must lie after the trading date")
        decay = np.exp(-p.kappa * h)
        # The long-term level is treated as a martingale in price terms: no drift is extrapolated
        # (a drift fitted on a few years of history is mostly noise) and its convexity term is
        # dropped, so the forward does not grow with maturity by construction. The short-term
        # factor keeps its lognormal mean correction.
        mean = seasonal_value(self.beta, days) + decay * self.state[0] + self.state[1]
        var_chi = p.sigma_chi ** 2 * (1 - np.exp(-2 * p.kappa * h)) / (1 - np.exp(-2 * p.kappa))
        return pd.Series(np.exp(mean + 0.5 * (var_chi + p.sigma_eps ** 2)), index=days, name="forward")

    def hpfc(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Hourly price forward curve (EUR/MWh) for delivery days start..end (local dates)."""
        days = pd.date_range(start, end, freq="D")
        daily = self.daily_forward(days)
        hours = pd.date_range(start, end + pd.Timedelta(days=1), freq="h", tz="Europe/Berlin",
                              inclusive="left")
        loc_day = hours.normalize().tz_localize(None)
        dt = _daytype(hours)
        f = self.shape.reindex(pd.MultiIndex.from_arrays([hours.month, dt])).to_numpy()
        factors = f[np.arange(len(hours)), hours.hour]
        return pd.Series(daily.reindex(loc_day).to_numpy() * factors, index=hours.tz_convert("UTC"),
                         name="hpfc")

    def product(self, start: pd.Timestamp, end: pd.Timestamp, kind: str = "base") -> float:
        """Forward price of a delivery period (base, peak or offpeak), average of the HPFC hours."""
        curve = self.hpfc(start, end)
        if kind == "base":
            return float(curve.mean())
        from .data import is_peak
        pk = is_peak(curve.index)
        return float(curve[pk].mean() if kind == "peak" else curve[~pk].mean())


def _daytype(index: pd.DatetimeIndex) -> np.ndarray:
    """0 = working day, 1 = Saturday, 2 = Sunday (public holidays not separated)."""
    dow = index.dayofweek
    return np.where(dow < 5, 0, np.where(dow == 5, 1, 2))


def shape_factors(prices: pd.Series) -> pd.DataFrame:
    """Hourly price relative to its daily mean, averaged by (month, day type) and local hour."""
    p = prices.tz_convert("Europe/Berlin")
    day = p.index.normalize()
    rel = p / p.groupby(day).transform("mean").clip(lower=FLOOR)
    df = pd.DataFrame({"rel": rel.to_numpy(), "month": p.index.month, "dt": _daytype(p.index),
                       "hour": p.index.hour})
    tab = df.pivot_table(index=["month", "dt"], columns="hour", values="rel", aggfunc="median")
    tab = tab.div(tab.mean(axis=1), axis=0)   # each daily shape averages one
    return tab.reindex(columns=range(24)).ffill(axis=1).bfill(axis=1)


def build_curve(prices: pd.Series, asof: pd.Timestamp, window_years: float = 3.0,
                start: TwoFactor | None = None) -> CurveModel:
    """Curve model on trading date `asof` from the hourly spot prices strictly before it."""
    asof = pd.Timestamp(asof).normalize()
    hist = prices[prices.index < asof.tz_localize("Europe/Berlin").tz_convert("UTC")]
    hist = hist[hist.index >= hist.index[-1] - pd.Timedelta(days=int(365 * window_years))]
    from .data import daily_base
    base = daily_base(hist)
    logp = np.log(base.clip(lower=FLOOR))
    beta = fit_seasonality(logp)
    y = logp.to_numpy() - seasonal_value(beta, logp.index)
    params = fit_two_factor(y, start)
    _, states, _ = kalman(y, params)
    # the state is for the last observed day; carry it to the trading date
    gap = (asof - base.index[-1]).days
    chi = states[-1, 0] * np.exp(-params.kappa * gap)
    xi = states[-1, 1] + params.mu * gap
    shape = shape_factors(hist[hist.index >= hist.index[-1] - pd.Timedelta(days=730)])
    return CurveModel(asof=asof, params=params, beta=beta, state=np.array([chi, xi]), shape=shape)
