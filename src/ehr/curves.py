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
def day_class(dates: pd.DatetimeIndex) -> np.ndarray:
    """Day of week, with nationwide public holidays and 24/31 December treated as Sunday and the
    working days between Christmas and New Year as Saturday (low-demand bridge days)."""
    from .book import holidays_de
    dow = np.asarray(dates.dayofweek).copy()
    hol = {h for y in range(dates.year.min() - 1, dates.year.max() + 2) for h in holidays_de(y)}
    for i, d in enumerate(dates):
        md = (d.month, d.day)
        if d.date() in hol or md in ((12, 24), (12, 31)):
            dow[i] = 6
        elif d.month == 12 and 27 <= d.day <= 30 and dow[i] < 5:
            dow[i] = 5
    return dow


def seasonal_design(dates: pd.DatetimeIndex) -> np.ndarray:
    """Month-of-year (11 dummies) and day-class (6 dummies) design, no intercept."""
    m = np.eye(12)[dates.month - 1][:, 1:]
    w = np.eye(7)[day_class(dates)][:, 1:]
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
    w = resid.groupby(day_class(resid.index)).median().reindex(range(7)).fillna(0.0)
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

    Written with scalars (the state is two-dimensional), which is far faster in pure Python than
    small numpy arrays. Returns the log-likelihood, the filtered states (n x 2) and the last
    state covariance.
    """
    phi, mu = p.phi, p.mu
    q11, q22 = p.sigma_chi ** 2, p.sigma_xi ** 2
    q12 = p.rho * p.sigma_chi * p.sigma_xi
    R = p.sigma_eps ** 2
    c, xi = (0.0, float(np.mean(y[:30]))) if x0 is None else (float(x0[0]), float(x0[1]))
    P11, P12, P22 = 1.0, 0.0, 1.0
    ll = 0.0
    log2pi = np.log(2 * np.pi)
    out = np.empty((len(y), 2))
    for i, yi in enumerate(y.tolist()):
        c = phi * c
        xi = xi + mu
        P11, P12, P22 = phi * phi * P11 + q11, phi * P12 + q12, P22 + q22
        S = P11 + 2 * P12 + P22 + R
        v = yi - c - xi
        k1, k2 = (P11 + P12) / S, (P12 + P22) / S
        c += k1 * v
        xi += k2 * v
        h1, h2 = P11 + P12, P12 + P22
        P11, P12, P22 = P11 - k1 * h1, P12 - k1 * h2, P22 - k2 * h2
        ll += -0.5 * (log2pi + np.log(S) + v * v / S)
        out[i, 0] = c
        out[i, 1] = xi
    return float(ll), out, np.array([[P11, P12], [P12, P22]])


def fit_two_factor(y: np.ndarray, start: TwoFactor | None = None) -> TwoFactor:
    """Maximum-likelihood parameters of the two-factor model on deseasonalised log prices."""
    if start is None:
        theta0 = np.array([np.log(0.05), np.log(0.15), 0.0, np.log(0.02), 0.0, np.log(0.05)])
    else:
        theta0 = np.array([np.log(start.kappa), np.log(start.sigma_chi), start.mu,
                           np.log(start.sigma_xi), np.arctanh(np.clip(start.rho, -0.95, 0.95)),
                           np.log(max(start.sigma_eps, 1e-3))])
    bounds = [(np.log(1e-3), np.log(2.0)), (np.log(1e-3), np.log(2.0)), (-0.01, 0.01),
              (np.log(1e-4), np.log(0.5)), (-2.5, 2.5), (np.log(1e-3), np.log(1.0))]
    theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])

    def nll(theta: np.ndarray) -> float:
        return -kalman(y, _unpack(theta))[0]

    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300})
    return _unpack(res.x)


# ---------------------------------------------------------------- curve on a trading date
HEAT_RATE = 1.82   # MWh gas per MWh electric, 55% efficient CCGT
EMISSION = 0.365   # t CO2 per MWh electric of that plant


def fuel_cost(ttf: pd.Series, eua: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Short-run cost of a CCGT per MWh electric from the last known TTF and EUA prices."""
    t = ttf.reindex(ttf.index.union(days)).ffill().reindex(days)
    e = eua.reindex(eua.index.union(days)).ffill().reindex(days)
    return (HEAT_RATE * t + EMISSION * e).rename("fuel")


@dataclass
class FuelAnchor:
    alpha: float      # log level of deseasonalised spot = alpha + b * log(fuel cost)
    b: float
    kappa_m: float    # per day: decay of the spot level's deviation from the fuel level
    fuel: float       # fuel cost on the trading date, EUR/MWh

    @property
    def level(self) -> float:
        return self.alpha + self.b * np.log(self.fuel)


def fit_fuel_anchor(y: pd.Series, fuel: pd.Series) -> tuple[float, float, float]:
    """OLS of monthly mean deseasonalised log spot on monthly mean log fuel cost, and the
    mean reversion of the residual (monthly AR(1) converted to a daily rate)."""
    df = pd.DataFrame({"y": y, "f": np.log(fuel.reindex(y.index))}).dropna()
    m = df.resample("MS").mean().dropna()
    A = np.column_stack([np.ones(len(m)), m["f"]])
    (alpha, b), *_ = np.linalg.lstsq(A, m["y"].to_numpy(), rcond=None)
    r = m["y"].to_numpy() - A @ np.array([alpha, b])
    rho = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.05, 0.95)) if len(r) > 6 else 0.5
    return float(alpha), float(b), float(-np.log(rho) / 30.4)


@dataclass
class CurveModel:
    asof: pd.Timestamp
    params: TwoFactor
    beta: np.ndarray
    state: np.ndarray       # (chi, xi) on the trading date
    shape: pd.DataFrame     # hourly shape factors: index (month, daytype), columns hour 0..23
    anchor: FuelAnchor | None = None

    def daily_forward(self, days: pd.DatetimeIndex) -> pd.Series:
        """Expected daily baseload price for future delivery days.

        log F = s(T) + e^{-kappa h} chi + e^{-kappa_m h} (xi - L) + L + 0.5 V_chi
        chi decays within days (weather, outages); the deviation of the spot level xi from the
        fuel-cost level L decays over months; L follows the fuel cost on the trading date (gas
        and carbon treated as martingales: flat at their last price). Without an anchor, L = xi.
        """
        p = self.params
        h = np.asarray((days - self.asof).days, dtype=float)
        if (h <= 0).any():
            raise ValueError("delivery days must lie after the trading date")
        decay = np.exp(-p.kappa * h)
        if self.anchor is None:
            level = self.state[1]
        else:
            L = self.anchor.level
            level = L + np.exp(-self.anchor.kappa_m * h) * (self.state[1] - L)
        mean = seasonal_value(self.beta, days) + decay * self.state[0] + level
        var_chi = p.sigma_chi ** 2 * (1 - np.exp(-2 * p.kappa * h)) / (1 - np.exp(-2 * p.kappa))
        return pd.Series(np.exp(mean + 0.5 * (var_chi + p.sigma_eps ** 2)), index=days, name="forward")

    def hpfc(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Hourly price forward curve (EUR/MWh) for delivery days start..end (local dates)."""
        days = pd.date_range(start, end, freq="D")
        daily = self.daily_forward(days)
        hours = pd.date_range(start, end + pd.Timedelta(1, unit="D"), freq="h", tz="Europe/Berlin",
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


def _fuels():
    from .data import eua_daily, ttf_daily
    return ttf_daily(), eua_daily()


def build_curve(prices: pd.Series, asof: pd.Timestamp, window_years: float = 3.0,
                start: TwoFactor | None = None, fuels: tuple | None = None,
                anchored: bool = True) -> CurveModel:
    """Curve model on trading date `asof` from spot (and fuel) prices strictly before it."""
    asof = pd.Timestamp(asof).normalize()
    hist = prices[prices.index < asof.tz_localize("Europe/Berlin").tz_convert("UTC")]
    hist = hist[hist.index >= hist.index[-1] - pd.Timedelta(int(365 * window_years), unit="D")]
    from .data import daily_base
    base = daily_base(hist)
    logp = np.log(base.clip(lower=FLOOR))
    beta = fit_seasonality(logp)
    y = logp.to_numpy() - seasonal_value(beta, logp.index)
    # The likelihood has several local optima (for example a long-term level stuck at an old
    # price regime with a fast-reverting short factor). Fit from a cold start, from the previous
    # month's parameters and from a slow-reversion start, and keep the best likelihood.
    starts = [None, TwoFactor(0.02, 0.15, 0.0, 0.03, 0.0, 0.05)] + ([start] if start else [])
    fits = [fit_two_factor(y, s0) for s0 in starts]
    lls = [kalman(y, f)[0] for f in fits]
    params = fits[int(np.argmax(lls))]
    _, states, _ = kalman(y, params)
    # the state is for the last observed day; carry it to the trading date
    gap = (asof - base.index[-1]).days
    chi = states[-1, 0] * np.exp(-params.kappa * gap)
    xi = states[-1, 1]
    shape = shape_factors(hist[hist.index >= hist.index[-1] - pd.Timedelta(730, unit="D")])
    anchor = None
    if anchored:
        ttf, eua = fuels if fuels is not None else _fuels()
        ttf, eua = ttf[ttf.index < asof], eua[eua.index < asof]
        fc = fuel_cost(ttf, eua, logp.index.union(pd.DatetimeIndex([asof])))
        alpha, b, km = fit_fuel_anchor(pd.Series(y, index=logp.index), fc)
        anchor = FuelAnchor(alpha, b, km, float(fc.iloc[-1]))
    return CurveModel(asof=asof, params=params, beta=beta, state=np.array([chi, xi]), shape=shape,
                      anchor=anchor)


def daily_forward_series(prices: pd.Series, cache: dict[pd.Timestamp, CurveModel],
                         start: str = "2020-02-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Modelled front-month and next-calendar-year baseload forwards on every business day.

    For each month the parameters, seasonality and fuel regression of that month's cached fit are
    kept; the Kalman filter runs over spot days before d and the fuel cost is that of day d (the
    curve is marked at the close of d), so no information after d enters.
    """
    from .data import daily_base
    base = daily_base(prices)
    logp = np.log(base.clip(lower=FLOOR))
    ttf, eua = _fuels()
    days = pd.bdate_range(start, end)
    fc = fuel_cost(ttf, eua, days)   # curve marked at the close of day d: fuel closes of d
    keys = sorted(cache)
    rows = []
    for i, d in enumerate(days):
        k = max((x for x in keys if x <= d), default=None)
        if k is None:
            continue
        m = cache[k]
        hist = logp[(logp.index < d) & (logp.index >= d - pd.Timedelta(365 * 3, unit="D"))]
        y = hist.to_numpy() - seasonal_value(m.beta, hist.index)
        _, st, _ = kalman(y, m.params)
        gap = (d - hist.index[-1]).days
        anchor = None
        if m.anchor is not None:
            anchor = FuelAnchor(m.anchor.alpha, m.anchor.b, m.anchor.kappa_m, float(fc.iloc[i]))
        cm = CurveModel(asof=d, params=m.params, beta=m.beta,
                        state=np.array([st[-1, 0] * np.exp(-m.params.kappa * gap), st[-1, 1]]),
                        shape=m.shape, anchor=anchor)
        m1 = d + pd.offsets.MonthBegin(1)
        cal = pd.date_range(f"{d.year + 1}-01-01", f"{d.year + 1}-12-31")
        rows.append((d, float(cm.daily_forward(pd.date_range(m1, m1 + pd.offsets.MonthEnd(0))).mean()),
                     float(cm.daily_forward(cal).mean())))
    return pd.DataFrame(rows, columns=["date", "power_m1", "power_cal1"]).set_index("date")
