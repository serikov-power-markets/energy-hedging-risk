"""Open input data, cached on disk under data/raw/.

- German day-ahead price (bidding zone DE-LU), wind, solar and grid load: SMARD (Bundesnetzagentur),
  hourly, CC BY 4.0. Weekly JSON chunks as published at smard.de/app/chart_data.
- TTF natural gas: continuous front-month future, daily closes (Yahoo Finance, symbol TTF=F).
- EU allowances: clearing prices of the EEX primary auctions, daily.
- Daily mean 2 m temperature of ten German cities (Open-Meteo historical weather API, ERA5-based).

Missing files are downloaded once; a rerun reads only the cache.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_REGION = "DE-LU"
SMARD_SERIES = {
    "price": 4169,          # day-ahead price DE-LU, EUR/MWh
    "wind_onshore": 4067,   # MWh per hour
    "wind_offshore": 1225,
    "solar": 4068,
    "load": 410,            # grid load, MWh per hour
}
TTF_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/TTF=F"
           "?period1=1508716800&period2={end}&interval=1d")
EUA_URL = ("https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
           "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
           "emission-spot-primary-market-auction-report-2012-2025-data.zip")
TEMP_FILE = "temperature_de_1991-01-01_2026-10-29.json"

_session = requests.Session()
_session.headers["User-Agent"] = "energy-hedging-risk (research; github.com/michaelserikov)"


def _get(url: str, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        r = _session.get(url, timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(url)


# ---------------------------------------------------------------- SMARD
def _smard_chunk(filter_id: int, ts: int) -> dict:
    path = RAW / "smard" / f"{filter_id}_{SMARD_REGION}" / f"{ts}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = f"{SMARD_BASE}/{filter_id}/{SMARD_REGION}/{filter_id}_{SMARD_REGION}_hour_{ts}.json"
    doc = _get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return doc


def smard_series(name: str, start: str = "2019-01-01", end: str = "2025-12-31") -> pd.Series:
    """Hourly SMARD series between start and end (dates in Europe/Berlin), index in UTC."""
    fid = SMARD_SERIES[name]
    folder = RAW / "smard" / f"{fid}_{SMARD_REGION}"
    if folder.exists():
        starts = sorted(int(p.stem) for p in folder.glob("*.json"))
    else:
        starts = _get(f"{SMARD_BASE}/{fid}/{SMARD_REGION}/index_hour.json").json()["timestamps"]
    t0 = pd.Timestamp(start, tz="Europe/Berlin").tz_convert("UTC")
    t1 = (pd.Timestamp(end, tz="Europe/Berlin") + pd.Timedelta(days=1)).tz_convert("UTC")
    week = 7 * 24 * 3600 * 1000
    rows = []
    for ts in starts:
        if ts + week < t0.value // 10**6 or ts > t1.value // 10**6:
            continue
        rows += [(t, v) for t, v in _smard_chunk(fid, ts).get("series", []) if v is not None]
    s = pd.Series(dict(rows), dtype=float)
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    s = s.sort_index()
    s = s[(s.index >= t0) & (s.index < t1)]
    s.name = name
    return s


def hourly_frame(start: str = "2019-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Price, wind, solar and load on one hourly UTC index."""
    df = pd.concat([smard_series(n, start, end) for n in SMARD_SERIES], axis=1)
    df["wind"] = df["wind_onshore"] + df["wind_offshore"]
    return df


# ---------------------------------------------------------------- fuels
def ttf_daily() -> pd.Series:
    """TTF front-month daily close, EUR/MWh, index = trading date."""
    path = RAW / "fuels" / "ttf_front_month.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(TTF_URL.format(end=int(time.time()))).content)
    res = json.loads(path.read_text())["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"], dtype=float,
                  index=pd.to_datetime(res["timestamp"], unit="s").normalize())
    s = s[~s.index.duplicated(keep="last")].dropna()
    s.name = "ttf"
    return s


def eua_daily(years: range = range(2018, 2026)) -> pd.Series:
    """EEX primary auction clearing price of EU allowances, EUR/t, by auction date."""
    path = RAW / "fuels" / "eex_eua_auctions_2012_2025.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(EUA_URL).content)
    z = zipfile.ZipFile(path)
    frames = []
    for year in years:
        name = next(n for n in z.namelist() if n.endswith((f"-{year}-data.xlsx", f"-{year}-data.xls")))
        raw = pd.read_excel(io.BytesIO(z.read(name)), header=None)
        hdr = next(i for i, v in enumerate(raw.iloc[:, 1]) if str(v).strip() == "Date")
        cols = [str(c).strip() for c in raw.iloc[hdr]]
        pc = next(i for i, c in enumerate(cols) if c.startswith("Auction Price"))
        if "Status" in cols:                     # 2020 on: cancelled auctions flagged
            df = raw.iloc[hdr + 1:, [1, pc, cols.index("Status")]]
            df.columns = ["date", "price", "status"]
            df = df[[str(v).strip().lower() == "successful" for v in df["status"]]]
        else:                                    # older files: no status column
            df = raw.iloc[hdr + 1:, [1, pc]]
            df.columns = ["date", "price"]
        df = df[pd.to_numeric(df["price"], errors="coerce").notna()]
        frames.append(df[["date", "price"]])
    out = pd.concat(frames)
    s = pd.Series(out["price"].astype(float).to_numpy(), index=pd.to_datetime(out["date"]).dt.normalize())
    s = s.groupby(level=0).mean().sort_index()
    s.name = "eua"
    return s


def temperature_daily() -> pd.Series:
    """Population-proxy mean of the daily mean temperature of ten German cities, deg C."""
    t = json.loads((RAW / "fuels" / TEMP_FILE).read_text())
    cols = {c: pd.Series(v["temperature_2m_mean"], index=pd.to_datetime(v["time"]), dtype=float)
            for c, v in t.items()}
    s = pd.DataFrame(cols).mean(axis=1)
    s.name = "temp"
    return s


def daily_base(prices: pd.Series) -> pd.Series:
    """Daily baseload average of an hourly price series (delivery day in Europe/Berlin)."""
    p = prices.tz_convert("Europe/Berlin")
    d = p.groupby(p.index.date).mean()
    d.index = pd.to_datetime(d.index)
    return d


def is_peak(index: pd.DatetimeIndex) -> np.ndarray:
    """EEX peak hours: Monday to Friday, 08:00-20:00 local time."""
    loc = index.tz_convert("Europe/Berlin")
    return (loc.dayofweek < 5) & (loc.hour >= 8) & (loc.hour < 20)
