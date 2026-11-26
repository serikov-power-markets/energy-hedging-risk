"""Quarter-hour data for the balancing-group study, cached under data/raw/.

- reBAP (uniform German imbalance price) and ID-AEP (volume-weighted intraday price of each quarter hour):
  netztransparenz.de WebAPI of the transmission system operators. Free client credentials from
  api-portal.netztransparenz.de, read from NTP_CLIENT_ID and NTP_CLIENT_SECRET; only needed for months
  that are not cached.
- Grid load, day-ahead load forecast and day-ahead price in quarter hours: SMARD (Bundesnetzagentur), CC BY 4.0.
"""

from __future__ import annotations

import io
import json
import os
import time

import pandas as pd
import requests

from .data import RAW, SMARD_BASE, _get

TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
NTP_BASE = "https://ds.netztransparenz.de/api/v1/data"
NTP_PATHS = {"rebap": "NrvSaldo/reBAP/Qualitaetsgesichert", "idaep": "IdAep"}
SMARD_QH = {"load": (410, "DE"), "load_forecast": (411, "DE"), "price": (4169, "DE-LU")}


def _german_number(s: pd.Series) -> pd.Series:
    """'1.234,56' -> 1234.56; empty or text -> NaN."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                         errors="coerce")


def parse_rebap(text: str) -> pd.DataFrame:
    """Columns short (paid by a short group) and long (paid to a long group), index = quarter start in UTC."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    if not (df["Zeitzone"].str.upper() == "UTC").all():
        raise ValueError("expected UTC timestamps")
    idx = pd.to_datetime(df["Datum"] + " " + df["von"], format="%d.%m.%Y %H:%M", utc=True)
    short = next(c for c in df.columns if "unterdeckt" in c.lower())
    long_ = next(c for c in df.columns if "berdeckt" in c.lower() and c != short)
    out = pd.DataFrame({"short": _german_number(df[short]).to_numpy(), "long": _german_number(df[long_]).to_numpy()},
                       index=idx)
    return out[~out.index.duplicated()].sort_index()


def parse_idaep(text: str) -> pd.Series:
    """ID-AEP series in EUR/MWh, index = quarter start in UTC (the file uses a decimal comma)."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    start = df.columns[1]
    idx = pd.to_datetime(df[df.columns[0]] + " " + df[start], format="%Y-%m-%d %H:%M", utc=True)
    s = pd.Series(_german_number(df[df.columns[-1]]).to_numpy(), index=idx, name="idaep")
    return s[~s.index.duplicated()].sort_index()


def _token() -> str:
    cid, secret = os.environ.get("NTP_CLIENT_ID"), os.environ.get("NTP_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError("set NTP_CLIENT_ID and NTP_CLIENT_SECRET to download months that are not cached")
    r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": cid,
                                       "client_secret": secret}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def _ntp_month(kind: str, month: pd.Timestamp) -> str:
    path = RAW / "netztransparenz" / kind / f"{month:%Y-%m}.csv"
    if path.exists():
        return path.read_text(encoding="utf-8")
    end = month + pd.offsets.MonthBegin(1)
    url = f"{NTP_BASE}/{NTP_PATHS[kind]}/{month:%Y-%m-%dT00:00:00}/{end:%Y-%m-%dT00:00:00}"
    for attempt in range(4):
        r = requests.get(url, headers={"Authorization": f"Bearer {_token()}"}, timeout=120)
        if r.status_code == 200:
            break
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def ntp_series(kind: str, start: str, end: str):
    months = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS")
    parse = parse_rebap if kind == "rebap" else parse_idaep
    out = pd.concat([parse(_ntp_month(kind, m)) for m in months])
    return out[~out.index.duplicated()].sort_index()


def _download_json(url: str, attempts: int = 6) -> dict:
    """GET with retries on dropped connections and server errors; the weekly cache makes a rerun resume."""
    for attempt in range(attempts):
        try:
            return _get(url).json()
        except (requests.RequestException, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(url)


def smard_quarter(name: str, start: str, end: str) -> pd.Series:
    """Quarter-hour SMARD series between two dates (Europe/Berlin), index in UTC."""
    fid, region = SMARD_QH[name]
    folder = RAW / "smard_quarterhour" / f"{fid}_{region}"
    folder.mkdir(parents=True, exist_ok=True)
    t0 = pd.Timestamp(start, tz="Europe/Berlin").tz_convert("UTC")
    t1 = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("Europe/Berlin").tz_convert("UTC")
    index_file = folder / f"index_{start}_{end}.json"
    if index_file.exists():
        stamps = json.loads(index_file.read_text())
    else:
        stamps = _download_json(f"{SMARD_BASE}/{fid}/{region}/index_quarterhour.json")["timestamps"]
    week = 7 * 24 * 3600 * 1000
    stamps = [ts for ts in stamps if ts + week >= t0.value // 10**6 and ts < t1.value // 10**6]
    index_file.write_text(json.dumps(stamps))
    rows = []
    for ts in stamps:
        path = folder / f"{ts}.json"
        if not path.exists():
            url = f"{SMARD_BASE}/{fid}/{region}/{fid}_{region}_quarterhour_{ts}.json"
            path.write_text(json.dumps(_download_json(url)))
        rows += [(t, v) for t, v in json.loads(path.read_text()).get("series", []) if v is not None]
    s = pd.Series(dict(rows), dtype=float)
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    s = s.sort_index()
    s.name = name
    return s[(s.index >= t0) & (s.index < t1)]
