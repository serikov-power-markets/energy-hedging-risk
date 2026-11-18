"""Guarantees of origin: results of the French state auctions run by EEX (open files).

Each monthly auction sells French guarantees of origin of one production month by region and
technology; the files give the volume sold and the volume-weighted average price (EUR/MWh).
The French auctions are one public price point for European GoOs; German GoOs trade bilaterally or
on EEX futures, whose history is not openly available.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from .data import RAW

BASE = ("https://www.eex.com/fileadmin/EEX/Downloads/Registry_Services/"
        "French_Auctions_for_Guarantees_of_Origin/")
FILES = ["GO_2019-2023_Global_Results.zip", "20251119_GO_2024_2025_GLOBAL_Results.zip",
         "20261022_GO_2026_GLOBAL_Results.zip"]
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}
TECH = {"Eolien onshore": "wind_onshore", "Eolien offshore": "wind_offshore", "Hydraulique": "hydro",
        "Solaire": "solar", "Thermique": "thermal"}


def _files() -> list[Path]:
    out = []
    for f in FILES:
        p = RAW / "goo" / f
        if not p.exists():
            from .data import _get
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_get(BASE + f).content)
        out.append(p)
    return out


def _parse(name: str, raw: bytes) -> pd.DataFrame:
    base = name.rsplit("/", 1)[-1]
    m = re.match(r"(\d{8})_([A-Za-z]+)_(\d{4})", base)
    if not m:
        return pd.DataFrame()
    auction = pd.Timestamp(m.group(1))
    prod = pd.Timestamp(year=int(m.group(3)), month=MONTHS[m.group(2)], day=1)
    df = pd.read_excel(io.BytesIO(raw), header=None)
    hdr = next(i for i, v in enumerate(df.iloc[:, 1]) if "Techno" in str(v))
    body = df.iloc[hdr + 1:, [0, 1, 3, 4]]
    body.columns = ["region", "tech", "volume", "price"]
    body = body[body["tech"].isin(TECH)]
    body["volume"] = pd.to_numeric(body["volume"], errors="coerce")
    body["price"] = pd.to_numeric(body["price"], errors="coerce")
    body = body.dropna(subset=["volume", "price"])
    body = body[body["volume"] > 0]
    body["tech"] = body["tech"].map(TECH)
    body["auction"] = auction
    body["production_month"] = prod
    return body


def auctions() -> pd.DataFrame:
    """All rows (region x technology x auction) with volume sold and weighted price."""
    frames = []
    seen = set()
    for p in _files():
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            base = n.rsplit("/", 1)[-1]
            if not base.endswith(".xlsx") or base in seen:
                continue
            seen.add(base)
            frames.append(_parse(n, z.read(n)))
    return pd.concat(frames, ignore_index=True)


def monthly_price(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Volume-weighted price by auction date and technology (EUR/MWh), plus all technologies."""
    df = auctions() if df is None else df
    df = df.assign(value=df["volume"] * df["price"])
    by = df.groupby(["auction", "tech"])[["value", "volume"]].sum()
    tab = (by["value"] / by["volume"]).unstack("tech")
    tot = df.groupby("auction")[["value", "volume"]].sum()
    tab["all"] = tot["value"] / tot["volume"]
    tab["volume_mwh"] = tot["volume"]
    return tab.sort_index()
