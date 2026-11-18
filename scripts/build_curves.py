"""Fit the forward-curve model on the first trading day of each month and cache it.

Each curve uses only spot prices before its trading date (expanding window, at most three years,
at least one year). Output: data/cache/curves.pkl, a dict {trading date: CurveModel}.
"""
from __future__ import annotations

import pickle
import sys
import time

import pandas as pd

from ehr import curves, data

OUT = data.ROOT / "data" / "cache" / "curves.pkl"


def trading_dates(start: str = "2020-01-01", end: str = "2025-12-01") -> list[pd.Timestamp]:
    """First working day of each month."""
    return [pd.bdate_range(m, periods=1)[0] for m in pd.date_range(start, end, freq="MS")]


def main() -> None:
    prices = data.smard_series("price")
    cache = pickle.loads(OUT.read_bytes()) if OUT.exists() else {}
    prev = None
    for t in trading_dates():
        if t in cache:
            prev = cache[t].params
            continue
        t0 = time.time()
        cache[t] = curves.build_curve(prices, t, start=prev)
        prev = cache[t].params
        OUT.write_bytes(pickle.dumps(cache))
        print(t.date(), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
