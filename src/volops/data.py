"""ECB euro reference rates: ingestion and the local store the service reads.

The store is one CSV (date x currency, units of currency per EUR). `ingest` appends only new
fixings, so the daily job is cheap and idempotent; the API only ever reads the file.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

CURRENCIES = ("USD", "GBP", "JPY", "CHF", "CNY", "AUD")
API = "https://data-api.ecb.europa.eu/service/data/EXR"
DEFAULT_STORE = Path("data/rates.csv")


def load_rates(path: Path = DEFAULT_STORE) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    if (frame.dropna(how="all") <= 0).any().any():
        raise ValueError("exchange rates must be strictly positive")
    return frame


def log_returns(rates: pd.DataFrame, currency: str) -> pd.Series:
    """Daily log return between consecutive fixings of one currency (a return needs two observed fixings)."""
    return np.log(rates[currency].dropna()).diff().dropna()


def fetch_since(start: str, currencies: tuple[str, ...] = CURRENCIES, timeout: float = 60.0) -> pd.DataFrame:
    url = f"{API}/D.{'+'.join(currencies)}.EUR.SP00.A"
    resp = httpx.get(url, params={"format": "csvdata", "startPeriod": start}, timeout=timeout)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    wide = raw.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide.sort_index()[list(currencies)]


def merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append rows newer than the store. Existing fixings are never rewritten: ECB reference rates are not
    revised, so a changed past value would signal a problem and must not silently alter the training data."""
    fresh = new[new.index > existing.index.max()]
    overlap = new.index.intersection(existing.index)
    if len(overlap) and not np.allclose(existing.loc[overlap].to_numpy(), new.loc[overlap].to_numpy(), equal_nan=True):
        raise ValueError("source disagrees with stored history; refusing to merge")
    return pd.concat([existing, fresh]).sort_index()


def ingest(path: Path = DEFAULT_STORE) -> tuple[int, pd.Timestamp]:
    """Fetch anything newer than the store and append it. Returns (rows added, last date)."""
    existing = load_rates(path)
    start = (existing.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")  # overlap lets merge() cross-check
    merged = merge(existing, fetch_since(start, tuple(existing.columns)))
    added = len(merged) - len(existing)
    if added:
        tmp = path.with_suffix(".tmp")
        merged.to_csv(tmp)
        tmp.replace(path)  # atomic: a reader never sees a half-written file
    return added, merged.index.max()
