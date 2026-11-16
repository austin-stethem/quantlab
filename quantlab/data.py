"""Price data with a local cache. yfinance for now; source-agnostic so it can be
swapped for Alpaca or a point-in-time vendor later without touching scorers.

Every frame this module returns has the same shape:
    index: DatetimeIndex (daily, tz-naive)
    columns: open, high, low, close, volume   (lowercase, adjusted close)
"""
from __future__ import annotations

import pathlib
import time
from typing import Iterable

import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache"
CACHE.mkdir(exist_ok=True)

COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "adj close" in df.columns:
        # scale OHLC to the adjusted series so splits/dividends don't fake signals
        ratio = df["adj close"] / df["close"]
        for c in ("open", "high", "low"):
            df[c] = df[c] * ratio
        df["close"] = df["adj close"]
    df = df[COLS].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna()


def fetch(ticker: str, start: str = "2012-01-01", end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """One ticker, cached to CSV. refresh=True re-downloads."""
    path = CACHE / f"{ticker.upper()}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if end is None or df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
            return df.loc[start:end]
    import yfinance as yf

    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
            if raw is not None and len(raw):
                break
        except Exception:  # network hiccup, retry
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"no data for {ticker}")
    df = _normalize(raw)
    df.to_csv(path)
    return df.loc[start:end]


def fetch_many(tickers: Iterable[str], start: str = "2012-01-01", end: str | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        try:
            out[t] = fetch(t, start, end, refresh)
        except Exception as e:  # keep going; report at the end
            print(f"  skip {t}: {e}")
    return out


def synthetic(n_days: int = 2500, seed: int = 7, drift: float = 0.0003, vol: float = 0.015) -> pd.DataFrame:
    """A regime-switching random walk for tests. Not for conclusions."""
    rng = np.random.default_rng(seed)
    regime = np.zeros(n_days)
    state = 0
    for i in range(n_days):
        if rng.random() < 0.01:
            state = 1 - state
        regime[i] = state
    mu = np.where(regime == 1, drift * 3, -drift)
    sig = np.where(regime == 1, vol * 0.8, vol * 1.4)
    rets = rng.normal(mu, sig)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n_days))
    volume = rng.lognormal(15, 0.4, n_days)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
