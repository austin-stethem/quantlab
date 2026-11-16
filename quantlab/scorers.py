"""Scorers. One interface, six of them, nothing clever inside any one of them.

    score(df) -> pd.Series in [0, 1], same index as df

Every scorer uses only data at or before each date (no lookahead). A high score
means "this says up over the next few weeks". 0.5 means "no opinion". The
consensus layer decides how much each one is worth; the scorers don't.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: pd.Series | np.ndarray, k: float = 1.0) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-k * x))


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    """Where today's value sits in its own trailing window, 0..1."""
    def pct(a):
        return (a[:-1] < a[-1]).mean()
    return s.rolling(window, min_periods=window // 2).apply(pct, raw=True)


# 1. Momentum: the 12-1 effect. Return over the last year, skipping the last month.
def momentum(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    r = c.shift(21) / c.shift(252) - 1.0
    return _rolling_pct_rank(r, 756).rename("momentum")


# 2. Trend: distance from the 200-day average and the slope of the 50 over the 200.
def trend(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    dist = (c / ma200 - 1.0) / c.pct_change().rolling(60).std().mul(np.sqrt(60))
    slope = (ma50 / ma200 - 1.0) * 10
    return _sigmoid(dist.fillna(0) + slope.fillna(0), k=1.5).rename("trend")


# 3. Short-term reversal: oversold bounces. RSI(14) inverted, as a weak voice.
def reversal(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return ((100 - rsi) / 100).clip(0, 1).rename("reversal")


# 4. Flow: volume surge in the direction of the move. Accumulation reads high.
def flow(df: pd.DataFrame) -> pd.Series:
    v = df["volume"]
    vz = (v - v.rolling(60).mean()) / v.rolling(60).std()
    r5 = df["close"].pct_change(5)
    signed = np.sign(r5) * vz.clip(-3, 3)
    return _sigmoid(signed.fillna(0), k=0.8).rename("flow")


# 5. Volatility regime: calm tape scores higher (the low-vol effect), inverted percentile.
def volatility(df: pd.DataFrame) -> pd.Series:
    rv = df["close"].pct_change().rolling(20).std()
    return (1.0 - _rolling_pct_rank(rv, 504)).rename("volatility")


# 6. Candlesticks: only the patterns with some evidence behind them, as weak nudges
#    that decay over three days. 0.5 baseline; a pattern moves it a little.
def candles(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    prev_o, prev_c = o.shift(1), c.shift(1)
    down_prev = prev_c < prev_o
    # bullish engulfing
    engulf = down_prev & (c > o) & (o <= prev_c) & (c >= prev_o)
    # hammer: small body near the top, long lower wick, after a down move
    lower_wick = (np.minimum(o, c) - l)
    hammer = (lower_wick > 2 * body) & ((h - np.maximum(o, c)) < body) & (c.shift(1) < c.rolling(10).mean().shift(1))
    # piercing line
    pierce = down_prev & (c > o) & (o < l.shift(1)) & (c > (prev_o + prev_c) / 2) & (c < prev_o)
    # bearish engulfing, the one that matters on the other side
    bear_engulf = (~down_prev) & (c < o) & (o >= prev_c) & (c <= prev_o)
    sig = (engulf.astype(float) * 0.6 + hammer.astype(float) * 0.4 + pierce.astype(float) * 0.4
           - bear_engulf.astype(float) * 0.6)
    decayed = sig.ewm(halflife=1.5, adjust=False).mean()
    return (0.5 + decayed.clip(-0.5, 0.5) * 0.6).rename("candles")


SCORERS = {
    "momentum": momentum,
    "trend": trend,
    "reversal": reversal,
    "flow": flow,
    "volatility": volatility,
    "candles": candles,
}


def score_all(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    names = names or list(SCORERS)
    out = pd.concat([SCORERS[n](df) for n in names], axis=1)
    return out
