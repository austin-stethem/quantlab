"""Quantlab v3 core: point-in-time market container, events, EAR, deciles, portfolios, event study.

Every array is indexed [day, firm]. Firms are keyed on SEC CIK. Tickers live only in the
security master as date-effective attributes and never key anything here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

HOLDOUT_START = pd.Timestamp("2024-11-14")
DEV_END = pd.Timestamp("2024-11-13")
ENTRY_LAG = 2          # enter at the close of day +2
POOL = 63              # trailing sessions for decile breakpoints
MIN_POOL = 50
AR_WIN = (-5, 62)      # event-study window, inclusive, relative to day 0
EST_WIN = (-260, -21)  # estimation window for BMP standardisation


@dataclass
class Market:
    days: pd.DatetimeIndex
    ciks: np.ndarray
    R: np.ndarray                 # T x N daily total return, nan when not trading
    member: np.ndarray            # T x N point-in-time S&P 500 membership
    spy: np.ndarray               # T
    mcap: np.ndarray              # T x N market cap, nan when unknown
    bm: np.ndarray                # T x N book-to-market known as of that day
    filings: pd.DataFrame         # cik, accepted (naive ET), form ('8-K' / '8-K/A')
    periods: pd.DataFrame         # cik, period_end, filed  (10-Q / 10-K)
    ff: pd.DataFrame | None = None        # index = days; Mkt-RF SMB HML RMW CMA Mom RF, decimals
    attention: pd.DataFrame | None = None  # cik, date, instown[, coverage]  as-of values
    meta: dict = field(default_factory=dict)

    @property
    def T(self): return len(self.days)

    @property
    def N(self): return len(self.ciks)


def dev_slice(m: Market, end: pd.Timestamp = DEV_END) -> Market:
    """The only door to data. Development work never sees the holdout."""
    k = int(np.searchsorted(m.days.values, np.datetime64(end), side="right"))
    cut = m.days[k - 1] if k else None
    f = m.filings[m.filings["accepted"] <= end + pd.Timedelta(hours=23, minutes=59)]
    p = m.periods[m.periods["filed"] <= end]
    att = None if m.attention is None else m.attention[m.attention["date"] <= end]
    ff = None if m.ff is None else m.ff.loc[: cut]
    return replace(m, days=m.days[:k], R=m.R[:k], member=m.member[:k], spy=m.spy[:k], mcap=m.mcap[:k],
                   bm=m.bm[:k], filings=f, periods=p, ff=ff, attention=att, meta={**m.meta, "slice": f"<= {end.date()}"})


# ----------------------------------------------------------------------------- events
def dedupe_events(filings: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """One event per company per fiscal quarter.

    Amendments (8-K/A) are dropped. For each 10-Q/10-K, the event is the LAST Item 2.02
    filing accepted after the period end and no later than the periodic report's filing day.
    Pre-announcements come earlier and lose; stray 2.02 filings outside any window never match.
    """
    f = filings[filings["form"].eq("8-K")].sort_values("accepted")[["cik", "accepted"]].copy()
    p = periods.copy()
    p["window_end"] = pd.to_datetime(p["filed"]) + pd.Timedelta(hours=23, minutes=59)
    p = p.sort_values("window_end")
    out = pd.merge_asof(p, f, left_on="window_end", right_on="accepted", by="cik", direction="backward")
    out = out[out["accepted"].notna() & (out["accepted"] > pd.to_datetime(out["period_end"]))]
    out = out.drop_duplicates(["cik", "accepted"]).sort_values("accepted").reset_index(drop=True)
    return out[["cik", "period_end", "filed", "accepted"]]


def assign_day0(accepted: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    """Day 0 = the session that first trades on the news.

    Before 09:30 or during the session -> that day (if a session). At or after 16:00 -> next session.
    Weekend/holiday -> next session.
    """
    acc = pd.to_datetime(accepted)
    d = acc.dt.normalize().values
    mins = (acc.dt.hour * 60 + acc.dt.minute).values
    base = np.searchsorted(days.values, d, side="left")
    is_session = (base < len(days)) & (days.values[np.minimum(base, len(days) - 1)] == d)
    return base + (is_session & (mins >= 16 * 60)).astype(int)


def ew_universe(m: Market) -> np.ndarray:
    x = np.where(m.member, m.R, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(x, axis=1)


def compute_ear(m: Market, ev: pd.DataFrame, ew: np.ndarray | None = None) -> pd.DataFrame:
    """EAR = stock return from close(-2) to close(+1) minus the equal-weight universe over the same days."""
    ew = ew_universe(m) if ew is None else ew
    col = pd.Series(np.arange(m.N), index=m.ciks)
    ev = ev[ev["cik"].isin(col.index)].copy()
    ev["col"] = col.loc[ev["cik"]].values
    if "day0" not in ev:
        ev["day0"] = assign_day0(ev["accepted"], m.days)
    ok = (ev["day0"] >= 1) & (ev["day0"] + ENTRY_LAG < m.T)
    ev = ev[ok].copy()
    d0, c = ev["day0"].values, ev["col"].values
    win = np.stack([m.R[d0 + k, c] for k in (-1, 0, 1)], axis=1)
    mk = np.stack([ew[d0 + k] for k in (-1, 0, 1)], axis=1)
    ev["ear"] = np.prod(1 + win, axis=1) - np.prod(1 + mk, axis=1)
    ev["member0"] = m.member[d0, c]
    ev = ev[np.isfinite(ev["ear"]) & ev["member0"]].copy()
    ev["signal_idx"] = ev["day0"] + 1
    ev["entry_idx"] = ev["day0"] + ENTRY_LAG
    return ev.drop(columns="member0").reset_index(drop=True)


def assign_deciles(ev: pd.DataFrame, col: str = "ear", pool: int = POOL, min_pool: int = MIN_POOL) -> pd.DataFrame:
    """Percentile of each event against events known in the trailing `pool` sessions (itself included).

    Uses only events whose signal day is <= this event's signal day: no look-ahead.
    """
    ev = ev.sort_values("signal_idx").reset_index(drop=True)
    s = ev["signal_idx"].values
    x = ev[col].values
    pct = np.full(len(ev), np.nan)
    for day in np.unique(s):
        lo = np.searchsorted(s, day - pool + 1, side="left")
        hi = np.searchsorted(s, day, side="right")
        if hi - lo < min_pool:
            continue
        ref = np.sort(x[lo:hi])
        idx = np.arange(np.searchsorted(s, day, side="left"), hi)
        a = np.searchsorted(ref, x[idx], side="left")
        b = np.searchsorted(ref, x[idx], side="right")
        pct[idx] = ((a + b) / 2) / len(ref)
    ev["pct"] = pct
    ev = ev[np.isfinite(ev["pct"])].copy()
    ev["decile"] = np.minimum((ev["pct"] * 10).astype(int) + 1, 10)
    return ev.reset_index(drop=True)


# ----------------------------------------------------------------------------- portfolios
def calendar(m: Market, ev: pd.DataFrame, H: int = 60, cost: float = 0.001, group: str = "decile",
             weight: np.ndarray | None = None, exit_after: np.ndarray | None = None) -> pd.DataFrame:
    """Calendar-time equal-weight (or `weight`-weighted) portfolios, one column per group value.

    A position enters at the close of entry_idx and earns returns on entry_idx+1 .. entry_idx+H
    (or entry_idx+exit_after). A delisting ends it on its last traded day. Cost is charged on the
    first and last held day.
    """
    groups = np.sort(ev[group].unique())
    gi = {g: i for i, g in enumerate(groups)}
    acc = np.zeros((len(groups), m.T))
    cnt = np.zeros((len(groups), m.T))
    w = np.ones(len(ev)) if weight is None else np.asarray(weight, float)
    hold = np.full(len(ev), H) if exit_after is None else np.asarray(exit_after, int)
    for k, (s, c, g, h) in enumerate(zip(ev["entry_idx"].values, ev["col"].values, ev[group].values, hold)):
        a, b = s + 1, min(s + 1 + h, m.T)
        if a >= b:
            continue
        r = m.R[a:b, c].copy()
        bad = ~np.isfinite(r)
        if bad.any():
            r = r[: int(np.argmax(bad))]
            if len(r) == 0:
                continue
        r[0] -= cost
        r[-1] -= cost
        i = gi[g]
        acc[i, a:a + len(r)] += w[k] * r
        cnt[i, a:a + len(r)] += w[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, acc / cnt, 0.0)
    df = pd.DataFrame(out.T, index=m.days, columns=groups)
    df.attrs["count"] = pd.DataFrame(cnt.T, index=m.days, columns=groups)
    return df


def legs(m: Market, cal: pd.DataFrame) -> pd.DataFrame:
    hi, lo = cal.columns.max(), cal.columns.min()
    return pd.DataFrame({"long": cal[hi], "short": cal[lo], "ls": cal[hi] - cal[lo],
                         "long_x_spy": cal[hi] - m.spy, "short_x_spy": cal[lo] - m.spy}, index=m.days)


def monthly(x: pd.Series) -> pd.Series:
    return x.groupby(x.index.to_period("M")).sum()


# ----------------------------------------------------------------------------- event study
def momentum_12_1(R: np.ndarray) -> np.ndarray:
    lr = np.log1p(np.where(np.isfinite(R), R, 0.0))
    n = np.isfinite(R).astype(float)
    cs, cn = np.cumsum(lr, axis=0), np.cumsum(n, axis=0)
    out = np.full(R.shape, np.nan)
    if R.shape[0] > 252:
        s = cs[231:-21] - cs[:-252]        # days t-251..t-21 for t = 252..T-1 ... aligned below
        c = cn[231:-21] - cn[:-252]
        out[252:] = np.where(c >= 200, np.expm1(s), np.nan)[: R.shape[0] - 252]
    return out


def dgtw_benchmark(m: Market, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """DGTW-style characteristic benchmark: size, then B/M within size, then 12-1 momentum, n x n x n cells.

    Formed at each month-end session on members with all three characteristics, value-weighted by the
    formation-date market cap, held until the next formation. Returns (cell[T,N], bench[T, n^3]).
    """
    T, N = m.R.shape
    mom = momentum_12_1(m.R)
    month_end = np.r_[np.diff(m.days.month) != 0, True]
    form_days = np.where(month_end)[0]
    cell = np.full((T, N), -1, int)
    wgt = np.zeros((T, N))
    for j, f in enumerate(form_days):
        nxt = form_days[j + 1] if j + 1 < len(form_days) else T - 1
        ok = m.member[f] & np.isfinite(m.mcap[f]) & np.isfinite(m.bm[f]) & np.isfinite(mom[f])
        if ok.sum() < n ** 3 * 3:
            continue
        idx = np.where(ok)[0]
        c = np.zeros(len(idx), int)
        s_rank = pd.qcut(pd.Series(m.mcap[f, idx]).rank(method="first"), n, labels=False).values
        for si in range(n):
            a = idx[s_rank == si]
            b_rank = pd.qcut(pd.Series(m.bm[f, a]).rank(method="first"), n, labels=False).values
            for bi in range(n):
                aa = a[b_rank == bi]
                m_rank = pd.qcut(pd.Series(mom[f, aa]).rank(method="first"), n, labels=False).values
                for mi in range(n):
                    cell[f + 1:nxt + 1, aa[m_rank == mi]] = si * n * n + bi * n + mi
                    wgt[f + 1:nxt + 1, aa[m_rank == mi]] = m.mcap[f, aa[m_rank == mi]]
    bench = np.full((T, n ** 3), np.nan)
    Rz = np.where(np.isfinite(m.R), m.R, 0.0)
    has = np.isfinite(m.R)
    for k in range(n ** 3):
        sel = (cell == k) & has
        ww = np.where(sel, wgt, 0.0)
        tot = ww.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bench[:, k] = np.where(tot > 0, (ww * Rz).sum(axis=1) / tot, np.nan)
    return cell, bench


def abnormal_returns(m: Market, ev: pd.DataFrame, cell: np.ndarray, bench: np.ndarray, ew: np.ndarray) -> np.ndarray:
    """AR[e, d] for d in AR_WIN against the event's DGTW cell as of day0-1 (EW universe if unassigned)."""
    lo, hi = AR_WIN
    offs = np.arange(lo, hi + 1)
    d0, c = ev["day0"].values, ev["col"].values
    cid = cell[np.maximum(d0 - 1, 0), c]
    rows = d0[:, None] + offs[None, :]
    valid = (rows >= 0) & (rows < m.T)
    rr = np.clip(rows, 0, m.T - 1)
    r = m.R[rr, c[:, None]]
    b = np.where(cid[:, None] >= 0, bench[rr, np.maximum(cid, 0)[:, None]], ew[rr])
    b = np.where(np.isfinite(b), b, ew[rr])
    ar = np.where(valid, r - b, np.nan)
    return ar


def estimation_sigma(m: Market, ev: pd.DataFrame, ew: np.ndarray) -> np.ndarray:
    lo, hi = EST_WIN
    offs = np.arange(lo, hi + 1)
    d0, c = ev["day0"].values, ev["col"].values
    rows = d0[:, None] + offs[None, :]
    valid = rows >= 0
    rr = np.clip(rows, 0, m.T - 1)
    ar = np.where(valid, m.R[rr, c[:, None]] - ew[rr], np.nan)
    n = np.isfinite(ar).sum(axis=1)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(ar, axis=1, ddof=1)
    return np.where(n >= 120, sd, np.nan)


def car(ar: np.ndarray, a: int, b: int) -> np.ndarray:
    """Sum of AR over event days a..b inclusive (nan-safe; nan if fewer than half the days exist)."""
    lo = AR_WIN[0]
    seg = ar[:, a - lo: b - lo + 1]
    n = np.isfinite(seg).sum(axis=1)
    s = np.nansum(seg, axis=1)
    return np.where(n >= max(1, (b - a + 1) // 2), s, np.nan)
