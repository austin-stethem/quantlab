"""A synthetic S&P-500-like market that carries every trap the real data has.

Index churn, delistings with final returns (acquisitions and bankruptcies), after-close vs pre-open
releases, late 8-K filings, 8-K/A amendments, pre-announcements, stray Item 2.02 filings, fiscal
periods with 10-Q/10-K filing dates, book equity known only after filing, and 13F-style ownership
published with a lag. Optionally plants a post-earnings drift whose strength depends on attention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Market


def make_market(seed: int = 7, start="2005-01-03", end="2026-11-13", n_members=500, n_extra=160,
                drift=0.0, friday_boost=0.0, busy_boost=0.0, inst_boost=0.0) -> Market:
    """drift: 60-day drift per 1 s.d. of announcement surprise (e.g. 0.006 = 0.6%).

    Boosts scale the drift: x (1 + friday_boost) on Friday releases, x (1 + busy_boost * busy_z),
    x (1 + inst_boost * (0.67 - instown) / 0.15).
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    T, N = len(days), n_members + n_extra

    # --- market, sectors, stocks
    m = np.empty(T); s2 = 0.011 ** 2; prev = 0.0
    for t in range(T):
        s2 = 2e-6 + 0.08 * prev ** 2 + 0.90 * s2
        prev = m[t] = 0.0003 + np.sqrt(s2) * rng.standard_normal()
    sector = rng.integers(0, 10, N)
    sec = rng.normal(0, 0.006, (T, 10))
    beta = rng.uniform(0.6, 1.5, N)
    sig = rng.uniform(0.010, 0.025, N)
    eps = rng.standard_t(5, (T, N)) * (sig / np.sqrt(5 / 3))
    R = beta * m[:, None] + sec[:, sector] + eps

    # --- life spans and point-in-time membership
    alive_from = np.zeros(N, int)
    alive_from[n_members:] = rng.integers(0, T // 2, n_extra)
    alive_to = np.full(N, T - 1)
    member = np.zeros((T, N), bool)
    in_idx = set(range(n_members))
    mem_start = {i: 0 for i in in_idx}
    month_end = np.where(np.r_[np.diff(days.month) != 0, False])[0]
    delist_ret = {}
    for f in month_end:
        for _ in range(rng.poisson(1.8)):
            if not in_idx:
                break
            out = int(rng.choice(sorted(in_idx)))
            in_idx.discard(out)
            member[mem_start.pop(out):f + 1, out] = True
            if rng.random() < 0.4:                      # delisted: acquired or bankrupt
                alive_to[out] = f
                delist_ret[out] = 0.25 if rng.random() < 0.7 else -0.55
            cands = [i for i in range(N) if i not in in_idx and alive_from[i] <= f and alive_to[i] > f + 60]
            if cands:
                new = int(rng.choice(cands)); in_idx.add(new); mem_start[new] = f + 1
    for i, s in mem_start.items():
        member[s:, i] = True
    for i in range(N):
        R[: alive_from[i], i] = np.nan
        R[alive_to[i] + 1:, i] = np.nan
        if i in delist_ret:
            R[alive_to[i], i] += delist_ret[i]

    # --- fiscal calendar, releases, filings
    instown = np.clip(rng.beta(6, 3, N), 0.05, 0.98)
    shares = rng.uniform(2e8, 3e9, N)
    book_ratio = rng.uniform(0.1, 0.9, N)
    filings, periods, rel = [], [], []
    off = (rng.random(N) < 0.1).astype(int)
    for i in range(N):
        a, b = days[alive_from[i]], days[alive_to[i]]
        qe = pd.date_range(a, b, freq="QE") + pd.offsets.MonthEnd(int(off[i]))
        for k, pe in enumerate(qe):
            q4 = pe.month in (12, 1)
            lag = int(rng.integers(35, 60) if q4 else rng.integers(20, 45))
            d = pe + pd.Timedelta(days=lag)
            if d.weekday() >= 5:
                d += pd.Timedelta(days=7 - d.weekday())
            if d.weekday() == 4 and rng.random() < 0.6:
                d -= pd.Timedelta(days=1)
            after = rng.random() < 0.55
            t_rel = d + (pd.Timedelta(hours=16, minutes=int(rng.integers(5, 25))) if after
                         else pd.Timedelta(hours=7, minutes=int(rng.integers(0, 60))))
            acc = t_rel + pd.Timedelta(minutes=int(rng.integers(1, 30)))
            late = rng.random() < 0.06
            if late:
                acc = (d + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=int(rng.integers(10, 17)))
            if acc > b + pd.Timedelta(days=1):
                continue
            filed = d + pd.Timedelta(days=int(rng.integers(5, 30) if q4 else rng.integers(0, 14)))
            filings.append((i, acc, "8-K"))
            periods.append((i, pe, filed, "10-K" if q4 else "10-Q"))
            rel.append((i, t_rel, d.weekday() == 4, acc))
            if rng.random() < 0.03:
                filings.append((i, acc + pd.Timedelta(days=int(rng.integers(1, 20))), "8-K/A"))
            if rng.random() < 0.04:
                filings.append((i, pe + pd.Timedelta(days=int(rng.integers(3, max(4, lag - 2)))) + pd.Timedelta(hours=8), "8-K"))
        for _ in range(rng.poisson(1.0)):
            filings.append((i, a + pd.Timedelta(days=int(rng.integers(0, max(1, (b - a).days)))) + pd.Timedelta(hours=9), "8-K"))

    # --- the true reaction (and optional drift) lands on the true release day
    rel = pd.DataFrame(rel, columns=["col", "t_rel", "friday", "acc"])
    tr = rel["t_rel"]
    base = np.searchsorted(days.values, tr.dt.normalize().values)
    sess = (base < T) & (days.values[np.minimum(base, T - 1)] == tr.dt.normalize().values)
    rel["d0"] = base + (sess & (tr.dt.hour >= 16)).astype(int)
    rel = rel[rel["d0"] < T - 1].reset_index(drop=True)
    busy = rel.groupby("d0")["col"].transform("size").values
    busy_z = (np.log(busy) - np.log(busy).mean()) / np.log(busy).std()
    for j, (c, d0, fri) in enumerate(zip(rel["col"].values, rel["d0"].values, rel["friday"].values)):
        if not np.isfinite(R[d0, c]):
            continue
        z = rng.standard_normal()
        R[d0, c] += 4 * sig[c] * z
        if drift:
            mult = (1 + friday_boost * fri) * max(0.1, 1 + busy_boost * busy_z[j]) * max(0.1, 1 + inst_boost * (0.67 - instown[c]) / 0.15)
            seg = slice(d0 + 1, min(d0 + 61, T))
            ok = np.isfinite(R[seg, c])
            R[seg, c] = np.where(ok, R[seg, c] + drift * z * mult / 60, R[seg, c])

    # --- market cap, book-to-market known after filing, factors, ownership
    lvl = np.cumprod(1 + np.nan_to_num(R), axis=0)
    mcap = np.where(np.isfinite(R), shares * 20 * lvl, np.nan)
    bm = np.full((T, N), np.nan)
    per = pd.DataFrame(periods, columns=["col", "period_end", "filed", "form"])
    for c, g in per.sort_values("filed").groupby("col"):
        idx = np.searchsorted(days.values, g["filed"].values)
        for k, s in enumerate(idx):
            if s >= T:
                continue
            e = idx[k + 1] if k + 1 < len(idx) else T
            bm[s:e, c] = book_ratio[c] * mcap[s, c] / np.where(np.isfinite(mcap[s:e, c]), mcap[s:e, c], np.nan)
    spy = np.nanmean(np.where(member, R, np.nan), axis=1)
    ff = pd.DataFrame({"Mkt-RF": m, "SMB": rng.normal(0, .005, T), "HML": rng.normal(0, .005, T),
                       "RMW": rng.normal(0, .004, T), "CMA": rng.normal(0, .004, T), "Mom": rng.normal(0, .007, T),
                       "RF": 0.0}, index=days)
    ciks = 1_000_000 + np.arange(N)
    q = pd.date_range(days[0], days[-1], freq="QE")
    att = pd.DataFrame([(ciks[c], d + pd.Timedelta(days=45), float(np.clip(instown[c] + rng.normal(0, .03), 0.02, 0.99)))
                        for c in range(N) for d in q], columns=["cik", "date", "instown"])
    fil = pd.DataFrame(filings, columns=["col", "accepted", "form"])
    fil["cik"] = ciks[fil["col"]]
    per["cik"] = ciks[per["col"]]
    return Market(days=days, ciks=ciks, R=R, member=member, spy=spy, mcap=mcap, bm=bm,
                  filings=fil[["cik", "accepted", "form"]].sort_values("accepted").reset_index(drop=True),
                  periods=per[["cik", "period_end", "filed", "form"]].reset_index(drop=True), ff=ff, attention=att,
                  meta={"synthetic": True, "seed": seed, "drift": drift, "late_share": 0.06,
                        "truth": pd.DataFrame({"cik": ciks[rel["col"].values], "accepted": rel["acc"].values, "true_d0": rel["d0"].values}),
                        "boosts": {"friday": friday_boost, "busy": busy_boost, "inst": inst_boost}})
