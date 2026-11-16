"""Pipeline trust: alignment audit, placebos, planted-drift recovery, power, replication."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import core as K
from . import stats as S

KEEP = ["cik", "period_end", "filed", "accepted", "col", "day0"]


def dating_quality(m: K.Market, ev: pd.DataFrame, ew: np.ndarray) -> dict:
    """Estimate the share of correctly dated events from the abnormal-variance profile around day 0.

    A filing accepted after the market already reacted puts the reaction on day -1. With v_d the mean
    squared abnormal return at offset d and a baseline from offsets +/-3..5, the late share is
    excess(-1) / (excess(-1) + excess(0)).
    """
    offs = np.arange(-5, 6)
    d0, c = ev["day0"].values, ev["col"].values
    rows = np.clip(d0[:, None] + offs[None, :], 0, m.T - 1)
    ar = m.R[rows, c[:, None]] - ew[rows]
    v = pd.Series(np.nanmean(ar ** 2, axis=0), index=offs)
    base = float(np.median(v.loc[[-5, -4, -3, 3, 4, 5]]))
    ex = (v - base).clip(lower=0)
    late = float(ex[-1] / (ex[-1] + ex[0])) if (ex[-1] + ex[0]) > 0 else np.nan
    return {"correct_share": 1 - late, "variance_ratio": {int(k): round(float(x / base), 2) for k, x in v.items()},
            "plus1_excess_vs_day0": float(ex[1] / ex[0]) if ex[0] > 0 else np.nan}


def alignment(m: K.Market, ev: pd.DataFrame, ew: np.ndarray, win: int = 3) -> dict:
    """Share of events whose largest |abnormal return| in day0-win..day0+win falls on day 0."""
    offs = np.arange(-win, win + 1)
    d0, c = ev["day0"].values, ev["col"].values
    rows = np.clip(d0[:, None] + offs[None, :], 0, m.T - 1)
    ar = np.abs(m.R[rows, c[:, None]] - ew[rows])
    ok = np.isfinite(ar).all(axis=1)
    arg = np.argmax(np.where(np.isfinite(ar), ar, -1), axis=1)[ok]
    share = pd.Series(offs[arg]).value_counts(normalize=True).sort_index()
    return {"day0_share": float(share.get(0, 0.0)), "within_1": float(share.reindex([-1, 0, 1]).fillna(0).sum()),
            "by_offset": {int(k): round(float(v), 4) for k, v in share.items()}, "n": int(ok.sum())}


def ls_monthly(m: K.Market, ev: pd.DataFrame, H: int = 60, cost: float = 0.001) -> pd.Series:
    return K.monthly(K.legs(m, K.calendar(m, ev, H, cost)).ls)


def headline(m: K.Market, ev: pd.DataFrame, H: int = 60) -> dict:
    mo = ls_monthly(m, ev, H)
    r = S.nw_mean(mo, 3)
    return {"ls_monthly": r["mean"], "t": r["t"], "months": r["n"]}


def _rebuild(m, ev, ew):
    return K.assign_deciles(K.compute_ear(m, ev[[c for c in KEEP if c in ev]], ew))


def placebo_random_dates(m, ev, ew, rng):
    e = ev[KEEP].copy()
    new = np.empty(len(e), int)
    for c, idx in e.groupby("col").indices.items():
        ok = np.where(m.member[:, c] & np.isfinite(m.R[:, c]))[0]
        ok = ok[(ok > 3) & (ok < m.T - 3)]
        new[idx] = rng.choice(ok, len(idx)) if len(ok) else e["day0"].values[idx]
    e["day0"] = new
    return _rebuild(m, e, ew)


def placebo_shuffle(ev, rng):
    e = ev.copy()
    wk = e["signal_idx"] // 5
    e["ear"] = e.groupby(wk)["ear"].transform(lambda x: rng.permutation(x.values))
    return K.assign_deciles(e.drop(columns=["pct", "decile"]))


def placebo_random_signal(ev, rng):
    e = ev.copy()
    e["ear"] = rng.standard_normal(len(e))
    return K.assign_deciles(e.drop(columns=["pct", "decile"]))


def run_placebos(m, ev, ew, seeds: int = 20, H: int = 60) -> dict:
    out = {}
    for name in ("random_dates", "within_week_shuffle", "random_signal"):
        ts = []
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            e = (placebo_random_dates(m, ev, ew, rng) if name == "random_dates"
                 else placebo_shuffle(ev, rng) if name == "within_week_shuffle" else placebo_random_signal(ev, rng))
            ts.append(headline(m, e, H)["t"])
        ts = np.array(ts)
        hits = int((np.abs(ts) > 2).sum())
        out[name] = {"t_values": [round(float(x), 2) for x in ts], "hits_abs_t_gt_2": hits, "mean_t": round(float(np.nanmean(ts)), 2),
                     "pass": hits <= 3}
    return out


def plant(m: K.Market, ev: pd.DataFrame, delta: float, H: int = 60, only: bool = False) -> K.Market:
    """Add delta*(pct-0.5)/0.9 spread evenly over each event's holding days. only=True plants into zeros."""
    R = np.zeros_like(m.R) if only else m.R.copy()
    add = delta * (ev["pct"].values - 0.5) / 0.9 / H
    for s, c, a in zip(ev["entry_idx"].values, ev["col"].values, add):
        seg = slice(s + 1, min(s + 1 + H, m.T))
        if only:
            R[seg, c] += np.where(np.isfinite(m.R[seg, c]), a, np.nan)
        else:
            R[seg, c] = np.where(np.isfinite(R[seg, c]), R[seg, c] + a, R[seg, c])
    if only:
        R = np.where(np.isfinite(m.R), R, np.nan)
    return replace(m, R=R)


def planted_size(m, ev, delta, H=60) -> float:
    """What the planted drift alone earns in the calendar-time long-short book (monthly)."""
    mo = K.monthly(K.legs(m, K.calendar(plant(m, ev, delta, H, only=True), ev, H, cost=0.0)).ls)
    return float(mo.mean())


def recovery(m, ev, ew, delta: float, H: int = 60) -> dict:
    base = headline(m, ev, H)["ls_monthly"]
    mi = plant(m, ev, delta, H)
    ev2 = K.assign_deciles(K.compute_ear(mi, ev[KEEP], ew))   # rebuilt from scratch on the planted market
    got = headline(mi, ev2, H)["ls_monthly"] - base
    want = planted_size(m, ev, delta, H)
    return {"delta_60d_spread": delta, "planted_monthly": want, "recovered_monthly": got,
            "ratio": got / want if want else np.nan, "pass": bool(want and 0.75 <= got / want <= 1.25)}


def power(m, ev, ew, deltas=(0.0, 0.005, 0.01, 0.015, 0.02, 0.03), seeds: int = 12, H: int = 60) -> dict:
    """Minimum detectable effect: plant drift into random-date placebo worlds and count t > 2."""
    rows = []
    for d in deltas:
        hits, sizes = 0, []
        for s in range(seeds):
            rng = np.random.default_rng(5000 + s)
            e = placebo_random_dates(m, ev, ew, rng)
            mi = plant(m, e, d, H) if d else m
            hits += headline(mi, e, H)["t"] > 2
            if s == 0:
                sizes.append(planted_size(m, e, d, H) if d else 0.0)
        rows.append({"delta_60d_spread": d, "planted_monthly": sizes[0], "power": hits / seeds})
    df = pd.DataFrame(rows)
    above = df[df["power"] >= 0.8]
    mde = None
    if len(above):
        j = above.index[0]
        if j > 0:
            a, b = df.loc[j - 1], df.loc[j]
            w = (0.8 - a["power"]) / max(1e-9, b["power"] - a["power"])
            mde = float(a["planted_monthly"] + w * (b["planted_monthly"] - a["planted_monthly"]))
        else:
            mde = float(df.loc[j, "planted_monthly"])
    return {"grid": df.to_dict(orient="records"), "mde_monthly": mde}


def replication(our_monthly: pd.Series, osap_csv: str | None, start="2005-01", end="2024-11") -> dict:
    if not osap_csv:
        return {"corr": None, "pass": None, "note": "no OSAP series for this market (synthetic)"}
    d = pd.read_csv(osap_csv, parse_dates=["date"])
    o = d[(d.signalname == "AnnouncementReturn") & (d.port == "LS")].set_index("date")["ret"] / 100
    o.index = o.index.to_period("M")
    j = pd.concat([our_monthly.rename("ours"), o.rename("osap")], axis=1, join="inner").loc[start:end].dropna()
    c = float(j.corr().iloc[0, 1]) if len(j) > 24 else np.nan
    return {"corr": c, "months": len(j), "pass": bool(np.isfinite(c) and c >= 0.5)}
