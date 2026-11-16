"""Holding window from the drift curve, triple-barrier exits, meta-labeling under CPCV, volatility targeting."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import core as K


def drift_curve(ev: pd.DataFrame, ar: np.ndarray) -> pd.DataFrame:
    """Mean cumulative abnormal return by decile from day -1 through day 62."""
    lo = K.AR_WIN[0]
    seg = ar[:, (-1 - lo):]
    cum = np.nancumsum(np.nan_to_num(seg, nan=0.0), axis=1)
    df = pd.DataFrame(cum, columns=np.arange(-1, K.AR_WIN[1] + 1))
    df["decile"] = ev["decile"].values
    return df.groupby("decile").mean()


def pick_window(curve: pd.DataFrame, choices=(20, 40, 60)) -> int:
    """Shortest window that captures 90% of the post-entry D10-D1 spread seen by day 62."""
    sp = (curve.loc[10] - curve.loc[1]) - (curve.loc[10, K.ENTRY_LAG] - curve.loc[1, K.ENTRY_LAG])
    post = sp.loc[K.ENTRY_LAG + 1:]
    top = post.max()
    if not np.isfinite(top) or top <= 0:
        return max(choices)
    reach = post[post >= 0.9 * top].index.min() - K.ENTRY_LAG
    return int(min([c for c in choices if c >= reach] or [max(choices)]))


def triple_barrier(m: K.Market, ev: pd.DataFrame, sigma: np.ndarray, H: int, k: float) -> np.ndarray:
    """Days held before the first touch of +/- k * sigma * sqrt(H) on the cumulative return, else H."""
    held = np.full(len(ev), H)
    for j, (s, c, sg) in enumerate(zip(ev["entry_idx"].values, ev["col"].values, sigma)):
        if not np.isfinite(sg):
            continue
        r = m.R[s + 1:min(s + 1 + H, m.T), c]
        r = r[: int(np.argmax(~np.isfinite(r)))] if (~np.isfinite(r)).any() else r
        if not len(r):
            continue
        path = np.cumprod(1 + r) - 1
        b = k * sg * np.sqrt(H)
        hit = np.where((path >= b) | (path <= -b))[0]
        held[j] = hit[0] + 1 if len(hit) else H
    return held


def cpcv_splits(order: np.ndarray, entry: np.ndarray, H: int, groups: int = 6, k: int = 2, embargo: int | None = None):
    """Combinatorial purged CV over events sorted in time. Yields (train_idx, test_idx)."""
    embargo = H if embargo is None else embargo
    parts = np.array_split(order, groups)
    span = [(entry[p].min(), entry[p].max() + H) for p in parts]
    for test in itertools.combinations(range(groups), k):
        te = np.concatenate([parts[g] for g in test])
        keep = np.ones(len(entry), bool)
        keep[te] = False
        for g in test:
            a, b = span[g]
            lab_end = entry + H
            keep &= ~((entry <= b) & (lab_end >= a))         # purge: label window overlaps test span
            keep &= ~((entry > b) & (entry <= b + embargo))   # embargo after the test span
        yield np.where(keep)[0], te


def meta_label(ev: pd.DataFrame, feats: list[str], y: np.ndarray, H: int, seed: int = 0) -> dict:
    d = ev.reset_index(drop=True)
    X = d[feats].values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    idx = np.where(ok)[0]
    entry = d["entry_idx"].values[idx]
    order = np.argsort(entry, kind="stable")
    yy = (y[idx] > 0).astype(int)
    prob = np.zeros(len(idx)); cnt = np.zeros(len(idx)); aucs = []
    for tr, te in cpcv_splits(order, entry, H):
        if len(np.unique(yy[tr])) < 2 or len(te) < 50:
            continue
        mdl = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed)
        mdl.fit(X[idx][tr], yy[tr])
        p = mdl.predict_proba(X[idx][te])[:, 1]
        prob[te] += p; cnt[te] += 1
        if len(np.unique(yy[te])) == 2:
            aucs.append(roc_auc_score(yy[te], p))
    out = np.full(len(d), np.nan)
    out[idx] = np.where(cnt > 0, prob / np.maximum(cnt, 1), np.nan)
    return {"prob": out, "auc_mean": float(np.mean(aucs)) if aucs else np.nan,
            "auc_min": float(np.min(aucs)) if aucs else np.nan, "splits": len(aucs)}


def vol_target(x: pd.Series, target_ann: float = 0.10, lookback: int = 60, cap: float = 1.5) -> pd.Series:
    vol = x.rolling(lookback, min_periods=20).std().shift(1) * np.sqrt(252)
    lev = (target_ann / vol).clip(upper=cap).fillna(0.0)
    return x * lev
