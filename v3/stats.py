"""Statistics: Newey-West, IC, Fama-MacBeth, factor alpha, BMP, deflated Sharpe, PBO (CSCV), bootstrap."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st


def nw_mean(x, lags: int) -> dict:
    x = pd.Series(x).dropna()
    if len(x) < 10 or x.std() == 0:
        return {"mean": float(x.mean()) if len(x) else np.nan, "t": np.nan, "n": len(x)}
    r = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": float(r.params[0]), "t": float(r.tvalues[0]), "n": len(x)}


def factor_alpha(y: pd.Series, ff: pd.DataFrame, lags: int, cols=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")) -> dict:
    d = pd.concat([y.rename("y"), ff[list(cols)]], axis=1, join="inner").dropna()
    d = d[d["y"] != 0]  # days with no positions carry no information
    if len(d) < 250:
        return {"alpha": np.nan, "t": np.nan, "n": len(d), "betas": {}}
    r = sm.OLS(d["y"], sm.add_constant(d[list(cols)])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha": float(r.params["const"]), "t": float(r.tvalues["const"]), "n": len(d),
            "betas": {c: round(float(r.params[c]), 3) for c in cols}}


def rank_ic(ev: pd.DataFrame, signal: str, outcome: str, by: str = "month", min_n: int = 20, lags: int = 3) -> dict:
    g = ev.dropna(subset=[signal, outcome]).groupby(by)
    ics = g.apply(lambda d: st.spearmanr(d[signal], d[outcome])[0] if len(d) >= min_n else np.nan,
                  include_groups=False).dropna()
    out = nw_mean(ics, lags)
    out["ic_series"] = ics
    return out


def fama_macbeth(ev: pd.DataFrame, y: str, xs: list[str], by: str = "month", min_n: int = 30, lags: int = 3) -> dict:
    rows = []
    for k, d in ev.dropna(subset=[y] + xs).groupby(by):
        if len(d) < min_n:
            continue
        r = sm.OLS(d[y].values, sm.add_constant(d[xs].values)).fit()
        rows.append(dict(zip(["const"] + xs, r.params)))
    co = pd.DataFrame(rows)
    return {x: nw_mean(co[x], lags) for x in xs} | {"months": len(co)}


def bmp(sar: np.ndarray) -> dict:
    s = sar[np.isfinite(sar)]
    if len(s) < 10:
        return {"mean_sar": np.nan, "t": np.nan, "n": len(s)}
    return {"mean_sar": float(s.mean()), "t": float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), "n": len(s)}


def welch(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(st.ttest_ind(a, b, equal_var=False).statistic)


def sharpe(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std() > 0 else np.nan


def deflated_sharpe(x: pd.Series, trial_srs: list[float]) -> dict:
    """Bailey & Lopez de Prado (2014). All Sharpe ratios per period (daily), not annualised."""
    x = pd.Series(x).dropna()
    T = len(x)
    sr = sharpe(x)
    srs = np.array([s for s in trial_srs if np.isfinite(s)])
    N = max(len(srs), 1)
    var = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * st.norm.ppf(1 - 1 / N) + g * st.norm.ppf(1 - 1 / (N * math.e))) if N > 1 else 0.0
    sk, ku = float(st.skew(x)), float(st.kurtosis(x, fisher=False))
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr * sr))
    dsr = float(st.norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))
    return {"dsr": dsr, "sr_daily": sr, "sr_ann": sr * math.sqrt(252), "sr0_daily": sr0, "trials": N}


def pbo_cscv(M: pd.DataFrame, S: int = 16) -> dict:
    """Probability of backtest overfitting, combinatorially symmetric cross-validation (Bailey et al. 2017).

    M: T x K matrix of per-period returns, one column per trial.
    """
    M = M.dropna(how="all").fillna(0.0)
    K = M.shape[1]
    if K < 2:
        return {"pbo": np.nan, "combos": 0, "trials": K}
    blocks = np.array_split(np.arange(len(M)), S)
    X = M.values
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in comb])
        te = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        def sr(a):
            sd = a.std(axis=0, ddof=1)
            return np.where(sd > 0, a.mean(axis=0) / sd, -np.inf)
        best = int(np.argmax(sr(X[tr])))
        oos = sr(X[te])
        w = (st.rankdata(oos)[best]) / (K + 1)
        logits.append(math.log(w / (1 - w)))
    lg = np.array(logits)
    return {"pbo": float((lg <= 0).mean()), "combos": len(lg), "trials": K}


def block_bootstrap_ci(x: pd.Series, block: int = 60, B: int = 2000, seed: int = 0, q=(0.025, 0.975)) -> tuple[float, float]:
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < block * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block, nb)
        means[b] = np.concatenate([x[s:s + block] for s in starts])[:n].mean()
    return tuple(float(v) for v in np.quantile(means, q))
