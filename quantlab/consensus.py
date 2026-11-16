"""Labels, the panel, walk-forward fitting, consensus, calibration, the gate.

The consensus is a logistic regression on the six scores: its coefficients are
the weights, learned out of sample, never typed in. A quorum sits on top:
a signal "fires" only when the calibrated probability clears a threshold AND at
least k scorers individually lean the same way. Calibration is isotonic,
fitted on the training fold, so the probability printed is the hit rate earned.

Walk-forward: train on everything up to year Y minus an embargo of one
horizon, test on year Y. No fold ever sees its own future.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .scorers import score_all, SCORERS


# ---------- labels ----------

def forward_label(df: pd.DataFrame, horizon: int = 20, min_move: float = 0.0) -> pd.Series:
    """1 if close rises more than min_move over the next `horizon` trading days."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    y = (fwd > min_move).astype(float)
    y[fwd.isna()] = np.nan
    return y.rename("y"), fwd.rename("fwd_ret")


# ---------- panel ----------

def build_panel(prices: dict[str, pd.DataFrame], horizon: int = 20, min_move: float = 0.0, names: list[str] | None = None,
                relative: bool = False, on_ticker=None) -> pd.DataFrame:
    """Stack every ticker into one long frame: date, ticker, six scores, label, forward return."""
    frames = []
    for t, df in prices.items():
        s = score_all(df, names)
        y, fwd = forward_label(df, horizon, min_move)
        p = pd.concat([s, y, fwd], axis=1)
        p["ticker"] = t
        frames.append(p)
        if on_ticker:
            on_ticker(t, len(p))
    panel = pd.concat(frames).reset_index().rename(columns={"index": "date", "Date": "date"})
    panel = panel.dropna(subset=list(names or SCORERS))
    if relative:
        # the question the gate asks: does this name beat the rest of the universe on this date?
        med = panel.groupby("date")["fwd_ret"].transform("median")
        panel["y"] = np.where(panel["fwd_ret"].isna(), np.nan, (panel["fwd_ret"] > med).astype(float))
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------- the model ----------

@dataclass
class FoldResult:
    test_year: int
    n_train: int
    n_test: int
    weights: dict
    base_rate: float
    brier_model: float
    brier_base: float
    fired: int
    fired_hit_rate: float
    fired_avg_ret: float


@dataclass
class RunResult:
    folds: list[FoldResult]
    predictions: pd.DataFrame  # date, ticker, prob, fired, y, fwd_ret
    calibration: pd.DataFrame  # bin, predicted, observed, n
    weights_latest: dict
    scorer_names: list[str]
    threshold: float
    quorum: int
    horizon: int
    gate: dict = field(default_factory=dict)


def _fit(X: np.ndarray, y: np.ndarray):
    lr = LogisticRegression(C=1.0, max_iter=500)
    lr.fit(X, y)
    raw = lr.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(raw, y)
    return lr, iso


def _predict(lr, iso, X: np.ndarray) -> np.ndarray:
    return iso.predict(lr.predict_proba(X)[:, 1])


def walk_forward(panel: pd.DataFrame, names: list[str] | None = None, threshold: float = 0.60, quorum: int = 4, on_fold=None,
                 lean: float = 0.55, horizon: int = 20, first_test_year: int | None = None, min_train_years: int = 3) -> RunResult:
    names = names or list(SCORERS)
    panel = panel.dropna(subset=["y"]).copy()
    panel["year"] = panel["date"].dt.year
    years = sorted(panel["year"].unique())
    first = first_test_year or years[min_train_years]
    embargo = pd.Timedelta(days=int(horizon * 1.6))  # calendar days ~ horizon trading days

    folds, preds = [], []
    weights_latest = {}
    for ty in years:
        if ty < first:
            continue
        test = panel[panel["year"] == ty]
        cutoff = test["date"].min() - embargo
        train = panel[panel["date"] < cutoff]
        if len(train) < 500 or len(test) == 0:
            continue
        lr, iso = _fit(train[names].values, train["y"].values)
        prob = _predict(lr, iso, test[names].values)
        agree = (test[names].values > lean).sum(axis=1)
        fired = (prob >= threshold) & (agree >= quorum)
        weights = {n: float(w) for n, w in zip(names, lr.coef_[0])}
        weights_latest = weights
        base = float(train["y"].mean())
        brier_model = float(np.mean((prob - test["y"].values) ** 2))
        brier_base = float(np.mean((base - test["y"].values) ** 2))
        f = test[fired]
        folds.append(FoldResult(ty, len(train), len(test), weights, base, brier_model, brier_base,
                                int(fired.sum()), float(f["y"].mean()) if len(f) else float("nan"),
                                float(f["fwd_ret"].mean()) if len(f) else float("nan")))
        out = test[["date", "ticker", "y", "fwd_ret"] + names].copy()
        out["prob"] = prob
        out["agree"] = agree
        out["fired"] = fired
        preds.append(out)
        if on_fold:
            on_fold(folds[-1])

    predictions = pd.concat(preds).reset_index(drop=True)
    calibration = _calibration_table(predictions["prob"].values, predictions["y"].values)
    return RunResult(folds, predictions, calibration, weights_latest, names, threshold, quorum, horizon)


def _calibration_table(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "predicted": float(prob[m].mean()),
                     "observed": float(y[m].mean()), "n": int(m.sum())})
    return pd.DataFrame(rows)


# ---------- paper P&L on fired signals ----------

def paper_pnl(res: RunResult, prices: dict[str, pd.DataFrame] | None = None, cost_bps: float = 10.0) -> pd.DataFrame:
    """Paper book on fired signals, marked with each ticker's ACTUAL daily returns.

    Capital is split into `horizon` sleeves. A day's fired signals share one sleeve
    equally, entered at that close and held `horizon` trading days. Idle sleeves sit
    in cash at 0. Costs: cost_bps on entry and on exit. Also returns a benchmark book
    built the same way but holding the whole universe every day, so the comparison is
    like for like: same sleeves, same holding period, same costs.
    """
    if prices is None:
        raise ValueError("paper_pnl needs prices to mark positions daily")
    rets = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()})
    p = res.predictions
    dates = rets.loc[p["date"].min():p["date"].max()].index
    rets = rets.loc[dates]
    pos = {t: i for i, t in enumerate(rets.columns)}
    h, cost = res.horizon, cost_bps / 1e4
    R = rets.fillna(0.0).values

    def book(signals: pd.DataFrame) -> np.ndarray:
        W = np.zeros_like(R)            # weight held in each ticker at each day's close
        C = np.zeros(len(dates))         # cost drag per day
        didx = {d: i for i, d in enumerate(dates)}
        for d, grp in signals.groupby("date"):
            i = didx.get(d)
            if i is None:
                continue
            w = 1.0 / h / len(grp)       # one sleeve, split across the day's names
            j = min(i + h, len(dates) - 1)
            for t in grp["ticker"]:
                W[i + 1:j + 1, pos[t]] += w
            C[min(i + 1, len(dates) - 1)] += cost / h
            C[j] += cost / h
        # return earned on day k comes from weights set going into day k
        daily = (W * R).sum(axis=1) - C
        return np.cumprod(1 + daily)

    fired = p[p["fired"]][["date", "ticker"]]
    every = p[["date", "ticker"]]       # the whole universe, every day
    return pd.DataFrame({"date": dates, "equity": book(fired), "benchmark": book(every)})


# ---------- the gate ----------

def _sharpe(equity: np.ndarray) -> float:
    r = np.diff(equity) / equity[:-1]
    r = r[r != 0] if (r == 0).mean() > 0.5 else r
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 30 and r.std() > 0 else float("nan")


def kill_gate(res: RunResult, prices: dict[str, pd.DataFrame], min_fired: int = 100) -> dict:
    """Four checks on out-of-sample results. Any FAIL and the honest answer is 'not yet'.

    The one that matters most is excess_return: in a market that rises 60% of
    20-day windows, anything long makes money. The question is whether the
    fired names beat the rest of the universe on the same dates.
    """
    p = res.predictions
    f = p[p["fired"]]
    base = float(p["y"].mean())
    brier_model = float(np.mean((p["prob"] - p["y"]) ** 2))
    brier_base = float(np.mean((base - p["y"]) ** 2))
    skill = 1 - brier_model / brier_base

    n, k = len(f), int(f["y"].sum()) if len(f) else 0
    hit = k / n if n else float("nan")
    # base rate for the comparison is the universe's hit rate ON THE SAME DATES the signals fired
    same_day_base = float(p[p["date"].isin(f["date"])]["y"].mean()) if n else base
    pval = float(stats.binomtest(k, n, same_day_base, alternative="greater").pvalue) if n else 1.0

    # excess: fired name's forward return minus that day's universe average, one value per signal
    day_mean = p.groupby("date")["fwd_ret"].transform("mean")
    excess = (p["fwd_ret"] - day_mean)[p["fired"]].dropna()
    # signals overlap for `horizon` days, so the effective sample is roughly n / horizon
    n_eff = max(len(excess) / res.horizon, 2)
    t_stat = float(excess.mean() / (excess.std() / np.sqrt(n_eff))) if len(excess) > 2 and excess.std() > 0 else 0.0

    curve = paper_pnl(res, prices)
    sharpe_book, sharpe_bench = _sharpe(curve["equity"].values), _sharpe(curve["benchmark"].values)

    checks = {
        "calibration_skill": {"value": round(skill, 4), "pass": bool(skill > 0.0),
                               "rule": "Brier skill vs base rate above 0: the probabilities carry information"},
        "fired_hit_rate": {"value": round(hit, 4) if n else None, "n": n, "base_rate": round(same_day_base, 4),
                           "p_value": round(pval, 4), "pass": bool(n >= min_fired and pval < 0.05),
                           "rule": f"fired names rise more often than the universe on the same days, p < 0.05, {min_fired}+ signals"},
        "excess_return": {"value": round(float(excess.mean()), 5) if len(excess) else None, "t_stat": round(t_stat, 2),
                          "pass": bool(t_stat > 2.0),
                          "rule": f"fired names beat the same-day universe average over {res.horizon} days, t > 2 on overlap-adjusted n"},
        "paper_sharpe": {"value": round(sharpe_book, 3) if sharpe_book == sharpe_book else None,
                         "benchmark": round(sharpe_bench, 3) if sharpe_bench == sharpe_bench else None,
                         "pass": bool(sharpe_book == sharpe_book and sharpe_book > max(0.5, sharpe_bench)),
                         "rule": "paper Sharpe after 10 bps each way above 0.5 AND above holding the whole universe"},
    }
    checks["verdict"] = "PASS" if all(c["pass"] for key, c in checks.items() if key != "verdict") else "FAIL"
    return checks


def summary(res: RunResult) -> str:
    g = res.gate
    lines = [f"folds: {len(res.folds)}   test rows: {len(res.predictions):,}   fired: {int(res.predictions['fired'].sum()):,}",
             f"weights (latest fold): " + ", ".join(f"{k} {v:+.2f}" for k, v in res.weights_latest.items()),
             f"calibration skill: {g['calibration_skill']['value']}  ({'PASS' if g['calibration_skill']['pass'] else 'FAIL'})",
             f"fired hit rate: {g['fired_hit_rate']['value']} vs same-day base {g['fired_hit_rate']['base_rate']}  n={g['fired_hit_rate']['n']}  p={g['fired_hit_rate']['p_value']}  ({'PASS' if g['fired_hit_rate']['pass'] else 'FAIL'})",
             f"excess return per signal: {g['excess_return']['value']}  t={g['excess_return']['t_stat']}  ({'PASS' if g['excess_return']['pass'] else 'FAIL'})",
             f"paper Sharpe: {g['paper_sharpe']['value']} vs universe {g['paper_sharpe']['benchmark']}  ({'PASS' if g['paper_sharpe']['pass'] else 'FAIL'})",
             f"GATE: {g['verdict']}"]
    return "\n".join(lines)
