"""Investor view of the round-1 paper book, computed from the same run as results/latest.json.

Rebuilds the walk-forward exactly as run.py does, rebuilds the sleeve book with the same math as
consensus.paper_pnl (and checks it matches), then writes results/investor.json plus CSV exports in
results/data/. Nothing here changes the research; it only measures it.

    python tools/build_investor.py      # after `python run.py`
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantlab import data, consensus as C          # noqa: E402
from run import UNIVERSE                           # noqa: E402

OUT = ROOT / "results"
SECTOR = {"SPY": "Broad market", "QQQ": "Broad market", "IWM": "Broad market", "XLK": "Technology", "AAPL": "Technology",
          "MSFT": "Technology", "XLF": "Financials", "JPM": "Financials", "XLE": "Energy", "XOM": "Energy", "XLV": "Health care",
          "JNJ": "Health care", "UNH": "Health care", "XLI": "Industrials", "CAT": "Industrials", "DE": "Industrials",
          "XLP": "Consumer staples", "PG": "Consumer staples", "XLY": "Consumer discretionary", "HD": "Consumer discretionary",
          "XLU": "Utilities", "XLB": "Materials"}
STRESS = {"2020 crash (19 Feb to 23 Mar 2020)": ("2020-02-19", "2020-03-23"), "2022 (full year)": ("2022-01-01", "2022-12-31")}
ST_TAX = 0.30   # illustrative short-term rate, same as results/economics.json


def sleeve_book(rets: pd.DataFrame, signals: pd.DataFrame, h: int, cost: float):
    """Same math as consensus.paper_pnl: one sleeve per signal day, 1/h of capital, held h sessions."""
    dates, cols = rets.index, list(rets.columns)
    R = rets.fillna(0.0).values
    W = np.zeros_like(R); Cd = np.zeros(len(dates)); entries = 0
    didx = {d: i for i, d in enumerate(dates)}
    for d, grp in signals.groupby("date"):
        i = didx.get(d)
        if i is None:
            continue
        w = 1.0 / h / len(grp); j = min(i + h, len(dates) - 1); entries += 1
        for t in grp["ticker"]:
            W[i + 1:j + 1, cols.index(t)] += w
        Cd[min(i + 1, len(dates) - 1)] += cost / h
        Cd[j] += cost / h
    gross = (W * R).sum(1)
    return gross - Cd, gross, W, entries


def metrics(r: pd.Series, spy: pd.Series, exposure: pd.Series | None = None, turnover: float | None = None) -> dict:
    r = r.dropna(); yrs = len(r) / 252
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    vol = r.std() * np.sqrt(252)
    down = r[r < 0].std() * np.sqrt(252)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    s = spy.reindex(r.index).fillna(0.0)
    beta = float(np.cov(r, s)[0, 1] / s.var())
    alpha = float((r.mean() - beta * s.mean()) * 252)
    ex = r - s
    out = {"cagr": float(cagr), "vol": float(vol), "sharpe": float(r.mean() * 252 / vol) if vol else None,
           "sortino": float(r.mean() * 252 / down) if down else None, "max_dd": float(dd.min()),
           "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else None, "beta": beta, "alpha": alpha,
           "info_ratio": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() else None, "growth": float(eq.iloc[-1])}
    if exposure is not None:
        out["time_invested"] = float((exposure > 1e-9).mean()); out["avg_exposure"] = float(exposure.mean())
    if turnover is not None:
        out["turnover"] = turnover
    return out


def after_tax(r: pd.Series, rate: float = ST_TAX) -> pd.Series:
    """Illustrative: tax each calendar year's net gain at the short-term rate, losses carried forward."""
    eq, carry, vals = 1.0, 0.0, []
    for _, grp in r.groupby(r.index.year):
        start = eq
        for x in grp:
            eq *= 1 + x; vals.append(eq)
        gain = eq - start + carry
        if gain > 0:
            eq -= gain * rate; carry = 0.0; vals[-1] = eq
        else:
            carry = gain
    return pd.Series(vals, index=r.index)


def main():
    L = json.loads((OUT / "latest.json").read_text())
    h, thr, quo = L["horizon"], L["threshold"], L["quorum"]
    prices = data.fetch_many(UNIVERSE, start="2012-01-01")
    panel = C.build_panel(prices, horizon=h)
    res = C.walk_forward(panel, threshold=thr, quorum=quo, horizon=h)
    p = res.predictions
    rets_all = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()})
    rets = rets_all.loc[p["date"].min():p["date"].max()]
    fired = p[p["fired"]][["date", "ticker", "prob", "agree", "fwd_ret"]]
    net, gross, W, entries = sleeve_book(rets, fired, h, 0.001)
    uni, _, _, _ = sleeve_book(rets, p[["date", "ticker"]], h, 0.001)
    ref = C.paper_pnl(res, prices)
    assert np.allclose(np.cumprod(1 + net), ref["equity"].values, rtol=1e-9), "sleeve book does not match paper_pnl"
    idx = rets.index
    book, bookg, unir = pd.Series(net, idx), pd.Series(gross, idx), pd.Series(uni, idx)
    spy, qqq = rets["SPY"].fillna(0.0), rets["QQQ"].fillna(0.0)
    expo = pd.Series(W.sum(1), idx)
    yrs = len(idx) / 252
    turnover = entries * (1 / h) * 2 / yrs                     # each sleeve buys then sells 1/h of capital
    m = {"Paper book (net of costs)": metrics(book, spy, expo, turnover), "Hold everything (same sleeves)": metrics(unir, spy),
         "SPY": metrics(spy, spy), "QQQ (Nasdaq-100)": metrics(qqq, spy)}
    eq = lambda r: (1 + r).cumprod()
    wk = idx[::5].append(idx[-1:]).unique()
    series = {"dates": [str(d.date()) for d in wk],
              "book_net": eq(book).reindex(wk).round(5).tolist(), "book_gross": eq(bookg).reindex(wk).round(5).tolist(),
              "book_after_tax": after_tax(book).reindex(wk).round(5).tolist(), "universe": eq(unir).reindex(wk).round(5).tolist(),
              "spy": eq(spy).reindex(wk).round(5).tolist(), "qqq": eq(qqq).reindex(wk).round(5).tolist(),
              "dd_book": (eq(book) / eq(book).cummax() - 1).reindex(wk).round(5).tolist(),
              "dd_spy": (eq(spy) / eq(spy).cummax() - 1).reindex(wk).round(5).tolist(),
              "exposure": expo.reindex(wk).round(4).tolist(),
              "roll12_excess": ((1 + book).rolling(252).apply(np.prod, raw=True) - (1 + spy).rolling(252).apply(np.prod, raw=True)).reindex(wk).round(5).tolist()}
    mon = (1 + book).groupby([idx.year, idx.month]).prod() - 1
    heat = {int(y): {int(mo): round(float(v), 5) for (yy, mo), v in mon.items() if yy == y} for y in sorted(set(idx.year))}
    sec = fired.assign(sector=fired["ticker"].map(SECTOR)).groupby("sector").size()
    sector = (sec / sec.sum()).sort_values(ascending=False).round(4).to_dict()
    stress = {}
    for name, (a, b) in STRESS.items():
        w = slice(a, b)
        stress[name] = {k: float((1 + s.loc[w]).prod() - 1) for k, s in (("book", book), ("spy", spy), ("qqq", qqq))}
        stress[name]["book_exposure"] = float(expo.loc[w].mean())
    # today's book: live read for the sessions the backtest cannot label yet, scored by the final model
    names = res.scorer_names
    train = panel.dropna(subset=["y"])
    lr, iso = C._fit(train[names].values, train["y"].values)
    last_dates = sorted(panel["date"].unique())[-h:]
    recent = panel[panel["date"].isin(last_dates)].copy()
    recent["prob"] = C._predict(lr, iso, recent[names].values)
    recent["agree"] = (recent[names].values > 0.55).sum(1)
    recent["fired"] = (recent["prob"] >= thr) & (recent["agree"] >= quo)
    trade_days = rets_all.index
    holds = []
    for r in recent[recent["fired"]].itertuples():
        close = prices[r.ticker]["close"]
        k = trade_days.get_loc(r.date)
        exit_d = trade_days[k + h] if k + h < len(trade_days) else (pd.Timestamp(r.date) + pd.offsets.BDay(h))
        holds.append({"ticker": r.ticker, "signal_date": str(pd.Timestamp(r.date).date()), "prob": round(float(r.prob), 3),
                      "agree": int(r.agree), "entry_close": round(float(close.loc[r.date]), 2), "last_close": round(float(close.iloc[-1]), 2),
                      "pnl": round(float(close.iloc[-1] / close.loc[r.date] - 1), 5), "planned_exit": str(pd.Timestamp(exit_d).date())})
    out = {"asof": L["asof"], "first_date": str(idx[0].date()), "last_marked": str(idx[-1].date()), "horizon": h,
           "cost_bps_each_way": 10, "tax_rate_illustrative": ST_TAX, "metrics": m, "series": series, "monthly": heat,
           "sector_mix": sector, "stress": stress, "book_today": holds, "recent_sessions": len(last_dates),
           "signals_total": int(len(fired)), "signal_days": entries}
    (OUT / "investor.json").write_text(json.dumps(out, indent=1))
    dd = OUT / "data"; dd.mkdir(exist_ok=True)
    pd.DataFrame([{"ticker": t["ticker"], "model_prob": t["prob"], "agree_of_6": t["agree"], "fired": t["fired"], **t["scores"]}
                  for t in L["today"]]).to_csv(dd / "signals.csv", index=False)
    fired.assign(exit_after_sessions=h).rename(columns={"date": "signal_date", "fwd_ret": "return_next_20_sessions"}) \
        .to_csv(dd / "trades.csv", index=False)
    pd.DataFrame({"date": idx.date, "book_net": eq(book).values, "book_gross": eq(bookg).values, "hold_everything": eq(unir).values,
                  "spy": eq(spy).values, "qqq": eq(qqq).values, "exposure": expo.values}).round(6).to_csv(dd / "equity.csv", index=False)
    print("investor.json written:", {k: round(v["cagr"], 4) for k, v in m.items()}, "| open positions:", len(holds))


if __name__ == "__main__":
    main()
