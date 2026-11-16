"""Live paper-trading log, kept separate from the backtest.

Each daily run appends that day's fired names (from results/latest.json) to results/track/ledger.csv
and the day itself to results/track/days.csv, then marks a sleeve book from the first logged day to
the latest close: same rules as the backtest (one sleeve per signal day, 1/h of capital, h sessions,
10 bps each way). The CI job commits these files back so the record survives between runs.

    python tools/track.py        # after `python run.py`
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantlab import data   # noqa: E402
from run import UNIVERSE    # noqa: E402

TR = ROOT / "results" / "track"


def main():
    TR.mkdir(parents=True, exist_ok=True)
    L = json.loads((ROOT / "results" / "latest.json").read_text())
    asof, h = L["asof"], L["horizon"]
    days_f, led_f = TR / "days.csv", TR / "ledger.csv"
    days = pd.read_csv(days_f) if days_f.exists() else pd.DataFrame(columns=["date", "fired"])
    led = pd.read_csv(led_f) if led_f.exists() else pd.DataFrame(columns=["date", "ticker", "prob", "agree", "entry_close"])
    if asof not in set(days["date"].astype(str)):
        fired = [t for t in L["today"] if t["fired"]]
        prices = data.fetch_many(UNIVERSE, start="2012-01-01")
        rows = [{"date": asof, "ticker": t["ticker"], "prob": t["prob"], "agree": t["agree"],
                 "entry_close": round(float(prices[t["ticker"]]["close"].iloc[-1]), 4)} for t in fired]
        led = pd.concat([led, pd.DataFrame(rows)], ignore_index=True) if rows else led
        days = pd.concat([days, pd.DataFrame([{"date": asof, "fired": len(fired)}])], ignore_index=True)
        days.to_csv(days_f, index=False); led.to_csv(led_f, index=False)
    start = str(days["date"].astype(str).min())
    prices = data.fetch_many(UNIVERSE, start="2012-01-01")
    rets = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()}).loc[start:]
    idx = rets.index
    W = np.zeros((len(idx), len(rets.columns))); Cd = np.zeros(len(idx)); cols = list(rets.columns)
    for d, grp in led.groupby("date"):
        i = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(d))))
        if i >= len(idx):
            continue
        w = 1.0 / h / len(grp); j = min(i + h, len(idx) - 1)
        for t in grp["ticker"]:
            W[i + 1:j + 1, cols.index(t)] += w
        if i + 1 < len(idx):
            Cd[i + 1] += 0.001 / h
        if i + h < len(idx):
            Cd[j] += 0.001 / h
    daily = (W * rets.fillna(0.0).values).sum(1) - Cd
    book = np.cumprod(1 + daily); spy = np.cumprod(1 + rets["SPY"].fillna(0.0).values)
    out = {"start": start, "asof": asof, "days_logged": int(len(days)), "signals": int(len(led)),
           "book_return": float(book[-1] - 1), "spy_return": float(spy[-1] - 1),
           "series": {"dates": [str(d.date()) for d in idx], "book": np.round(book, 6).tolist(), "spy": np.round(spy, 6).tolist()},
           "recent_signals": led.tail(12).to_dict(orient="records")}
    (TR / "track.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"track: start {start}, {len(days)} days, {len(led)} signals, book {out['book_return']:+.2%} vs SPY {out['spy_return']:+.2%}")


if __name__ == "__main__":
    main()
