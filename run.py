"""Run quantlab v2 on a real universe and write results/latest.json for the dashboard.

    python run.py                 # default universe, 20-day horizon
    python run.py --refresh       # re-download prices
    python run.py --horizon 10
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from quantlab import data, consensus as C

UNIVERSE = [
    # broad market and sectors
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
    # large caps across sectors
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "CAT", "HD", "DE", "UNH",
]

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--quorum", type=int, default=4)
    ap.add_argument("--relative", action="store_true", help="label = beats the same-day universe median")
    a = ap.parse_args()

    print("prices...")
    prices = data.fetch_many(UNIVERSE, start="2012-01-01", refresh=a.refresh)
    print(f"  {len(prices)} tickers, {min(len(d) for d in prices.values())}-{max(len(d) for d in prices.values())} days each")

    print("panel...")
    panel = C.build_panel(prices, horizon=a.horizon, relative=a.relative)
    print(f"  {len(panel):,} rows")

    print("walk-forward...")
    res = C.walk_forward(panel, threshold=a.threshold, quorum=a.quorum, horizon=a.horizon)
    res.gate = C.kill_gate(res, prices)
    print(C.summary(res))

    # today's read: score the latest bar per ticker with the latest fold's model
    names = res.scorer_names
    train = panel.dropna(subset=["y"])
    lr, iso = C._fit(train[names].values, train["y"].values)
    latest = panel.sort_values("date").groupby("ticker").tail(1).copy()
    latest["prob"] = C._predict(lr, iso, latest[names].values)
    latest["agree"] = (latest[names].values > 0.55).sum(axis=1)
    latest["fired"] = (latest["prob"] >= a.threshold) & (latest["agree"] >= a.quorum)
    latest = latest.sort_values("prob", ascending=False)

    curve = C.paper_pnl(res, prices)

    payload = {
        "asof": str(panel["date"].max().date()),
        "horizon": a.horizon, "threshold": a.threshold, "quorum": a.quorum, "label": "beats universe median" if a.relative else "rises",
        "universe": list(prices),
        "scorers": names,
        "gate": res.gate,
        "weights": res.weights_latest,
        "folds": [f.__dict__ for f in res.folds],
        "calibration": res.calibration.to_dict(orient="records"),
        "equity": {"date": [str(d.date()) for d in curve["date"]], "book": [round(float(x), 5) for x in curve["equity"]],
                   "benchmark": [round(float(x), 5) for x in curve["benchmark"]]},
        "today": [
            {"ticker": r.ticker, "prob": round(float(r.prob), 3), "agree": int(r.agree), "fired": bool(r.fired),
             "scores": {n: round(float(getattr(r, n)), 3) for n in names}}
            for r in latest.itertuples()
        ],
        "recent_fired": [
            {"date": str(r.date.date()), "ticker": r.ticker, "prob": round(float(r.prob), 3),
             "fwd_ret": None if pd.isna(r.fwd_ret) else round(float(r.fwd_ret), 4), "hit": None if pd.isna(r.y) else bool(r.y)}
            for r in res.predictions[res.predictions["fired"]].sort_values("date").tail(40).itertuples()
        ],
    }
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return None if np.isnan(o) else float(o)
        return o
    (OUT / "latest.json").write_text(json.dumps(clean(payload), indent=1))
    print(f"wrote {OUT / 'latest.json'}")


if __name__ == "__main__":
    main()
