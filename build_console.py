"""Bake the real run plus per-ticker price context into console.html.

Adds, per ticker: last close, last close date, 20-day daily vol, and the last 90
closes (for sparklines). The console's live tape starts from these.
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
data = json.loads((HERE / "results" / "latest.json").read_text())

quotes = {}
for t in data["universe"]:
    df = pd.read_csv(HERE / "cache" / f"{t}.csv", index_col=0, parse_dates=True)
    c = df["close"].dropna()
    vol = float(c.pct_change().tail(20).std())
    quotes[t] = {
        "last": round(float(c.iloc[-1]), 2),
        "prev": round(float(c.iloc[-2]), 2),
        "date": str(c.index[-1].date()),
        "vol": round(vol, 5),
        "spark": [round(float(x), 2) for x in c.tail(90)],
    }
data["quotes"] = quotes

# v3 edge lab: compact summaries of each tested world (synthetic worlds are labeled as such on the page)
v3 = {}
for w in ("synthA", "synthB", "real"):
    f = HERE / "results" / "v3" / w / "report.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    v3[w] = {"curve": r["event_study"]["curve"], "curve_days": r["event_study"]["curve_days"],
             "car_hold": r["event_study"]["car_hold_by_decile"], "H": r["holding_window"],
             "dating": r["dating"], "placebos": r["placebos"], "planted": r["planted"],
             "power": {"mde_monthly": r["power"]["mde_monthly"]}, "replication": r["replication"],
             "ls": r["calendar"]["ls_monthly"], "dsr": r["deflated_sharpe"]["dsr"], "pbo": r["pbo"]["pbo"],
             "checks": r["gate"]["checks"], "verdict": r["gate"]["verdict"]}
data["v3"] = v3

# console upgrade: investor view, pricing pilot, economics, live track record, provenance
def _load(rel):
    f = HERE / "results" / rel
    return json.loads(f.read_text()) if f.exists() else None

data["investor"] = _load("investor.json")
data["pilot"] = _load("pilot/report.json")
data["pilot_grid"] = _load("pilot/grid.json")
data["economics"] = _load("economics.json")
track = _load("track/track.json")
if track:
    track.pop("series", None)
data["track"] = track

import os
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

sha = os.environ.get("GITHUB_SHA", "")
if not sha:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
etfs = [t for t in data["universe"] if t.startswith("XL") or t in ("SPY", "QQQ", "IWM")]
data["prov"] = {
    "source": "Yahoo Finance daily closes (yfinance)",
    "built": datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M ") + "Central",
    "commit": sha[:7] or "local",
    "repo": "https://github.com/austin-stethem/quantlab",
    "survivorship": f"{len(etfs)} index and sector funds plus {len(data['universe']) - len(etfs)} large caps picked today, "
                    "so the single stocks are known survivors and the backtest flatters them",
}

import sys
tpl_name = sys.argv[1] if len(sys.argv) > 1 else "console_reactor_template.html"
tpl = (HERE / tpl_name).read_text()
page = tpl.replace("/*__DATA__*/null", json.dumps(data))
if not page.lstrip().lower().startswith("<!doctype"):
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n') + page + "\n</html>\n"
(HERE / "results" / "console.html").write_text(page)
print("wrote results/console.html from", tpl_name, round(len(json.dumps(data)) / 1024), "KB data")
