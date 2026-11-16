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

import sys
tpl_name = sys.argv[1] if len(sys.argv) > 1 else "console_reactor_template.html"
tpl = (HERE / tpl_name).read_text()
page = tpl.replace("/*__DATA__*/null", json.dumps(data))
if not page.lstrip().lower().startswith("<!doctype"):
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n') + page + "\n</html>\n"
(HERE / "results" / "console.html").write_text(page)
print("wrote results/console.html from", tpl_name, round(len(json.dumps(data)) / 1024), "KB data")
