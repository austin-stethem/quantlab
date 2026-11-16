"""Freeze the study's round-1 numbers and the public economics into result files the README is built from.

    python tools/freeze_results.py     # needs results/latest.json from `python run.py`
"""
import json, math, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def drawdown(x: pd.Series) -> float:
    return float((x / x.cummax() - 1).min())


def round1():
    import sys; sys.path.insert(0, str(ROOT))
    from quantlab import data, consensus as C
    from run import UNIVERSE
    L = json.loads((OUT / "latest.json").read_text())
    eq = pd.DataFrame({"book": L["equity"]["book"], "universe": L["equity"]["benchmark"]}, index=pd.to_datetime(L["equity"]["date"]))
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    prices = data.fetch_many(UNIVERSE, start="2012-01-01")
    res = C.walk_forward(C.build_panel(prices, horizon=L["horizon"]), horizon=L["horizon"])
    p = res.predictions.dropna(subset=["fwd_ret"]).copy()
    p["base"] = p.groupby("date")["y"].transform("mean")
    bars = []
    for thr in (0.55, 0.60, 0.65, 0.70, 0.80):
        f = p[(p.prob >= thr) & (p.agree >= L["quorum"])]
        bars.append({"bar": thr, "signals": len(f), "rose": float(f.y.mean()), "base": float(f.base.mean())})
    g = L["gate"]
    return {"asof": L["asof"], "universe_size": len(L["universe"]), "first_test_year": int(L["folds"][0]["test_year"]),
            "last_test_year": int(L["folds"][-1]["test_year"]), "horizon": L["horizon"], "threshold": L["threshold"], "quorum": L["quorum"],
            "verdict": g["verdict"],
            "calibration_skill": g["calibration_skill"]["value"],
            "hit_rate": g["fired_hit_rate"]["value"], "base_rate": g["fired_hit_rate"]["base_rate"], "hit_p": g["fired_hit_rate"]["p_value"],
            "excess": g["excess_return"]["value"], "excess_t": g["excess_return"]["t_stat"],
            "sharpe_book": g["paper_sharpe"]["value"], "sharpe_universe": g["paper_sharpe"]["benchmark"],
            "growth_book": float(eq.book.iloc[-1] / eq.book.iloc[0]), "growth_universe": float(eq.universe.iloc[-1] / eq.universe.iloc[0]),
            "mdd_book": drawdown(eq.book), "mdd_universe": drawdown(eq.universe),
            "worst_year_book": [int(yearly.book.idxmin().year), float(yearly.book.min())],
            "worst_year_universe": [int(yearly.universe.idxmin().year), float(yearly.universe.min())],
            "bars": bars}


def economics(st=0.30, lt=0.20, mkt=0.08, yrs=10, rt_cost=0.0020, hold_months=2.86):
    """Illustrative public inputs, not the owner's: 30% short-term, 20% long-term, 8% market, 10 years."""
    drag = rt_cost * 12 / hold_months
    spy = 1 + ((1 + mkt) ** yrs - 1) * (1 - lt)
    grow = lambda e, taxed=True: (1 + (mkt + e) * (1 - st)) ** yrs if taxed else (1 + mkt + e) ** yrs
    lo, hi = 0.0, 0.2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if grow(mid) < spy else (lo, mid)
    eras = {}
    for name, ls in (("2005-2024", 0.0061), ("since 2015", 0.0036)):
        net = ls / 2 * 12 - drag
        eras[name] = {"ls_monthly": ls, "long_gross_annual": ls / 2 * 12, "long_net_annual": net,
                      "taxable_vs_spy_per_dollar_yr": (grow(net) - spy) / yrs,
                      "taxfree_vs_spy_per_dollar_yr": (grow(net, False) - (1 + mkt) ** yrs) / yrs}
    sens = []
    for st2 in (0.22, 0.30, 0.37):
        lo2, hi2 = 0.0, 0.2
        g2 = lambda e: (1 + (mkt + e) * (1 - st2)) ** yrs
        for _ in range(60):
            m2 = (lo2 + hi2) / 2
            lo2, hi2 = (m2, hi2) if g2(m2) < spy else (lo2, m2)
        sens.append({"short_term_rate": st2, "hurdle": m2})
    return {"sensitivity": sens, "inputs": {"short_term_rate": st, "long_term_rate": lt, "market": mkt, "years": yrs, "round_trip_cost": rt_cost,
                       "hold_months": hold_months}, "cost_drag_annual": drag, "tax_hurdle_annual": mid, "eras": eras}


if __name__ == "__main__":
    (OUT / "round1").mkdir(parents=True, exist_ok=True)
    (OUT / "round1" / "summary.json").write_text(json.dumps(round1(), indent=1))
    (OUT / "economics.json").write_text(json.dumps(economics(), indent=1))
    print((OUT / "round1" / "summary.json").read_text()[:1500]); print((OUT / "economics.json").read_text())
