"""Quantlab v3: post-earnings drift measured by the announcement return, tested end to end.

    python run_v3.py --world synthA      # synthetic market, no drift   (gate must FAIL)
    python run_v3.py --world synthB      # synthetic market, planted attention-driven drift (gate should PASS)
    python run_v3.py --world real        # real data (needs the EDGAR + EODHD pulls)
    add --open-holdout once, at the very end
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd

from v3 import core as K, stats as S, trust as T, mechanism as MX, exits as X, synth

HERE = pathlib.Path(__file__).parent
OSAP = HERE.parent / "qlraw" / "osap" / "osap_ports.csv"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (pd.Period, pd.Timestamp)):
        return str(o)
    return o


def load(world: str) -> K.Market:
    if world == "synthA":
        return synth.make_market(seed=7)
    if world == "synthB":
        return synth.make_market(seed=7, drift=0.009, friday_boost=1.5, busy_boost=0.8, inst_boost=0.5)
    if world == "real":
        from v3 import realdata
        return realdata.load_market()
    raise SystemExit(f"unknown world {world}")


def build_events(m, ew):
    ev = K.compute_ear(m, K.dedupe_events(m.filings, m.periods), ew)
    return ev


def run(world: str, seeds: int, power_seeds: int, open_holdout: bool):
    t0 = time.time()
    out = HERE / "results" / "v3" / world
    out.mkdir(parents=True, exist_ok=True)
    full = load(world)
    D = K.dev_slice(full)
    ew = K.ew_universe(D)
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
    rep = {"world": world, "dev": [str(D.days[0].date()), str(D.days[-1].date())],
           "meta": {k: v for k, v in full.meta.items() if k != "truth"}}

    # 3. event integrity
    raw = K.dedupe_events(D.filings, D.periods)
    ev = K.compute_ear(D, raw, ew)
    rep["events"] = {"item202_filings": int((D.filings["form"] == "8-K").sum()), "amendments_dropped": int((D.filings["form"] == "8-K/A").sum()),
                     "one_per_quarter": len(raw), "with_ear_in_universe": len(ev)}
    rep["alignment"] = T.alignment(D, ev, ew)
    rep["dating"] = T.dating_quality(D, ev, ew)
    if "truth" in full.meta:
        tr = full.meta["truth"]
        j = ev.merge(tr, on=["cik", "accepted"], how="inner")
        rep["dating"]["true_correct_share"] = float((j["day0"] == j["true_d0"]).mean())
    log("dating", {k: rep["dating"].get(k) for k in ("correct_share", "true_correct_share")}, "raw max-share", round(rep["alignment"]["day0_share"], 3))
    ev = K.assign_deciles(ev)
    ev["month"] = D.days[ev["entry_idx"].values].to_period("M")
    rep["events"]["ranked"] = len(ev)

    # 4. power first
    rep["power"] = T.power(D, ev, ew, seeds=power_seeds)
    log("power MDE monthly", rep["power"]["mde_monthly"])

    # 5. pipeline trust
    rep["placebos"] = T.run_placebos(D, ev, ew, seeds=seeds)
    log("placebos", {k: v["hits_abs_t_gt_2"] for k, v in rep["placebos"].items()})
    rep["planted"] = T.recovery(D, ev, ew, delta=0.02)
    log("planted ratio", rep["planted"]["ratio"])

    # 6. event study
    cell, bench = K.dgtw_benchmark(D)
    ar = K.abnormal_returns(D, ev, cell, bench, ew)
    sig = K.estimation_sigma(D, ev, ew)
    curve = X.drift_curve(ev, ar)
    H = X.pick_window(curve)
    rep["holding_window"] = H
    ev["car_hold"] = K.car(ar, K.ENTRY_LAG + 1, K.ENTRY_LAG + H)
    for h in (5, 20, 40, 60):
        ev[f"car_{h}"] = K.car(ar, K.ENTRY_LAG + 1, K.ENTRY_LAG + h)
    sar = ev["car_hold"].values / (sig * math.sqrt(H))
    dec = ev["decile"].values
    rep["event_study"] = {
        "car_hold_by_decile": {int(d): float(np.nanmean(ev["car_hold"].values[dec == d])) for d in range(1, 11)},
        "bmp_by_decile": {int(d): S.bmp(sar[dec == d]) for d in (1, 10)},
        "welch_t_d10_vs_d1": S.welch(sar[dec == 10], sar[dec == 1]),
        "curve": {int(d): [round(float(v), 5) for v in curve.loc[d].values] for d in curve.index},
        "curve_days": [int(c) for c in curve.columns]}
    log("holding window", H, "D10-D1 CAR", rep["event_study"]["car_hold_by_decile"][10] - rep["event_study"]["car_hold_by_decile"][1])

    # primary: calendar-time portfolios
    cal = K.calendar(D, ev, H)
    L = K.legs(D, cal)
    mo = K.monthly(L["ls"])
    rep["calendar"] = {
        "ls_monthly": S.nw_mean(mo, 3),
        "ls_ff6_alpha_daily": S.factor_alpha(L["ls"], D.ff, lags=H) if D.ff is not None else None,
        "long_x_spy_daily": S.nw_mean(L["long_x_spy"][cal.attrs["count"][10] > 0], H),
        "short_x_spy_daily": S.nw_mean(L["short_x_spy"][cal.attrs["count"][1] > 0], H),
        "ls_bootstrap_ci_monthly": S.block_bootstrap_ci(mo, block=6),
        "bug_flag": bool(mo.mean() > 0.012)}
    rep["replication"] = T.replication(mo, str(OSAP) if world == "real" else None)
    ic = {h: S.rank_ic(ev, "pct", f"car_{h}", lags=max(1, h // 21 + 1)) for h in (5, 20, 40, 60)}
    rep["ic_by_horizon"] = {h: {k: v for k, v in r.items() if k != "ic_series"} for h, r in ic.items()}
    means = [rep["event_study"]["car_hold_by_decile"][d] for d in range(1, 11)]
    rep["monotonicity"] = {"spearman_decile_vs_car": float(pd.Series(means).corr(pd.Series(range(10)), method="spearman")),
                           "adjacent_inversions": int(sum(b < a for a, b in zip(means, means[1:])))}
    ev["pct_c"] = ev["pct"] - 0.5
    rep["fama_macbeth"] = S.fama_macbeth(ev, "car_hold", ["pct_c"])
    log("calendar LS monthly", rep["calendar"]["ls_monthly"], "FF alpha t", (rep["calendar"]["ls_ff6_alpha_daily"] or {}).get("t"))

    # 7. mechanism
    ea = MX.attention_vars(D, ev)
    rep["mechanism"] = MX.interactions(ea, "car_hold")
    log("mechanism", {k: v.get("t") for k, v in rep["mechanism"].items()})

    # 8. exits and sizing (long-only top decile is the tradeable version)
    trials = {}
    def add_trial(name, x):
        x = x.copy()
        trials[name] = x
    top = ev[ev["decile"] == 10].reset_index(drop=True)
    sig_top = sig[ev["decile"].values == 10]
    for h in (20, 40, 60):
        c = K.calendar(D, ev, h)
        add_trial(f"long_hold{h}", (c[10] - D.spy).where(c.attrs["count"][10] > 0, 0.0))
        add_trial(f"ls_hold{h}", c[10] - c[1])
    for k in (1.0, 1.5, 2.0):
        held = X.triple_barrier(D, top, sig_top, H, k)
        c = K.calendar(D, top, H, exit_after=held)
        add_trial(f"long_barrier{k}", (c[10] - D.spy).where(c.attrs["count"][10] > 0, 0.0))
    ea_top = ea[ea["decile"] == 10].copy()
    ea_top["abs_ear"] = ea_top["ear"].abs()
    ea_top["presig"] = sig[ev["decile"].values == 10][:len(ea_top)] if len(ea_top) == len(top) else np.nan
    size = np.log(D.mcap[np.maximum(top["day0"].values - 1, 0), top["col"].values])
    ea_top = ea_top.set_index(["cik", "accepted"]).reindex(pd.MultiIndex.from_frame(top[["cik", "accepted"]])).reset_index()
    ea_top["logsize"] = size
    ea_top["presig"] = sig_top
    feats = [f for f in ["ear", "abs_ear", "logsize", "presig", "friday", "busy", "instown"] if f in ea_top and ea_top[f].notna().any()]
    ml = X.meta_label(ea_top.assign(abs_ear=ea_top["ear"].abs()), feats, ea_top["car_hold"].values, H)
    take = np.nan_to_num(ml["prob"], nan=0.0) > 0.5
    if take.sum() > 50:
        c = K.calendar(D, top[take].reset_index(drop=True), H)
        add_trial("long_meta", (c[10] - D.spy).where(c.attrs["count"][10] > 0, 0.0))
    w = 1 / np.where(np.isfinite(sig_top) & (sig_top > 0), sig_top, np.nanmedian(sig_top))
    c = K.calendar(D, top, H, weight=w)
    add_trial("long_invvol_voltarget", X.vol_target((c[10] - D.spy).where(c.attrs["count"][10] > 0, 0.0)))
    rep["meta_label"] = {"auc_mean": ml["auc_mean"], "auc_min": ml["auc_min"], "splits": ml["splits"], "features": feats,
                         "taken": int(take.sum()), "of": len(top)}
    TR = pd.DataFrame(trials)
    rows = []
    for n, x in TR.items():
        r = S.nw_mean(x[x != 0], H)
        rows.append({"trial": n, "mean_daily": r["mean"], "t": r["t"], "sr_ann": S.sharpe(x[x != 0]) * math.sqrt(252),
                     "monthly_mean": float(K.monthly(x).mean())})
    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(out / "trials.csv", index=False)
    head = f"long_hold{H}"
    longs = [c for c in TR if c.startswith("long")]
    rep["exits"] = {"headline": head, "table": trials_df.round(5).to_dict(orient="records"),
                    "barrier_beats_hold": {k: bool(trials_df.set_index("trial").loc[f"long_barrier{k}", "sr_ann"] >
                                                   trials_df.set_index("trial").loc[head, "sr_ann"]) for k in (1.0, 1.5, 2.0)}}
    hx = TR[head][TR[head] != 0]
    rep["deflated_sharpe"] = S.deflated_sharpe(hx, [S.sharpe(TR[c][TR[c] != 0]) for c in longs])
    rep["pbo"] = S.pbo_cscv(TR[longs])
    log("DSR", rep["deflated_sharpe"]["dsr"], "PBO", rep["pbo"]["pbo"])

    # 9. gate
    mech = {k: v for k, v in rep["mechanism"].items() if k in ("friday", "busy", "instown") and v.get("available")}
    ff = rep["calendar"]["ls_ff6_alpha_daily"] or {}
    g = {
        "dating_85": rep["dating"]["correct_share"] >= 0.85,
        "placebos_empty": all(v["pass"] for v in rep["placebos"].values()),
        "planted_recovered": bool(rep["planted"]["pass"]),
        "replication_corr_05": rep["replication"]["pass"],
        "calendar_alpha_t2": bool(ff.get("t") is not None and ff["t"] > 2 and ff["alpha"] > 0),
        "long_beats_spy_t2": bool(rep["calendar"]["long_x_spy_daily"]["t"] > 2),
        "mechanism_2_of_3": sum(v["pass"] for v in mech.values()) >= 2,
        "dsr_095": rep["deflated_sharpe"]["dsr"] >= 0.95,
        "holdout": None,
    }
    if open_holdout:
        g["holdout"], rep["holdout"] = holdout(full, world, H, mo)
    required = [k for k, v in g.items() if v is not None]
    rep["gate"] = {"checks": g, "verdict": "PASS" if all(g[k] for k in required) and g["holdout"] is not None else
                   ("FAIL" if not all(g[k] for k in required) else "PENDING HOLDOUT"),
                   "not_applicable": [k for k, v in g.items() if v is None]}
    log("GATE", rep["gate"]["verdict"], g)

    # console payload
    rep["series"] = {"dates": [str(d.date()) for d in D.days[::5]],
                     "ls_equity": list(np.round(np.cumprod(1 + L["ls"].values)[::5], 5)),
                     "long_equity": list(np.round(np.cumprod(1 + L["long"].values)[::5], 5)),
                     "spy_equity": list(np.round(np.cumprod(1 + D.spy)[::5], 5))}
    latest = ev[ev["signal_idx"] >= ev["signal_idx"].max() - K.POOL][["cik", "day0", "ear", "pct", "decile"]]
    rep["latest_book"] = {"buy": latest[latest.decile == 10].cik.astype(str).tolist()[-20:],
                          "sell_avoid": latest[latest.decile == 1].cik.astype(str).tolist()[-20:]}
    (out / "report.json").write_text(json.dumps(clean(rep), indent=1))
    log("wrote", out / "report.json")
    return rep


def holdout(full: K.Market, world: str, H: int, dev_mo: pd.Series):
    flag = HERE / "results" / "v3" / world / "HOLDOUT_OPENED.txt"
    if world == "real" and flag.exists():
        raise SystemExit("holdout already opened once: " + flag.read_text())
    ew = K.ew_universe(full)
    ev = K.assign_deciles(K.compute_ear(full, K.dedupe_events(full.filings, full.periods), ew))
    ev = ev[full.days[ev["entry_idx"].values] >= K.HOLDOUT_START]
    mo = K.monthly(K.legs(full, K.calendar(full, ev, H)).ls)
    mo = mo[mo.index >= K.HOLDOUT_START.to_period("M")]
    r = S.nw_mean(mo, 3)
    ok = bool(np.sign(r["mean"]) == np.sign(dev_mo.mean()) and abs(r["mean"]) >= 0.5 * abs(dev_mo.mean()))
    flag.write_text(f"opened {pd.Timestamp.now()} for world={world}\n")
    return ok, {"ls_monthly": r, "dev_ls_monthly": float(dev_mo.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="synthA")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--power-seeds", type=int, default=12)
    ap.add_argument("--open-holdout", action="store_true")
    a = ap.parse_args()
    run(a.world, a.seeds, a.power_seeds, a.open_holdout)
