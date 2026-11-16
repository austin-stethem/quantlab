"""A simulated multi-site price pilot: the quick read vs the disciplined read.

The network: 120 locations in 6 regions, weekly visits and average ticket since January 2024, with
seasonality, holiday dips, a company-wide list-price increase every January, regional growth
momentum, location-level drift and noise. The export runs through a partial current week.

The pilot as run: leadership picks its 20 fastest-growing locations and raises prices 3% for 12 weeks.
The truth, known only to the simulation: clients respond with an elasticity of -1.0, so visits fall
about as much as the ticket rises, and revenue barely moves.

Quick read: pilot comp vs the rest of the network. Disciplined read: a matched holdout against each
location's own pre-selection run rate, then placebo launches at past dates using the same selection
rule (what the design "finds" when nothing changed), then a power check from the placebo spread.

    python pilot/pilot.py      # writes results/pilot/report.json and results/pilot/pilot_weekly.csv
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "pilot"
SITES, REGIONS, PILOT_N, WIN = 120, 6, 20, 12
PRICE_UP, ELASTICITY = 0.03, -1.0
HURDLE = 0.02                      # roll out only on a revenue lift of at least 2 points
WEEKS = pd.date_range("2024-01-07", "2026-11-22", freq="W-SUN")   # last week is partial (data through Mon 16 Nov)
LAUNCH = int(np.searchsorted(WEEKS, pd.Timestamp("2026-08-23")))
FAKE_LAUNCHES = list(range(92, LAUNCH - WIN + 1, 2))            # untreated past dates with enough history


def simulate(seed: int, elasticity: float = ELASTICITY, treat: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    T = len(WEEKS)
    region = np.repeat(np.arange(REGIONS), SITES // REGIONS)
    base_visits = 180 * np.exp(rng.normal(0, 0.35, SITES))
    base_ticket = 240 * np.exp(rng.normal(0, 0.12, SITES))
    woy = WEEKS.isocalendar().week.values.astype(float)
    season = 1 + 0.06 * np.sin(2 * np.pi * (woy - 10) / 52)
    hol = np.array([0.88 if (d.month == 11 and 22 <= d.day <= 30) else 0.82 if (d.month == 12 and d.day >= 24)
                    else 0.92 if (d.month == 1 and d.day <= 7) else 1.0 for d in WEEKS])
    g = np.zeros((T, REGIONS))
    for t in range(1, T):
        g[t] = 0.90 * g[t - 1] + rng.normal(0, 0.0008, REGIONS)
    region_lvl = np.cumsum(g + 0.0006, axis=0)
    site_trend = rng.normal(0, 0.00015, SITES)
    u = np.zeros((T, SITES))
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + rng.normal(0, 0.05, SITES)
    visits = np.exp(np.log(base_visits)[None, :] + region_lvl[:, region] + site_trend[None, :] * np.arange(T)[:, None] + u) \
        * (season * hol)[:, None]
    jan = np.cumsum([1 if (d.month == 1 and d.day <= 7) else 0 for d in WEEKS])
    ticket = base_ticket[None, :] * (1.035 ** (jan - jan[0]))[:, None] * np.exp(rng.normal(0, 0.03, (T, SITES)))
    pilot = select(visits * ticket, LAUNCH)
    if treat:
        ramp = np.clip((np.arange(T) - LAUNCH + 1) / 4, 0, 1)[:, None]
        on = (np.arange(T) >= LAUNCH)[:, None] & pilot[None, :]
        ticket = np.where(on, ticket * (1 + PRICE_UP), ticket)
        visits = np.where(on, visits * (1 + PRICE_UP) ** (elasticity * ramp), visits)
    return {"visits": visits, "ticket": ticket, "rev": visits * ticket, "pilot": pilot, "region": region}


def comp(x, a, n=WIN):
    """n weeks starting at week a vs the same weeks a year earlier."""
    return x[a:a + n].sum(0) / x[a - 52:a - 52 + n].sum(0) - 1


def select(rev, L, n=PILOT_N):
    """Leadership's rule: the n locations with the strongest comp over the 13 weeks before launch."""
    p = np.zeros(rev.shape[1], bool)
    p[np.argsort(-comp(rev, L - 13, 13))[:n]] = True
    return p


def read(s, L, pilot):
    """Quick read and holdout read at launch week L for a given pilot group."""
    out = {}
    base = comp(s["rev"], L - 39, 26)                     # own run rate, before the weeks used to pick the pilot
    size = np.log(s["rev"][L - 39:L - 13].sum(0))
    Z = np.column_stack([base, size]); Z = (Z - Z.mean(0)) / Z.std(0)
    pi, ci = np.where(pilot)[0], np.where(~pilot)[0]
    match = ci[np.argsort(((Z[pi][:, None] - Z[ci][None]) ** 2).sum(-1), axis=1)[:, :2]]
    for k in ("rev", "visits", "ticket"):
        c = comp(s[k], L)
        if k == "rev":
            out["quick"] = float(c[pilot].mean() - c[~pilot].mean())
            out["pilot_comp"], out["rest_comp"] = float(c[pilot].mean()), float(c[~pilot].mean())
        chg = c - comp(s[k], L - 39, 26)
        out[k] = float((chg[pi] - chg[match].mean(1)).mean())
    return out


def disciplined(s):
    """Holdout read, its range, the placebo record of both reads, the smallest lift this design can detect,
    and the price/volume split. Rule: no rollout unless the holdout read clears the hurdle, the design can
    detect a lift the size of the hurdle, and the read beats what placebo launches produce by chance."""
    obs = read(s, LAUNCH, s["pilot"])
    plc = [read(s, L, select(s["rev"], L)) for L in FAKE_LAUNCHES]
    pr, pq = np.array([p["rev"] for p in plc]), np.array([p["quick"] for p in plc])
    sd = float(pr.std(ddof=1))
    res = {"quick": obs["quick"], "pilot_comp": obs["pilot_comp"], "rest_comp": obs["rest_comp"],
           "holdout": obs["rev"], "range90": [obs["rev"] - 1.645 * sd, obs["rev"] + 1.645 * sd],
           "ticket": obs["ticket"], "visits": obs["visits"], "placebo_launches": len(plc),
           "quick_placebo_clears_hurdle": float((pq >= HURDLE).mean()), "quick_placebo_mean": float(pq.mean()),
           "holdout_placebo_sd": sd, "holdout_placebo_beats_observed": float((pr >= obs["rev"]).mean()),
           "mde_80": 2.8 * sd}
    res["quick_rolls_out"] = bool(obs["quick"] >= HURDLE)
    res["disciplined_rolls_out"] = bool(obs["rev"] >= HURDLE and res["mde_80"] <= HURDLE
                                        and res["holdout_placebo_beats_observed"] <= 0.05)
    return res


def randomized_mde(n_sites: int, sims: int = 300) -> float:
    """Spread of the holdout read for a RANDOMLY assigned pilot when nothing changed."""
    rng = np.random.default_rng(77)
    est = []
    for i in range(sims):
        s = simulate(20_000 + i, treat=False)
        p = np.zeros(SITES, bool); p[rng.choice(SITES, n_sites, replace=False)] = True
        est.append(read(s, LAUNCH, p)["rev"])
    return float(2.8 * np.std(est, ddof=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = simulate(1)
    d = disciplined(ex)
    true_rev = float((1 + PRICE_UP) ** (1 + ELASTICITY) - 1)
    annual = float(ex["rev"][LAUNCH - 52:LAUNCH].sum())
    many_null = [disciplined(simulate(100 + i)) for i in range(500)]
    many_real = [disciplined(simulate(3000 + i, elasticity=-0.2)) for i in range(300)]
    rate = lambda R, k: float(np.mean([r[k] for r in R]))
    rep = {
        "network": {"locations": SITES, "regions": REGIONS, "pilot_locations": PILOT_N, "pilot_weeks": WIN,
                    "price_increase": PRICE_UP, "hurdle": HURDLE, "launch_week": str(WEEKS[LAUNCH].date()),
                    "readout_week": str(WEEKS[LAUNCH + WIN - 1].date()), "annual_revenue": annual,
                    "true_revenue_effect": true_rev, "true_elasticity": ELASTICITY, "placebo_launches": len(FAKE_LAUNCHES)},
        "example": d,
        "cost_of_wrong_call": {"promised_annual": d["quick"] * annual, "true_annual": true_rev * annual,
                               "plan_miss_annual": (d["quick"] - true_rev) * annual},
        "power": {"as_run_mde": d["mde_80"], "randomized_40_mde": randomized_mde(40), "randomized_60_mde": randomized_mde(60)},
        "repeated_no_effect": {"networks": len(many_null), "quick_rolls_out": rate(many_null, "quick_rolls_out"),
                               "disciplined_rolls_out": rate(many_null, "disciplined_rolls_out")},
        "repeated_real_effect": {"networks": len(many_real), "true_revenue_effect": float((1.03) ** 0.8 - 1),
                                 "quick_rolls_out": rate(many_real, "quick_rolls_out"),
                                 "disciplined_rolls_out": rate(many_real, "disciplined_rolls_out")},
    }
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    T = len(WEEKS)
    rows = []
    for s in range(SITES):
        for t in range(T):
            f = 1 / 7 if t == T - 1 else 1.0
            v = ex["visits"][t, s] * f
            rows.append((f"L{s + 1:03d}", f"R{ex['region'][s] + 1}", WEEKS[t].date(), int(round(v)), round(v * ex["ticket"][t, s], 2),
                         round(ex["ticket"][t, s], 2), bool(ex["pilot"][s]), t == T - 1))
    pd.DataFrame(rows, columns=["location", "region", "week_ending", "visits", "revenue", "avg_ticket", "pilot", "partial_week"]) \
        .to_csv(OUT / "pilot_weekly.csv", index=False)
    print(json.dumps({k: rep[k] for k in ("example", "cost_of_wrong_call", "power", "repeated_no_effect", "repeated_real_effect")}, indent=1, default=float))


if __name__ == "__main__":
    main()
