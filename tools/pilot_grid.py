"""Pilot sizing grid for the console calculator: the smallest lift a RANDOMIZED pilot can detect,
by number of pilot locations and pilot length, on the same simulated 120-location network.

Uses untreated networks, so any "lift" the holdout read reports is noise; 2.8 x its spread is the
smallest lift detectable at 80% power. Every pilot length ends at the example's readout week, so longer
pilots start earlier. Same networks, assignment draws and read as pilot.randomized_mde: the report's 40-
and 60-location figures are the first 300 of these 2,000 networks (checked below), and 2,000 gives a steadier
estimate. Writes results/pilot/grid.json.

    python tools/pilot_grid.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot"))
import pilot as P   # noqa: E402

SIZES, WEEKS, SIMS = [10, 20, 30, 40, 50, 60], [8, 12, 16, 20, 26], 2000
start = lambda w: P.LAUNCH + P.WIN - w        # each window ends where the example's 12-week read ends


def holdout_read(s, launch, pilot, weeks):
    rev = s["rev"]
    base = P.comp(rev, launch - 39, 26)
    size = np.log(rev[launch - 39:launch - 13].sum(0))
    Z = np.column_stack([base, size]); Z = (Z - Z.mean(0)) / Z.std(0)
    pi, ci = np.where(pilot)[0], np.where(~pilot)[0]
    match = ci[np.argsort(((Z[pi][:, None] - Z[ci][None]) ** 2).sum(-1), axis=1)[:, :2]]
    chg = P.comp(rev, launch, weeks) - base
    return float((chg[pi] - chg[match].mean(1)).mean())


def main():
    nets = [P.simulate(20_000 + i, treat=False) for i in range(SIMS)]     # same networks as randomized_mde
    grid, first300 = {}, {}
    for n in SIZES:
        for w in WEEKS:
            rng = np.random.default_rng(77)                                  # same draws as randomized_mde
            est = []
            for s in nets:
                p = np.zeros(P.SITES, bool); p[rng.choice(P.SITES, n, replace=False)] = True
                est.append(holdout_read(s, start(w), p, w))
            grid[f"{n}x{w}"] = round(2.8 * float(np.std(est, ddof=1)), 5)
            if w == P.WIN and n in (40, 60):
                first300[f"{n}x{w}"] = round(2.8 * float(np.std(est[:300], ddof=1)), 5)
    rep = json.loads((ROOT / "results" / "pilot" / "report.json").read_text())["power"]
    for n, k in ((40, "randomized_40_mde"), (60, "randomized_60_mde")):
        assert abs(first300[f"{n}x12"] - rep[k]) < 1e-4, (n, first300[f"{n}x12"], rep[k])
    out = {"locations": SIZES, "weeks": WEEKS, "sims_per_cell": SIMS, "network_locations": P.SITES, "mde_80": grid, "report_first_300": first300,
           "readout_week": str(P.WEEKS[P.LAUNCH + P.WIN - 1].date()),
           "note": "randomized assignment, matched holdout, each location vs its own pre-period run rate; every length ends at the same readout week; simulated network"}
    (ROOT / "results" / "pilot" / "grid.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(grid))


if __name__ == "__main__":
    main()
