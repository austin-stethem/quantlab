"""Write README.md from the result files. Every number in the README comes from this script.

    python tools/make_readme.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
J = lambda p: json.loads((R / p).read_text())

r1, eco, pil = J("round1/summary.json"), J("economics.json"), J("pilot/report.json")
A, B = J("v3/synthA/report.json"), J("v3/synthB/report.json")
ex, net, cost, pw = pil["example"], pil["network"], pil["cost_of_wrong_call"], pil["power"]


def pct(x, d=1, sign=False):
    s = f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return s.replace("-", "−") if x < 0 else s


def pts(x, d=1):
    return (f"+{x * 100:.{d}f}" if x >= 0 else f"−{abs(x) * 100:.{d}f}") + " pts"


def money(x):
    return "$0" if abs(x) < 5e4 else f"${x / 1e6:.1f}M"


def ptw(x, d=1):
    return f"{abs(x) * 100:.{d}f}-point"


def hits(r):
    return ", ".join(str(r["placebos"][k]["hits_abs_t_gt_2"]) for k in ("random_dates", "within_week_shuffle", "random_signal"))


old, new = eco["eras"]["2005-2024"], eco["eras"]["since 2015"]
bars = "\n".join(f"| {b['bar']:.0%} | {b['signals']:,} | {pct(b['rose'])} | {pct(b['base'])} |" for b in r1["bars"])

md = f"""# Quantlab

**How I underwrite a pricing initiative without fooling myself or the board:** the read a private equity-backed, multi-site operator needs before a network-wide price change.
Designed and directed by Austin Stethem · [Live console](https://austin-stethem.github.io/quantlab/), rebuilt after every weekday close

## Recommendation

1. **Don't approve a price rollout on a pilot comp.** In a simulated {net['pilot_locations']}-location price pilot, the quick read showed a {ptw(ex['quick'])} lift, worth "{money(cost['promised_annual'])} a year." The disciplined read found no lift ({pts(ex['holdout'])}): the price stuck, but visits fell just as much.
2. **Hold every initiative to four guardrails before it reaches the board:** a return-on-investment hurdle written down in advance, a matched control group, a placebo, and a power check.
3. **The same discipline stopped my own trading strategy before it cost real money.** It could not beat simply holding the index after tax.

## Thesis

A pricing strategy is only as good as the read of its pilot, and pilots get read in ways that flatter them:
- hand-picked locations
- comps against the rest of the network
- no check of what the method "finds" when nothing changed
- no check of what the test is even able to detect

Fixing the read, and being willing to challenge assumptions behind it, is worth more than running another pilot. The expensive mistake is the rollout that was never real.

## Worked example: a 3% price pilot across {net['locations']} locations

This network is simulated. It has {net['locations']} locations in {net['regions']} regions, with weekly visits and average ticket since January 2024. The data includes seasonality, holiday dips, a company-wide list-price increase every January, regional momentum, and a partial current week. Leadership picks its {net['pilot_locations']} fastest-growing locations and raises prices {pct(net['price_increase'], 0)} for {net['pilot_weeks']} weeks. The rollout hurdle is a {ptw(net['hurdle'], 0)} revenue lift, standing in for the return on investment the rollout must earn after its costs and attrition risk. In truth, clients respond with an elasticity of {net['true_elasticity']:.1f}, so revenue barely moves. The raw weekly export is in [`results/pilot/pilot_weekly.csv`](results/pilot/pilot_weekly.csv).

| | Quick read | Disciplined read |
|---|---|---|
| Comparison | Pre- and post-performance: pilot comp ({pct(ex['pilot_comp'], 1, True)}) vs rest of network ({pct(ex['rest_comp'], 1, True)}) | Each pilot location vs a matched control group of two locations, net of its own run rate before selection |
| Incremental revenue | **{pts(ex['quick'])}** | **{pts(ex['holdout'])}**, a range of outcomes from {pts(ex['range90'][0])} to {pts(ex['range90'][1])} (90%) |
| Key drivers | Not split | Price, volume and mix bridge: the price leg (ticket) {pct(ex['ticket'], 1, True)}, the volume leg (visits) {pct(ex['visits'], 1, True)}. The price stuck and clients left |
| Placebo: same method at {ex['placebo_launches']} past dates, no price change | "Found" a lift above the hurdle at {pct(ex['quick_placebo_clears_hurdle'], 0)} of dates | With no price change, this read still swings about ±{1.645 * ex['holdout_placebo_sd'] * 100:.1f} pts from date to date |
| Smallest lift this test can detect | Never checked | {pts(ex['mde_80'])}, against a {pts(net['hurdle'], 0)} hurdle |
| Call | **Roll out: "{money(cost['promised_annual'])} a year"** | **Not proven. Do not roll out.** |

**What the wrong call costs:**
- The plan and forecast book {money(cost['promised_annual'])} of annual incremental revenue that is not there, since the true effect is {money(cost['true_annual'])}.
- Rolled out network-wide, visits would fall about {pct(abs(ex['visits']))}: clients lost, not just revenue that never arrived.
- Scenario analysis: across {pil['repeated_no_effect']['networks']} simulated networks where the price change does nothing to revenue, the quick read approved the rollout **{pct(pil['repeated_no_effect']['quick_rolls_out'], 0)}** of the time. The disciplined read approved it {pct(pil['repeated_no_effect']['disciplined_rolls_out'])} of the time.

**What a test that can answer the question needs.** Even a properly randomized test of 40 locations can only detect about {pts(pw['randomized_40_mde'])}, and 60 locations can detect about {pts(pw['randomized_60_mde'])}. A {ptw(net['hurdle'], 0)} revenue hurdle cannot be read from {net['pilot_weeks']} weeks of location revenue. The fix is to measure price realization and visit loss directly, which gives the elasticity, or to run the test longer. When the true effect *is* a real {pct(pil['repeated_real_effect']['true_revenue_effect'], 1)}, this pilot design still cannot prove it.

## Diligence performed

**Round 1: market signals, real prices.**
- **Universe:** {r1['universe_size']} large caps and sector ETFs.
- **Signals:** six chart signals combined by a calibrated logistic regression, a statistical model that has to earn its probabilities. A signal fires at {pct(r1['threshold'], 0)} probability with {r1['quorum']} of 6 signals agreeing.
- **Test design:** walk-forward from {r1['first_test_year']} through {r1['last_test_year']}, judged {r1['horizon']} trading days out.
- **Gate: {r1['verdict']}.**

| Check | Result |
|---|---|
| Calibration skill vs base rate | {r1['calibration_skill']:.3f} (no better than guessing the base rate) |
| Hit rate when fired | {pct(r1['hit_rate'])} vs {pct(r1['base_rate'])} for the same days (p = {r1['hit_p']:.2f}) |
| Excess return per signal | {pct(r1['excess'], 2, True)} (t = {r1['excess_t']:.2f}) |
| Paper book vs holding everything | Sharpe {r1['sharpe_book']:.2f} vs {r1['sharpe_universe']:.2f}; growth {r1['growth_book']:.2f}x vs {r1['growth_universe']:.2f}x |
| Max drawdown | {pct(r1['mdd_book'], 1)} vs {pct(r1['mdd_universe'], 1)}, shallower only because the book sat in cash between signals |
| Worst calendar year | {r1['worst_year_book'][0]}: {pct(r1['worst_year_book'][1], 1, True)} vs {r1['worst_year_universe'][0]}: {pct(r1['worst_year_universe'][1], 1, True)} |

Raising the bar did not help. The model's confidence did not track reality:

| Probability bar | Signals | Actually rose | Same days, all names |
|---|---|---|---|
{bars}

**The selection bias in round 1, named.** The 22 names are giants *today*. Picking them in hindsight flatters any long strategy, because the companies that fell apart along the way were never in the test. Round 2 was rebuilt to remove exactly that.

**Round 2: one pre-registered edge (post-earnings drift), on a bench proven before use.** The design uses the S&P 500 as it stood on each date, including companies that later died or were bought, with every event dated by its SEC filing time. Before buying any real data, I tested the bench on two simulated markets carrying the real data's traps. In one, no edge exists. In the other, an edge is planted on purpose.

| Check | No edge exists | Edge planted |
|---|---|---|
| Event dating, estimated vs true | {pct(A['dating']['correct_share'])} vs {pct(A['dating']['true_correct_share'])} | {pct(B['dating']['correct_share'])} vs {pct(B['dating']['true_correct_share'])} |
| Placebo false alarms (3 placebos × 20 runs) | {hits(A)} | {hits(B)} |
| Planted signal recovered | {pct(A['planted']['ratio'], 0)} | {pct(B['planted']['ratio'], 0)} |
| Factor-adjusted alpha, t (regression on market, size, value, profitability, investment and momentum) | {A['calendar']['ls_ff6_alpha_daily']['t']:.1f} | {B['calendar']['ls_ff6_alpha_daily']['t']:.1f} |
| Overfitting check (deflated Sharpe / PBO) | {A['deflated_sharpe']['dsr']:.2f} / {A['pbo']['pbo']:.2f} | {B['deflated_sharpe']['dsr']:.2f} / {B['pbo']['pbo']:.2f} |
| **Gate** | **{A['gate']['verdict']}** | **{B['gate']['verdict']}** |

The bench finds an edge when one exists and refuses to invent one when it does not. Testing it also exposed a flaw in my own rule for checking event dates before any real data was touched. The rule was replaced, and the change is logged as Amendment 9 in [TRIALS.md](TRIALS.md).

## Key risks and mitigants

| Risk | Mitigant |
|---|---|
| Selection bias: hand-picked pilot locations, or today's survivors | Matched holdout, point-in-time universe, placebo launches at past dates |
| A test too small to see the effect | Power check before the read; the smallest detectable lift is compared to the hurdle |
| Trying variants until one works | Rules written before results ([PREREG.md](PREREG.md)), every variant logged, deflated Sharpe |
| Data integrity: bad dates or dirty data | Dating audit, raw files fingerprinted, tests that fail on any look-ahead |
| Simulated evidence mistaken for real evidence | Stated plainly: round 1 used real prices; the pilot and the round 2 bench are simulations |

## Returns vs hurdle

The illustrative inputs are a {pct(eco['inputs']['short_term_rate'], 0)} tax rate on short-term gains, {pct(eco['inputs']['long_term_rate'], 0)} on long-term gains, an {pct(eco['inputs']['market'], 0)} market return and a {eco['inputs']['years']}-year hold. These are not anyone's actual tax situation.

A strategy that closes positions every few months is taxed every year. Holding the index defers that tax and pays a lower rate. Just to **tie** the index after tax, the strategy has to beat it by **{pct(eco['tax_hurdle_annual'], 2)} a year after trading costs**.

| Published edge, long side, after costs | Per year | vs the {pct(eco['tax_hurdle_annual'], 2)} hurdle |
|---|---|---|
| 2005 to 2024 | {pct(old['long_net_annual'], 2)} | Clears it by {pct(old['long_net_annual'] - eco['tax_hurdle_annual'], 2)} |
| Since 2015 (the one that matters for a decision today) | {pct(new['long_net_annual'], 2)} | **Falls short by {pct(eco['tax_hurdle_annual'] - new['long_net_annual'], 2)}** |

Sensitivity analysis on the tax rate alone: {", ".join(f"at {pct(x['short_term_rate'], 0)} the hurdle is {pct(x['hurdle'], 2)}" for x in eco['sensitivity'])}.

**Decision: the stop rule fired, no data was bought, and the strategy stays on paper.**

## What would change my mind

- **For the trading strategy:** any one of these would reopen it:
  - a tax-advantaged account large enough to matter
  - automation that cuts the hours to near zero
  - evidence of a larger edge in large caps than the published research shows
- **For a pricing initiative:** a randomized test sized so the smallest detectable lift sits at or below the hurdle, read on price realization and visit loss as well as revenue before any profitability gain is claimed, followed by a post-launch review that checks realized results against the forecast.

---

## Technical appendix

| Term in the code | What it means in business terms |
|---|---|
| Pre-registration | The win condition is written down before anyone looks at results |
| Matched holdout | Locations that did not get the change, chosen to look like the ones that did |
| Placebo test | Running the same read where nothing changed, to prove it reports nothing |
| Power / minimum detectable effect | The smallest lift the test could reliably see; if that is above the hurdle, the test cannot approve anything |
| Walk-forward | Every year is predicted using only earlier years |
| Point-in-time universe | Companies as they stood on each date, including those that later died |
| Deflated Sharpe / PBO | Marking a result down for how many versions were tried |
| Newey-West t-statistic | Confidence that holds up when overlapping periods make results look more certain than they are |

**Run it:**

```
pip install -r requirements.txt
python -m pytest -q tests          # no look-ahead, holdout invisible, planted signal recovered
python run.py --refresh            # round 1 on the latest close (feeds the live console)
python run_v3.py --world synthA    # round 2 bench: no-edge market    (gate must FAIL)
python run_v3.py --world synthB    # round 2 bench: planted edge      (gate should PASS)
python pilot/pilot.py              # the price-pilot worked example
python tools/freeze_results.py     # freeze round 1 numbers and the economics
python tools/make_readme.py        # write this README from the result files
```

Every number in this README is generated from the result files by [`tools/make_readme.py`](tools/make_readme.py).
"""

import re
md = re.sub(r"(?m)^(?!\s*[-|\d#`]|\s*$)(.+)\n(?=\s*- )", r"\1\n\n", md)   # blank line before lists, for every renderer
assert "—" not in md, "no em dashes"
(ROOT / "README.md").write_text(md)
print("README.md written,", len(md.split()), "words")
