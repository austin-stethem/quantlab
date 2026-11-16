# Quantlab

**How I underwrite a pricing initiative without fooling myself or the board:** the read a private equity-backed, multi-site operator needs before a network-wide price change.
Designed and directed by Austin Stethem · [Live console](https://austin-stethem.github.io/quantlab/), rebuilt after every weekday close

## Recommendation

1. **Don't approve a price rollout on a pilot comp.** In a simulated 20-location price pilot, the quick read showed a 3.1-point lift, worth "$9.2M a year." The disciplined read found no lift (+0.0 pts): the price stuck, but visits fell just as much.
2. **Hold every initiative to four guardrails before it reaches the board:** a return-on-investment hurdle written down in advance, a matched control group, a placebo, and a power check.
3. **The same discipline stopped my own trading strategy before it cost real money.** It could not beat simply holding the index after tax.

## Thesis

A pricing strategy is only as good as the read of its pilot, and pilots get read in ways that flatter them:

- hand-picked locations
- comps against the rest of the network
- no check of what the method "finds" when nothing changed
- no check of what the test is even able to detect

Fixing the read, and being willing to challenge assumptions behind it, is worth more than running another pilot. The expensive mistake is the rollout that was never real.

## Worked example: a 3% price pilot across 120 locations

This network is simulated. It has 120 locations in 6 regions, with weekly visits and average ticket since January 2024. The data includes seasonality, holiday dips, a company-wide list-price increase every January, regional momentum, and a partial current week. Leadership picks its 20 fastest-growing locations and raises prices 3% for 12 weeks. The rollout hurdle is a 2-point revenue lift, standing in for the return on investment the rollout must earn after its costs and attrition risk. In truth, clients respond with an elasticity of -1.0, so revenue barely moves. The raw weekly export is in [`results/pilot/pilot_weekly.csv`](results/pilot/pilot_weekly.csv).

| | Quick read | Disciplined read |
|---|---|---|
| Comparison | Pre- and post-performance: pilot comp (+6.3%) vs rest of network (+3.3%) | Each pilot location vs a matched control group of two locations, net of its own run rate before selection |
| Incremental revenue | **+3.1 pts** | **+0.0 pts**, a range of outcomes from −4.7 pts to +4.8 pts (90%) |
| Key drivers | Not split | Price, volume and mix bridge: the price leg (ticket) +3.7%, the volume leg (visits) −3.4%. The price stuck and clients left |
| Placebo: same method at 17 past dates, no price change | "Found" a lift above the hurdle at 100% of dates | With no price change, this read still swings about ±4.7 pts from date to date |
| Smallest lift this test can detect | Never checked | +8.1 pts, against a +2 pts hurdle |
| Call | **Roll out: "$9.2M a year"** | **Not proven. Do not roll out.** |

**What the wrong call costs:**

- The plan and forecast book $9.2M of annual incremental revenue that is not there, since the true effect is $0.
- Rolled out network-wide, visits would fall about 3.4%: clients lost, not just revenue that never arrived.
- Scenario analysis: across 500 simulated networks where the price change does nothing to revenue, the quick read approved the rollout **94%** of the time. The disciplined read approved it 0.2% of the time.

**What a test that can answer the question needs.** Even a properly randomized test of 40 locations can only detect about +4.3 pts, and 60 locations can detect about +3.5 pts. A 2-point revenue hurdle cannot be read from 12 weeks of location revenue. The fix is to measure price realization and visit loss directly, which gives the elasticity, or to run the test longer. When the true effect *is* a real 2.4%, this pilot design still cannot prove it.

## Diligence performed

**Round 1: market signals, real prices.**

- **Universe:** 22 large caps and sector ETFs.
- **Signals:** six chart signals combined by a calibrated logistic regression, a statistical model that has to earn its probabilities. A signal fires at 60% probability with 4 of 6 signals agreeing.
- **Test design:** walk-forward from 2017 through 2026, judged 20 trading days out.
- **Gate: FAIL.**

| Check | Result |
|---|---|
| Calibration skill vs base rate | -0.029 (no better than guessing the base rate) |
| Hit rate when fired | 58.6% vs 57.6% for the same days (p = 0.13) |
| Excess return per signal | +0.16% (t = 0.49) |
| Paper book vs holding everything | Sharpe 0.55 vs 0.82; growth 1.82x vs 3.49x |
| Max drawdown | −25.4% vs −35.0%, shallower only because the book sat in cash between signals |
| Worst calendar year | 2018: −10.1% vs 2022: −6.5% |

Raising the bar did not help. The model's confidence did not track reality:

| Probability bar | Signals | Actually rose | Same days, all names |
|---|---|---|---|
| 55% | 6,814 | 58.2% | 58.8% |
| 60% | 3,848 | 58.5% | 58.8% |
| 65% | 809 | 58.1% | 59.3% |
| 70% | 118 | 60.2% | 57.1% |
| 80% | 98 | 57.1% | 56.7% |

**The selection bias in round 1, named.** The 22 names are giants *today*. Picking them in hindsight flatters any long strategy, because the companies that fell apart along the way were never in the test. Round 2 was rebuilt to remove exactly that.

**Round 2: one pre-registered edge (post-earnings drift), on a bench proven before use.** The design uses the S&P 500 as it stood on each date, including companies that later died or were bought, with every event dated by its SEC filing time. Before buying any real data, I tested the bench on two simulated markets carrying the real data's traps. In one, no edge exists. In the other, an edge is planted on purpose.

| Check | No edge exists | Edge planted |
|---|---|---|
| Event dating, estimated vs true | 97.2% vs 96.7% | 97.2% vs 96.7% |
| Placebo false alarms (3 placebos × 20 runs) | 0, 1, 1 | 2, 1, 1 |
| Planted signal recovered | 98% | 99% |
| Factor-adjusted alpha, t (regression on market, size, value, profitability, investment and momentum) | -0.9 | 6.5 |
| Overfitting check (deflated Sharpe / PBO) | 0.04 / 0.79 | 0.97 / 0.27 |
| **Gate** | **FAIL** | **PASS** |

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

The illustrative inputs are a 30% tax rate on short-term gains, 20% on long-term gains, an 8% market return and a 10-year hold. These are not anyone's actual tax situation.

A strategy that closes positions every few months is taxed every year. Holding the index defers that tax and pays a lower rate. Just to **tie** the index after tax, the strategy has to beat it by **1.69% a year after trading costs**.

| Published edge, long side, after costs | Per year | vs the 1.69% hurdle |
|---|---|---|
| 2005 to 2024 | 2.82% | Clears it by 1.13% |
| Since 2015 (the one that matters for a decision today) | 1.32% | **Falls short by 0.37%** |

Sensitivity analysis on the tax rate alone: at 22% the hurdle is 0.69%, at 30% the hurdle is 1.69%, at 37% the hurdle is 2.76%.

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
