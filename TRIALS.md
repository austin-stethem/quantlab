# Quantlab v3 trial log

Every variant that produces a result gets a row in `results/v3/trials.csv`, which the code
writes automatically. This file records the decisions made by hand, in order. The
amendments below were all made on 2026-11-16, before any v3 result on real or synthetic
data existed.

## Pre-result amendments to PREREG.md

1. **Published comparison, fixed now.** Open Source Asset Pricing (Chen and Zimmermann),
   AnnouncementReturn long-short, all US stocks, monthly %:

   | Period | Mean monthly % | t |
   |---|---|---|
   | 1972 to 1992 | 1.14 | 13.5 |
   | 1993 to 2004 | 1.29 | 7.8 |
   | 2005 to 2014 | 0.85 | 5.7 |
   | 2015 to 2024-11 | 0.36 | 1.9 |
   | 2005 to 2024-11 (our development window) | 0.61 | 5.0 |

   EarningsSurprise over 2005 to 2024-11: 0.14, t 1.0.

   The "probable bug" flag fires if our S&P 500 long-short monthly mean is above 1.2%
   (twice the 0.61 all-stock figure). Large caps should come in *below* the all-stock number.

2. **Disclosure.** While loading that file, the OSAP figures for 2024-11 to 2025-12 were
   printed: 0.51% monthly, t 0.71, n 14. That overlaps our holdout window on a different
   universe. It is noted here rather than hidden.

3. **Replication threshold.** Our monthly long-short series must correlate at 0.5 or more
   with OSAP AnnouncementReturn LS over 2005-01 to 2024-11. Theirs covers all CRSP stocks
   and ours covers the S&P 500 only, so a correlation near 1 is not expected.

4. **Analyst coverage dropped from the mechanism tests.** Historical analyst coverage only
   exists in I/B/E/S, which is institutional-only. The mechanism test becomes Friday, busy
   day and institutional ownership, and the gate requires 2 of these 3.

5. **Benchmark.** The DGTW benchmark uses 3x3x3 sequential sorts (27 cells) instead of
   5x5x5. 125 cells would leave about 4 S&P 500 names per cell.

6. **Correction.** PREREG gate item 4 said "deflated Sharpe ratio above 0." The DSR is a
   probability, so the intended rule is **DSR of at least 0.95**.

7. **Gate thresholds for the new items.**
   - Placebos: each of the three placebos passes if 3 or fewer of 20 seeds show |t| > 2.
   - Planted drift: the recovered increment must land within 25% of the planted size.
   - Long-only leg: the daily excess over SPY after costs must have a Newey-West t above 2.
   - Mechanism tests: one-sided p < 0.05, in the predicted direction.

8. **Known confound for the random-date placebo.** Any 3-day return can predict reversal or
   momentum on its own. If the random-date placebo fails while the other two pass, that
   failure is reported as a real price effect, not hidden. The gate still counts it as a
   fail.

## Trials

See `results/v3/trials.csv`.

## Amendment 9 (2026-11-16, after a SYNTHETIC debug run, before any real data)

The step-3 rule "the largest absolute return lands on day 0 at least 85% of the time" cannot be
met even with near-perfect dating. On the synthetic market, where about 97% of events are dated
correctly by construction, it scored 64%. Small surprises mean the announcement day often is not
the week's biggest move.

**The replacement.** The correctly dated share is estimated from the abnormal-variance profile.
Late filings put the reaction on day -1, so:

```
late share = excess variance(-1) / (excess variance(-1) + excess variance(0))
```

The baseline variance comes from offsets +/-3 to 5. The gate requires an estimated correct share
of 85% or more. The estimator is checked against the known true share on synthetic data, and the
raw max-share is still reported. The synthetic "hypothesis true" world (synthB) uses a stronger
planted drift and stronger attention boosts. It exists only to show the pipeline can pass when
the effect is real.

## Economics check (2026-11-16, before any real data)

The owner's account size, tax bracket and hours are private and are not part of this repo.

The method:
- Short-term gains are taxed as ordinary income each year.
- SPY held long-term is taxed once, at long-term rates.
- Costs: 20 bps per round trip on roughly 60-session holds, plus hours valued at an hourly rate.
- The long leg is assumed to carry half the published long-short spread.

**Verdict: the stop rule fired.** For a small taxable account with these holding periods, the
published edge cannot beat simply holding SPY after tax at any account size a careful scale-up
would reach. Since 2015 it falls short at every size. No market data was bought, and the
strategy remains a paper-only research project.
