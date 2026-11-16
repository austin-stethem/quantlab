# Quantlab v3 pre-registration

Written 2026-11-16, before any v3 code or v3 result exists. Nothing below changes after
the first result is seen. Any change is logged in TRIALS.md as a new trial, with the reason.

## Hypothesis

Post-earnings announcement drift (PEAD), with the surprise measured by the market's own
reaction: the earnings announcement return (EAR).

Names whose stock jumps the most on an earnings release keep outperforming, over the next
several weeks, the names whose stock drops the most on theirs.

**Who is on the other side.** Investors who anchor on their old view of the company and
update slowly, and analysts who revise in small steps. **Why it persists.** Attention is
limited (dozens of firms report on the same day), and arbitrage is limited: holding a
long-short book for weeks ties up capital and carries risk.

**Known weakness, stated up front.** Published work finds the drift has shrunk over time and
is weakest in large, liquid stocks, which is exactly what the S&P 500 is. Failing the gate
is a live, expected outcome. The gate is built to report that, not to avoid it.

## Candidates considered and set aside

- **Short-term reversal (1 month).** The story is real: liquidity providers get paid by
  traders who need to trade right away. But in large caps the edge is mostly eaten by the
  turnover it requires. A possible second test.
- **12-1 cross-sectional momentum.** The most replicated edge, but it *is* the momentum
  factor. It cannot show alpha after an FF5+momentum adjustment, so step 7 fails it by
  construction.
- **Analyst revisions.** Needs a history of estimates (I/B/E/S), which is institutional
  pricing. Out of scope.

## Design

- **Universe.** S&P 500 members as of each date (point-in-time membership), dead names
  included where the price source has them.
- **Event.** An SEC 8-K carrying Item 2.02 (Results of Operations), timed by its EDGAR
  acceptance timestamp. Day 0 is the first session that opens after acceptance, so a filing
  after 16:00 ET belongs to the next day.
- **Signal.** EAR = the stock's return from the close of day -2 to the close of day +1,
  minus the equal-weight universe return over the same window.
- **No look-ahead.**
  - Entry is the close of day +2.
  - Decile breakpoints come only from events in the trailing 63 sessions.
- **Portfolios.** Calendar-time portfolios.
  - Long: top-decile EAR names, held 60 sessions.
  - Short (sell/avoid): bottom-decile names, held 60 sessions.
  - Equal weight within each side.
  - 10 bps cost each way.
- **Second check.** A seasonal random-walk SUE, built from SEC XBRL EPS with filing dates:
  this quarter versus the same quarter last year, scaled. No analyst estimates needed.

## Evaluation order

1. Daily rank IC at 5, 20, 40 and 60 sessions (decay curve).
2. Decile spread and monotonicity.
3. Fama-MacBeth, plus FF5+momentum regressions on the long-short returns, with Newey-West
   t-statistics (lags = holding period).
4. Combinatorial purged cross-validation, with a 60-session embargo.
5. Triple-barrier exits plus meta-labeling, against hold-60, with volatility-targeted sizing.

## Honesty controls

- **TRIALS.md.** Every variant gets logged, and the trial count feeds the deflated Sharpe
  and the PBO calculation.
- **Placebos, which must come back empty.**
  - EAR assigned to random dates.
  - EAR shuffled across names within each date.
  - A pure random signal.
- **Published comparison.** Our spread is compared to the published PEAD and EAR sizes,
  pulled from the papers before our number is read. More than twice the published size is
  treated as a probable bug.

## Holdout

**Locked: 2024-11-14 through 2026-11-13.** No v3 code may load returns from these dates
until the final gate run. That run happens once, after the design above is frozen.
Development sample: 2005-01-01 through 2024-11-13 (Item 2.02 exists from August 2004).

## Kill gate (all must pass)

1. The 20- or 60-session rank IC is positive, with a Newey-West t above 2.
2. The long-short spread stays positive after 10 bps each way.
3. FF5+momentum alpha is positive, with a Newey-West t above 2.
4. The deflated Sharpe ratio is above 0, given the logged trial count.
5. All three placebos fail (|t| < 2).
6. The holdout's long-short spread has the same sign, and is at least half the development
   size.

## Forward test (only if the gate passes)

- These rules are copied into RULES.md, frozen, and run on Alpaca paper trading for 6 months
  with no changes.
- The console shows the paper fills against the backtest expectation.
