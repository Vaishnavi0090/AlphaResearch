# Data Leakage Audit

Look-ahead bias is the single most common reason a backtest looks good and a live strategy does not. It is also easy to introduce accidentally and nearly invisible once present. So this project treats the absence of leakage as a claim requiring *evidence*, not assertion.

Run the audit:

```bash
python -c "
from finrisk.config.settings import default_settings
from finrisk.data.market_data import load_bundle, align_to_calendar
from finrisk.evaluation.leakage import run_full_audit
s = default_settings(); b = align_to_calendar(load_bundle(s.data))
print(run_full_audit(b, s).render())"
```

## Result: 8 checks, 0 failures

```
[PASS] fundamental_availability
        2400 records; 0 available before period end.
        Publication lag median 61d, range [45, 73]d.
[PASS] asof_accessor
        sampled 12 dates; 0 returned unpublished filings
[PASS] factor_truncation_invariance
        max absolute difference between full-sample and truncated factor
        values across 6 dates: 0.000e+00
[PASS] cross_sectional_normalisation
        per-date normalised panels have |mean| <= 1.63e-15 and |std - 1| <= 0.000
[PASS] regime_causality
        filtered state at 2020-07-10 is 'risk_off_high_vol' (conf 1.000)
        whether or not future data is present.
        Max |filtered - smoothed| = 0.872
[PASS] execution_lag
        156 rebalances; 0 executed on or before the decision date.
        Lag in sessions: min 1, max 1.
[PASS] survivorship_bias
        synthetic universe: all 40 assets exist for the full sample by
        construction. Real-data runs with a fixed current ticker list DO
        have survivorship bias.
[PASS] shifted_signal_placebo
        honest: CAGR 6.18%, Sharpe 0.33.
        one-period look-ahead: CAGR 33.22%, Sharpe 1.85.
        Advantage from cheating: +27.04% CAGR, +1.53 Sharpe.
```

## The placebo control — the check that actually matters

Static code review catches obvious leakage. It does not catch the subtle kind. So the headline test is empirical.

`test_shifted_signal_advantage` runs the identical strategy twice on identical data with identical costs. The only difference: in the second run, each decision date receives the factor scores belonging to the **next** rebalance date — deliberate one-period look-ahead.

| Run | CAGR | Sharpe |
|---|---|---|
| Honest | 6.18% | 0.33 |
| One-period look-ahead | **33.22%** | **1.85** |
| Advantage from cheating | +27.04 pp | +1.53 |

The logic is a two-sided argument:

- If the honest run performed *like* the cheating run, the honest run would already be cheating.
- Because cheating is worth +27 percentage points of CAGR, the information barrier is demonstrably load-bearing.

This is a far stronger statement than "I reviewed the code." It is the financial equivalent of a placebo arm, and it is the first thing I would want to see from someone else's backtest.

A note on implementation honesty: the cheating run has to *re-stamp* the borrowed scores with the current date, because the backtester's own runtime assertion would otherwise refuse to run them. That assertion is itself tested — `test_backtester_raises_on_injected_future_signal` tampers with a context and confirms the `AssertionError` fires.

## The eight checks in detail

### 1. Fundamental availability
Every fundamental record carries both `period_end` (the fiscal period it describes) and `available_from` (when it became public). The audit asserts `available_from >= period_end` for all 2,400 records. Median publication lag is 61 days, range 45–73.

### 2. The as-of accessor
`MarketDataBundle.fundamentals_asof(d)` is the *only* sanctioned way to read fundamentals. It filters on `available_from <= d` and returns the latest such filing per ticker. The audit samples dates across the history and confirms no unpublished filing is ever returned.

This matters because the naive alternative — indexing fundamentals by `period_end` — leaks roughly two months of hindsight into every value factor, which is enough to manufacture an impressive and entirely fictitious backtest.

### 3. Factor truncation invariance
The factor panel computed at date `d` on the full dataset must equal the panel computed at `d` on data truncated at `d`. If they differ, something in the factor engine is reading forward.

Measured maximum absolute difference: **0.000e+00** across sampled dates. Exact, not approximate.

### 4. Cross-sectional normalisation scope
Z-scores are computed within each date's cross-section. Pooled normalisation across the full history would leak the future distribution into every past date — a subtle and common error.

The audit verifies each date's normalised panel has `|mean| <= 1.63e-15`, which is only possible if normalisation happened inside that date.

Winsorization precedes z-scoring, both within-date. The order matters: z-scoring first and clipping second would leave the outlier's influence baked into the mean and standard deviation. (This is why the standard deviation can sit marginally below 1.)

### 5. Regime causality
Two properties are asserted together:

- The filtered state at `d` is unchanged when future data is appended (causality holds).
- Filtered and smoothed probabilities *differ*, with max difference **0.872**.

The second is the interesting one. `smoothed_states()` runs the full forward–backward recursion and therefore uses future observations; it exists in the codebase, is clearly documented as future-using, and is **never** called from any decision path. Its measurable divergence from the filtered path is proof that the distinction is real rather than cosmetic. A codebase where filtered and smoothed agreed would be one where filtering was secretly cheating.

`tests/test_regimes.py` asserts both directions: `test_filtering_is_causal` and `test_smoothing_is_not_causal`.

### 6. Execution lag
Across 156 rebalances, every execution date follows its decision date by exactly one session. `BacktestConfig` validates `execution_lag_days >= 1`, so a zero-lag config raises at construction rather than producing plausible-looking results.

### 7. Survivorship bias
Reported honestly as a property of the *universe*, not the code.

On the synthetic dataset there is no survivorship bias — all 40 assets exist throughout by construction. But there is also no delisting risk, which means the synthetic results are optimistic in a different way: a real universe contains companies that fail.

For real-data runs, a universe drawn from currently-listed tickers **does** have survivorship bias, and this project does not solve it. Doing so requires a point-in-time constituent list (CRSP or similar). This is listed in `LIMITATIONS.md`, not hidden.

### 8. Shifted-signal placebo
See *The placebo control* above. Unlike the other seven, this check is expected to show a large effect: it passes only if deliberately peeking one period ahead buys more than 1% CAGR and 0.10 Sharpe (`gap_cagr > 0.01 and gap_sharpe > 0.10`). If cheating bought little, the honest run would probably be leaking already.

## Independent test: scrambling the future

The placebo proves the pipeline is *sensitive* to look-ahead. It does not prove the honest run never peeks: a pipeline with a small leak would also show a large placebo gap. A second, independent test closes that gap from the other side.

`test_signals_ignore_the_future` and `test_decisions_ignore_the_future` (in `tests/test_backtest.py`) take the dataset and replace **everything after a cutoff date with random noise**: prices, benchmark, risk-free rate, volumes and every filing published after the cutoff. They then assert that factor scores, regime states and the **portfolio weights decided on or before the cutoff are identical** to the honest run. A pipeline that uses only information available on each decision date cannot tell the difference.

Measured on the full synthetic dataset (cutoff 2019-09-30, 94 decisions): maximum weight difference **0.00e+00** for each of the five ablation rungs B–F. The test runs two strategies: B (factor selection only) and F (the full pipeline: regime overlay, shrinkage covariance, optimiser, constraints).

It also covers a class the placebo cannot: a leak in the *optimiser's inputs* (for example, estimating covariance from the full return history) leaves the factor signals untouched and so does not move the placebo, but it changes the weights and fails this test. That was checked by injecting exactly that leak and confirming the test fails.

## What is deliberately *not* claimed

- **No claim of zero bias.** Survivorship bias exists for real-data runs, and the synthetic DGP has no delistings.
- **No claim about transaction costs being realistic.** They are a linear approximation; see `BACKTESTING.md`.
- **No claim that the audit is exhaustive.** It covers the leakage channels I identified. An unknown channel would, by definition, not be here — which is why the placebo control matters more than the seven targeted checks: it is sensitive to leakage regardless of mechanism.
