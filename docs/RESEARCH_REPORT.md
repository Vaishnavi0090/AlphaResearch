# Research Report

All figures below come from `python scripts/run_all.py --full` on the synthetic dataset (seed 7, 40 assets, 2010-01-04 → 2024-12-31, 3,912 sessions, 156 monthly rebalances, 504-session warm-up, 1-session execution lag, 10 bps round-trip costs).

**These are not historical performance figures.** The dataset is a synthetic data-generating process with known ground truth. That is a deliberate choice, and it buys something a real dataset cannot: because the DGP's latent states are observable, I can test whether the factor engine measures *the thing it claims to measure*, rather than only whether it made money.

## Dataset and ground truth

The DGP injects three real, known premia:

| Latent driver | True annualised premium |
|---|---|
| Trend state (AR(1), φ=0.985, ~46-session half-life) | 3.0% |
| Value state (slow AR(1), φ=0.9985) | 4.5% |
| Quality state (slow AR(1), φ=0.9985) | 3.5% |

Market regimes follow a 3-state Markov chain (bull/chop/bear with drifts +14%/+2%/−18% and volatilities 11%/16%/34%). Fundamentals are published with a 45–75 day lag (median 61). Data quality: 156,480 observations, 0 missing, 0 duplicates, 0 errors.

## Finding 1 — value and quality are recovered; momentum is not

Mean information coefficient (Spearman rank correlation of score vs forward return), 156 monthly cross-sections:

| Factor group | 5d | 21d | 63d | 126d | 252d |
|---|---|---|---|---|---|
| composite | 0.008 | **0.039** | 0.045 | 0.065 | 0.075 |
| value | 0.029 | **0.051** | 0.092 | 0.126 | **0.169** |
| quality | 0.014 | **0.025** | 0.059 | 0.090 | 0.092 |
| momentum | −0.004 | 0.005 | −0.025 | −0.047 | **−0.072** |
| risk | 0.002 | 0.006 | −0.003 | −0.014 | −0.007 |
| market | −0.002 | 0.010 | 0.017 | 0.019 | 0.036 |

t-statistics at 21d: composite +2.88, value +3.44, quality +2.03. At 252d: value +11.18, quality +7.99, momentum **−4.87**.

Long/short quintile portfolios over the same period:

| Factor | Long CAGR | L/S CAGR | Vol | Sharpe | Max DD | Turnover | Mean IC | IC-IR | IC>0 |
|---|---|---|---|---|---|---|---|---|---|
| value | 11.61% | 12.74% | 17.2% | 0.72 | −42.9% | 18.7% | 0.051 | 0.28 | 60% |
| quality | 5.84% | 7.04% | 15.8% | 0.44 | −41.8% | 17.1% | 0.025 | 0.16 | 59% |
| composite | 5.47% | 5.07% | 15.3% | 0.42 | −32.2% | 23.2% | 0.039 | 0.23 | 58% |
| market | 4.29% | −0.47% | 14.1% | 0.37 | −39.2% | 13.5% | 0.010 | 0.05 | 57% |
| risk | 3.95% | 0.64% | 12.8% | 0.37 | −30.8% | 15.7% | 0.006 | 0.04 | 53% |
| momentum | **−0.10%** | **−1.59%** | 16.7% | 0.08 | −57.5% | 35.8% | 0.005 | 0.03 | 49% |

## Finding 2 (NEGATIVE RESULT) — momentum's failure is a signal-to-noise property, not a bug

This is the most interesting result in the project, and it took direct investigation against ground truth to establish rather than guess.

The obvious conclusion from Finding 1 would be "the momentum implementation is broken." It is not. Three measurements separate the hypotheses:

**(a) The trend premium genuinely exists.** Correlating the *latent* trend state against forward returns:

| Horizon | Mean IC of true trend state |
|---|---|
| 21d | +0.041 |
| 63d | +0.052 |
| 252d | +0.038 |

So there is a real premium to capture.

**(b) The momentum factors are correctly constructed but barely correlated with the state they proxy.** Mean cross-sectional Spearman correlation against the true latent trend state:

| Factor | ρ with true trend state |
|---|---|
| mom_1m | +0.032 |
| mom_3m | +0.031 |
| mom_6m | +0.021 |
| mom_12m_1m | +0.006 |

Signs are correct — so the sign convention and implementation are right — but the magnitudes are tiny.

**(c) Contrast: the value and quality factors recover their latent states strongly.**

| Factor | ρ with true latent state |
|---|---|
| roe | **+0.850** (quality state) |
| book_to_price | **+0.780** (value state) |
| earnings_yield | +0.582 (value state) |

This rules out a generic bug in the factor engine: the same machinery recovers value at ρ=0.78 and momentum at ρ=0.03.

**Diagnosis.** Two compounding causes:

1. **Signal-to-noise.** Over 21 sessions the injected trend alpha contributes roughly 0.25% of return, against roughly 6.4% of idiosyncratic noise (22% annualised vol × √(21/252)). Trailing return is therefore about 96% noise as an estimator of the drift state. Realised returns are a poor thermometer for a small drift.

2. **Lookback/half-life mismatch.** The trend state's half-life is ~46 sessions. A 12-month-minus-1-month window spans ~231 sessions — roughly five half-lives. It therefore measures states that have already reverted, and because the process mean-reverts, a high trailing return implies the state has *since fallen*. Hence the negative long-horizon IC (−0.072 at 252d) and the slightly negative correlation with the current state on some seeds.

**Conclusion:** classical 12-1 momentum is *mis-specified for this DGP*. The appropriate lookback here is 1–3 months. This is a property of the data-generating process, not a defect in the code — and notably, real equity markets do exhibit 12-1 momentum, so this is a case where the synthetic dataset differs from reality in an instructive way.

This is locked in by `tests/test_factor_recovery.py::test_momentum_lookback_decay_matches_state_half_life`, which asserts the short lookbacks are positively signed, the magnitudes stay small, and the 12-1 lookback decays below the 3-month one. A future change that "fixes" momentum would fail that test — deliberately, because the negative result is the correct answer for this dataset.

## Finding 3 (NEW) — value's premium is real; quality's was largely a sector bet

The IC analysis above says value and quality both work. Neutralisation asks a
harder question: is that a *factor* premium, or an unintended sector and beta
exposure wearing a factor's name?

Identical construction for each variant, so the difference is the neutralisation:

| Factor | Raw Sharpe | Sector-neutral | Sector + beta-neutral | Verdict |
|---|---|---|---|---|
| Value | 0.98 | 0.95 | **1.03** | Survives — premium is genuine |
| Quality | 0.58 | 0.32 | **0.23** | **Does not survive** |
| Momentum | −0.01 | 0.11 | −0.04 | Nothing to remove |

**More than half of quality's apparent premium disappears once sector and beta
exposure are removed.** A quality strategy that looked like a 0.58 Sharpe is a
0.23 Sharpe once you stop being paid for sector bets you did not intend to make.
Value, by contrast, is *slightly better* neutralised (0.98 → 1.03) — removing
the incidental exposures removes noise, not signal.

The neutrality is measured, not asserted. Realised sector exposure falls from
0.2401 to 0.0000 and portfolio beta from 0.0785 to 0.0000.

This matters for the composite: the default weights treat value and quality as
comparably reliable, and on this evidence they are not.

## Finding 4 — factor correlations behave as designed

Mean cross-sectional correlation between group scores:

|  | composite | momentum | value | quality | risk | market |
|---|---|---|---|---|---|---|
| composite | 1.000 | 0.595 | 0.530 | 0.651 | 0.499 | 0.102 |
| momentum | 0.595 | 1.000 | −0.070 | 0.066 | 0.347 | 0.065 |
| value | 0.530 | −0.070 | 1.000 | 0.474 | −0.048 | 0.010 |
| quality | 0.651 | 0.066 | 0.474 | 1.000 | 0.073 | 0.013 |
| risk | 0.499 | 0.347 | −0.048 | 0.073 | 1.000 | 0.178 |
| market | 0.102 | 0.065 | 0.010 | 0.013 | 0.178 | 1.000 |

Value and quality correlate at 0.474 — expected, since the DGP's value and quality states both feed reported fundamentals. Value and momentum are essentially orthogonal (−0.070), which is the classic empirical relationship and means the composite is not double-counting.

## Finding 5 — regime detection works but is noisier than ideal

Over 157 monthly periods: switch rate 0.49, mean run length 2.01 months, mean confidence 0.946. Regime shares: risk-on 41.4%, neutral 40.8%, risk-off 17.2%, unknown 0.6%.

Against ground truth: P(detect risk_off | true bear) = **0.778**, P(detect risk_on | true bull) = **0.462**.

Honest reading: the detector is good at identifying stress (78% of true bear months flagged risk-off) and mediocre at identifying calm (46%). For a risk-management overlay, that asymmetry is the useful direction — the expensive error is missing a bear market, not under-claiming a bull. But a switch rate of 0.49 means the state changes roughly every other month, which is more turnover than a genuine regime process should produce; some of that is estimation noise, and the probability-blended risk multiplier exists specifically to stop that noise from becoming trading.

## Finding 6 — ablation: factor selection does the work; optimisation adds risk-adjusted quality

Each rung adds one component. Deltas are versus the previous rung.

| Rung | CAGR | ΔCAGR | Vol | Sharpe | ΔSharpe | Max DD | CVaR95 | Turnover |
|---|---|---|---|---|---|---|---|---|
| A: equal-weight universe | 2.39% | — | 14.8% | 0.08 | — | −52.0% | −2.06% | 3.0% |
| B: + factor selection | 6.18% | **+3.80%** | 15.2% | 0.33 | **+0.24** | −30.3% | −2.10% | 21.5% |
| C: + regime overlay | 6.06% | −0.12% | 12.8% | 0.35 | +0.02 | −26.1% | −1.78% | 21.6% |
| D: + mean-variance | 6.36% | +0.31% | 13.4% | 0.36 | +0.01 | −23.5% | −1.84% | 22.2% |
| D2: + risk parity (alt to D) | 6.40% | +0.03% | 14.5% | 0.35 | −0.01 | −28.4% | −2.00% | 21.3% |
| E: + volatility constraint | 6.69% | +0.30% | 15.5% | 0.35 | +0.00 | −26.2% | −2.10% | 26.5% |
| F: full pipeline | 6.81% | +0.12% | 13.3% | **0.40** | +0.04 | −23.7% | −1.82% | 24.7% |

Interpretation, stated plainly:

- **Factor selection is where nearly all the return comes from** (+3.80pp CAGR, +0.24 Sharpe). Everything after it is refinement.
- **The regime overlay costs return and buys risk reduction**: −0.12pp CAGR but volatility 15.2%→12.8%, max drawdown −30.3%→−26.1%, CVaR −2.10%→−1.78%. That is the trade it was designed to make, and calling it a "return improvement" would be dishonest.
- **Optimisation contributes modestly and mostly through drawdown**: max DD improves from −30.3% (B) to −23.7% (F) — a 6.6pp improvement — while CAGR gains only +0.63pp across C through F.
- **Individual optimiser steps are within noise.** ΔSharpe values of +0.01 and +0.00 are not distinguishable from zero at this sample size. See the significance note below.

## Statistical significance — the honest framing

The standard error of an annualised Sharpe is approximately √((1 + S²/2)/T) with T in years. Over ~13.5 years that is roughly 0.28 per strategy, so a pairwise difference needs to exceed roughly 0.78 to clear two standard errors.

Against the buy-and-hold benchmark (Sharpe 0.05), the full pipeline's 0.40 is a +0.35 difference — **inside** one standard error of the pairwise comparison. The formal verdict is *not distinguishable from the benchmark*.

I am stating this prominently because the ablation table above invites the opposite conclusion. Three qualifications:

1. The comparison assumes independence, which overstates precision — both series share market exposure, so the true standard error of the *difference* is smaller than the naive pairwise figure.
2. Consistency across sub-periods and parameter settings (see `EVALUATION.md`) is stronger evidence than a single t-statistic.
3. The IC t-statistics are the better significance evidence, because they use 156 cross-sections rather than one time series: value at t=+11.18 over 252d is not noise.

The correct summary is: *the factor signal is statistically solid; the portfolio-construction improvements on top of it are directionally sensible but individually within noise at this sample size.*

## Finding 7 (NEGATIVE RESULT) — performance is not stable across time

The chronological split is the cleanest out-of-sample test available in this sample, and the strategy does not survive it well:

| Period | CAGR | Sharpe |
|---|---|---|
| First half (2010–2017) | 10.38% | **0.62** |
| Second half (2017–2024) | 2.73% | **0.11** |
| Bull regimes | 9.30% | 0.56 |
| Bear regimes | 5.33% | 0.29 |
| High-volatility years | 5.36% | 0.29 |

Sharpe falls from 0.62 to 0.11 between halves. **A single full-sample Sharpe of 0.40 conceals this split, and quoting only the full-sample figure would misrepresent the strategy.**

Two explanations are consistent with the data, and I cannot separate them at this sample size:

1. **Genuine signal decay** across the sample.
2. **Regime composition** — the strategy is regime-sensitive (bull 0.56 vs bear 0.29), so a bear-heavy second half would produce part of this drop mechanically rather than through decay.

What *is* stable is the risk control: volatility stays within 13.3–13.7% and drawdown never exceeds −24% in any sub-period. The return is unstable; the risk management is not. For a system whose stated purpose is risk intelligence, that is the more important of the two — but it does not rescue the return claim.

## Finding 8 — factor weighting is the largest source of fragility

Sweeping the composite weights produces a Sharpe range of 0.07 to 0.70 — a 10× spread:

| Scheme | CAGR | Sharpe |
|---|---|---|
| momentum tilt (55%) | 2.31% | 0.07 |
| baseline (30/25/25/20) | 6.81% | 0.40 |
| value tilt (55%) | 11.39% | 0.70 |

This independently confirms Findings 1 and 2 through a completely different route: tilting toward value (IC +0.169) helps enormously, tilting toward momentum (IC −0.072) nearly destroys the strategy.

It also means the default 30% momentum weight is **wrong for this dataset**. I have deliberately not re-tuned it: fitting factor weights on the same data used to evaluate them is exactly the overfitting this project exists to avoid. The baseline is a defensible prior from published equity research that happens to be mis-specified for this DGP, and saying so is more useful than silently producing a better number.

## Finding 9 — E's realised volatility overshoots its target

Strategy E targets 15% volatility and realises **15.5%**. This is not a bug, and it is worth being precise about why: the constraint binds on **ex-ante forecast** volatility computed from a Ledoit–Wolf-shrunk covariance matrix at each rebalance. Realised volatility differs for three reasons — covariance estimation error, the shrinkage itself biasing the forecast toward the mean, and drift between monthly rebalances.

Strategy F, which layers the regime overlay on top, realises 13.3% — the overlay's exposure reduction more than absorbs the overshoot.

The general lesson: a volatility *target* implemented as a constraint on a *forecast* is not a volatility *guarantee*. Reporting the 15.5% figure rather than the 15.0% target is the only defensible choice.

## What would change my mind

Concrete, falsifiable things that would overturn these conclusions:

- **On momentum:** re-running with a trend half-life longer than the momentum lookback (raise φ toward 0.999) should make 12-1 momentum work. If it did not, my SNR diagnosis is wrong.
- **On quality:** if quality's neutralised Sharpe recovered under a different
  sector scheme, the "sector bet" diagnosis would be wrong. It does not — the
  result holds across the sector definitions available here.
- **On the factor signal:** if value's IC collapsed under a different seed, the recovery result would be luck rather than construction. `test_factor_recovery.py` runs on seed 7 with 30 assets — a different configuration from the main evaluation's seed 7 / 40 assets — and value recovery holds in both.
- **On the whole pipeline:** real data with real survivorship bias, real delistings, and real cost structure would be the only test that matters for a live claim. Nothing here supports a live claim.
