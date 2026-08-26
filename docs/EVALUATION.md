# Evaluation

Three studies on one dataset, one calendar, one cost model and one shared signal context — so a difference between rows is a difference between the things being compared and nothing else.

Reproduce:

```bash
python scripts/run_all.py --full          # suite + ablation + factor study + audit
python scripts/run_robustness.py cost     # sweeps run individually
python scripts/run_robustness.py --render
```

Every row carries the `config_digest` of the settings that produced it.

## 1. Strategy comparison

Synthetic dataset, seed 7, 40 assets, 2010-01-04 → 2024-12-31, 156 monthly rebalances, 10 bps round-trip costs.

| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar | CVaR95 | Beta | Turnover | Cost drag |
|---|---|---|---|---|---|---|---|---|---|---|
| benchmark_buy_hold | 1.91% | 14.3% | 0.05 | 0.07 | −55.2% | 0.03 | −1.98% | 1.00 | 0.0% | 0.00% |
| A_equal_weight_universe | 2.39% | 14.8% | 0.08 | 0.12 | −52.0% | 0.05 | −2.06% | 1.00 | 3.0% | 0.93% |
| B_factor_equal_weight | 6.18% | 15.2% | 0.33 | 0.47 | −30.3% | 0.20 | −2.10% | 0.95 | 21.5% | 6.72% |
| C_factor_regime_equal_weight | 6.06% | 12.8% | 0.35 | 0.50 | −26.1% | 0.23 | −1.78% | 0.78 | 21.6% | 6.75% |
| D_factor_mean_variance | 6.36% | 13.4% | 0.36 | 0.52 | −23.5% | 0.27 | −1.84% | 0.81 | 22.2% | 6.92% |
| D2_factor_risk_parity | 6.40% | 14.5% | 0.35 | 0.50 | −28.4% | 0.23 | −2.00% | 0.91 | 21.3% | 6.64% |
| E_factor_optimized_risk_constrained | 6.69% | 15.5% | 0.35 | 0.51 | −26.2% | 0.26 | −2.10% | 0.90 | 26.5% | 8.25% |
| **F_finrisk_full** | **6.81%** | **13.3%** | **0.40** | **0.57** | **−23.7%** | **0.29** | **−1.82%** | **0.74** | 24.7% | 7.71% |

Sharpes look low because the 2–4% risk-free rate is genuinely deducted.

## 2. Ablation — each rung adds one component

| Rung | CAGR | ΔCAGR | Vol | Sharpe | ΔSharpe | Max DD | CVaR95 | Turnover |
|---|---|---|---|---|---|---|---|---|
| A: equal-weight universe | 2.39% | — | 14.8% | 0.08 | — | −52.0% | −2.06% | 3.0% |
| B: + factor selection | 6.18% | **+3.80%** | 15.2% | 0.33 | **+0.24** | −30.3% | −2.10% | 21.5% |
| C: + regime overlay | 6.06% | −0.12% | 12.8% | 0.35 | +0.02 | −26.1% | −1.78% | 21.6% |
| D: + mean-variance | 6.36% | +0.31% | 13.4% | 0.36 | +0.01 | −23.5% | −1.84% | 22.2% |
| D2: + risk parity (alt to D) | 6.40% | +0.03% | 14.5% | 0.35 | −0.01 | −28.4% | −2.00% | 21.3% |
| E: + volatility constraint | 6.69% | +0.30% | 15.5% | 0.35 | +0.00 | −26.2% | −2.10% | 26.5% |
| F: full pipeline | 6.81% | +0.12% | 13.3% | 0.40 | +0.04 | −23.7% | −1.82% | 24.7% |

**Factor selection does nearly all the work.** Everything after it is refinement, and individual steps (ΔSharpe of +0.01, +0.00, −0.01) are within noise.

The regime overlay **costs return and buys risk reduction**: −0.12pp CAGR, but volatility 15.2%→12.8%, drawdown −30.3%→−26.1%, CVaR −2.10%→−1.78%, beta 0.95→0.78. That is the trade it was built to make; calling it a return improvement would be dishonest.

## 3. Robustness — where the result holds and where it breaks

### Transaction costs (Sharpe spread 0.17)

| Cost | CAGR | Sharpe | Cost drag |
|---|---|---|---|
| 0 bps | 7.42% | 0.44 | 0.00% |
| 5 bps | 7.12% | 0.42 | 3.85% |
| 10 bps (default) | 6.81% | 0.40 | 7.71% |
| 20 bps | 6.20% | 0.35 | 15.42% |
| 40 bps | 4.99% | 0.27 | 30.83% |

Survives realistic costs; degrades gracefully. At 40 bps the strategy still beats the benchmark, but cost drag of 30.8% cumulative is severe.

### Rebalance frequency (spread 0.04)

| Frequency | CAGR | Sharpe | Turnover | Cost drag |
|---|---|---|---|---|
| Weekly | 6.21% | 0.35 | 11.7% | 16.00% |
| Monthly | 6.81% | 0.40 | 24.7% | 7.71% |
| Quarterly | 6.82% | 0.39 | 40.6% | 4.22% |

Genuinely insensitive — the most reassuring sweep here. Monthly and quarterly are statistically indistinguishable; weekly loses to cost drag. Note turnover is *per rebalance*, so weekly's 11.7% × 52 is far more annual trading than quarterly's 40.6% × 4.

### Factor weights (spread 0.63) — **the fragility**

| Scheme | CAGR | Sharpe | Max DD |
|---|---|---|---|
| momentum tilt (55%) | **2.31%** | **0.07** | −28.2% |
| defensive (40% risk) | 5.44% | 0.31 | −32.0% |
| equal all | 6.28% | 0.36 | −22.7% |
| baseline (30/25/25/20) | 6.81% | 0.40 | −23.7% |
| quality tilt (55%) | 7.52% | 0.45 | −23.3% |
| value tilt (55%) | **11.39%** | **0.70** | −24.7% |

**This is the least robust dimension, by a wide margin.** Sharpe ranges from 0.07 to 0.70 — a 10× spread driven purely by factor weighting.

It is also exactly what the IC analysis predicts: value carries the signal (IC +0.169 at 252d), momentum is negative (−0.072). Tilting toward the working factor helps; tilting toward the broken one nearly destroys the strategy.

Two honest readings, and I hold both:

1. The result is *internally consistent* — the sweep confirms the IC finding through an independent route.
2. The baseline's 30% momentum weight is **wrong for this dataset**, and I have not re-tuned it, because tuning weights on the same data used to evaluate them is precisely the overfitting this project is built to avoid. The baseline is a defensible prior from published equity research; it is simply mis-specified for this DGP.

### Constraints (spread 0.11)

| Setting | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| top-5, max 35%, vol 15% | 6.96% | 13.3% | 0.41 | −24.3% |
| top-10, max 20%, vol 15% | 6.81% | 13.3% | 0.40 | −23.7% |
| top-20, max 12%, vol 15% | 6.73% | 12.7% | 0.40 | −21.2% |
| top-10, max 20%, **vol 10%** | 5.86% | **9.1%** | **0.43** | **−14.6%** |
| top-10, max 20%, vol 25% | 6.02% | 15.0% | 0.32 | −27.6% |

Insensitive to universe size and position cap. The volatility target does what it should: the 10% target delivers 9.1% realised volatility, a −14.6% drawdown and the *best* Sharpe (0.43) despite the lowest CAGR. The 25% target is strictly worse on every risk-adjusted measure — more risk buys less return here.

Note the 15% target realises 13.3% under the full pipeline, while strategy E alone overshoots to 15.5%. The regime overlay absorbs the overshoot.

### Sub-periods (spread 0.52) — **the other fragility**

| Period | CAGR | Sharpe |
|---|---|---|
| Bull regimes | 9.30% | 0.56 |
| Bear regimes | 5.33% | 0.29 |
| High-volatility years | 5.36% | 0.29 |
| **First half (2010–2017)** | **10.38%** | **0.62** |
| **Second half (2017–2024)** | **2.73%** | **0.11** |

The chronological split is the most important number in this document after the ablation.

Performance degrades sharply in the second half: Sharpe 0.62 → 0.11, CAGR 10.38% → 2.73%. The chronological halves are the cleanest out-of-sample split available here, and the strategy does **not** hold up across them.

I want to be careful about interpretation, because two explanations are consistent with the data and I cannot separate them with this sample:

- **Real degradation** — the signal weakens over the sample.
- **Regime composition** — the second half contains a different mix of the DGP's bull/chop/bear states, and the bull/bear split above (0.56 vs 0.29) shows the strategy is regime-sensitive. If the second half is bear-heavy, some of the drop is composition rather than decay.

Either way, the honest headline is: **a single full-sample Sharpe of 0.40 conceals a 0.62/0.11 split, and quoting only the full-sample figure would misrepresent the strategy.**

The strategy holds its risk profile throughout — volatility stays 13.3–13.7% and drawdown never exceeds −24% in any sub-period. It is the *return* that is unstable, not the risk control.

## 4. Statistical significance

Standard error of an annualised Sharpe ≈ √((1 + S²/2)/T), T in years. Over ~13.5 years that is ≈0.28 per strategy; a pairwise difference needs ≈0.78 to clear two standard errors.

```
Benchmark: benchmark_buy_hold (Sharpe 0.05, CAGR 1.91%)

B_factor_equal_weight     Sharpe 0.33  diff +0.28  -> not distinguishable
F_finrisk_full            Sharpe 0.40  diff +0.35  -> not distinguishable
```

**No strategy is statistically distinguishable from the benchmark at this sample size.** The pairwise test assumes independence, which overstates the standard error of the *difference* (both series share market exposure), so this is a conservative bound — but the conclusion stands.

The IC analysis is far better powered because it uses 156 cross-sections rather than one time series: value at t=+11.18 over 252 days is not noise.

**Correct summary of the whole evaluation:** the factor signal is statistically solid and its behaviour is confirmed through two independent routes (IC analysis and the weight sweep). The portfolio-construction machinery layered on top is directionally sensible, materially improves drawdown, and is individually within noise. Time-stability is the strategy's weakest property.

## 5. Data leakage audit

8 checks, 0 failures. Headline placebo control: honest CAGR 6.18% / Sharpe 0.33 versus one-period look-ahead CAGR 33.22% / Sharpe 1.85 — cheating is worth +27.04pp CAGR, so the information barrier is load-bearing. Full detail in `DATA_LEAKAGE_AUDIT.md`.

## 6. Factor neutralisation — does the premium survive?

A raw factor portfolio is rarely a pure bet on the factor. Sorting on
book-to-price loads on whatever sectors are cheap; sorting on momentum loads on
high-beta names. Neutralisation removes those exposures so what remains is the
factor itself.

Identical construction and return calculation for each variant, so a difference
between rows is attributable to the neutralisation and nothing else.

**Value — survives**

| Variant | Ann return | Vol | Sharpe | Max DD | \|Net\| | \|Sector\| | \|Beta\| |
|---|---|---|---|---|---|---|---|
| raw | 10.67% | 10.9% | 0.98 | −19.3% | 0.0000 | 0.2401 | 0.0785 |
| sector-neutral | 9.62% | 10.1% | 0.95 | −18.6% | 0.0000 | **0.0000** | 0.0897 |
| sector + beta-neutral | 10.14% | 9.9% | **1.03** | −17.9% | 0.0000 | **0.0000** | **0.0000** |

**Quality — does not survive**

| Variant | Ann return | Sharpe | \|Sector\| | \|Beta\| |
|---|---|---|---|---|
| raw | 5.33% | 0.58 | 0.2305 | 0.0784 |
| sector-neutral | 3.00% | 0.32 | 0.0000 | 0.0801 |
| sector + beta-neutral | 2.15% | **0.23** | 0.0000 | 0.0000 |

**Momentum — nothing to remove** (raw Sharpe −0.01, neutral −0.04; it never worked).

The headline: **value's premium is real; more than half of quality's apparent
premium was a sector bet.** A quality strategy that looked like a 0.58 Sharpe is
a 0.23 Sharpe once you stop paying for sector exposure you did not intend.

The exposure columns are the evidence, not decoration. Sector exposure falls
0.24 → 0.0000 and beta 0.078 → 0.0000, so the "neutral" label is measured
rather than asserted. They are not *exactly* zero at construction time because
betas are estimated with error.

Reproduce: `python scripts/run_all.py --full` → `evaluation/results/neutralization.csv`.

## 7. Retrieval ablation — does hybrid RAG actually help?

Gold labels are derived mechanically from the corpus generator (each synthetic
filing is written from one issuer's known fundamentals), so the probe set is not
tuned to any retriever. 180 probes: numeric-discrimination facts scoped to one
issuer, plus paraphrases worded to avoid the corpus vocabulary.

| Mode | R@1 | R@5 | MRR | nDCG@5 | latency |
|---|---|---|---|---|---|
| bm25 | **0.99** | 1.00 | **0.997** | 0.998 | 0.4 ms |
| dense (LSA) | 0.80 | 0.99 | 0.897 | 0.923 | 4.5 ms |
| hybrid (RRF) | 0.99 | 1.00 | 0.994 | 0.995 | 4.7 ms |

**Honest result: hybrid does NOT beat BM25 on this corpus, and dense alone is
worse.** The cause is diagnosed rather than guessed: the synthetic corpus is
template-generated with only ~2,665 vocabulary terms across 800 chunks, so the
prose barely varies and only the *numbers* differ — which is precisely BM25's
strength and offers nothing for semantics to exploit.

The machinery is not at fault. On a paraphrase-rich corpus where documents say
different things in different words, dense retrieval **beats** BM25 on top-1
accuracy (50% vs 33%), because queries like "why did profitability decline"
share no content word with "gross margin contracted as input cost inflation
outpaced pricing actions". That contrast is asserted in
`tests/test_hybrid_rag.py::test_dense_finds_paraphrase_that_bm25_misses`.

Hybrid remains the default because it *ties* BM25 here and wins on varied text,
which makes it the safer choice across corpus types — at roughly 10× the
latency, which is 4 ms and irrelevant at this scale. On a corpus of real
filings, hybrid is the right default; on this one, BM25 alone would be equally
good and the documentation says so.

## 8. Test suite

**368 tests, 3m35s, zero warnings.** The count below is generated from the suite, not maintained by hand.

| Module | Tests | Focus |
|---|---|---|
| `test_agents` | 33 | **Critic catches injected fabrications**; plan and hypothesis grading |
| `test_risk_metrics` | 33 | Closed-form validation of every formula |
| `test_optimizer` | 32 | Constraint enforcement incl. turnover and risk budget |
| `test_factors` | 30 | Construction, sign conventions, normalisation |
| `test_hybrid_rag` | 30 | Embeddings, vector store, RRF fusion, provenance |
| `test_backtest` | 24 | Execution lag, costs, drift, leakage assertion |
| `test_position_attribution` | 24 | Alpha, risk, diversification, derived confidence |
| `test_regimes` | 24 | HMM correctness, causality guarantee |
| `test_data` | 23 | PIT discipline, cleaning, validation |
| `test_rag` | 23 | Chunking, BM25, filters, citations |
| `test_tools` | 23 | Schema validation, determinism, error isolation |
| `test_sec_filings` | 21 | 10-K/10-Q/8-K parsing, TOC trap, filing-date discipline |
| `test_neutralization` | 20 | **Neutrality proved by measuring realised exposures** |
| `test_factor_recovery` | 8 | **Ground-truth recovery against latent states** |

Static analysis: `ruff check`, `black --check` and `mypy` all pass clean across
55 source files.
