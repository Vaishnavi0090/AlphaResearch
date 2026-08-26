# Limitations

Written deliberately as the document I would most want a sceptical reader to find. Every claim this project makes elsewhere is bounded by something here.

## 1. The results are synthetic-data validation, not performance

**This is the limitation that subsumes most others.** Network egress to every financial data provider (Yahoo Finance, FRED, SEC EDGAR — all HTTP 403) was blocked in the build environment, so the evaluation runs on a seeded synthetic data-generating process.

What that means:
- **No number in this repository is historical performance.** A 6.81% CAGR describes a synthetic universe, not a market.
- The DGP was written by the same person who wrote the factors. That is a real conflict: I injected value and quality premia and then found value and quality premia. The momentum result is the honest counterweight — I injected a trend premium and my momentum factors *failed* to capture it.
- Real market microstructure, liquidity, delistings, corporate actions, and crowding are absent.

What it does buy: ground truth. Because latent states are observable, `tests/test_factor_recovery.py` can verify the engine measures what it claims (book-to-price recovers the latent value state at ρ=0.78) — a check no real dataset permits.

Real providers are implemented (`YFinanceProvider`, `FredProvider`, `EdgarProvider`) and work outside a sandbox. Switch via `settings.data.provider = "yfinance"`.

## 2. Survivorship bias (real-data runs)

The synthetic universe has no survivorship bias — and no delistings either, which is optimistic in its own way, since real portfolios hold companies that fail.

For real data with a fixed current ticker list, survivorship bias **is present and unsolved**. Companies that went bankrupt are silently excluded, which inflates returns. Fixing it requires a point-in-time constituent list (CRSP or similar) that I do not have access to. The audit reports this as a `warning`, not a pass.

## 3. Statistical power is low, and the ablation table oversells

Over ~13.5 years the standard error of an annualised Sharpe is roughly 0.28. Pairwise, a difference must exceed roughly 0.78 to clear two standard errors.

The full pipeline's Sharpe of 0.40 versus the benchmark's 0.05 is a +0.35 difference — **not statistically distinguishable**. Individual ablation steps (ΔSharpe of +0.01, +0.00, −0.01) are unambiguously noise.

The IC analysis is better-powered because it uses 156 cross-sections rather than one time series (value at t=+11.18 over 252d). So: the *signal* is solid; the *construction improvements layered on top of it* are directionally sensible but individually unproven.

## 3b. Performance is not stable across time — the most serious limitation

Chronological halves: Sharpe **0.62** (2010–2017) versus **0.11** (2017–2024); CAGR 10.38% versus 2.73%.

This is the cleanest out-of-sample split the sample permits, and the strategy does not hold up across it. I cannot distinguish genuine signal decay from regime composition (the strategy is regime-sensitive: bull 0.56, bear 0.29), and the sample is too short to settle it.

Risk control *is* stable — volatility 13.3–13.7% and drawdown within −24% in every sub-period. So the honest statement is that the risk machinery generalises and the return does not.

Anyone quoting the full-sample Sharpe of 0.40 without this split would be misrepresenting the result.

## 3c. Factor weights are the largest fragility

Sharpe ranges 0.07 (momentum tilt) to 0.70 (value tilt) — a 10× spread from weighting alone. The default 30% momentum weight is mis-specified for this dataset.

I have not re-tuned it, deliberately: fitting weights on the evaluation data is the overfitting this project is built to avoid. But it means the headline configuration is *not* the best available one, and a reader should know that the reported 6.81% CAGR sits well below the 11.39% a value tilt would have produced.

## 3d. Quality's premium does not survive neutralisation

Sector- and beta-neutralising quality cuts its Sharpe from 0.58 to 0.23. Most of
the apparent premium was sector exposure, not stock selection.

The composite weights quality at 25%, equal to value — and on this evidence that
is not justified. As with the momentum weighting, I have not re-tuned it, for the
same reason: fitting weights on the evaluation data is the overfitting this
project exists to avoid. But a reader should know the default configuration
over-weights a factor whose premium is largely incidental exposure.

## 3e. Hybrid retrieval does not improve on the synthetic corpus

Measured: BM25 MRR 0.997, hybrid 0.994, dense 0.897. The template-generated
corpus has ~2,665 vocabulary terms across 800 chunks — the prose barely varies
and only numbers differ, which is BM25's strength.

Hybrid is kept as the default because it ties here and wins on paraphrase-rich
text (dense beats BM25 50% vs 33% top-1 on such a corpus), but **no claim is
made that hybrid RAG improves retrieval in this project's own evaluation.** On
real filings it should; that is untested here.

The dense encoder is corpus-fitted LSA, not a pretrained model, because model
weights are unreachable (huggingface.co blocked). LSA generalises less well and
needs refitting when the corpus changes.

## 4. Volatility targeting overshoots

Strategy E targets 15% and realises **15.5%**. The constraint binds on *ex-ante forecast* volatility from a shrunk covariance matrix; realised volatility differs through estimation error, shrinkage bias, and drift between monthly rebalances.

A volatility target implemented as a constraint on a forecast is not a volatility guarantee. Anyone reading "15% vol target" as a promise would be misreading it.

## 5. Transaction costs are a linear approximation

Costs are `notional × (commission + spread/2 + slippage)`, defaulting to 10 bps round trip. Missing:

- **Market impact**, which scales roughly with (order size / ADV)^0.6 and would penalise concentrated positions disproportionately.
- **Spread variation** — spreads widen in exactly the stressed conditions where the regime overlay wants to trade.
- **Borrow costs**, irrelevant here (long-only) but required for any short extension.

Cost drag is material at 6.6–8.3% cumulative, so this approximation matters. The robustness sweep runs 0/5/10/20/40 bps precisely because the assumption is uncertain.

## 6. Momentum is mis-specified for this DGP

Documented at length in `RESEARCH_REPORT.md`. Briefly: the trend state's ~46-session half-life is far shorter than a 231-session 12-1 lookback, and trailing return is ~96% noise as a drift estimator. Momentum's long-horizon IC is −0.072.

Real equity markets *do* exhibit 12-1 momentum. So this is a case where the synthetic dataset diverges from reality, and the composite's default 30% momentum weight is inappropriate for this data specifically.

## 7. Regime detection is noisy

Switch rate 0.49 means the state changes roughly every other month — more than a genuine regime process should produce. P(detect risk_on | true bull) is only **0.462**.

The detector is much better at stress (P(risk_off | true bear) = 0.778) than at calm. For a risk overlay that asymmetry is the useful direction, but the model is not a reliable bull-market detector, and the probability-blended multiplier exists to stop the noise becoming turnover.

## 8. The LLM layer is a stub by default

`StubLLM` is a **deterministic template narrator, not a language model**. It is honestly named and exists so the graph runs in CI with no API key and tests assert exact strings.

Consequently: the committee's adversarial structure is verified, but the *quality* of real LLM reasoning is not evaluated here. With `AnthropicLLM` the critic still enforces numeric grounding, but no benchmark in this repo measures how often a real model attempts a fabrication.

## 9. RAG retrieval is lexical

Retrieval is now hybrid (BM25 + dense, fused by RRF), but the dense half is
corpus-fitted LSA rather than a pretrained bi-encoder, because model weights are
unreachable here. LSA learns semantics only from co-occurrence within the
indexed documents, so it generalises less well than a pretrained model and must
be refitted when the corpus changes materially.

And, per §3e, hybrid retrieval does **not** improve results on this corpus.

The corpus is synthetic, generated from the same fundamentals the factors use, so retrieval answers are internally checkable but say nothing about real filing language.

## 10. Single-asset-class, long-only, daily

No fixed income, FX, commodities, or derivatives. No shorting, no leverage. Daily bars only — no intraday, no overnight-gap modelling. Rebalance frequencies limited to weekly/monthly/quarterly.

## 11. Not investment advice

Research and decision-support software. It has never traded real capital. Backtested and simulated results do not indicate future returns.

## What I would fix first, in order

1. Real point-in-time data with delistings (fixes #1, #2 — and makes #3 answerable).
2. Longer sample / more assets for statistical power (#3).
3. Market-impact cost model (#5).
4. Re-tune momentum lookback to the data, or drop the factor (#6); reconsider
   quality's composite weight given §3d.
5. A pretrained bi-encoder in place of corpus-fitted LSA (#9), and a corpus with
   real linguistic variety so the dense half has something to contribute.
