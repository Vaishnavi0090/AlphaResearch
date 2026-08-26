# Resume Traceability

Every claim below maps to code, a test, and a measured number. Nothing here is aspirational, and nothing implies live trading or historical market performance.

## Suggested bullets

> **Built an agentic quantitative portfolio platform with a structurally enforced separation between LLM reasoning and numerical decisions** — weights come only from a convex optimiser; a validation agent scrubs any figure not traceable to a deterministic tool. Verified by injecting fabricated metrics and citations into the pipeline (24 agent tests).
>
> *Code:* `finrisk/agents/critic.py`, `finrisk/agents/graph.py`, `finrisk/tools/finance_tools.py`
> *Test:* `test_critic_removes_fabricated_metric`, `test_llm_cannot_change_weights`

> **Proved absence of look-ahead bias empirically rather than by inspection** — built a placebo control running the identical strategy with signals shifted one period forward: honest 6.18% CAGR / 0.33 Sharpe vs 33.22% / 1.85 when cheating, a +27pp gap confirming the information barrier is load-bearing. 8-check automated audit, 0 failures.
>
> *Code:* `finrisk/evaluation/leakage.py`
> *Numbers:* `evaluation/results/leakage_audit.json`

> **Implemented a Gaussian HMM regime detector from scratch (Baum–Welch EM)** with a tested causality guarantee separating filtered (forward-only) from smoothed inference; optimised EM from 156s to 2.16s per fit. Detects market stress at 77.8% recall against known ground truth.
>
> *Code:* `finrisk/regimes/hmm.py`, `finrisk/regimes/detector.py`
> *Test:* `test_filtering_is_causal`, `test_smoothing_is_not_causal`, `test_em_log_likelihood_is_monotonically_non_decreasing`

> **Built a walk-forward backtester with 1-day execution lag, transaction costs and modelled weight drift**, and used it for a 7-rung ablation isolating each component's contribution: factor selection contributed +3.80pp CAGR while portfolio optimisation improved max drawdown from −30.3% to −23.7%.
>
> *Code:* `finrisk/backtesting/engine.py`, `finrisk/backtesting/strategies.py`
> *Numbers:* `evaluation/results/ablation.csv`

> **Diagnosed and documented a negative result**: momentum factors failed in the target dataset. Established via latent-state ground truth that the implementation was correct (positive sign, ρ=+0.03 with the true trend state) but that trailing return is a ~96%-noise proxy for a drift process whose 46-session half-life is far shorter than a 231-session lookback. Value factors recovered their latent state at ρ=0.78 using identical machinery.
>
> *Code:* `tests/test_factor_recovery.py`
> *Doc:* `docs/RESEARCH_REPORT.md` Finding 2

> **Implemented six portfolio optimisers including CVaR-constrained optimisation via the Rockafellar–Uryasev linear program**, with an independent post-hoc constraint audit that does not trust solver status. 0 solver fallbacks across the full evaluation.
>
> *Code:* `finrisk/portfolio/optimizer.py`
> *Test:* `test_cvar_constraint_is_binding`, `test_risk_parity_equalises_risk_contributions`

> **Built an Euler risk-decomposition engine** attributing portfolio volatility exactly to positions and sectors, plus historical/parametric/Cornish–Fisher VaR, CVaR with tail attribution, and factor risk decomposition (R²=0.70, 30% residual).
>
> *Code:* `finrisk/risk/metrics.py`, `finrisk/risk/attribution.py`
> *Test:* `test_euler_decomposition_sums_to_volatility`

> **Built factor-neutral research tooling** (sector-neutral, beta-neutral and general OLS residualisation) and used it to show that value's premium survives neutralisation (Sharpe 0.98 → 1.03) while more than half of quality's does not (0.58 → 0.23) — i.e. quality's apparent edge was largely unintended sector exposure. Neutrality is verified by measuring realised exposures (sector 0.2401 → 0.0000, beta 0.0785 → 0.0000), not asserted.
>
> *Code:* `finrisk/factors/neutralization.py`
> *Test:* `test_capping_preserves_neutrality`, `test_sector_only_neutral_leaves_beta_exposure`
> *Numbers:* `evaluation/results/neutralization.csv`

> **Built a hybrid retrieval system** (BM25 + dense embeddings fused by Reciprocal Rank Fusion, with a vector store and section/page/filing-date provenance) and an IR evaluation harness with generator-derived gold labels. Measured honestly: hybrid ties BM25 on the synthetic corpus (MRR 0.994 vs 0.997) and wins on paraphrase-rich text (top-1 50% vs 33%).
>
> *Code:* `finrisk/rag/{embeddings,vector_store,retriever}.py`, `finrisk/evaluation/rag_eval.py`
> *Test:* `test_dense_finds_paraphrase_that_bm25_misses`, `test_ndcg_never_exceeds_one`

> **Added turnover and Euler risk-budget constraints to the optimiser**, both verified to bind (turnover 0.689 → 0.300; max single-name risk share 0.495 → 0.150) with infeasible limits surfaced as violations rather than silently breached.
>
> *Code:* `finrisk/portfolio/optimizer.py`
> *Test:* `test_turnover_constraint_binds`, `test_risk_budget_constraint_binds`

> **Wrote 368 tests covering numerical correctness against closed-form values, point-in-time data discipline, and adversarial validation of the agent layer.** Full suite runs in 3m35s; `ruff`, `black` and `mypy` pass clean across 55 source files.

## Claims I am deliberately NOT making

These would be unsupported, and I would rather state that than let a reader infer them:

| Not claimed | Why |
|---|---|
| "Achieved 6.81% CAGR" without qualification | Synthetic data. Not historical performance. |
| "Outperformed the benchmark" | Not statistically distinguishable at this sample size (diff +0.35 Sharpe, SE ≈0.28). |
| "Beat the market" / "generated alpha" | No real market was involved. |
| "Production trading system" | Never traded capital. Research and decision support only. |
| "Deployed at scale" | Runs locally; no production deployment. |
| "Eliminated look-ahead bias" | Eliminated for the channels tested. Survivorship bias remains for real-data runs. |
| "Robust strategy" | Sharpe 0.62 → 0.11 across chronological halves; factor-weight sensitivity spans 0.07–0.70. |
| "Hybrid RAG improved retrieval" | It did not, on this corpus (MRR 0.994 vs BM25's 0.997). It wins only on paraphrase-rich text. |
| "Built a semantic search engine" | The dense encoder is corpus-fitted LSA, not a pretrained transformer — model weights were unreachable. |
| "Quality factor generates alpha" | Its premium is largely sector exposure; neutralised Sharpe is 0.23. |

## Interview-safe framing

If asked "did it make money?", the accurate answer is:

> "It made money on synthetic data, which mostly tells you the pipeline works end to end. The results I'd actually defend are the ones that don't depend on returns — the placebo control showing look-ahead is worth 27 points of CAGR, and the ground-truth test showing value factors recover their latent state at 0.78 while momentum manages 0.03. The Sharpe difference versus benchmark isn't statistically significant, and performance halves across the sample, both of which are in the limitations doc."
