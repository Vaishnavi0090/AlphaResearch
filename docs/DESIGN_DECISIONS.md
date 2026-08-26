# Design Decisions

Each entry: the decision, the alternatives, why this one, and what it costs.

---

### 1. LLM never produces numbers that reach a decision

**Alternatives:** let the LLM propose weights; let it propose and then validate; keep it purely descriptive.

**Chosen:** purely descriptive, enforced structurally — tools are the only interface, the report is assembled from state rather than prose, and a critic scrubs ungrounded figures.

**Why:** an LLM-produced weight has no error bound, no reproducibility guarantee and no audit trail. In a domain that allocates capital that is disqualifying. Prompting ("please only use provided numbers") is not enforcement.

**Cost:** the agent layer is less impressive as an autonomy demo. It cannot surprise you with a novel allocation, because it is not allowed to invent one.

---

### 2. Synthetic data as the default provider

**Alternatives:** ship without data; fabricate plausible-looking results; use synthetic data with known ground truth.

**Chosen:** synthetic DGP, with real providers implemented but unusable in the build environment (all financial hosts returned HTTP 403).

**Why:** fabricating results was never an option. Beyond necessity, ground truth buys something real data cannot — I can test whether a factor *measures what it claims* (book-to-price recovers the latent value state at ρ=0.78) rather than only whether it made money.

**Cost:** no result here supports a claim about markets. Every figure is labelled synthetic-data validation, and `LIMITATIONS.md` opens with it. There is also a self-confirmation risk — I wrote both the DGP and the factors — which the momentum failure partly offsets.

---

### 3. Fixed DAG instead of free-form agent routing

**Alternatives:** supervisor picks the next agent each step; fully autonomous ReAct loop; fixed DAG with optional branches.

**Chosen:** fixed DAG.

**Why:** quantitative dependencies are real dependencies. Risk contributions need weights; weights need factor scores and a covariance estimate. Letting a model reorder that adds failure modes and buys nothing. The genuine discretion — which optional branches run — is preserved.

**Cost:** less "agentic" in the buzzword sense. I consider this correct rather than a limitation, but it is a deliberate departure from how these systems are usually demoed.

---

### 4. Hand-written Gaussian HMM instead of `hmmlearn`

**Alternatives:** `hmmlearn`; a rolling volatility heuristic; hand-written EM.

**Chosen:** hand-written.

**Why:** the causality guarantee *is* the point. I needed to control exactly which recursion runs and to expose `filter_states` (forward-only) separately from `smoothed_states` (future-using), so the distinction could be tested rather than trusted.

**Cost:** ~300 lines to maintain and debug. Performance work was required — see decision 12.

---

### 5. scipy rather than cvxpy for optimisation

**Alternatives:** cvxpy (elegant DSL, proves convexity); scipy SLSQP/HiGHS; hand-rolled gradient methods.

**Chosen:** scipy.

**Why:** already a dependency, handles every problem here including the CVaR LP, and keeps the install reproducible without a heavyweight optional package.

**Cost:** I write constraint Jacobians myself, and the solver does not certify convexity. Mitigated by `check_constraints`, an independent post-hoc audit that does not trust solver status — measured fallback count is 0 across all six methods.

---

### 6. BM25 rather than embeddings for retrieval

**Alternatives:** dense embeddings; hybrid; BM25.

**Chosen:** BM25.

**Why:** exact financial terminology matters ("goodwill impairment" ≠ "asset impairment"), numbers must match exactly, no model drift, no inference dependency.

**Cost:** genuine paraphrase is missed. A hybrid retriever is the acknowledged upgrade path (`LIMITATIONS.md` §9).

---

### 7. Missing factors become neutral, not dropped

**Alternatives:** drop the asset; impute from sector median; assign neutral 0 with coverage tracking.

**Chosen:** neutral with coverage tracking.

**Why:** dropping introduces selection bias correlated with data availability — small and newly-listed companies have worse coverage, so dropping systematically tilts the universe. Sector-median imputation invents information.

**Cost:** an asset with thin coverage gets a composite score dominated by whichever factors it does have. `coverage` is reported so this is visible.

---

### 8. Winsorize then z-score, strictly within each date

**Alternatives:** z-score then clip; pooled normalisation across history; per-date winsorize-then-z-score.

**Chosen:** the last.

**Why:** order matters (z-scoring first bakes the outlier into the mean and sd). Scope matters more — pooled normalisation leaks the future distribution into every past date. The leakage audit verifies per-date `|mean| ≤ 1.63e-15`.

**Cost:** none identified.

---

### 9. Execution lag is structurally impossible to disable

**Alternatives:** configurable with 0 default; configurable with 1 default; validated minimum of 1.

**Chosen:** `BacktestConfig` rejects `execution_lag_days=0` at construction.

**Why:** same-day execution is the single most common way a backtest becomes fiction. A default is a suggestion; a validator is a guarantee.

**Cost:** a legitimate use case (open-price execution on the signal day) cannot be expressed. Acceptable — that would need an intraday model anyway.

---

### 10. D and D2 forced fully invested

**Alternatives:** let mean-variance choose cash; force full investment.

**Chosen:** force it (`cash_allowed=False`), with the rationale in code.

**Why:** unconstrained, mean-variance correctly solved its stated objective by collapsing to ~all cash (0.9% volatility). Technically right, useless as an ablation rung — it compares a cash portfolio against equity portfolios, so the "improvement" measured would be market exposure, not allocation skill.

**Cost:** D is not the unconstrained optimum. Documented at the rung.

---

### 11. Critic scrubs rather than rejects

**Alternatives:** reject the whole report on any ungrounded claim; scrub and log; warn only.

**Chosen:** scrub and log.

**Why:** a report with fabrications removed plus a record of what was removed is more useful than no report, and the failure stays visible rather than being silently tolerated.

**Cost:** a caller ignoring `validation.findings` sees a clean report and misses that something was stripped. The findings are surfaced in the API response, the dashboard and the CLI output.

---

### 12. Performance work on the HMM

**Problem:** initial EM fit took 156 seconds, making the 156-rebalance backtest unusable.

**Fixes:** vectorised the ξ accumulation over the time axis; replaced `scipy.special.logsumexp` with a hand-rolled version (the recursions are sequential, so it is called O(T) times per iteration on tiny arrays where scipy's dispatch overhead dominates the arithmetic); tightened the forward/backward loops.

**Result:** 156s → 2.16s per fit (44 iterations). Full backtest context builds in ~27s.

**Cost:** hand-rolled `logsumexp` needs its own numerical-stability tests — `test_logsumexp_is_stable_for_large_values` covers the overflow case.

---

### 13. Shared research context across compared strategies

**Alternatives:** recompute signals per strategy; share one precomputed context.

**Chosen:** share.

**Why:** correctness first — a difference between two rows must be a difference between the strategies, not between two independently-computed signal sets. Speed is the secondary benefit (robustness sweeps went from ~20 context builds to 10).

**Cost:** a strategy needing genuinely different signals cannot share a context. The API allows passing a distinct one.

---

### 14. Factor weights left at a published prior, not tuned

**Alternatives:** tune on the data; tune on a holdout; leave at the prior.

**Chosen:** leave at 30/25/25/20.

**Why:** the robustness sweep shows a value tilt would produce Sharpe 0.70 versus the baseline's 0.40. Tuning to capture that would mean fitting weights on the same data used to evaluate them — the exact overfitting this project is built to avoid.

**Cost:** the headline configuration is knowingly not the best available one, and the reported 6.81% CAGR sits well below the 11.39% a value tilt would have given. Stated explicitly in `EVALUATION.md` and `LIMITATIONS.md` rather than quietly improved.
