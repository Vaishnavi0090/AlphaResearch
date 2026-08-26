# FINRISK AI

**Agentic quantitative portfolio & risk intelligence platform.**

A factor research, portfolio construction, risk management and agentic decision-support system built around one rule: **the language model never produces a number that reaches a decision.**

Weights come from a convex optimiser. Risk comes from closed-form estimators. Regimes come from a Hidden Markov Model. The LLM reads that output and explains it — and a validation agent strips any figure it can't trace back to a deterministic tool.

> ⚠️ **All results are synthetic-data validation, not historical performance.**
> Financial data providers were unreachable in the build environment, so the default dataset is a seeded data-generating process with known ground truth. Real providers are implemented. See [LIMITATIONS.md](docs/LIMITATIONS.md).
>
> Research and decision-support software. Not investment advice. Has never traded capital.

---

## Quick start

```bash
pip install -e ".[all]"          # or: pip install -r requirements.txt

python scripts/run_all.py --quick   # ~2 min  — every stage, smoke scale
python scripts/run_all.py           # ~6 min  — suite + ablation + audit
python scripts/run_all.py --full    # ~15 min — adds factor study

python scripts/run_robustness.py cost        # sweeps run individually
python scripts/run_robustness.py --render

ruff check . && black --check . && mypy finrisk/

pytest                                        # 368 tests, ~3m35s
uvicorn finrisk.api.app:app --reload          # API at :8000/docs
streamlit run finrisk/application/dashboard.py
```

No API key needed. Without `ANTHROPIC_API_KEY` the agent layer uses `StubLLM`, a deterministic template narrator — honestly named, not a simulated model.

---

## Headline results

Synthetic dataset · seed 7 · 40 assets · 2010–2024 · 156 monthly rebalances · 10 bps costs · 1-day execution lag

### Data leakage: proven empirically, not asserted

The same strategy, run honestly and then with signals shifted one period into the future:

| | CAGR | Sharpe |
|---|---|---|
| Honest | 6.18% | 0.33 |
| One-period look-ahead | **33.22%** | **1.85** |
| Advantage from cheating | **+27.04 pp** | **+1.53** |

If the honest run scored near the cheating run, it would already be leaking. It doesn't. **8 audit checks, 0 failures.**

### Ablation: factor selection does the work

| Rung | CAGR | Sharpe | Max DD |
|---|---|---|---|
| A: equal-weight universe | 2.39% | 0.08 | −52.0% |
| B: + factor selection | 6.18% (**+3.80**) | 0.33 | −30.3% |
| C: + regime overlay | 6.06% | 0.35 | −26.1% |
| D: + mean-variance | 6.36% | 0.36 | −23.5% |
| E: + volatility constraint | 6.69% | 0.35 | −26.2% |
| **F: full pipeline** | **6.81%** | **0.40** | **−23.7%** |

Optimisation contributes mostly through **drawdown** (−30.3% → −23.7%), not return.

### Factor neutralisation: is the premium real?

A raw factor portfolio is rarely a pure bet on the factor. Removing sector and
beta exposure asks whether what remains is stock selection:

| Factor | Raw Sharpe | Sector + beta-neutral | Verdict |
|---|---|---|---|
| Value | 0.98 | **1.03** | Survives — genuine |
| Quality | 0.58 | **0.23** | **Mostly a sector bet** |
| Momentum | −0.01 | −0.04 | Nothing to remove |

Realised sector exposure 0.2401 → **0.0000**, beta 0.0785 → **0.0000** — the
neutrality is measured, not asserted.

### And the parts that don't work

Reported as prominently as the parts that do:

- **Not statistically significant.** Sharpe +0.35 vs benchmark; standard error ≈0.28. Not distinguishable.
- **Not stable in time.** Sharpe **0.62** in the first half of the sample, **0.11** in the second.
- **Fragile to factor weights.** Sharpe ranges 0.07 (momentum tilt) to 0.70 (value tilt).
- **Momentum fails entirely.** IC −0.072 at 252 days — diagnosed, not hidden.
- **Quality is mostly sector exposure.** Sharpe 0.58 → 0.23 when neutralised.
- **Hybrid RAG doesn't beat BM25 here.** MRR 0.994 vs 0.997 — the template
  corpus has no semantics to exploit. Reported, not buried.

---

## The momentum negative result

The most interesting finding here is a failure, established against latent ground truth rather than guessed at:

| Measurement | Value | Meaning |
|---|---|---|
| True trend state → forward returns | IC **+0.041** | The premium genuinely exists |
| `mom_1m` → true trend state | ρ **+0.032** | Correct sign, tiny magnitude |
| `mom_12m_1m` → true trend state | ρ **+0.006** | Essentially uncorrelated |
| `book_to_price` → true value state | ρ **+0.780** | Same machinery, works fine |
| `roe` → true quality state | ρ **+0.850** | Same machinery, works fine |

**Diagnosis:** over 21 sessions the injected alpha contributes ~0.25% of return against ~6.4% idiosyncratic noise — trailing return is ~96% noise as a drift estimator. And the trend's 46-session half-life is far shorter than a 231-session 12-1 window, so that window measures already-reverted states.

Classical 12-1 momentum is **mis-specified for this DGP**. Real markets do exhibit it — an instructive divergence. Locked in by a test so a future change can't silently "fix" a result that is supposed to be negative.

---

## Architecture

```
Interfaces          FastAPI · Streamlit · CLI
       │
Agent layer         LangGraph supervisor · committee · critic
       │            (typed tool calls only)
Tool layer          13 Pydantic-validated deterministic tools
       │
Core       Factors · Regimes · Portfolio · Risk · RAG
       │
Backtesting         walk-forward · costs · execution lag
       │
Data                providers · point-in-time · validation
```

| Capability | Implementation |
|---|---|
| Factors | 20 definitions, 5 groups, per-date winsorised z-scores, IC/decay/quantile study |
| Neutralisation | Sector-neutral, beta-neutral, general OLS residualisation; neutral L/S books |
| Regimes | Gaussian HMM from scratch (Baum–Welch EM), filtered-only inference, BIC selection |
| Alpha | Grinold–Kahn `α = IC × σ × z`, IC from training window only, clipped |
| Portfolio | 6 optimisers incl. CVaR-constrained LP; turnover and Euler risk-budget caps |
| Risk | Euler decomposition, 3 VaR estimators, CVaR + tail attribution, stress suite |
| Backtesting | Walk-forward, 1-day lag, drift modelling, cost on traded notional |
| RAG | Hybrid BM25 + dense (LSA/ST) via RRF, vector store, section/page/date provenance |
| Agents | Fixed DAG, pre-registered hypotheses with falsifiers, adversarial committee, critic |
| Explainability | Per-position alpha, risk, diversification, tail share, derived confidence |
| Filings | 10-K/10-Q/8-K parsing with Item sections and filing-date discipline |

---

## What makes this defensible

**The LLM/quant barrier is structural.** Tools are the only interface; the report is assembled from state, never prose; the critic scrubs ungrounded figures. Tested by *injecting* fabrications — `test_critic_removes_fabricated_metric` plants a fake metric next to a real one and asserts only the fake is stripped.

**Look-ahead is tested, not assumed.** A placebo control, plus checks on PIT fundamentals (2,400 records, 0 violations), factor truncation invariance (max diff 0.000e+00), per-date normalisation (|mean| ≤ 1.6e-15), and execution lag (exactly 1 session × 156 rebalances).

**Ground truth makes factor claims falsifiable.** Because the DGP exposes latent states, tests verify factors *measure what they claim* rather than merely making money.

**Negative results are documented, not buried.** Momentum's failure, the time-stability collapse, the factor-weight fragility, and the volatility-target overshoot all appear in the docs — and several are asserted in tests.

**368 tests against closed-form values.** Portfolio variance against a hand-expanded quadratic form, parametric VaR against `norm.ppf`, drawdown against a hand-computed path.

---

## Documentation

| Doc | Contents |
|---|---|
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Layers, data flow, extension points |
| [DESIGN_DECISIONS](docs/DESIGN_DECISIONS.md) | 14 decisions with alternatives and costs |
| [QUANTITATIVE_METHODS](docs/QUANTITATIVE_METHODS.md) | Every formula and why it was chosen |
| [DATA_PIPELINE](docs/DATA_PIPELINE.md) | Providers, PIT discipline, cleaning |
| [DATA_LEAKAGE_AUDIT](docs/DATA_LEAKAGE_AUDIT.md) | The placebo control and 7 further checks |
| [BACKTESTING](docs/BACKTESTING.md) | Execution lag, costs, drift, strategy ladder |
| [RISK_MODEL](docs/RISK_MODEL.md) | Euler decomposition, tail risk, stress |
| [AGENT_ARCHITECTURE](docs/AGENT_ARCHITECTURE.md) | Graph, committee, critic |
| [RAG](docs/RAG.md) | BM25 rationale, two bugs found |
| [EVALUATION](docs/EVALUATION.md) | Full results incl. all robustness sweeps |
| [RESEARCH_REPORT](docs/RESEARCH_REPORT.md) | Findings, including 3 negative results |
| [LIMITATIONS](docs/LIMITATIONS.md) | Everything that bounds the claims above |
| [INTERVIEW_GUIDE](docs/INTERVIEW_GUIDE.md) | Hard questions, honest answers |
| [RESUME_TRACEABILITY](docs/RESUME_TRACEABILITY.md) | Every claim → code → test → number |

---

## Project layout

```
finrisk/          config data factors regimes alpha portfolio risk
                  backtesting rag tools agents evaluation api application
scripts/          run_all.py  run_robustness.py
tests/            14 modules, 368 tests
docs/             14 documents
evaluation/       results/ (json+csv)  reports/
```

## Security

No secrets in code or config — `Settings` holds no credentials; keys come from the environment. No endpoint accepts code, file paths or shell commands. Ticker inputs are validated against the loaded universe; numeric parameters are bounded by Pydantic. `.env` is gitignored and `.env.example` documents the variables.

## License

MIT
