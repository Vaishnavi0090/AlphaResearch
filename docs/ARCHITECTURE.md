# Architecture

## Design principle

One rule shapes everything else: **the language model never produces a number that reaches a decision.**

Weights come from a convex optimiser. Risk comes from closed-form estimators. Regimes come from a Hidden Markov Model. The LLM reads those outputs and explains them, and a critic checks every figure it quotes against the tool results before the report is emitted.

This is not a stylistic preference. An LLM asked for a portfolio weight will produce a plausible number with no error bound, no reproducibility guarantee, and no audit trail. That is unacceptable in a domain where the output allocates capital. So the architecture makes it structurally impossible rather than discouraged.

## Layers

```
                        ┌──────────────────────────────┐
                        │  Interfaces                  │
                        │  FastAPI · Streamlit · CLI   │
                        └───────────────┬──────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │  Agent layer (LangGraph)     │
                        │  supervisor · committee      │
                        │  critic · explainer          │
                        └───────────────┬──────────────┘
                                        │ typed tool calls only
                        ┌───────────────▼──────────────┐
                        │  Tool layer (13 tools)       │
                        │  Pydantic in / Pydantic out  │
                        └───────────────┬──────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        │               │               │               │               │
   ┌────▼────┐   ┌──────▼─────┐  ┌──────▼─────┐  ┌──────▼─────┐  ┌─────▼─────┐
   │ Factors │   │  Regimes   │  │ Portfolio  │  │    Risk    │  │    RAG    │
   │ 20 defs │   │  HMM/EM    │  │ 6 methods  │  │ VaR/CVaR   │  │  BM25     │
   │ IC study│   │  filtered  │  │ scipy      │  │ Euler      │  │ citations │
   └────┬────┘   └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └─────┬─────┘
        └───────────────┴───────────────┼───────────────┴───────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │  Backtesting                 │
                        │  walk-forward · costs · lag  │
                        └───────────────┬──────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │  Data layer                  │
                        │  providers · PIT · validation│
                        └──────────────────────────────┘
```

Dependencies point downward only. The risk engine does not know an agent exists. The optimiser does not know what a factor is — it receives expected returns and a covariance matrix. This is what makes each layer testable in isolation, and it is why the 250-test suite can assert numerical properties without constructing an agent.

## Module map

| Module | Responsibility | Key entry points |
|---|---|---|
| `finrisk/config` | Typed settings, validation, config digest | `Settings`, `default_settings()` |
| `finrisk/data` | Providers, point-in-time discipline, quality checks, SEC filings | `load_bundle`, `fundamentals_asof`, `parse_filing` |
| `finrisk/factors` | 20 factor definitions, cross-sectional engine, IC research, neutralisation | `FactorEngine.compute`, `study_factor`, `build_neutral_portfolio` |
| `finrisk/regimes` | Gaussian HMM from scratch, causal detector | `GaussianHMM.filter_states`, `RegimeDetector.detect` |
| `finrisk/alpha` | Grinold–Kahn signal scaling | `AlphaEngine.generate` |
| `finrisk/portfolio` | Six optimisers + constraint audit | `PortfolioOptimizer.optimize` |
| `finrisk/risk` | Metrics, attribution, stress, per-position explainability | `RiskEngine.analyze_portfolio`, `compute_position_attribution` |
| `finrisk/backtesting` | Walk-forward engine, strategy ladder | `WalkForwardBacktester.run` |
| `finrisk/rag` | Chunking, hybrid BM25+dense retrieval, vector store, citations | `DocumentRetriever.retrieve`, `build_encoder` |
| `finrisk/tools` | The agent/quant boundary | `ToolRegistry.call` |
| `finrisk/agents` | Graph, committee, critic, LLM adapters | `run_research`, `Critic` |
| `finrisk/evaluation` | Benchmark, ablation, robustness, leakage audit, retrieval IR metrics | `run_ablation`, `run_full_audit`, `run_retrieval_ablation` |
| `finrisk/api` | FastAPI + observability | `app`, `Telemetry` |
| `finrisk/application` | Streamlit viewer | `dashboard.py` |

## Data flow for one rebalance

```
decision date d
   │
   ├─ prices[:d] ─────────────► factor panel (20 factors, winsorised, z-scored per date)
   ├─ fundamentals_asof(d) ──►  (only filings with available_from <= d)
   │                                  │
   │                                  ▼
   │                            composite score → top-N selection
   │                                  │
   ├─ benchmark[:d] ──► HMM filter_states → regime + risk multiplier
   │                                  │
   │                                  ▼
   │                    alpha = IC × σ × z   (IC from training window only)
   │                                  │
   ├─ returns[:d] ──► covariance (Ledoit–Wolf shrinkage)
   │                                  │
   │                                  ▼
   │                          convex optimiser → weights
   │                                  │
   │                                  ▼
   │                          constraint audit (independent re-check)
   │
   ▼
execution date d+1  ── trade at next close, pay costs on traded notional
   │
   ▼
drift until next rebalance (weights move with prices; no free rebalancing)
```

The single most important arrow is the last one but two: **decision at `d`, execution at `d+1`**. `BacktestConfig` rejects `execution_lag_days=0` at construction, so a zero-lag configuration cannot be expressed.

## Why LangGraph, and why a fixed DAG

LangGraph is used because the workflow is stateful, has conditional branches (documents / committee / backtest are optional), needs per-node error isolation, and benefits from one serialisable state object that can be traced.

It is a **fixed DAG rather than free-form routing**, which is a deliberate departure from typical "agents choose their own path" designs. Quantitative dependencies are real dependencies: you cannot compute risk contributions before you have weights, and you cannot have weights before you have factor scores and a covariance estimate. Letting a model choose that order adds failure modes and buys nothing. The supervisor's real discretion is over *which optional branches to run*, not over the order of the mandatory ones.

`run_sequential()` is a topologically identical fallback, so the system remains usable and testable if LangGraph is absent.

## The critic as an architectural component

The critic is not a post-hoc filter bolted on for safety theatre. It sits on the only path from prose back to the report and performs four checks:

1. every number in `quantitative_evidence` must match a value the tools actually computed (2% relative tolerance);
2. every citation must resolve to a chunk that exists in the index;
3. percentages appearing in the narrative must be traceable to tool output;
4. weights are independently re-validated against configured limits — the critic does **not** trust the optimiser's own compliance report.

Ungrounded content is *stripped and recorded* rather than causing a hard failure, because a report with the fabrications removed plus a log of what was removed is more useful than no report. `tests/test_agents.py` verifies this by injecting fabrications and asserting they do not survive.

## Extension points

- **New data provider** — implement the `DataProvider` protocol (`fetch`) and return a `MarketDataBundle`. `YFinanceProvider`, `FredProvider` and `EdgarProvider` are working examples.
- **New factor** — add a `FactorSpec` to `FACTOR_REGISTRY` with `higher_is_better=True`; negate the raw value if lower is better. Nothing else changes.
- **New dense encoder** — implement the `DenseEncoder` protocol (`fit`, `encode`) returning unit-norm vectors; `build_encoder` selects it. `LSAEncoder` and `SentenceTransformerEncoder` are working examples.
- **New neutralisation axis** — pass any exposure matrix to `neutralize()`; sector dummies and beta are the supplied cases.
- **New optimiser** — add a method to `PortfolioOptimizer` returning `PortfolioWeights`; shared constraint machinery and the audit apply automatically.
- **New strategy rung** — implement the `Strategy` protocol and register in `all_strategies()`; it joins the comparison and ablation tables.
- **Real LLM** — set `ANTHROPIC_API_KEY` and `agents.llm_provider="anthropic"`. `StubLLM` remains the CI path.
