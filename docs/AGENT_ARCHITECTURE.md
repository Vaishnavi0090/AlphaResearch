# Agent Architecture

## The one rule

**The LLM never produces a number that reaches a decision.**

Weights come from a convex optimiser. Risk comes from closed-form estimators. Regimes come from an HMM. The LLM reads structured tool output and explains it — and a critic checks every figure it quotes before the report is emitted.

This is enforced by three mechanisms, not by prompting:

1. **Tools are the only interface.** Agents call Pydantic-validated tools; they cannot compute. Asking for a CVaR figure means calling `calculate_cvar`, and the number returned is the same number the backtester would compute.
2. **The report is assembled from state, not from prose.** `build_report()` reads `state["portfolio"]["weights"]` — the optimiser's output — never the narrative.
3. **The critic scrubs ungrounded numbers.** Any figure not matching a tool result is removed and logged.

`test_llm_cannot_change_weights` asks the system to "recommend 90% in one name" and asserts the resulting weights equal the optimiser's output exactly and respect the position cap.

## The graph

```
START
  │
  ▼
factor_agent ──────► calculate_factor_scores
  │
  ▼
market_agent ──────► get_market_data, detect_market_regime
  │
  ▼
fundamental_agent ─► get_fundamentals  (narrowed to the factor candidates)
  │
  ▼
portfolio_agent ───► optimize_portfolio      ◄── the ONLY producer of weights
  │
  ▼
risk_agent ────────► calculate_risk, calculate_cvar,
  │                  calculate_risk_contribution, stress_test
  │
  ├─(optional)─────► backtest_agent → run_backtest
  ▼
research_agent ────► retrieve_financial_documents + LLM synthesis
  │
  ▼
explain_agent ─────► per-position explanations (structured data only)
  │
  ├─(optional)─────► committee_agent → bull / bear / risk / quant → debate
  ▼
critic_agent ──────► validate everything above
  │
  ▼
END
```

**Why a fixed DAG rather than free-form routing.** Quantitative dependencies are real dependencies: risk contributions require weights; weights require factor scores and a covariance estimate. Letting a model choose that order adds failure modes and buys nothing. The supervisor's genuine discretion is over which *optional* branches run (documents, committee, backtest).

Factors run first so the market and fundamental agents can narrow their queries to the candidate set instead of pulling the whole universe.

## Why LangGraph

Stateful workflow, conditional branches, per-node error isolation, and one serialisable state object that can be traced. `run_sequential()` is a topologically identical fallback, so the system stays usable and testable without LangGraph — verified by `test_sequential_runner_matches_graph_nodes`.

## The investment committee

Four roles see identical facts and argue distinct briefs:

| Role | Brief |
|---|---|
| Bull | Strongest case **for**, using only supplied numbers |
| Bear | Strongest case **against** |
| Risk | Limit compliance, concentration, tail risk only |
| Quant | Methodology, reproducibility, estimation robustness |

The debate node does **not** pick the most favourable argument. It reports agreements, disagreements, and unresolved uncertainty, then maps stances to an action:

- Risk role not supportive → `reject`
- Bear opposes or concerns raised → `proceed_with_limits`
- Otherwise → `proceed`

A bear objection becomes an **explicit condition** rather than being discarded. On the full run: bull=support, bear=oppose, risk=support, quant=conditional → `proceed_with_limits`.

`test_committee_does_not_simply_follow_the_bull` asserts the action is never plain `proceed` when the bear opposes, and that unresolved uncertainty is populated.

## The critic

Four checks on the only path from prose back to the report:

1. **Numeric grounding** — every value in `quantitative_evidence` must match a tool-computed number within 2% relative tolerance. Unmatched values are removed and logged as errors.
2. **Citation validity** — every citation must resolve to a chunk that exists in the index. A plausible "10-K, page 42" backed by nothing is a fabrication.
3. **Prose grounding** — percentages in the narrative are checked against the grounded set, catching the common failure where structured fields are clean but the paragraph invents "outperformed by 340 bps".
4. **Constraint and leakage integrity** — weights re-checked independently against configured limits. The critic deliberately does **not** trust `pw.constraints`; the optimiser reporting its own compliance is worth little without a second opinion.

Findings are `error` for fabrication or limit breach, `warning` for weaker issues (one-sided evidence, overconfidence above 0.8).

**Scrub rather than reject.** A report with fabrications removed plus a log of what was removed is more useful than no report, and the removals stay visible.

Tested by injection, not by inspection: `test_critic_removes_fabricated_metric` plants `secret_alpha: 0.42` alongside a real CVaR figure and asserts only the fabrication is stripped. Similar tests cover invented citations, ungrounded prose percentages, weight-cap breaches, negative weights, leverage, and look-ahead.

## The LLM backends

**`StubLLM`** — a deterministic template narrator, **not** a language model. Honestly named. It composes prose from the structured facts handed to it, so the graph runs in CI with no API key, no network and no non-determinism, and tests can assert exact strings.

**`AnthropicLLM`** — the real backend (`claude-sonnet-4-6`), used when `ANTHROPIC_API_KEY` is set and `agents.llm_provider="anthropic"`. Tracks token usage.

Both satisfy the same protocol, and neither can emit a weight. Facts are injected into prompts inside `<facts:name>...</facts:name>` blocks so the critic can diff prose against exactly what the model was shown.

## Failure handling

Node failures are recorded in `state["errors"]` and surfaced as critic findings rather than crashing the run. `test_graph_survives_a_failing_node` runs the graph with no document index and confirms a portfolio and research conclusion are still produced, with `sources == []`.

Observed on the full run: 9 trace steps, 16 tool calls, 0 failures, critic PASS with 0 findings.
