# Backtesting Methodology

## The core discipline: decide at close `d`, execute at close `d+1`

```
day d        signals computed from data available at close d
             optimiser produces target weights
             ─────────────────────────────────────────────
day d+1      trade at close, pay costs on traded notional
             ─────────────────────────────────────────────
d+2 .. d+n   weights DRIFT with prices — no free rebalancing
             ─────────────────────────────────────────────
next date    repeat
```

`BacktestConfig` validates `execution_lag_days >= 1`, so a zero-lag configuration **cannot be expressed**. Measured across 156 rebalances: lag exactly 1 session, minimum and maximum both 1.

A runtime assertion re-checks every signal's `as_of` against the decision date on every rebalance. `test_backtester_raises_on_injected_future_signal` tampers with a context to prove the assertion actually fires rather than being decorative.

## Weight drift is modelled

Between rebalances, positions move with prices:

$$w_i^{(t+1)} = \frac{w_i^{(t)}(1 + r_i^{(t+1)})}{\sum_j w_j^{(t)}(1 + r_j^{(t+1)})}$$

This matters more than it looks. Holding weights constant between rebalances implicitly assumes free, continuous rebalancing — which both understates turnover and silently adds a rebalancing premium the strategy never earned. Cost is charged on notional traded *against the drifted portfolio*, so positions that drift toward their target need less trading.

Evidence it is real: `A_equal_weight_universe` shows 3.0% average turnover. A non-drifting equal-weight portfolio would show ~0%; the 3.0% is drift being traded back.

## Walk-forward, expanding window

Everything is estimated using only data available at the decision date:

| Component | Estimation scope |
|---|---|
| Factor z-scores | That date's cross-section only |
| Covariance | Trailing 504 sessions |
| IC for alpha scaling | Training window only, clipped [0, 0.15] |
| HMM parameters | Expanding window, refit every 252 sessions |
| HMM state | `filter_states()` — forward pass only |

The HMM refit cadence is a deliberate compromise: refitting every rebalance is more responsive but ~156× the cost and adds parameter churn that shows up as turnover. Feature standardisation uses training-window statistics only.

## Cost model

$$\text{cost} = \text{notional} \times (\text{commission} + \text{spread}/2 + \text{slippage})$$

Default 10 bps round trip (5 commission + 5 spread). Turnover is reported **one-way**: $\tfrac12\sum_i|w_i^{new} - w_i^{old}|$, so full portfolio replacement is 100%. `_traded_notional` is twice that.

Cost drag is material — 6.6% to 8.3% cumulative across the factor strategies — so the robustness sweep varies costs across 0/5/10/20/40 bps.

## The strategy ladder

Each rung adds exactly one component, so the ablation isolates contributions.

| Strategy | Adds |
|---|---|
| `benchmark_buy_hold` | Cap-weighted benchmark, zero turnover |
| `A_equal_weight_universe` | Rebalancing only, no signal |
| `B_factor_equal_weight` | Factor selection |
| `C_factor_regime_equal_weight` | Regime overlay |
| `D_factor_mean_variance` | Mean-variance optimisation |
| `D2_factor_risk_parity` | Risk parity (alternative to D) |
| `E_factor_optimized_risk_constrained` | Volatility constraint |
| `F_finrisk_full` | Full pipeline |

**A note on D and D2.** Both are forced fully invested (`cash_allowed=False`). Without that, mean-variance correctly solved its stated objective by collapsing to ~all cash (0.9% volatility) — technically right, useless as an ablation rung, because it compares a cash portfolio against equity portfolios. Forcing full investment keeps the comparison about *allocation* rather than about *market exposure*. D2 is an alternative to D, not a further step, so its delta is measured against D.

## Shared research context

All strategies in a comparison share one precomputed `ResearchContext` (factor panels, regime path, IC estimates). Two benefits:

1. **Correctness** — a difference between two rows is a difference between the strategies, not between two independently-computed signal sets.
2. **Cost** — context construction dominates runtime; the robustness study reuses contexts wherever the factor and regime config are unchanged, cutting ~20 rebuilds to 10.

## Reproducibility

Every result carries a `config_digest` — a SHA-256 over the settings that produced it. `test_config_digest_is_stable_and_discriminating` verifies identical configs digest identically and that changing `max_weight` changes the digest.

`test_results_are_reproducible` asserts two runs produce byte-identical equity curves.

## Metrics reported

CAGR, annualised return, annualised volatility, Sharpe, Sortino, max drawdown, Calmar, VaR95, CVaR95, hit rate, average turnover, cumulative cost drag, beta, information ratio, and rebalance count.

CAGR and annualised return are both shown because they differ by volatility drag; reporting only the arithmetic figure flatters a strategy.

## What this methodology does not fix

- Sample-size limits on significance (see `LIMITATIONS.md` §3)
- Survivorship bias on real data (§2)
- Market impact (§5)
- Multiple-testing across strategies tried during development — the ablation ladder was designed up front rather than selected after the fact, but this is not a formal correction
