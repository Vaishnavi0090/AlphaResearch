# Quantitative Methods

Every formula the system uses, with the reasoning behind the choice. All of it is tested in `tests/test_risk_metrics.py` against closed-form values rather than against the code's own prior output.

## Returns and aggregation

Simple returns for compounding, log returns for additivity:

$$r_t = \frac{P_t}{P_{t-1}} - 1 \qquad \ell_t = \ln\frac{P_t}{P_{t-1}}$$

Portfolio return uses simple returns because weights aggregate linearly in simple space, not log space. This is the kind of detail that silently corrupts results if got wrong: $w^\top \ell \neq \ln(1 + w^\top r)$.

**CAGR vs annualised return.** Both are reported, and they are not interchangeable:

$$\text{CAGR} = \left(\prod_t (1+r_t)\right)^{252/T} - 1 \qquad \text{AnnRet} = \bar{r} \cdot 252$$

CAGR is geometric (what an investor actually earns); annualised return is arithmetic. Volatility drag makes CAGR strictly lower whenever returns vary — quoting only the arithmetic figure flatters a strategy, so `test_cagr_below_arithmetic_mean_when_volatile` asserts the inequality holds.

## Risk-adjusted performance

**Sharpe ratio**, with the risk-free rate genuinely subtracted:

$$S = \frac{\bar{r} - r_f/252}{\sigma(r)}\sqrt{252}$$

The reported Sharpes look low (0.05–0.40) precisely because the 2–4% risk-free rate is deducted. Omitting it would inflate every figure.

**Sortino ratio** uses downside deviation with a **full-sample denominator**:

$$\sigma_d = \sqrt{\frac{\sum_t \min(r_t, 0)^2}{T-1}}\sqrt{252}$$

The subtlety: dividing by the count of *negative* observations instead of $T-1$ is a common implementation that inflates Sortino for strategies with few but large losses. Using $T-1$ keeps it comparable to Sharpe.

**Maximum drawdown** on the cumulative wealth path:

$$\text{MDD} = \min_t \left(\frac{W_t}{\max_{s \le t} W_s} - 1\right)$$

**Calmar** = CAGR / |MDD|.

## Value at Risk — three estimators, all reported

VaR is reported three ways because the disagreement between them *is* the information.

**Historical** (empirical quantile, no distributional assumption):
$$\text{VaR}_\alpha = Q_{1-\alpha}(\{r_t\})$$

**Parametric Gaussian**:
$$\text{VaR}_\alpha = \mu + z_{1-\alpha}\sigma$$

**Cornish–Fisher**, which corrects the Gaussian quantile for skewness $S$ and excess kurtosis $K$:

$$z_{CF} = z + \frac{S}{6}(z^2-1) + \frac{K}{24}(z^3-3z) - \frac{S^2}{36}(2z^3-5z)$$

Why all three: financial returns have fat tails, so the Gaussian estimate systematically understates loss. When historical VaR is materially worse than parametric, that gap quantifies the non-normality — and the tool reports `gaussian_understates_tail` as an explicit boolean. `test_gaussian_var_understates_fat_tails` verifies this on Student-t data.

**CVaR / Expected Shortfall** — the average loss *given* the tail is breached:

$$\text{CVaR}_\alpha = \mathbb{E}[r \mid r \le \text{VaR}_\alpha]$$

CVaR is preferred to VaR for optimisation for two reasons: it is coherent (subadditive — VaR is not, so VaR can perversely reward concentration), and it is sensitive to tail *shape* rather than only to the threshold.

## Euler risk decomposition

Volatility is homogeneous of degree 1 in weights, so Euler's theorem gives an exact additive decomposition. This is the mathematical fact that makes "which position is responsible for our risk" a well-posed question.

Marginal contribution:
$$\text{MCR}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

Component contribution, which sums **exactly** to total volatility:
$$\text{CCR}_i = w_i \cdot \text{MCR}_i \qquad \sum_i \text{CCR}_i = \sigma_p$$

A weight-based risk report is misleading whenever correlations are non-trivial: a 20% position can easily carry 35% of the risk. The system reports both, and flags positions where risk share materially exceeds weight share.

**One implementation subtlety, found during testing.** The decomposition sums to the volatility implied by the *covariance model*, $\sqrt{w^\top \Sigma w}$, not to the *realised* sample standard deviation of the portfolio return stream. These differ because $\Sigma$ is shrunk. Conflating them makes the Euler identity appear to fail. The system therefore reports `model_volatility` and `ann_volatility` (realised) as separate fields — measured on a sample portfolio: model 0.13739, components sum 0.13739 (exact), realised 0.13863.

## Covariance estimation

Sample covariance is badly conditioned when $T$ is not much larger than $N$, and the optimiser will happily exploit that noise — this is the well-known "error maximisation" failure of naive mean-variance.

**Ledoit–Wolf shrinkage** is the default:
$$\Sigma_{\text{shrunk}} = (1-\delta)S + \delta F$$

with $F$ a structured target and $\delta$ chosen analytically to minimise expected squared error. `test_ledoit_wolf_is_better_conditioned_than_sample` verifies the condition number improves on a deliberately short sample.

**PSD repair** via eigenvalue clipping is applied where needed:
$$\Sigma^+ = V \max(\Lambda, 0) V^\top$$

## Alpha generation — Grinold–Kahn

Factor scores are z-scores, not return forecasts. Converting one to the other requires the forecasting-rule identity:

$$\alpha_i = \text{IC} \times \sigma_i \times z_i$$

where IC is the information coefficient, $\sigma_i$ the asset's forecast volatility, and $z_i$ the standardised score. The structure is what makes it defensible: expected return scales with *both* the signal's demonstrated skill and the asset's volatility, so a weak signal on a volatile asset does not masquerade as a strong forecast.

Three safeguards:

- IC is estimated **only from the training window** available at each decision date, never from the evaluation period.
- IC is clipped to $[0, 0.15]$. An IC above 0.15 is implausible for a public equity factor; without the clip, a noisy early estimate produces absurd return forecasts and a wildly concentrated portfolio.
- The regime multiplier scales alpha *magnitude*, never its *sign*. A regime model should adjust conviction, not reverse a signal.

## Portfolio construction — six methods

| Method | Objective | Purpose |
|---|---|---|
| Equal weight | — | Ablation baseline; hard to beat after costs |
| Mean-variance | $\max w^\top\mu - \tfrac{\lambda}{2}w^\top\Sigma w$ | Classical Markowitz |
| Max Sharpe | $\max (w^\top\mu - r_f)/\sqrt{w^\top\Sigma w}$ | Tangency portfolio |
| Risk parity (ERC) | equalise $\text{CCR}_i$ | Ignores return forecasts entirely |
| Risk-constrained | $\max w^\top\mu$ s.t. $\sqrt{w^\top\Sigma w} \le \sigma^*$ | Volatility targeting |
| CVaR-constrained | $\max w^\top\mu$ s.t. $\text{CVaR}_\alpha \le c$ | Tail-risk budgeting |

Constraints common to all: long-only, per-name cap, sector cap, leverage cap, optional full-investment.

**CVaR via Rockafellar–Uryasev.** CVaR appears non-convex but linearises exactly with auxiliary variables — VaR level $\zeta$ and per-scenario excess $u_s$:

$$\text{CVaR} \approx \zeta + \frac{1}{(1-\alpha)S}\sum_s u_s, \quad u_s \ge -(r_s^\top w) - \zeta, \quad u_s \ge 0$$

This turns the problem into a linear program, solved with scipy's HiGHS. `test_cvar_constraint_is_binding` confirms a tighter limit actually reduces realised scenario CVaR.

**Why scipy rather than cvxpy.** cvxpy is more elegant for expressing convex programs, but scipy is a dependency this project already has, `SLSQP`/`HiGHS` handle every problem here, and avoiding a heavyweight optional dependency keeps the install reproducible. The trade-off is that I write the constraint Jacobians myself — which is why `check_constraints` performs an independent post-hoc audit rather than trusting solver status alone.

**Fallback behaviour is counted, not hidden.** If a solver fails, the optimiser falls back to equal weight, records `fallback_*` in `optimization_status`, and increments `fallback_count`. On the full evaluation the count is **0** across all six methods.

## Regime detection — Gaussian HMM from scratch

Implemented directly (Baum–Welch EM with a diagonal-covariance Gaussian emission) rather than via `hmmlearn`, because the causality guarantee is the entire point and I needed to control exactly which recursion runs.

Forward recursion in log space:
$$\log\alpha_t(j) = \log p(x_t \mid j) + \text{logsumexp}_i\left[\log\alpha_{t-1}(i) + \log A_{ij}\right]$$

`logsumexp` is hand-rolled: the recursions are inherently sequential, so it is called $O(T)$ times per EM iteration on tiny arrays where scipy's per-call dispatch overhead dominates the arithmetic. Replacing scipy's version cut fit time from 156s to 14s, and further vectorising the $\xi$ accumulation brought a single fit to 2.16s over 44 iterations.

**The critical distinction:**

- `filter_states()` — forward pass only, uses data up to $t$. **The only method any decision path may call.**
- `smoothed_states()` — full forward–backward, uses the entire sample. Explicitly named as future-using; called only in diagnostics.

Measured max divergence between them: **0.872**. `tests/test_regimes.py` asserts filtering is causal and smoothing is not.

Model selection by BIC: $\text{BIC} = -2\ln L + k\ln T$. The detector refits on an expanding window every 252 sessions, standardising features using training-window statistics only.

EM's monotonicity guarantee is testable because `HMMParams.ll_history` retains the per-iteration likelihood — `test_em_log_likelihood_is_monotonically_non_decreasing` asserts it never decreases. Note what is *not* asserted: that more states give higher likelihood. With deterministic quantile initialisation a larger model can converge to a worse local optimum, so that would be an invalid assumption about the algorithm rather than a property of it.

## Factor construction

20 factors across 5 groups. Two conventions do the heavy lifting:

**Higher is always better.** Factors where low values are desirable (`low_volatility`, `low_beta`, `low_leverage`, `small_drawdown`) are stored **negated**. A sign error here silently inverts the strategy, so `test_all_factors_declared_higher_is_better` asserts the invariant registry-wide and `test_low_beta_factor_recovers_true_betas` confirms the negation against ground-truth betas (ρ < −0.7).

**Winsorize, then z-score, within each date's cross-section:**
$$z_i = \frac{\tilde{x}_i - \text{mean}(\tilde{x})}{\text{sd}(\tilde{x})}$$

Order matters — z-scoring first would bake the outlier into the mean and standard deviation. Per-date scope matters more: pooled normalisation leaks the future distribution into every past date, and the leakage audit verifies per-date `|mean| <= 1.63e-15`.

Missing factors become **neutral (0) with coverage tracking**, rather than dropping the asset. Dropping introduces selection bias correlated with data availability — small and newly-listed companies have worse coverage.

## Information coefficient analysis

$$\text{IC}_t = \text{Spearman}(\text{score}_t, r_{t \to t+h})$$

Spearman rather than Pearson: the signal is a rank, outliers are common, and the relationship need only be monotone.

Reported as mean IC, IC information ratio (mean/sd), and t-statistic $\bar{\text{IC}}\sqrt{n}/\text{sd}$. IC decay across horizons {5, 21, 63, 126, 252} days reveals how fast a signal ages — which is what determines the correct rebalance frequency.

## Transaction costs

$$\text{cost} = \text{notional traded} \times (\text{commission} + \text{spread}/2 + \text{slippage})$$

Default 10 bps round trip. Cost is charged on **traded notional against the drifted portfolio**, not on target weights — positions that drift toward their target need less trading, and ignoring drift overstates turnover.

Turnover is reported one-way: $\tfrac{1}{2}\sum_i |w_i^{new} - w_i^{old}|$, so full replacement is 100%.

Limitation: costs are linear and do not model market impact, which scales roughly with (order size / ADV)^0.6 and would penalise concentrated positions more heavily. See `LIMITATIONS.md`.
