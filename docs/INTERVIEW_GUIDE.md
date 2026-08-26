# Interview Guide

Questions a quant, an ML engineer, or a hiring manager would actually ask — with honest answers and pointers to the evidence.

## Data and bias

**Q: How do you know there's no look-ahead bias?**

I don't rely on knowing — I test it. The placebo control (`test_shifted_signal_advantage`) runs the identical strategy twice: once honestly, once with signals shifted one period forward. Honest gives 6.18% CAGR / 0.33 Sharpe; cheating gives 33.22% / 1.85. The +27pp gap proves the barrier is load-bearing. If the honest run had scored near 33%, it would already be leaking.

Seven targeted checks back that up: fundamentals never available before period end (2,400 records, 0 violations), factor scores identical under truncation (max diff 0.000e+00), per-date normalisation (|mean| ≤ 1.6e-15), execution lag exactly 1 session across 156 rebalances.

**Q: What about survivorship bias?**

Present and unsolved for real data. The synthetic universe has none by construction — but it also has no delistings, which is optimistic differently. Fixing it needs a point-in-time constituent list I don't have. The audit reports it as a `warning`, not a pass.

**Q: Why synthetic data? Isn't that a cop-out?**

Network egress to every financial provider returned 403 in the build environment. Given that, the options were: ship nothing, fabricate results, or use a DGP with known ground truth. The third is the only defensible one, and it buys something real data can't — I can verify a factor *measures what it claims*. Book-to-price recovers the latent value state at ρ=0.78; ROE recovers quality at ρ=0.85. No real dataset lets you check that.

The honest caveat: I wrote both the DGP and the factors, so there's a self-confirmation risk. The momentum failure is the counterweight — I injected a trend premium and my momentum factors failed to capture it.

## Methods

**Q: Why implement the HMM yourself instead of using hmmlearn?**

Because the causality guarantee is the whole point. I needed `filter_states()` (forward-only) and `smoothed_states()` (full forward-backward, future-using) as separately callable methods so the distinction is testable rather than trusted. Measured divergence between them is 0.872 — proof the distinction is real. Only the filtered path touches any decision.

**Q: Your Sharpe ratios look low.**

Because I subtract the risk-free rate, which is 2–4% over the sample. Omitting it would roughly double every figure. The full pipeline's 0.40 against a benchmark's 0.05 is the honest comparison.

**Q: Why scipy instead of cvxpy?**

Already a dependency, handles every problem here including the CVaR linear program, keeps the install reproducible. The cost is that I write constraint Jacobians myself and don't get a convexity certificate — mitigated by `check_constraints`, an independent audit that doesn't trust solver status. Fallback count is 0 across all six optimisers.

**Q: Explain the CVaR optimisation.**

CVaR looks non-convex but linearises exactly via Rockafellar–Uryasev: introduce the VaR level ζ and per-scenario excess variables u_s, then CVaR ≈ ζ + (1/((1−α)S))Σu_s with u_s ≥ −(r_sᵀw) − ζ and u_s ≥ 0. That's a linear program, solved with HiGHS. `test_cvar_constraint_is_binding` confirms a tighter limit actually reduces realised scenario CVaR.

I use CVaR over VaR because VaR isn't subadditive — it can perversely reward concentration — and CVaR is sensitive to tail shape rather than just a threshold.

## Results — including the uncomfortable ones

**Q: Did it beat the benchmark?**

On the numbers, yes: 6.81% vs 1.91% CAGR, 0.40 vs 0.05 Sharpe. Statistically, no — the standard error of an annualised Sharpe over 13.5 years is about 0.28, so a +0.35 difference doesn't clear two standard errors. I'd say the factor signal is solid (value IC t=+11.18 over 156 cross-sections) and the return claim isn't.

**Q: What's the weakest part of this project?**

Time stability. Sharpe is 0.62 in the first half of the sample and 0.11 in the second. That's the cleanest out-of-sample split available and the strategy doesn't survive it well. I can't separate genuine signal decay from regime composition — the strategy is regime-sensitive (bull 0.56, bear 0.29), so a bear-heavy second half would produce part of the drop mechanically.

What *is* stable is the risk control: volatility 13.3–13.7% and drawdown within −24% in every sub-period. The return is unstable; the risk management isn't.

**Q: Your momentum factor doesn't work. Is it broken?**

No, and establishing that took direct investigation. Three measurements:

1. The latent trend state *does* predict returns (IC +0.04 to +0.05), so the premium exists.
2. My momentum factors correlate positively but weakly with that state (ρ = +0.032 for 1-month, +0.006 for 12-1) — correct sign, so the implementation and sign convention are right.
3. The same machinery recovers value at ρ=0.78, ruling out a generic engine bug.

Diagnosis: over 21 sessions the injected alpha contributes ~0.25% of return against ~6.4% idiosyncratic noise, so trailing return is ~96% noise as a drift estimator. And the trend's 46-session half-life is far shorter than a 231-session 12-1 window, so that window measures already-reverted states — hence the negative long-horizon IC of −0.072.

Conclusion: 12-1 momentum is mis-specified for *this DGP*. Real equity markets do exhibit it, so this is a case where the synthetic data diverges from reality instructively.

**Q: If you know momentum is broken, why is it still weighted 30%?**

Because re-tuning weights on the same data I evaluate them on is exactly the overfitting this project is built to avoid. The sweep shows a value tilt would give Sharpe 0.70 vs the baseline's 0.40 — I could have quietly adopted that and reported a better number. I'd rather report 0.40 and explain why. The baseline is a defensible published prior that happens to be wrong for this dataset.

## Agents

**Q: What stops the LLM from hallucinating a number?**

Three structural mechanisms, not prompting. Tools are the only interface — agents call Pydantic-validated tools and cannot compute. The report is assembled from state (`state["portfolio"]["weights"]`), never from prose. And the critic checks every figure in the conclusion against tool output, removing what doesn't match.

Tested by injection, not inspection: `test_critic_removes_fabricated_metric` plants a fake `secret_alpha: 0.42` next to a real CVaR figure and asserts only the fake is stripped. `test_llm_cannot_change_weights` asks the system to "recommend 90% in one name" and confirms the weights still equal the optimiser's output and respect the cap.

**Q: Is the committee just four prompts producing agreement?**

The debate node maps stances to actions rather than picking the nicest argument. If the risk role isn't supportive it's `reject`; if the bear opposes or concerns are raised it's `proceed_with_limits`. A bear objection becomes an explicit condition rather than being discarded. On the full run: bull=support, bear=oppose, risk=support, quant=conditional → `proceed_with_limits`. `test_committee_does_not_simply_follow_the_bull` asserts the action is never plain `proceed` when the bear opposes.

**Q: Why not let agents choose their own execution order?**

Quantitative dependencies are real dependencies — risk contributions need weights, weights need factors and a covariance estimate. Reordering that adds failure modes and buys nothing. The real discretion, which optional branches to run, is preserved. I'd rather defend a fixed DAG than a demo that looks more autonomous and is less correct.

## Engineering

**Q: Walk me through a bug you found.**

Factor risk decomposition was returning R² = 0.0. The cause: I was building factor-mimicking portfolios from a single date's scores, producing one weight row and a degenerate time-series regression. Factor attribution is inherently a time-series regression — it needs factor returns across the whole lookback. Fixed by rebuilding score panels on a monthly grid. R² went to 0.698 with market explaining 65.6% and 30.2% residual.

A subtler one: the Euler identity was reporting `False`. Not a bug in the decomposition — the components sum to the volatility implied by the *shrunk covariance model*, while I was comparing against the *realised* sample standard deviation. Two different estimators. Fixed by reporting `model_volatility` (0.13739, exact match) and realised `ann_volatility` (0.13863) as separate fields, so the estimation gap is visible rather than hidden.

**Q: How do you know the tests are meaningful?**

They assert against closed-form values, not against the code's prior output. Portfolio variance against a hand-expanded quadratic form; parametric VaR against `norm.ppf`; max drawdown against a hand-computed path. Several tests are adversarial — they inject fabrications, tamper with signal timestamps to confirm the leakage assertion fires, and assert the *negative* momentum result so a future change can't silently "fix" it.

Four of my own test expectations turned out wrong where the code was right — including an invalid assumption that more HMM states must raise likelihood. With deterministic init a larger model can land in a worse local optimum, so I exposed EM's likelihood history and tested the real guarantee: monotonicity.
