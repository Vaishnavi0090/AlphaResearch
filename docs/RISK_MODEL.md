# Risk Model

## Philosophy

Weight-based risk reporting is misleading whenever correlations are non-trivial. A 20% position can carry 35% of portfolio risk. So every risk report answers "where is the risk actually coming from", not "what do we own".

## Euler decomposition

Volatility is homogeneous of degree 1 in weights, so it decomposes exactly:

$$\text{MCR}_i = \frac{(\Sigma w)_i}{\sigma_p} \qquad \text{CCR}_i = w_i\,\text{MCR}_i \qquad \sum_i \text{CCR}_i = \sigma_p$$

Measured on a sample portfolio: model volatility 0.13739, components sum to 0.13739 — exact.

**A subtlety worth stating.** The components sum to the volatility implied by the *covariance model*, not to the *realised* sample standard deviation, because the covariance is shrunk. Conflating the two makes the identity appear to fail. The system therefore reports `model_volatility` (0.13739) and realised `ann_volatility` (0.13863) as separate fields — the gap is estimation error, and hiding it would be dishonest.

## Tail risk

Three VaR estimators are always reported together — historical, parametric Gaussian, and Cornish–Fisher — because their disagreement quantifies non-normality. The tool returns an explicit `gaussian_understates_tail` boolean.

CVaR (expected shortfall) is preferred for optimisation: it is coherent (subadditive, where VaR is not — VaR can perversely reward concentration) and sensitive to tail shape rather than only to a threshold.

Per-position **tail attribution** decomposes the worst 5% of days by contributor, answering which holdings actually hurt when things go wrong — often a different list from the largest positions.

## Factor risk decomposition

Portfolio returns are regressed on factor-mimicking portfolio returns (long-short quintile portfolios built from group scores), decomposing variance into factor and residual shares.

Measured on a sample portfolio: **R² = 0.698**, market 65.6%, value 4.3%, residual **30.2%**.

The residual share is reported prominently and deliberately. A decomposition claiming to explain everything is either overfitted or hiding its errors; 30% idiosyncratic is a realistic figure for a 6-position portfolio.

**A bug worth recording.** This originally returned R² = 0.0. The cause: factor-mimicking portfolios were being built from a *single date's* scores, producing one weight row and a degenerate time-series regression. Factor attribution is inherently a time-series regression, so it needs factor returns across the whole lookback. Fixed by rebuilding score panels on a monthly grid across the window.

## Concentration

- **Herfindahl index** $\sum w_i^2$ and **effective N** $1/\sum w_i^2$
- **Diversification ratio** — weighted average volatility over portfolio volatility
- **Mean pairwise correlation**
- **Sector risk shares** via the same Euler decomposition

`risk_budget_check` raises explicit flags when concentration limits are approached.

## Stress testing

**Historical replay** — the portfolio's current weights applied to returns from labelled windows (GFC 2008, COVID 2020, rates 2022).

**Hypothetical shocks**:
- *Volatility shock* — scale return dispersion
- *Correlation shock* — floor pairwise correlations and resample, modelling the "correlations go to 1 in a crisis" failure
- *Market shock* — first-order beta response
- *Sector shock* — targeted single-sector decline

Every scenario reports portfolio return, benchmark return, max drawdown, CVaR, and worst contributors.

These are **conditional scenarios, not forecasts**, and the tool output says so explicitly. A stress test answers "if this happened, what would it cost", never "this will happen".

## Covariance

Ledoit–Wolf shrinkage by default. Sample covariance is badly conditioned when T is not much larger than N, and an optimiser will exploit that noise — the classic error-maximisation failure. PSD repair via eigenvalue clipping is applied where needed.
