"""Core portfolio mathematics (Phase 9) and risk statistics (Phase 10).

All functions are pure and operate on numpy/pandas. They are the numerical
foundation of the project, so every one of them has a closed-form or
hand-computed test in `tests/test_risk_metrics.py`.

Conventions
-----------
* Returns are **simple** daily returns unless stated otherwise.
* `periods_per_year = 252`.
* VaR and CVaR are returned as **negative numbers** representing a loss
  (e.g. -0.021 = a 2.1% loss). This avoids the ubiquitous sign confusion in
  risk code: a more negative number is always worse, everywhere.
* Compounding uses geometric aggregation for returns and sqrt-time scaling for
  volatility, the standard approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """r_t = P_t / P_{t-1} - 1"""
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Log returns. Additive across time, which is why they are used for
    multi-period aggregation, but NOT additive across assets — so portfolio
    aggregation always uses simple returns."""
    return np.log(prices).diff().iloc[1:]


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """R_p = w' mu"""
    return float(np.dot(weights, mu))


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    """sigma_p^2 = w' Sigma w  (clipped at 0 for numerical noise)"""
    return float(max(weights @ cov @ weights, 0.0))


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(weights, cov)))


def portfolio_returns_series(weights: pd.Series, asset_returns: pd.DataFrame) -> pd.Series:
    """Return series of a *fixed-weight* portfolio (no drift, no rebalancing)."""
    common = [c for c in asset_returns.columns if c in weights.index]
    w = weights.reindex(common).fillna(0.0)
    return (asset_returns[common].fillna(0.0) * w).sum(axis=1)


# ----------------------------------------------------------------------
# Aggregate performance
# ----------------------------------------------------------------------


def total_return(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualised growth rate."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    growth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0:
        return 0.0
    if growth <= 0:  # total wipe-out
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Arithmetic annualisation of the mean. Reported alongside CAGR because the
    two differ by roughly sigma^2/2 and quoting only the flattering one is a
    common sleight of hand."""
    r = returns.dropna()
    return float(r.mean() * periods_per_year) if len(r) else 0.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def excess_returns(returns: pd.Series, risk_free_annual: pd.Series | float) -> pd.Series:
    """Subtract the daily-equivalent risk-free rate."""
    r = returns.dropna()
    if isinstance(risk_free_annual, (int, float)):
        return r - float(risk_free_annual) / TRADING_DAYS
    rf = risk_free_annual.reindex(r.index).ffill().bfill().fillna(0.0)
    return r - rf / TRADING_DAYS


def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: pd.Series | float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Sharpe = (R_p - R_f) / sigma_p, annualised."""
    ex = excess_returns(returns, risk_free_annual)
    if len(ex) < 2:
        return 0.0
    sd = ex.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(ex.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised deviation of returns below a minimum acceptable return.

    Denominator is the FULL sample size, not the count of downside
    observations. That is the Sortino convention: a portfolio with few but
    severe losses should not be rewarded for the rarity."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    short = np.minimum(r - mar, 0.0)
    return float(np.sqrt((short**2).sum() / (len(r) - 1)) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_annual: pd.Series | float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    ex = excess_returns(returns, risk_free_annual)
    dd = downside_deviation(ex, 0.0, periods_per_year)
    if dd < 1e-12:
        return 0.0
    return float(ex.mean() * periods_per_year / dd)


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    r = returns.dropna()
    return initial * (1.0 + r).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    return curve / curve.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline. Returned as a negative number."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd < 1e-9:
        return 0.0
    return float(cagr(returns, periods_per_year) / mdd)


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else 0.0


# ----------------------------------------------------------------------
# Benchmark-relative
# ----------------------------------------------------------------------


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """beta = Cov(R_p, R_m) / Var(R_m)"""
    df = pd.concat([returns, benchmark], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    var_m = float(df.iloc[:, 1].var(ddof=1))
    if var_m < 1e-16:
        return float("nan")
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / var_m)


def alpha_annualized(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_annual: pd.Series | float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Jensen's alpha from a single-factor regression."""
    rp = excess_returns(returns, risk_free_annual)
    rm = excess_returns(benchmark, risk_free_annual)
    df = pd.concat([rp, rm], axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    b = beta(df.iloc[:, 0], df.iloc[:, 1])
    return float((df.iloc[:, 0].mean() - b * df.iloc[:, 1].mean()) * periods_per_year)


def tracking_error(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    df = pd.concat([returns, benchmark], axis=1).dropna()
    if len(df) < 3:
        return 0.0
    active = df.iloc[:, 0] - df.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    df = pd.concat([returns, benchmark], axis=1).dropna()
    if len(df) < 3:
        return 0.0
    active = df.iloc[:, 0] - df.iloc[:, 1]
    sd = active.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(active.mean() / sd * np.sqrt(periods_per_year))


# ----------------------------------------------------------------------
# Tail risk
# ----------------------------------------------------------------------


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Empirical quantile of the loss distribution.

    Returned as a negative number. No distributional assumption, but it can
    never see a loss worse than the worst observed day — its key weakness.
    """
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    return float(np.quantile(r.to_numpy(), 1.0 - confidence))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian VaR: mu + z_{1-c} * sigma.

    Understates tail risk for the fat-tailed, negatively skewed distributions
    that actual asset returns follow, which is exactly why the historical and
    parametric numbers are both reported side by side.
    """
    from scipy.stats import norm

    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    return float(r.mean() + norm.ppf(1.0 - confidence) * r.std(ddof=1))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall: mean loss GIVEN the loss exceeds VaR.

    Preferred over VaR for optimisation because (a) it describes the severity
    of the tail rather than just its threshold, and (b) it is a coherent,
    convex risk measure, so a CVaR-constrained problem stays tractable while
    a VaR-constrained one is generally non-convex.
    """
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    var = np.quantile(r.to_numpy(), 1.0 - confidence)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def cornish_fisher_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Skew/kurtosis-adjusted VaR. Cheap improvement over Gaussian VaR without
    resorting to Monte Carlo."""
    from scipy.stats import norm

    r = returns.dropna()
    if len(r) < 30:
        return 0.0
    z = norm.ppf(1.0 - confidence)
    s = float(r.skew())
    k = float(r.kurtosis())  # excess
    z_cf = z + (z**2 - 1) * s / 6.0 + (z**3 - 3 * z) * k / 24.0 - (2 * z**3 - 5 * z) * s**2 / 36.0
    return float(r.mean() + z_cf * r.std(ddof=1))


# ----------------------------------------------------------------------
# Risk decomposition
# ----------------------------------------------------------------------


def marginal_risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """d(sigma_p)/d(w_i) = (Sigma w)_i / sigma_p"""
    vol = portfolio_volatility(weights, cov)
    if vol < 1e-12:
        return np.zeros_like(weights)
    return (cov @ weights) / vol


def component_risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """w_i * d(sigma_p)/d(w_i). Sums exactly to sigma_p (Euler decomposition)."""
    return weights * marginal_risk_contribution(weights, cov)


def percent_risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    comp = component_risk_contribution(weights, cov)
    total = comp.sum()
    if abs(total) < 1e-14:
        return np.zeros_like(comp)
    return comp / total


def herfindahl(weights: np.ndarray) -> float:
    """Sum of squared weights. 1/H is the 'effective number of positions'."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w**2))


def effective_n(weights: np.ndarray) -> float:
    h = herfindahl(weights)
    return float(1.0 / h) if h > 1e-12 else 0.0


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Weighted average asset vol / portfolio vol. 1.0 = no diversification."""
    sd = np.sqrt(np.diag(cov))
    pv = portfolio_volatility(weights, cov)
    if pv < 1e-12:
        return 1.0
    return float(np.dot(np.abs(weights), sd) / pv)


# ----------------------------------------------------------------------
# Covariance estimation
# ----------------------------------------------------------------------


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov(ddof=1)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage toward a scaled identity.

    With N assets you need ~N^2/2 covariance parameters from T observations.
    At N=40, T=252 the sample matrix is badly conditioned and a mean-variance
    optimiser will happily exploit the estimation error, producing extreme
    weights that fall apart out of sample. Shrinkage is the cheapest effective
    fix and is why it is the default.
    """
    from sklearn.covariance import LedoitWolf

    r = returns.dropna(axis=1, how="all").dropna()
    if r.shape[1] < 2 or r.shape[0] < 10:
        return sample_covariance(returns)
    lw = LedoitWolf(assume_centered=False).fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    periods_per_year: int = TRADING_DAYS,
    annualize: bool = True,
) -> pd.DataFrame:
    cov = ledoit_wolf_covariance(returns) if method == "ledoit_wolf" else sample_covariance(returns)
    return cov * periods_per_year if annualize else cov


def nearest_psd(cov: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping eigenvalues.

    Shrinkage estimators are PSD by construction, but a matrix that has been
    sliced, reindexed and re-symmetrised can pick up a tiny negative eigenvalue
    that makes scipy's optimiser diverge. This guard is cheap insurance.
    """
    a = np.asarray(cov, dtype=float)
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, epsilon, None)
    return vecs @ np.diag(vals) @ vecs.T


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def mean_pairwise_correlation(cov: np.ndarray) -> float:
    corr = correlation_from_cov(cov)
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(np.nanmean(corr[iu]))


# ----------------------------------------------------------------------
# Turnover / cost
# ----------------------------------------------------------------------


def turnover(prev: dict[str, float] | pd.Series, new: dict[str, float] | pd.Series) -> float:
    """One-way turnover = 0.5 * sum |w_new - w_prev|.

    The 0.5 makes "100% turnover" mean "the whole portfolio was replaced",
    which is the convention transaction-cost estimates are quoted against.
    """
    p = pd.Series(prev, dtype=float)
    n = pd.Series(new, dtype=float)
    idx = p.index.union(n.index)
    return float(0.5 * (n.reindex(idx).fillna(0.0) - p.reindex(idx).fillna(0.0)).abs().sum())
