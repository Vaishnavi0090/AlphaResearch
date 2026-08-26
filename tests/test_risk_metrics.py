"""Numerical validation of the core mathematics (Phase 9/31).

Assertions are against closed-form or hand-computed values, not against the
code's own prior output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from finrisk.risk import metrics as m


def test_simple_returns_definition():
    px = pd.Series([100.0, 110.0, 99.0])
    r = m.simple_returns(px)
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(-0.10)


def test_log_returns_are_additive():
    px = pd.Series([100.0, 110.0, 121.0])
    lr = m.log_returns(px)
    assert lr.sum() == pytest.approx(np.log(1.21))


def test_portfolio_return_and_variance_closed_form():
    w = np.array([0.6, 0.4])
    mu = np.array([0.10, 0.05])
    cov = np.array([[0.04, 0.006], [0.006, 0.09]])
    assert m.portfolio_return(w, mu) == pytest.approx(0.08)
    expected = 0.36 * 0.04 + 0.16 * 0.09 + 2 * 0.6 * 0.4 * 0.006
    assert m.portfolio_variance(w, cov) == pytest.approx(expected)
    assert m.portfolio_volatility(w, cov) == pytest.approx(np.sqrt(expected))


def test_variance_never_negative():
    w = np.array([1.0, -1.0])
    cov = np.array([[1e-18, 0.0], [0.0, 1e-18]])
    assert m.portfolio_variance(w, cov) >= 0.0


def test_sharpe_matches_definition():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert m.sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_subtracts_risk_free():
    r = pd.Series([0.001] * 500)
    assert m.sharpe_ratio(r, risk_free_annual=0.0) == 0.0  # zero vol -> 0 by guard
    r2 = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 500))
    assert m.sharpe_ratio(r2, 0.05) < m.sharpe_ratio(r2, 0.0)


def test_cagr_geometric():
    # +10% then -10% -> total -1%. With 2 periods at 2 periods/year that is
    # exactly one year, so CAGR == total return.
    r = pd.Series([0.10, -0.10])
    assert m.total_return(r) == pytest.approx(-0.01)
    assert m.cagr(r, periods_per_year=2) == pytest.approx(-0.01)
    # Over two years (1 period/year) the annualised rate is the geometric root.
    assert m.cagr(r, periods_per_year=1) == pytest.approx(0.99**0.5 - 1)


def test_cagr_below_arithmetic_mean_when_volatile():
    """Volatility drag: geometric < arithmetic. Quoting only the arithmetic
    figure is a common way to flatter a strategy."""
    rng = np.random.default_rng(21)
    r = pd.Series(rng.normal(0.0005, 0.02, 3000))
    assert m.cagr(r) < m.annualized_return(r)


def test_cagr_handles_total_loss():
    assert m.cagr(pd.Series([-1.0, 0.0])) == -1.0


def test_max_drawdown_hand_computed():
    px = pd.Series([100.0, 120.0, 90.0, 150.0])
    r = m.simple_returns(px)
    assert m.max_drawdown(r) == pytest.approx(90 / 120 - 1)


def test_drawdown_is_zero_for_monotone_series():
    r = pd.Series([0.01] * 50)
    assert m.max_drawdown(r) == pytest.approx(0.0)


def test_sortino_uses_full_sample_denominator():
    r = pd.Series([0.02, -0.01, 0.02, -0.01] * 50)
    dd = m.downside_deviation(r)
    manual = np.sqrt((np.minimum(r, 0.0) ** 2).sum() / (len(r) - 1)) * np.sqrt(252)
    assert dd == pytest.approx(manual)


def test_sortino_exceeds_sharpe_for_right_skewed():
    rng = np.random.default_rng(5)
    r = pd.Series(np.abs(rng.normal(0, 0.01, 2000)) - 0.002)
    assert m.sortino_ratio(r) > m.sharpe_ratio(r)


def test_beta_of_series_against_itself_is_one():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0, 0.01, 500))
    assert m.beta(b, b) == pytest.approx(1.0)


def test_beta_scales_linearly():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0, 0.01, 500))
    assert m.beta(2.0 * b, b) == pytest.approx(2.0)


def test_beta_of_constant_is_zero():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0, 0.01, 500))
    assert m.beta(pd.Series(0.0, index=b.index), b) == pytest.approx(0.0)


def test_historical_var_is_empirical_quantile():
    r = pd.Series(np.linspace(-0.10, 0.10, 201))
    assert m.historical_var(r, 0.95) == pytest.approx(np.quantile(r, 0.05))


def test_parametric_var_matches_gaussian_formula():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0, 0.01, 5000))
    expected = r.mean() + norm.ppf(0.05) * r.std(ddof=1)
    assert m.parametric_var(r, 0.95) == pytest.approx(expected)


def test_cvar_is_at_least_as_severe_as_var():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.standard_t(4, 5000) * 0.01)
    assert m.conditional_var(r, 0.95) <= m.historical_var(r, 0.95)
    assert m.conditional_var(r, 0.99) <= m.conditional_var(r, 0.95)


def test_gaussian_var_understates_fat_tails():
    rng = np.random.default_rng(13)
    r = pd.Series(rng.standard_t(3, 20000) * 0.01)
    # parametric is LESS negative => understates the loss
    assert m.parametric_var(r, 0.99) > m.historical_var(r, 0.99)


def test_var_returns_zero_on_short_sample():
    assert m.historical_var(pd.Series([0.01] * 5)) == 0.0


def test_euler_decomposition_sums_to_volatility(toy_returns):
    cov = toy_returns.cov(ddof=1).to_numpy() * 252
    w = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    comp = m.component_risk_contribution(w, cov)
    assert comp.sum() == pytest.approx(m.portfolio_volatility(w, cov), rel=1e-12)
    assert m.percent_risk_contribution(w, cov).sum() == pytest.approx(1.0)


def test_marginal_contribution_zero_for_zero_vol():
    w = np.array([0.5, 0.5])
    assert np.allclose(m.marginal_risk_contribution(w, np.zeros((2, 2))), 0.0)


def test_effective_n_and_herfindahl():
    w = np.array([0.25] * 4)
    assert m.herfindahl(w) == pytest.approx(0.25)
    assert m.effective_n(w) == pytest.approx(4.0)
    assert m.effective_n(np.array([1.0])) == pytest.approx(1.0)


def test_diversification_ratio_is_one_for_single_asset():
    assert m.diversification_ratio(np.array([1.0]), np.array([[0.04]])) == pytest.approx(1.0)


def test_diversification_ratio_exceeds_one_when_uncorrelated():
    cov = np.eye(4) * 0.04
    w = np.full(4, 0.25)
    assert m.diversification_ratio(w, cov) > 1.5


def test_turnover_convention_is_one_way():
    # full replacement of the portfolio == 100% one-way turnover
    assert m.turnover({"A": 1.0}, {"B": 1.0}) == pytest.approx(1.0)
    assert m.turnover({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}) == pytest.approx(0.0)
    assert m.turnover({"A": 1.0}, {"A": 0.5, "B": 0.5}) == pytest.approx(0.5)


def test_ledoit_wolf_is_better_conditioned_than_sample(toy_returns):
    short = toy_returns.iloc[:20]  # T < N*4 -> ill-conditioned
    s = m.sample_covariance(short).to_numpy()
    lw = m.ledoit_wolf_covariance(short).to_numpy()
    assert np.linalg.cond(lw) < np.linalg.cond(s)


def test_nearest_psd_removes_negative_eigenvalues():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    fixed = m.nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() >= 0.0
    assert np.allclose(fixed, fixed.T)


def test_correlation_from_cov_has_unit_diagonal():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    corr = m.correlation_from_cov(cov)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr[0, 1] == pytest.approx(0.01 / np.sqrt(0.04 * 0.09))


def test_tracking_error_zero_against_self():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 300))
    assert m.tracking_error(r, r) == pytest.approx(0.0)


def test_information_ratio_zero_against_self():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 300))
    assert m.information_ratio(r, r) == pytest.approx(0.0)


def test_calmar_is_cagr_over_drawdown():
    r = pd.Series([0.01, -0.05, 0.02] * 100)
    assert m.calmar_ratio(r) == pytest.approx(m.cagr(r) / abs(m.max_drawdown(r)))
