"""Portfolio optimiser: constraint enforcement and objective correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.config.settings import PortfolioConfig
from finrisk.portfolio.optimizer import OptimizationInputs, PortfolioOptimizer
from finrisk.risk import metrics as m

AS_OF = pd.Timestamp("2024-01-31")
METHODS = [
    "equal_weight",
    "mean_variance",
    "max_sharpe",
    "risk_parity",
    "risk_constrained",
    "cvar_constrained",
]


@pytest.fixture
def inputs():
    rng = np.random.default_rng(42)
    n, t = 10, 900
    R = rng.normal(0.0004, 0.011, (t, n))
    R[:, 0] *= 2.5  # one deliberately volatile asset
    tickers = [f"A{i}" for i in range(n)]
    return OptimizationInputs(
        tickers=tickers,
        expected_returns=rng.normal(0.07, 0.04, n),
        cov=np.cov(R.T) * 252,
        scenario_returns=R,
        sectors={t: ["Tech", "Fin", "Health"][i % 3] for i, t in enumerate(tickers)},
        risk_free=0.02,
    )


@pytest.fixture
def cfg():
    return PortfolioConfig(
        top_n=10,
        max_weight=0.25,
        target_volatility=0.15,
        max_cvar_95=0.025,
        max_sector_weight=0.50,
    )


# ------------------------------------------------------------ constraints
@pytest.mark.parametrize("method", METHODS)
def test_all_methods_respect_constraints(inputs, cfg, method):
    opt = PortfolioOptimizer(cfg)
    pw = opt.optimize(inputs, AS_OF, method)
    assert pw.constraints is not None
    assert pw.constraints.feasible, pw.constraints.violations
    assert max(pw.weights.values()) <= cfg.max_weight + 1e-6
    assert min(pw.weights.values()) >= -1e-8
    assert sum(pw.weights.values()) <= cfg.max_leverage + 1e-6


@pytest.mark.parametrize("method", METHODS)
def test_no_solver_fallbacks_on_wellposed_problem(inputs, cfg, method):
    opt = PortfolioOptimizer(cfg)
    pw = opt.optimize(inputs, AS_OF, method)
    assert "fallback" not in pw.optimization_status
    assert opt.fallback_count == 0


def test_max_weight_binds_when_tight(inputs):
    cfg = PortfolioConfig(top_n=10, max_weight=0.12, target_volatility=None)
    pw = PortfolioOptimizer(cfg).mean_variance(inputs, AS_OF)
    assert max(pw.weights.values()) <= 0.12 + 1e-6


def test_sector_cap_binds(inputs):
    cfg = PortfolioConfig(top_n=10, max_weight=1.0, max_sector_weight=0.40, target_volatility=None)
    pw = PortfolioOptimizer(cfg).mean_variance(inputs, AS_OF)
    agg: dict[str, float] = {}
    for t, w in pw.weights.items():
        s = inputs.sectors[t]
        agg[s] = agg.get(s, 0.0) + w
    assert max(agg.values()) <= 0.40 + 1e-4


def test_volatility_target_is_respected(inputs):
    cfg = PortfolioConfig(top_n=10, max_weight=1.0, target_volatility=0.08)
    pw = PortfolioOptimizer(cfg).risk_constrained(inputs, AS_OF)
    assert pw.expected_volatility <= 0.08 + 5e-3


def test_tighter_vol_target_gives_lower_or_equal_return(inputs):
    loose = PortfolioOptimizer(
        PortfolioConfig(top_n=10, max_weight=0.5, target_volatility=0.20)
    ).risk_constrained(inputs, AS_OF)
    tight = PortfolioOptimizer(
        PortfolioConfig(top_n=10, max_weight=0.5, target_volatility=0.06)
    ).risk_constrained(inputs, AS_OF)
    assert tight.expected_return <= loose.expected_return + 1e-9


def test_cash_not_allowed_forces_full_investment(inputs):
    cfg = PortfolioConfig(top_n=10, max_weight=0.30, cash_allowed=False, target_volatility=None)
    pw = PortfolioOptimizer(cfg).mean_variance(inputs, AS_OF)
    assert sum(pw.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_infeasible_config_rejected_at_construction():
    with pytest.raises(ValueError):
        PortfolioConfig(top_n=3, max_weight=0.20, cash_allowed=False)


def test_invalid_max_weight_rejected():
    with pytest.raises(ValueError):
        PortfolioConfig(max_weight=1.5)


# ------------------------------------------------------------ objectives
def test_equal_weight_is_uniform(inputs):
    pw = PortfolioOptimizer(PortfolioConfig(top_n=10, max_weight=1.0)).equal_weight(inputs, AS_OF)
    vals = list(pw.weights.values())
    assert np.allclose(vals, vals[0])


def test_risk_parity_equalises_risk_contributions(inputs):
    cfg = PortfolioConfig(top_n=10, max_weight=1.0, target_volatility=None, cash_allowed=False)
    pw = PortfolioOptimizer(cfg).risk_parity(inputs, AS_OF)
    w = np.array([pw.weights[t] for t in inputs.tickers])
    pct = m.percent_risk_contribution(w, inputs.cov)
    assert pct.std() < 1e-3
    assert np.allclose(pct, 1.0 / len(w), atol=2e-3)


def test_risk_parity_underweights_the_volatile_asset(inputs):
    cfg = PortfolioConfig(top_n=10, max_weight=1.0, target_volatility=None)
    pw = PortfolioOptimizer(cfg).risk_parity(inputs, AS_OF)
    vol = np.sqrt(np.diag(inputs.cov))
    most_volatile = inputs.tickers[int(np.argmax(vol))]
    assert pw.weights[most_volatile] < 1.0 / len(inputs.tickers)


def test_max_sharpe_beats_equal_weight_ex_ante(inputs):
    cfg = PortfolioConfig(top_n=10, max_weight=0.30, target_volatility=None)
    opt = PortfolioOptimizer(cfg)
    assert opt.max_sharpe(inputs, AS_OF).sharpe >= opt.equal_weight(inputs, AS_OF).sharpe


def test_mean_variance_higher_risk_aversion_lowers_volatility(inputs):
    lo = PortfolioOptimizer(
        PortfolioConfig(
            top_n=10, max_weight=1.0, risk_aversion=1.0, target_volatility=None, cash_allowed=False
        )
    ).mean_variance(inputs, AS_OF)
    hi = PortfolioOptimizer(
        PortfolioConfig(
            top_n=10, max_weight=1.0, risk_aversion=50.0, target_volatility=None, cash_allowed=False
        )
    ).mean_variance(inputs, AS_OF)
    assert hi.expected_volatility < lo.expected_volatility


def test_cvar_constraint_is_binding(inputs):
    """A tight CVaR limit must actually reduce realised scenario CVaR."""
    loose = PortfolioOptimizer(
        PortfolioConfig(top_n=10, max_weight=0.5, target_volatility=None, max_cvar_95=0.10)
    ).cvar_constrained(inputs, AS_OF)
    tight = PortfolioOptimizer(
        PortfolioConfig(top_n=10, max_weight=0.5, target_volatility=None, max_cvar_95=0.012)
    ).cvar_constrained(inputs, AS_OF)
    wl = np.array([loose.weights[t] for t in inputs.tickers])
    wt = np.array([tight.weights[t] for t in inputs.tickers])
    cvar_l = m.conditional_var(pd.Series(inputs.scenario_returns @ wl), 0.95)
    cvar_t = m.conditional_var(pd.Series(inputs.scenario_returns @ wt), 0.95)
    assert abs(cvar_t) <= abs(cvar_l) + 1e-9
    assert abs(cvar_t) <= 0.012 + 1e-3


def test_cvar_falls_back_without_scenarios(inputs, cfg):
    bare = OptimizationInputs(
        tickers=inputs.tickers,
        expected_returns=inputs.expected_returns,
        cov=inputs.cov,
        scenario_returns=None,
        sectors=inputs.sectors,
    )
    pw = PortfolioOptimizer(cfg).cvar_constrained(bare, AS_OF)
    assert pw.method == "risk_constrained"


# ------------------------------------------------------------ robustness
def test_single_asset_universe(cfg):
    inp = OptimizationInputs(["ONLY"], np.array([0.08]), np.array([[0.04]]))
    pw = PortfolioOptimizer(cfg).optimize(inp, AS_OF, "max_sharpe")
    assert pw.optimization_status == "single_asset"
    assert pw.weights["ONLY"] == pytest.approx(cfg.max_weight)


def test_empty_universe_raises(cfg):
    inp = OptimizationInputs([], np.array([]), np.zeros((0, 0)))
    with pytest.raises(ValueError):
        PortfolioOptimizer(cfg).optimize(inp, AS_OF, "max_sharpe")


def test_unknown_method_raises(inputs, cfg):
    with pytest.raises(ValueError):
        PortfolioOptimizer(cfg).optimize(inputs, AS_OF, "magic")


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        OptimizationInputs(["A", "B"], np.array([0.1]), np.eye(2))


def test_singular_covariance_is_handled(cfg):
    tickers = ["A", "B", "C"]
    cov = np.ones((3, 3)) * 0.04  # perfectly correlated => singular
    inp = OptimizationInputs(tickers, np.array([0.05, 0.06, 0.07]), cov)
    pw = PortfolioOptimizer(cfg).optimize(inp, AS_OF, "risk_parity")
    assert np.isfinite(pw.expected_volatility)
    assert pw.constraints.feasible


def test_dust_positions_are_zeroed(inputs, cfg):
    pw = PortfolioOptimizer(cfg).max_sharpe(inputs, AS_OF)
    assert all(w == 0.0 or abs(w) > 1e-6 for w in pw.weights.values())


def test_risk_metrics_populated(inputs, cfg):
    pw = PortfolioOptimizer(cfg).risk_constrained(inputs, AS_OF)
    for k in ("expected_volatility", "effective_n", "cvar_95_1d", "diversification_ratio"):
        assert k in pw.risk_metrics


def test_check_constraints_detects_violation(cfg):
    opt = PortfolioOptimizer(cfg)
    rep = opt.check_constraints({"A": 0.9, "B": 0.5}, None, vol=0.10)
    assert not rep.feasible
    assert any("max_weight" in v for v in rep.violations)
    assert any("leverage" in v for v in rep.violations)


def test_results_are_deterministic(inputs, cfg):
    a = PortfolioOptimizer(cfg).risk_constrained(inputs, AS_OF)
    b = PortfolioOptimizer(cfg).risk_constrained(inputs, AS_OF)
    assert a.weights == b.weights


# ------------------------------------------------------------ turnover
def test_turnover_constraint_binds(inputs):
    """A turnover ceiling must actually reduce trading, not just be recorded."""
    prev = {inputs.tickers[0]: 0.5, inputs.tickers[1]: 0.5}
    inp = OptimizationInputs(
        tickers=inputs.tickers,
        expected_returns=inputs.expected_returns,
        cov=inputs.cov,
        scenario_returns=inputs.scenario_returns,
        sectors=inputs.sectors,
        prev_weights=prev,
    )

    def realised(pw):
        names = set(pw.weights) | set(prev)
        return 0.5 * sum(abs(pw.weights.get(t, 0.0) - prev.get(t, 0.0)) for t in names)

    free = PortfolioOptimizer(
        PortfolioConfig(top_n=10, max_weight=0.3, target_volatility=None)
    ).max_sharpe(inp, AS_OF)
    capped = PortfolioOptimizer(
        PortfolioConfig(top_n=10, max_weight=0.3, target_volatility=None, max_turnover=0.30)
    ).max_sharpe(inp, AS_OF)
    assert realised(free) > 0.30
    assert realised(capped) <= 0.30 + 1e-3
    assert capped.constraints.feasible


def test_turnover_constraint_inert_without_previous_weights(inputs):
    """Nothing to compare against means the constraint cannot bind."""
    cfg = PortfolioConfig(top_n=10, max_weight=0.3, target_volatility=None, max_turnover=0.01)
    pw = PortfolioOptimizer(cfg).max_sharpe(inputs, AS_OF)
    assert pw.constraints.feasible


def test_infeasible_turnover_is_reported_not_hidden(inputs):
    """An impossible limit must surface as a violation rather than silently
    returning a portfolio that breaches it."""
    prev = {inputs.tickers[0]: 1.0}
    inp = OptimizationInputs(
        tickers=inputs.tickers,
        expected_returns=inputs.expected_returns,
        cov=inputs.cov,
        sectors=None,
        prev_weights=prev,
    )
    cfg = PortfolioConfig(
        top_n=10, max_weight=0.12, target_volatility=None, max_turnover=0.001, cash_allowed=True
    )
    pw = PortfolioOptimizer(cfg).max_sharpe(inp, AS_OF)
    assert not pw.constraints.feasible
    assert any("turnover" in v for v in pw.constraints.violations)


# ------------------------------------------------------------ risk budget
@pytest.mark.parametrize("cap", [0.30, 0.25, 0.15])
def test_risk_budget_constraint_binds(inputs, cap):
    """A weight cap does not bound risk share; this constraint does."""
    cfg = PortfolioConfig(
        top_n=10, max_weight=0.5, target_volatility=None, max_risk_contribution=cap
    )
    pw = PortfolioOptimizer(cfg).max_sharpe(inputs, AS_OF)
    w = np.array([pw.weights.get(t, 0.0) for t in inputs.tickers])
    shares = m.percent_risk_contribution(w, inputs.cov)
    assert shares.max() <= cap + 5e-3, f"max risk share {shares.max():.4f} > {cap}"
    assert pw.constraints.feasible


def test_risk_budget_is_tighter_than_weight_cap(inputs):
    """The motivating case: an unconstrained-by-risk portfolio can put a large
    share of volatility in one name even while respecting the weight cap."""
    cfg = PortfolioConfig(top_n=10, max_weight=0.5, target_volatility=None)
    pw = PortfolioOptimizer(cfg).max_sharpe(inputs, AS_OF)
    w = np.array([pw.weights.get(t, 0.0) for t in inputs.tickers])
    shares = m.percent_risk_contribution(w, inputs.cov)
    assert shares.max() > w.max(), "fixture should show risk share exceeding weight share"


def test_risk_budget_audit_flags_breach(inputs):
    """The audit recomputes risk shares rather than trusting the solver."""
    cfg = PortfolioConfig(top_n=10, max_weight=1.0, max_risk_contribution=0.10)
    rep = PortfolioOptimizer(cfg).check_constraints(
        {inputs.tickers[0]: 1.0},
        None,
        vol=0.2,
        cov=inputs.cov,
        tickers=inputs.tickers,
    )
    assert not rep.feasible
    assert any("risk budget" in v for v in rep.violations)


def test_combined_constraints_all_hold(inputs):
    """Turnover, risk budget, weight cap and volatility target together."""
    prev = dict.fromkeys(inputs.tickers, 0.1)
    inp = OptimizationInputs(
        tickers=inputs.tickers,
        expected_returns=inputs.expected_returns,
        cov=inputs.cov,
        scenario_returns=inputs.scenario_returns,
        sectors=inputs.sectors,
        prev_weights=prev,
    )
    cfg = PortfolioConfig(
        top_n=10,
        max_weight=0.25,
        max_sector_weight=0.60,
        target_volatility=0.18,
        max_turnover=0.40,
        max_risk_contribution=0.35,
    )
    pw = PortfolioOptimizer(cfg).risk_constrained(inp, AS_OF)
    w = np.array([pw.weights.get(t, 0.0) for t in inputs.tickers])
    assert pw.constraints.feasible, pw.constraints.violations
    assert w.max() <= 0.25 + 1e-6
    assert pw.expected_volatility <= 0.18 + 5e-3
    assert m.percent_risk_contribution(w, inputs.cov).max() <= 0.35 + 5e-3
    turn = 0.5 * sum(abs(pw.weights.get(t, 0.0) - prev.get(t, 0.0)) for t in inputs.tickers)
    assert turn <= 0.40 + 1e-3
