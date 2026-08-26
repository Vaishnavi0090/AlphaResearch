"""Tool layer: schema validation, determinism, and error isolation."""

from __future__ import annotations

import pandas as pd
import pytest

from finrisk.tools.finance_tools import TOOLS, ToolContext, ToolRegistry


@pytest.fixture(scope="module")
def registry(small_bundle, small_settings):
    return ToolRegistry(ToolContext(small_bundle, small_settings))


@pytest.fixture(scope="module")
def as_of(small_bundle):
    return str(small_bundle.dates[-1].date())


@pytest.fixture(scope="module")
def weights(registry, as_of):
    res = registry.call("optimize_portfolio", as_of=as_of)
    assert "error" not in res, res
    return res["weights"]


def test_all_tools_expose_a_schema(registry):
    schema = registry.schema()
    assert len(schema) == len(TOOLS)
    for s in schema:
        assert s["name"] and s["description"]
        assert s["input_schema"]["type"] == "object"


def test_unknown_tool_returns_error_not_exception(registry):
    res = registry.call("does_not_exist")
    assert "error" in res
    assert "available" in res


def test_invalid_input_is_rejected(registry, weights):
    assert "error" in registry.call("calculate_var", weights=weights, confidence=1.5)
    assert "error" in registry.call("retrieve_financial_documents", query="x")
    assert "error" in registry.call("calculate_risk", weights=weights, lookback_days=1)


def test_missing_required_field_rejected(registry):
    assert "error" in registry.call("calculate_risk")


def test_date_snapping_never_looks_forward(registry, small_bundle):
    """A weekend request must resolve BACKWARD to the prior session."""
    ctx = registry.ctx
    sunday = pd.Timestamp("2019-06-16")  # a Sunday
    assert ctx.date(str(sunday.date())) <= sunday


def test_date_before_history_raises(registry):
    with pytest.raises(ValueError):
        registry.ctx.date("1990-01-01")


def test_get_market_data(registry, as_of):
    res = registry.call("get_market_data", as_of=as_of, lookback_days=252)
    assert res["n_sessions"] > 200
    assert res["tickers"]
    assert set(res["last_price"]) == set(res["tickers"])


def test_get_market_data_unknown_tickers(registry, as_of):
    assert "error" in registry.call("get_market_data", tickers=["NOPE"], as_of=as_of)


def test_get_fundamentals_is_point_in_time(registry, as_of):
    res = registry.call("get_fundamentals", as_of=as_of)
    for rec in res["records"].values():
        assert pd.Timestamp(rec["available_from"]) <= pd.Timestamp(as_of)
        assert pd.Timestamp(rec["available_from"]) >= pd.Timestamp(rec["period_end"])


def test_calculate_factor_scores_ranking(registry, as_of):
    res = registry.call("calculate_factor_scores", as_of=as_of, top_n=5)
    ranking = res["ranking"]
    assert len(ranking) == 5
    assert [r["rank"] for r in ranking] == sorted(r["rank"] for r in ranking)
    assert sum(res["composite_weights"].values()) == pytest.approx(1.0)


def test_detect_market_regime_is_filtered(registry, as_of):
    res = registry.call("detect_market_regime", as_of=as_of)
    assert "filtered" in res["inference"]
    assert sum(res["state_probabilities"].values()) == pytest.approx(1.0)
    assert 0.5 <= res["risk_multiplier"] <= 1.0


def test_calculate_risk_reports_both_var_methods(registry, weights, as_of):
    res = registry.call("calculate_risk", weights=weights, as_of=as_of)
    assert res["var_95_hist"] <= 0.0
    assert res["var_95_param"] <= 0.0
    assert res["cvar_95"] <= res["var_95_hist"] + 1e-12
    assert res["effective_n"] > 0


def test_calculate_var_flags_non_gaussian(registry, weights, as_of):
    res = registry.call("calculate_var", weights=weights, as_of=as_of)
    assert isinstance(res["gaussian_understates_tail"], bool)
    assert "var_cornish_fisher" in res


def test_calculate_cvar_tail_attribution_sums_to_one(registry, weights, as_of):
    res = registry.call("calculate_cvar", weights=weights, as_of=as_of)
    tail = res["tail_contribution"]
    assert tail
    assert sum(tail.values()) == pytest.approx(1.0, abs=0.02)


def test_risk_contribution_satisfies_euler(registry, weights, as_of):
    res = registry.call("calculate_risk_contribution", weights=weights, as_of=as_of)
    assert res["euler_identity_holds"]
    assert res["sum_of_components"] == pytest.approx(res["model_volatility"], rel=1e-6)
    # abs tolerance, not rel: the tool rounds pct_of_risk to 5dp for JSON
    # transport, so the reported shares sum to 1 only up to that rounding.
    # Exactness of the underlying identity is asserted in test_risk_metrics.
    assert sum(c["pct_of_risk"] for c in res["contributions"]) == pytest.approx(1.0, abs=1e-4)


def test_optimize_portfolio_respects_overrides(registry, as_of):
    res = registry.call(
        "optimize_portfolio",
        method="risk_constrained",
        as_of=as_of,
        max_weight=0.15,
        target_volatility=0.10,
    )
    assert max(res["weights"].values()) <= 0.15 + 1e-6
    assert res["expected_volatility_ann"] <= 0.10 + 5e-3
    assert res["constraints_satisfied"]


def test_optimize_portfolio_all_methods(registry, as_of):
    for method in ("equal_weight", "max_sharpe", "risk_parity", "risk_constrained"):
        res = registry.call("optimize_portfolio", method=method, as_of=as_of)
        assert "error" not in res, method
        assert res["weights"], method


def test_stress_test_produces_scenarios(registry, weights, as_of):
    res = registry.call("stress_test", weights=weights, as_of=as_of)
    assert res["scenarios"]
    names = {s["name"] for s in res["scenarios"]}
    assert any("vol_shock" in n for n in names)
    assert any("correlation_shock" in n for n in names)


def test_analyze_portfolio_factor_decomposition(registry, weights, as_of):
    res = registry.call("analyze_portfolio", weights=weights, as_of=as_of)
    fr = res["factor_risk_decomposition"]
    assert 0.0 <= fr["r_squared"] <= 1.0
    assert 0.0 <= fr["residual_share"] <= 1.0
    assert res["factor_exposures"]


def test_retrieve_without_index_is_graceful(registry):
    res = registry.call("retrieve_financial_documents", query="anything")
    assert res["citations"] == []


def test_tool_calls_are_accounted(small_bundle, small_settings, as_of):
    reg = ToolRegistry(ToolContext(small_bundle, small_settings))
    reg.call("detect_market_regime", as_of=as_of)
    reg.call("nope")
    stats = reg.stats()
    assert stats["n_calls"] == 2
    assert stats["n_failed"] == 1
    assert all(isinstance(v, int) for v in stats["by_tool"].values())


def test_tools_are_deterministic(registry, as_of):
    a = registry.call("optimize_portfolio", as_of=as_of)
    b = registry.call("optimize_portfolio", as_of=as_of)
    assert a["weights"] == b["weights"]


def test_langchain_adapter_builds(registry):
    from finrisk.tools.finance_tools import to_langchain_tools

    tools = to_langchain_tools(registry)
    if tools:  # langchain_core installed
        assert len(tools) == len(TOOLS)
        assert all(hasattr(t, "name") and hasattr(t, "args_schema") for t in tools)
