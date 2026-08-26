"""Per-position attribution: allocation vs alpha vs risk vs diversification.

Assertions are against analytically-known answers wherever possible — an
equal-weight book of identical uncorrelated assets has exactly 1/N risk share
and a known diversification benefit, so the engine can be checked rather than
merely exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.risk.position_attribution import (
    build_position_explanation,
    compute_confidence,
    compute_position_attribution,
    render_explanations,
)


@pytest.fixture
def uncorrelated_returns():
    """Four independent assets with equal volatility."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    return pd.DataFrame(rng.normal(0.0004, 0.012, (600, 4)), index=idx, columns=list("ABCD"))


@pytest.fixture
def lopsided_returns():
    """One asset three times as volatile as the others."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=600)
    data = rng.normal(0.0003, 0.010, (600, 4))
    data[:, 0] *= 3.0
    return pd.DataFrame(data, index=idx, columns=list("ABCD"))


# ---------------------------------------------------------------- basics
def test_attribution_covers_every_held_position(uncorrelated_returns):
    w = {"A": 0.4, "B": 0.3, "C": 0.3}
    out = compute_position_attribution(w, uncorrelated_returns)
    assert set(out) == set(w)


def test_zero_weight_positions_are_excluded(uncorrelated_returns):
    out = compute_position_attribution({"A": 0.5, "B": 0.5, "C": 0.0}, uncorrelated_returns)
    assert "C" not in out


def test_empty_weights_return_empty(uncorrelated_returns):
    assert compute_position_attribution({}, uncorrelated_returns) == {}


def test_unknown_tickers_are_ignored(uncorrelated_returns):
    out = compute_position_attribution({"ZZZ": 1.0}, uncorrelated_returns)
    assert out == {}


# ---------------------------------------------------------------- risk
def test_equal_weight_uncorrelated_gives_equal_risk_shares(uncorrelated_returns):
    """Analytic case: identical uncorrelated assets each carry 1/N of risk."""
    w = dict.fromkeys("ABCD", 0.25)
    out = compute_position_attribution(w, uncorrelated_returns)
    shares = [out[t]["pct_of_portfolio_risk"] for t in "ABCD"]
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    assert all(s == pytest.approx(0.25, abs=0.05) for s in shares)


def test_risk_shares_always_sum_to_one(lopsided_returns):
    w = {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2}
    out = compute_position_attribution(w, lopsided_returns)
    assert sum(v["pct_of_portfolio_risk"] for v in out.values()) == pytest.approx(1.0, abs=1e-6)


def test_volatile_asset_carries_more_risk_than_its_weight(lopsided_returns):
    """The motivating case for reporting risk share at all."""
    w = dict.fromkeys("ABCD", 0.25)
    out = compute_position_attribution(w, lopsided_returns)
    assert out["A"]["pct_of_portfolio_risk"] > 0.25
    assert out["A"]["risk_vs_weight_gap"] > 0.0
    assert out["B"]["risk_vs_weight_gap"] < 0.0


def test_standalone_volatility_matches_direct_calculation(lopsided_returns):
    from finrisk.risk import metrics as m

    w = dict.fromkeys("ABCD", 0.25)
    out = compute_position_attribution(w, lopsided_returns)
    direct = m.annualized_volatility(lopsided_returns["A"])
    # Shrunk covariance pulls the estimate toward the mean, so allow a band.
    assert out["A"]["standalone_volatility"] == pytest.approx(direct, rel=0.25)


# ---------------------------------------------------------------- diversification
def test_uncorrelated_book_shows_high_diversification(uncorrelated_returns):
    w = dict.fromkeys("ABCD", 0.25)
    out = compute_position_attribution(w, uncorrelated_returns)
    for t in "ABCD":
        assert out[t]["diversification_benefit"] > 0.4
        assert abs(out[t]["avg_correlation_to_book"]) < 0.2


def test_perfectly_correlated_book_shows_no_diversification():
    """If everything is the same asset, correlation absorbs nothing."""
    idx = pd.bdate_range("2020-01-01", periods=400)
    base = np.random.default_rng(3).normal(0.0004, 0.011, 400)
    rets = pd.DataFrame({"A": base, "B": base, "C": base}, index=idx)
    out = compute_position_attribution(dict.fromkeys("ABC", 1 / 3), rets)
    for t in "ABC":
        assert out[t]["diversification_benefit"] < 0.15
        assert out[t]["avg_correlation_to_book"] > 0.9


def test_single_position_has_no_correlation_to_book(uncorrelated_returns):
    out = compute_position_attribution({"A": 1.0}, uncorrelated_returns)
    assert out["A"]["avg_correlation_to_book"] == 0.0
    assert out["A"]["pct_of_portfolio_risk"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- alpha
def test_alpha_contribution_is_weight_times_forecast(uncorrelated_returns):
    w = {"A": 0.5, "B": 0.5}
    alphas = {"A": 0.08, "B": 0.02}
    out = compute_position_attribution(w, uncorrelated_returns, alphas=alphas)
    assert out["A"]["alpha_contribution"] == pytest.approx(0.04)
    assert out["B"]["alpha_contribution"] == pytest.approx(0.01)


def test_alpha_shares_sum_to_one(uncorrelated_returns):
    out = compute_position_attribution(
        dict.fromkeys("ABCD", 0.25),
        uncorrelated_returns,
        alphas={"A": 0.08, "B": 0.02, "C": 0.05, "D": 0.01},
    )
    assert sum(v["pct_of_portfolio_alpha"] for v in out.values()) == pytest.approx(1.0)


def test_missing_alphas_do_not_break_attribution(uncorrelated_returns):
    out = compute_position_attribution({"A": 1.0}, uncorrelated_returns)
    assert out["A"]["alpha_contribution"] == 0.0
    assert out["A"]["pct_of_portfolio_alpha"] == 0.0


# ---------------------------------------------------------------- tail
def test_tail_shares_sum_to_one(lopsided_returns):
    out = compute_position_attribution(dict.fromkeys("ABCD", 0.25), lopsided_returns)
    total = sum(v["pct_of_tail_loss"] for v in out.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_volatile_asset_dominates_the_tail(lopsided_returns):
    out = compute_position_attribution(dict.fromkeys("ABCD", 0.25), lopsided_returns)
    assert out["A"]["pct_of_tail_loss"] > 0.4


# ---------------------------------------------------------------- confidence
def test_confidence_rises_with_coverage_agreement_and_evidence():
    strong = {"coverage": 1.0, "momentum": 1.2, "value": 0.9, "quality": 0.7, "risk": 0.5}
    weak = {"coverage": 0.3, "momentum": 1.2, "value": -0.9, "quality": 0.7, "risk": -0.5}
    attrib = {"diversification_benefit": 0.5}
    hi, _ = compute_confidence("A", strong, attrib, n_citations=3)
    lo, _ = compute_confidence("A", weak, attrib, n_citations=0)
    assert hi > lo
    assert 0.0 <= lo <= hi <= 1.0


def test_confidence_is_bounded_and_explained():
    score, basis = compute_confidence("A", {"coverage": 1.0}, {"diversification_benefit": 1.0}, 10)
    assert 0.0 <= score <= 1.0
    assert basis and any("coverage" in b for b in basis)


def test_confidence_never_set_by_narrative():
    """Confidence must come from measurable inputs; with none it stays low."""
    score, _ = compute_confidence("A", None, {}, 0)
    assert score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- explanation
def test_explanation_reports_contradicting_evidence(lopsided_returns):
    out = compute_position_attribution(dict.fromkeys("ABCD", 0.25), lopsided_returns)
    exp = build_position_explanation(
        "A",
        out["A"],
        {"coverage": 1.0, "momentum": 1.5, "value": -1.2, "quality": 0.4},
        {"regime": "risk_on_low_vol", "confidence": 0.9, "risk_multiplier": 1.0},
        evidence=[],
    )
    assert exp.contradicting_evidence, "a mixed, risk-heavy position must show counterpoints"
    assert any("No supporting document evidence" in c for c in exp.contradicting_evidence)
    assert exp.drivers_positive and exp.drivers_negative


def test_explanation_populates_every_numeric_field(uncorrelated_returns):
    out = compute_position_attribution(
        {"A": 0.5, "B": 0.5}, uncorrelated_returns, alphas={"A": 0.06, "B": 0.02}
    )
    exp = build_position_explanation("A", out["A"], {"coverage": 1.0, "value": 0.8}, None, [])
    for field in (
        "alpha_contribution",
        "alpha_forecast",
        "pct_of_portfolio_alpha",
        "pct_of_portfolio_risk",
        "risk_vs_weight_gap",
        "standalone_volatility",
        "marginal_risk",
        "diversification_benefit",
        "avg_correlation_to_book",
    ):
        assert np.isfinite(getattr(exp, field)), field
    assert exp.confidence > 0
    assert exp.confidence_basis


def test_high_risk_gap_appears_in_the_risk_note(lopsided_returns):
    out = compute_position_attribution(dict.fromkeys("ABCD", 0.25), lopsided_returns)
    exp = build_position_explanation("A", out["A"], {"coverage": 1.0}, None, [])
    assert "risk budget" in exp.risk_note.lower()


def test_render_explanations_table(uncorrelated_returns):
    out = compute_position_attribution({"A": 0.6, "B": 0.4}, uncorrelated_returns)
    exps = [build_position_explanation(t, out[t], {"coverage": 1.0}, None, []) for t in ("A", "B")]
    text = render_explanations(exps)
    assert "Position Attribution" in text
    assert "A" in text and "B" in text
    assert "never assigned by the language model" in text


def test_render_empty():
    assert "No positions" in render_explanations([])
