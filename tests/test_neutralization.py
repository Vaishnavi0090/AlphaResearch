"""Factor neutralisation: proofs that exposures are actually removed.

Neutral construction is a *claim*. These tests are the evidence. Each one
measures a realised exposure rather than trusting the construction, and the
tolerances are stated explicitly so a regression shows up as a number rather
than as a vague failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.factors.neutralization import (
    beta_neutralize,
    build_neutral_portfolio,
    estimate_betas,
    exposure_report,
    neutral_vs_raw,
    neutralize,
    render_neutrality_report,
    sector_dummies,
    sector_neutralize,
)


@pytest.fixture
def biased_cross_section():
    """Scores deliberately contaminated with a sector tilt AND a beta tilt, so
    neutralisation has something real to remove."""
    rng = np.random.default_rng(0)
    tickers = [f"A{i}" for i in range(30)]
    sectors = {t: ["Tech", "Fin", "Health"][i % 3] for i, t in enumerate(tickers)}
    betas = pd.Series(rng.uniform(0.5, 1.8, 30), index=tickers)
    raw = (
        rng.normal(0, 1, 30)
        + np.array([1.5 if sectors[t] == "Tech" else 0.0 for t in tickers])
        + 0.8 * betas.to_numpy()
    )
    return pd.Series(raw, index=tickers), sectors, betas


# ---------------------------------------------------------------- dummies
def test_sector_dummies_are_one_hot(biased_cross_section):
    _, sectors, _ = biased_cross_section
    D = sector_dummies(list(sectors), sectors)
    assert (D.sum(axis=1) == 1.0).all()
    assert set(D.columns) == {"Tech", "Fin", "Health"}


def test_unmapped_ticker_becomes_unknown_not_dropped():
    D = sector_dummies(["A", "B"], {"A": "Tech"})
    assert len(D) == 2
    assert D.loc["B", "Unknown"] == 1.0


# ---------------------------------------------------------------- sector
def test_sector_neutralization_zeroes_sector_means(biased_cross_section):
    scores, sectors, _ = biased_cross_section
    grp = pd.Series(sectors)
    raw_spread = scores.groupby(grp).mean().abs().max()
    out = sector_neutralize(scores, sectors)
    assert raw_spread > 0.5, "fixture should have a real sector tilt"
    assert out.groupby(grp).mean().abs().max() < 1e-9


def test_sector_neutralization_preserves_within_sector_ranking(biased_cross_section):
    scores, sectors, _ = biased_cross_section
    out = sector_neutralize(scores, sectors)
    tech = [t for t, s in sectors.items() if s == "Tech"]
    assert scores[tech].rank().equals(out[tech].rank())


def test_singleton_sector_is_not_zeroed():
    """Demeaning a one-member sector would destroy its signal entirely."""
    scores = pd.Series({"A": 2.0, "B": -1.0, "C": 0.5})
    out = sector_neutralize(scores, {"A": "Solo", "B": "Pair", "C": "Pair"})
    assert out["A"] != 0.0


# ---------------------------------------------------------------- beta
def test_beta_neutralization_removes_beta_correlation(biased_cross_section):
    scores, _, betas = biased_cross_section
    assert abs(scores.corr(betas)) > 0.1, "fixture should have a real beta tilt"
    out = beta_neutralize(scores, betas)
    assert abs(out.corr(betas)) < 1e-6


def test_estimate_betas_recovers_known_beta():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0, 0.01, 500))
    rets = pd.DataFrame({"HIGH": 1.8 * bench, "LOW": 0.4 * bench})
    est = estimate_betas(rets, bench)
    assert est["HIGH"] == pytest.approx(1.8, abs=1e-6)
    assert est["LOW"] == pytest.approx(0.4, abs=1e-6)


def test_estimate_betas_nan_on_short_history():
    bench = pd.Series(np.random.default_rng(2).normal(0, 0.01, 10))
    est = estimate_betas(pd.DataFrame({"A": bench}), bench, min_obs=60)
    assert np.isnan(est["A"])


# ---------------------------------------------------------------- general
def test_residual_is_orthogonal_to_every_exposure(biased_cross_section):
    scores, sectors, betas = biased_cross_section
    X = pd.concat(
        [sector_dummies(list(scores.index), sectors), pd.DataFrame({"beta": betas})],
        axis=1,
    )
    out = neutralize(scores, X)
    for col in X.columns:
        assert abs(out.corr(X[col])) < 1e-6, f"residual still loads on {col}"


def test_neutralize_preserves_nan(biased_cross_section):
    scores, _, betas = biased_cross_section
    scores = scores.copy()
    scores.iloc[3] = np.nan
    out = beta_neutralize(scores, betas)
    assert np.isnan(out.iloc[3]), "a missing signal must not become a synthetic 0"


def test_neutralize_returns_input_when_too_few_observations():
    s = pd.Series({"A": 1.0, "B": 2.0})
    out = neutralize(s, pd.DataFrame({"x": [1.0, 2.0]}, index=["A", "B"]))
    pd.testing.assert_series_equal(out, s)


def test_neutralize_survives_collinear_exposures(biased_cross_section):
    """Sector dummies plus an intercept are exactly collinear."""
    scores, sectors, _ = biased_cross_section
    D = sector_dummies(list(scores.index), sectors)
    X = pd.concat([D, D.iloc[:, [0]].rename(columns={D.columns[0]: "dup"})], axis=1)
    out = neutralize(scores, X)
    assert out.notna().all()
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- portfolio
def test_portfolio_is_dollar_sector_and_beta_neutral(biased_cross_section):
    scores, sectors, betas = biased_cross_section
    w = build_neutral_portfolio(scores, sectors, betas, gross_exposure=2.0)
    rep = exposure_report(w, sectors, betas)
    assert abs(rep["net_exposure"]) < 1e-8
    assert abs(rep["portfolio_beta"]) < 1e-6
    assert rep["max_abs_sector_exposure"] < 1e-6
    assert rep["gross_exposure"] == pytest.approx(2.0, abs=1e-6)


def test_portfolio_holds_both_longs_and_shorts(biased_cross_section):
    scores, sectors, betas = biased_cross_section
    w = build_neutral_portfolio(scores, sectors, betas)
    rep = exposure_report(w, sectors, betas)
    assert rep["n_long"] >= 5 and rep["n_short"] >= 5


@pytest.mark.parametrize("cap", [0.20, 0.15, 0.08])
def test_capping_preserves_neutrality(biased_cross_section, cap):
    """Regression test for a real bug: capping positions reintroduced the
    imbalance that neutralisation removed (measured residual beta ~1e-2).
    Alternating projection between the cap and the neutral subspace fixes it."""
    scores, sectors, betas = biased_cross_section
    w = build_neutral_portfolio(scores, sectors, betas, 2.0, max_weight=cap)
    rep = exposure_report(w, sectors, betas)
    assert rep["max_abs_weight"] <= cap + 1e-9
    assert abs(rep["net_exposure"]) < 1e-5
    assert abs(rep["portfolio_beta"]) < 1e-4, f"cap={cap} broke beta neutrality"
    assert rep["max_abs_sector_exposure"] < 1e-4


def test_gross_exposure_is_honoured(biased_cross_section):
    scores, sectors, betas = biased_cross_section
    for gross in (1.0, 2.0, 3.0):
        w = build_neutral_portfolio(scores, sectors, betas, gross_exposure=gross)
        assert w.abs().sum() == pytest.approx(gross, abs=1e-6)


def test_sector_only_neutral_leaves_beta_exposure(biased_cross_section):
    """The contrast case: neutralising sectors alone does NOT make the book
    beta-neutral. Without this, 'neutral' could be claimed too broadly."""
    scores, sectors, betas = biased_cross_section
    w = build_neutral_portfolio(scores, sectors, None)
    assert abs(exposure_report(w, sectors, betas)["portfolio_beta"]) > 1e-3


def test_degenerate_inputs_return_empty():
    assert build_neutral_portfolio(pd.Series({"A": 1.0})).empty
    assert build_neutral_portfolio(pd.Series(dtype=float)).empty


def test_constant_scores_produce_no_position():
    w = build_neutral_portfolio(pd.Series({f"A{i}": 1.0 for i in range(8)}))
    assert w.abs().sum() < 1e-9


# ---------------------------------------------------------------- comparison
def test_neutral_vs_raw_reports_exposures(small_bundle, small_settings):
    from finrisk.factors.engine import FactorEngine

    fe = FactorEngine(small_settings.factors)
    dates = (
        pd.Series(small_bundle.dates)
        .groupby([small_bundle.dates.year, small_bundle.dates.month])
        .last()
        .tolist()
    )
    dates = [d for d in dates if d >= small_bundle.dates[504]][:24]
    panel = fe.composite_series(small_bundle, dates)
    out = neutral_vs_raw(
        panel, small_bundle.prices, small_bundle.sector_map(), small_bundle.benchmark
    )
    assert set(out.index) == {"raw", "sector_neutral", "sector_beta_neutral"}
    # The neutral variants must show materially lower realised exposures.
    assert (
        out.loc["sector_beta_neutral", "mean_abs_portfolio_beta"]
        < out.loc["raw", "mean_abs_portfolio_beta"]
    )
    assert (
        out.loc["sector_neutral", "mean_abs_sector_exposure"]
        < out.loc["raw", "mean_abs_sector_exposure"]
    )
    assert "Factor Neutralisation Comparison" in render_neutrality_report(out)
