"""Factor construction, normalisation, sign conventions and signal recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.factors.definitions import (
    FACTOR_REGISTRY,
    GROUPS,
    _tail_return,
    factors_in_group,
    momentum_factors,
    risk_factors,
)
from finrisk.factors.engine import (
    FactorEngine,
    winsorize_cross_section,
    zscore_cross_section,
)
from finrisk.factors.research import (
    ic_summary,
    information_coefficient,
    quantile_portfolios,
    study_factor,
)


# ---------------------------------------------------------------- registry
def test_every_factor_belongs_to_a_known_group():
    for spec in FACTOR_REGISTRY.values():
        assert spec.group in GROUPS


def test_groups_are_non_empty():
    for g in ("momentum", "value", "quality", "risk", "market"):
        assert factors_in_group(g)


def test_all_factors_declared_higher_is_better():
    """The sign convention is the whole safety property: a factor stored with
    the wrong sign silently inverts the strategy."""
    assert all(s.higher_is_better for s in FACTOR_REGISTRY.values())


# ---------------------------------------------------------------- primitives
def test_tail_return_computes_window():
    px = pd.DataFrame({"A": np.linspace(100, 200, 300)})
    r = _tail_return(px, 100)
    assert r["A"] == pytest.approx(px["A"].iloc[-1] / px["A"].iloc[-101] - 1)


def test_tail_return_nan_when_insufficient_history():
    px = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    assert np.isnan(_tail_return(px, 100)["A"])


def test_tail_return_respects_skip():
    px = pd.DataFrame({"A": np.arange(1, 301, dtype=float)})
    r = _tail_return(px, 50, skip=21)
    assert r["A"] == pytest.approx(px["A"].iloc[-22] / px["A"].iloc[-72] - 1)


def test_momentum_positive_for_uptrend():
    px = pd.DataFrame({"UP": np.linspace(100, 300, 400), "DOWN": np.linspace(300, 100, 400)})
    f = momentum_factors(px, [21, 63, 126, 252])
    for name in ("mom_1m", "mom_3m", "mom_6m", "mom_12m_1m"):
        assert f[name]["UP"] > 0 > f[name]["DOWN"]


def test_px_to_ma200_positive_above_average():
    px = pd.DataFrame({"A": np.linspace(100, 200, 300)})
    assert momentum_factors(px, [21])["px_to_ma200"]["A"] > 0


def test_low_volatility_factor_is_negated():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({"CALM": rng.normal(0, 0.002, 300), "WILD": rng.normal(0, 0.03, 300)})
    f = risk_factors(rets)
    # higher (less negative) score = calmer asset
    assert f["low_volatility"]["CALM"] > f["low_volatility"]["WILD"]
    assert f["low_volatility"]["WILD"] < 0


def test_small_drawdown_is_negative_or_zero():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame({"A": rng.normal(0, 0.01, 300)})
    assert risk_factors(rets)["small_drawdown"]["A"] <= 0


# ---------------------------------------------------------------- normalisation
def test_zscore_has_zero_mean_unit_std():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore_cross_section(s)
    assert z.mean() == pytest.approx(0.0)
    assert z.std(ddof=1) == pytest.approx(1.0)


def test_zscore_handles_constant_series():
    z = zscore_cross_section(pd.Series([3.0] * 5))
    assert (z == 0.0).all()


def test_zscore_preserves_nan():
    z = zscore_cross_section(pd.Series([1.0, 2.0, np.nan, 4.0]))
    assert pd.isna(z.iloc[2])


def test_winsorize_clips_outlier():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
    w = winsorize_cross_section(s, 0.2)
    assert w.max() < 1000.0


def test_winsorize_noop_for_tiny_cross_section():
    s = pd.Series([1.0, 500.0])
    assert winsorize_cross_section(s, 0.2).max() == 500.0


def test_winsorize_reduces_zscore_of_outlier():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    raw = zscore_cross_section(s).abs().max()
    wins = zscore_cross_section(winsorize_cross_section(s, 0.15)).abs().max()
    assert wins < raw


# ---------------------------------------------------------------- engine
def test_raw_panel_has_expected_factors(small_bundle, small_settings):
    fe = FactorEngine(small_settings.factors)
    panel = fe.raw_panel(small_bundle, small_bundle.dates[-1])
    assert not panel.empty
    for expected in ("mom_12m_1m", "low_volatility", "low_beta", "earnings_yield", "roe"):
        assert expected in panel.columns


def test_compute_returns_scores_and_ranks(small_bundle, small_settings):
    fe = FactorEngine(small_settings.factors)
    scores, diag = fe.compute(small_bundle, small_bundle.dates[-1])
    assert scores and not diag.empty
    ranks = sorted(s.rank for s in scores.values())
    assert ranks == list(range(1, len(scores) + 1))
    assert diag["composite"].is_monotonic_decreasing


def test_ranks_are_consistent_with_composite(small_bundle, small_settings):
    fe = FactorEngine(small_settings.factors)
    scores, _ = fe.compute(small_bundle, small_bundle.dates[-1])
    best = min(scores.values(), key=lambda s: s.rank)
    assert best.composite == max(s.composite for s in scores.values())


def test_percentile_helper(small_bundle, small_settings):
    fe = FactorEngine(small_settings.factors)
    scores, _ = fe.compute(small_bundle, small_bundle.dates[-1])
    best = min(scores.values(), key=lambda s: s.rank)
    assert best.percentile == pytest.approx(100.0)


def test_composite_weights_are_honoured(small_bundle, small_settings):
    """A 100% momentum composite must rank identically to the momentum group."""
    cfg = small_settings.factors.model_copy(update={"composite_weights": {"momentum": 1.0}})
    fe = FactorEngine(cfg)
    _, diag = fe.compute(small_bundle, small_bundle.dates[-1])
    assert diag["composite"].corr(diag["grp_momentum"]) == pytest.approx(1.0, abs=1e-9)


def test_missing_factor_becomes_neutral_not_dropped(small_bundle, small_settings):
    """Assets with no fundamentals must remain in the cross-section."""
    b = small_bundle.slice(None, None)
    b.fundamentals = b.fundamentals.iloc[0:0]
    fe = FactorEngine(small_settings.factors)
    scores, diag = fe.compute(b, b.dates[-1])
    assert len(scores) == small_bundle.prices.shape[1]
    assert "grp_value" not in diag.columns  # no value data at all
    assert diag["composite"].notna().all()


def test_coverage_reflects_available_factors(small_bundle, small_settings):
    fe = FactorEngine(small_settings.factors)
    scores, _ = fe.compute(small_bundle, small_bundle.dates[-1])
    assert all(0.0 < s.coverage <= 1.0 for s in scores.values())


def test_normalisation_is_per_date_not_pooled(small_bundle, small_settings):
    fe = FactorEngine(small_settings.factors)
    for d in small_bundle.dates[[-1, -300]]:
        normed = fe.normalize(fe.raw_panel(small_bundle, d))
        assert normed.mean().abs().max() < 1e-10


# ---------------------------------------------------------------- research
def test_information_coefficient_perfect_signal():
    dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
    cols = list("ABCDEFGH")
    scores = pd.DataFrame(
        [np.arange(8, dtype=float), np.arange(8, dtype=float)], index=dates, columns=cols
    )
    fwd = scores.copy()  # forward return exactly equals the score
    ic = information_coefficient(scores, fwd)
    assert ic.min() == pytest.approx(1.0)


def test_information_coefficient_inverted_signal():
    dates = pd.to_datetime(["2020-01-31"])
    cols = list("ABCDEFGH")
    scores = pd.DataFrame([np.arange(8, dtype=float)], index=dates, columns=cols)
    ic = information_coefficient(scores, -scores)
    assert ic.iloc[0] == pytest.approx(-1.0)


def test_ic_summary_reports_ir_and_tstat():
    ic = pd.Series([0.05, 0.03, 0.07, 0.01, 0.04] * 10)
    s = ic_summary(ic)
    assert s["mean_ic"] == pytest.approx(ic.mean())
    assert s["ic_ir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
    assert s["t_stat"] > 0 and s["n"] == 50


def test_ic_summary_empty():
    assert ic_summary(pd.Series(dtype=float))["n"] == 0


def test_quantile_portfolios_partition_universe():
    dates = pd.to_datetime(["2020-01-31"])
    cols = [f"A{i}" for i in range(20)]
    scores = pd.DataFrame([np.arange(20, dtype=float)], index=dates, columns=cols)
    buckets = quantile_portfolios(scores, 5)
    counts = sum(int(b.loc[dates[0]].sum()) for b in buckets.values())
    assert counts == 20
    # top bucket holds the highest scores
    assert buckets[5].loc[dates[0], "A19"]
    assert buckets[1].loc[dates[0], "A0"]


def test_study_factor_produces_complete_report(small_bundle, small_settings):
    """The study must return a fully populated FactorPerformance, including the
    long-short leg and the IC decay curve."""
    fe = FactorEngine(small_settings.factors)
    dates = (
        pd.Series(small_bundle.dates)
        .groupby([small_bundle.dates.year, small_bundle.dates.month])
        .last()
        .tolist()
    )
    dates = [d for d in dates if d >= small_bundle.dates[504]]
    panel = fe.composite_series(small_bundle, dates)
    perf = study_factor("composite", panel, small_bundle.prices)
    assert perf.long_short_ann_return is not None
    assert perf.n_rebalances == len(panel)
    assert 0.0 <= perf.turnover <= 1.0
    assert set(perf.ic_by_horizon) == {"5d", "21d", "63d", "126d", "252d"}
    assert perf.max_drawdown <= 0.0
