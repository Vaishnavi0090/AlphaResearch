"""Backtester: schedule, execution lag, costs, drift and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.backtesting.engine import (
    WalkForwardBacktester,
    _traded_notional,
    config_digest,
    rebalance_dates,
    summarize_results,
)
from finrisk.backtesting.strategies import all_strategies


@pytest.fixture(scope="module")
def ctx_and_bt(small_bundle, small_settings):
    bt = WalkForwardBacktester(small_settings)
    return bt, bt.prepare_context(small_bundle)


# ---------------------------------------------------------------- schedule
def test_rebalance_dates_respect_warmup(small_bundle):
    dates = rebalance_dates(small_bundle.dates, "M", warmup=504)
    assert dates[0] >= small_bundle.dates[504]
    assert all(d in set(small_bundle.dates) for d in dates)


def test_rebalance_dates_are_sorted_and_unique(small_bundle):
    dates = rebalance_dates(small_bundle.dates, "M", 504)
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)


def test_rebalance_frequency_ordering(small_bundle):
    w = rebalance_dates(small_bundle.dates, "W", 504)
    m = rebalance_dates(small_bundle.dates, "M", 504)
    q = rebalance_dates(small_bundle.dates, "Q", 504)
    assert len(w) > len(m) > len(q)


def test_rebalance_dates_excludes_final_date(small_bundle):
    dates = rebalance_dates(small_bundle.dates, "M", 504)
    assert dates[-1] < small_bundle.dates[-1]


def test_rebalance_dates_rejects_short_history(small_bundle):
    with pytest.raises(ValueError):
        rebalance_dates(small_bundle.dates[:100], "M", warmup=504)


def test_unsupported_frequency_rejected(small_bundle):
    with pytest.raises(ValueError):
        rebalance_dates(small_bundle.dates, "D", 504)


def test_execution_lag_cannot_be_zero():
    from finrisk.config.settings import BacktestConfig

    with pytest.raises(ValueError):
        BacktestConfig(execution_lag_days=0)


# ---------------------------------------------------------------- mechanics
def test_execution_strictly_follows_decision(small_bundle, ctx_and_bt):
    bt, ctx = ctx_and_bt
    res = bt.run(small_bundle, all_strategies()["B_factor_equal_weight"], ctx)
    assert res.events
    for e in res.events:
        assert pd.Timestamp(e.execution_date) > pd.Timestamp(e.decision_date)


def test_all_weights_within_caps(small_bundle, small_settings, ctx_and_bt):
    bt, ctx = ctx_and_bt
    res = bt.run(small_bundle, all_strategies()["F_finrisk_full"], ctx)
    cap = small_settings.portfolio.max_weight
    for e in res.events:
        assert not e.weights or max(e.weights.values()) <= cap + 1e-6
        assert all(w >= -1e-9 for w in e.weights.values())
        assert sum(e.weights.values()) <= small_settings.portfolio.max_leverage + 1e-6


def test_turnover_is_bounded(small_bundle, ctx_and_bt):
    bt, ctx = ctx_and_bt
    res = bt.run(small_bundle, all_strategies()["B_factor_equal_weight"], ctx)
    for e in res.events:
        assert 0.0 <= e.turnover <= 1.0 + 1e-9


def test_traded_notional_is_twice_one_way_turnover():
    prev, new = {"A": 1.0}, {"B": 1.0}
    assert _traded_notional(prev, new) == pytest.approx(2.0)


def test_costs_reduce_performance(small_bundle, small_settings, ctx_and_bt):
    """The core cost-model test: identical signals, different cost assumption."""
    _, ctx = ctx_and_bt
    free = small_settings.model_copy(deep=True)
    free.backtest.costs.commission_bps = 0.0
    free.backtest.costs.spread_bps = 0.0
    free.backtest.costs.slippage_bps = 0.0
    expensive = small_settings.model_copy(deep=True)
    expensive.backtest.costs.commission_bps = 50.0
    expensive.backtest.costs.spread_bps = 50.0
    expensive.backtest.costs.slippage_bps = 0.0

    strat = all_strategies()["B_factor_equal_weight"]
    r_free = WalkForwardBacktester(free).run(small_bundle, strat, ctx)
    r_exp = WalkForwardBacktester(expensive).run(small_bundle, strat, ctx)
    assert r_exp.metrics.cagr < r_free.metrics.cagr
    assert r_exp.metrics.total_cost_drag > r_free.metrics.total_cost_drag
    assert r_free.metrics.total_cost_drag == pytest.approx(0.0)


def test_zero_turnover_strategy_pays_no_cost(small_bundle, small_settings):
    bt = WalkForwardBacktester(small_settings)
    res = bt.run(small_bundle, all_strategies()["benchmark_buy_hold"])
    assert res.metrics.total_cost_drag == pytest.approx(0.0)
    assert res.n_rebalances == 0


def test_weights_drift_between_rebalances(small_bundle, small_settings, ctx_and_bt):
    """Positions must NOT be held at constant weight between rebalances.
    Constant weights would imply free continuous rebalancing."""
    bt, ctx = ctx_and_bt
    res = bt.run(small_bundle, all_strategies()["A_equal_weight_universe"], ctx)
    # An equal-weight strategy that never drifted would show ~zero turnover;
    # drift forces real trades back to equal weight each period.
    assert res.metrics.avg_turnover > 0.001


def test_equity_curve_is_positive_and_continuous(small_bundle, ctx_and_bt):
    bt, ctx = ctx_and_bt
    res = bt.run(small_bundle, all_strategies()["F_finrisk_full"], ctx)
    curve = pd.Series(res.equity_curve)
    assert (curve > 0).all()
    assert len(curve) > 200
    assert curve.index.is_monotonic_increasing


def test_metrics_are_internally_consistent(small_bundle, ctx_and_bt):
    bt, ctx = ctx_and_bt
    res = bt.run(small_bundle, all_strategies()["E_factor_optimized_risk_constrained"], ctx)
    mt = res.metrics
    assert mt.max_drawdown <= 0.0
    assert mt.ann_volatility >= 0.0
    assert 0.0 <= mt.hit_rate <= 1.0
    assert mt.cvar_95 <= mt.var_95 + 1e-12
    assert mt.n_days > 0


def test_config_digest_is_stable_and_discriminating(small_settings):
    a = config_digest(small_settings, "X")
    b = config_digest(small_settings, "X")
    c = config_digest(small_settings, "Y")
    assert a == b and a != c
    other = small_settings.model_copy(deep=True)
    other.portfolio.max_weight = 0.11
    assert config_digest(other, "X") != a


def test_results_are_reproducible(small_bundle, small_settings, ctx_and_bt):
    bt, ctx = ctx_and_bt
    strat = all_strategies()["F_finrisk_full"]
    a = bt.run(small_bundle, strat, ctx)
    b = bt.run(small_bundle, strat, ctx)
    assert a.equity_curve == b.equity_curve
    assert a.config_digest == b.config_digest


def test_summarize_results_frame(small_bundle, ctx_and_bt):
    bt, ctx = ctx_and_bt
    results = [
        bt.run(small_bundle, all_strategies()[n], ctx)
        for n in ("A_equal_weight_universe", "B_factor_equal_weight")
    ]
    df = summarize_results(results)
    assert len(df) == 2
    for col in ("CAGR", "Sharpe", "MaxDD", "AvgTurnover", "CostDrag"):
        assert col in df.columns


# ---------------------------------------------------------------- leakage
def test_signal_timestamps_never_exceed_decision_date(small_bundle, ctx_and_bt):
    _, ctx = ctx_and_bt
    for d, scores in ctx.factor_scores.items():
        for s in scores.values():
            assert pd.Timestamp(s.as_of) <= pd.Timestamp(d)


def test_regime_timestamps_never_exceed_decision_date(ctx_and_bt):
    _, ctx = ctx_and_bt
    for d, st in ctx.regimes.items():
        assert pd.Timestamp(st.as_of) <= pd.Timestamp(d)


def test_backtester_raises_on_injected_future_signal(small_bundle, small_settings, ctx_and_bt):
    """Tamper with the context to prove the runtime assertion actually fires."""
    bt, ctx = ctx_and_bt
    import copy

    tampered = copy.deepcopy(ctx)
    d = tampered.dates[10]
    future = pd.Timestamp(d) + pd.Timedelta(days=40)
    tampered.factor_scores[d] = {
        t: s.model_copy(update={"as_of": future.date()})
        for t, s in tampered.factor_scores[d].items()
    }
    with pytest.raises(AssertionError, match="look-ahead"):
        bt.run(small_bundle, all_strategies()["B_factor_equal_weight"], tampered)


@pytest.fixture(scope="module")
def scrambled_future(small_bundle, small_settings, ctx_and_bt):
    """The same bundle with EVERYTHING after a cutoff replaced by noise.

    Prices, benchmark, risk-free rate, volumes and every filing published after
    the cutoff are overwritten. A pipeline that uses only information available
    on each decision date cannot tell the difference for decisions up to the cutoff.
    """
    import copy

    bt, ctx = ctx_and_bt
    cutoff = ctx.dates[int(len(ctx.dates) * 0.6)]
    rng = np.random.default_rng(12345)
    scr = copy.deepcopy(small_bundle)
    after = scr.prices.index > cutoff
    n = int(after.sum())
    scr.prices.loc[after, :] = 100 * np.exp(
        np.cumsum(rng.normal(0, 0.05, (n, scr.prices.shape[1])), axis=0)
    )
    scr.benchmark.loc[after] = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, n)))
    scr.risk_free.loc[after] = rng.uniform(0.0, 0.2, n)
    scr.volumes.loc[after, :] = rng.uniform(1e3, 1e7, (n, scr.volumes.shape[1]))
    published_later = pd.to_datetime(scr.fundamentals["available_from"]) > cutoff
    numeric = [
        c for c in scr.fundamentals.columns if c not in ("ticker", "period_end", "available_from")
    ]
    scr.fundamentals.loc[published_later, numeric] = rng.normal(
        5, 50, (int(published_later.sum()), len(numeric))
    )
    assert n > 100 and published_later.any(), "the scramble must actually change something"

    # Signals are only built for decision dates up to the cutoff; the scrambled
    # future is still sitting in the bundle for any leak to find.
    early = [d for d in ctx.dates if d <= cutoff]
    return cutoff, scr, bt.prepare_context(scr, early)


def test_signals_ignore_the_future(ctx_and_bt, scrambled_future):
    """Factor scores and regimes at every decision date <= cutoff are unchanged
    when all later data is replaced by noise."""
    _, honest = ctx_and_bt
    _, _, scrambled = scrambled_future
    checked = 0
    for d in scrambled.dates:
        a = {t: s.composite for t, s in honest.factor_scores[d].items()}
        b = {t: s.composite for t, s in scrambled.factor_scores[d].items()}
        assert a == pytest.approx(b, abs=1e-12), f"factor scores at {d.date()} depend on the future"
        ra, rb = honest.regimes.get(d), scrambled.regimes.get(d)
        assert (ra is None) == (rb is None)
        if ra is not None:
            assert ra.regime == rb.regime
            assert ra.confidence == pytest.approx(rb.confidence, abs=1e-9)
        checked += 1
    assert checked > 20


@pytest.mark.parametrize("name", ["B_factor_equal_weight", "F_finrisk_full"])
def test_decisions_ignore_the_future(small_bundle, ctx_and_bt, scrambled_future, name):
    """Independent look-ahead test, complementary to the shifted-signal placebo.

    The placebo shows the pipeline WOULD benefit from peeking. This shows it does
    not peek: portfolio weights decided on or before the cutoff are identical when
    everything after it is random noise. Covers the factor-only rung (B) and the
    full pipeline (regime overlay, covariance estimation, optimiser, constraints).
    """
    bt, ctx = ctx_and_bt
    cutoff, scr, scr_ctx = scrambled_future
    strat = all_strategies()[name]
    honest = {
        e.decision_date: e
        for e in bt.run(small_bundle, strat, ctx).events
        if pd.Timestamp(e.decision_date) <= cutoff
    }
    scrambled = {e.decision_date: e for e in bt.run(scr, strat, scr_ctx).events}
    assert set(honest) <= set(scrambled) and len(honest) > 20
    for d, event in honest.items():
        w1, w2 = event.weights, scrambled[d].weights
        for t in set(w1) | set(w2):
            assert w1.get(t, 0.0) == pytest.approx(
                w2.get(t, 0.0), abs=1e-9
            ), f"{name}: weight of {t} decided on {d} depends on post-cutoff data"


def test_ic_estimates_are_clipped(ctx_and_bt):
    _, ctx = ctx_and_bt
    assert all(0.0 <= v <= 0.15 for v in ctx.ic_estimates.values())


def test_all_strategies_run_without_error(small_bundle, ctx_and_bt):
    bt, ctx = ctx_and_bt
    for name, strat in all_strategies().items():
        res = bt.run(small_bundle, strat, ctx)
        assert res.metrics.n_days > 0, name
        assert np.isfinite(res.metrics.cagr), name
