"""Data ingestion, normalisation and point-in-time discipline."""

from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from finrisk.data.fundamentals import stamp_availability
from finrisk.data.market_data import (
    drop_sparse_columns,
    flag_abnormal_returns,
    limited_ffill,
    normalize_index,
)
from finrisk.data.synthetic import SyntheticProvider
from finrisk.data.validation import (
    assert_no_lookahead_in_frame,
    check_monotone_availability,
    validate_bundle,
)


def test_normalize_index_strips_timezone_and_sorts():
    # Fixed offset rather than a named zone: named zones need the `tzdata`
    # package, which is not a dependency of this project.
    eastern = timezone(timedelta(hours=-5))
    idx = pd.to_datetime(["2020-01-03 16:00", "2020-01-02 16:00"]).tz_localize(eastern)
    df = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    out = normalize_index(df)
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    assert (out.index == out.index.normalize()).all()


def test_normalize_index_keeps_last_duplicate():
    idx = pd.to_datetime(["2020-01-02", "2020-01-02"])
    df = pd.DataFrame({"A": [1.0, 99.0]}, index=idx)
    out = normalize_index(df)
    assert len(out) == 1
    assert out["A"].iloc[0] == 99.0  # corrected republished bar wins


def test_limited_ffill_does_not_fill_long_gaps():
    df = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0]})
    out = limited_ffill(df, limit=2)
    assert out["A"].iloc[1] == 1.0
    assert out["A"].iloc[2] == 1.0
    assert pd.isna(out["A"].iloc[3])  # a dead stock must NOT keep trading


def test_limited_ffill_zero_limit_is_noop():
    df = pd.DataFrame({"A": [1.0, np.nan]})
    assert pd.isna(limited_ffill(df, limit=0)["A"].iloc[1])


def test_drop_sparse_columns():
    df = pd.DataFrame({"good": [1.0] * 10, "bad": [1.0] + [np.nan] * 9})
    out = drop_sparse_columns(df, min_coverage=0.6)
    assert list(out.columns) == ["good"]


def test_flag_abnormal_returns_finds_bad_tick():
    df = pd.DataFrame({"A": [100.0, 101.0, 500.0, 102.0]})
    flags = flag_abnormal_returns(df, threshold=0.5)
    assert len(flags) == 2  # spike up and back down
    assert set(flags["ticker"]) == {"A"}


def test_flag_abnormal_returns_empty_when_clean():
    df = pd.DataFrame({"A": [100.0, 101.0, 102.0]})
    assert flag_abnormal_returns(df, 0.5).empty


def test_stamp_availability_adds_lag():
    df = pd.DataFrame({"period_end": ["2023-03-31"], "value": [1.0]})
    out = stamp_availability(df, lag_days=90)
    assert out["available_from"].iloc[0] == pd.Timestamp("2023-06-29")


def test_stamp_availability_clamps_impossible_dates():
    df = pd.DataFrame(
        {"period_end": ["2023-03-31"], "available_from": ["2023-01-01"], "value": [1.0]}
    )
    out = stamp_availability(df)
    assert out["available_from"].iloc[0] >= out["period_end"].iloc[0]


def test_synthetic_provider_is_deterministic():
    a = SyntheticProvider(seed=5, n_assets=6).fetch(None, "2015-01-01", "2018-12-31")
    b = SyntheticProvider(seed=5, n_assets=6).fetch(None, "2015-01-01", "2018-12-31")
    pd.testing.assert_frame_equal(a.prices, b.prices)
    pd.testing.assert_series_equal(a.benchmark, b.benchmark)


def test_synthetic_provider_seed_changes_data():
    a = SyntheticProvider(seed=5, n_assets=6).fetch(None, "2015-01-01", "2018-12-31")
    b = SyntheticProvider(seed=6, n_assets=6).fetch(None, "2015-01-01", "2018-12-31")
    assert not a.prices.equals(b.prices)


def test_synthetic_prices_are_positive(small_bundle):
    assert (small_bundle.prices > 0).all().all()


def test_fundamentals_asof_never_returns_future_filings(small_bundle):
    for d in small_bundle.dates[::400]:
        got = small_bundle.fundamentals_asof(d)
        if len(got):
            assert (pd.to_datetime(got["available_from"]) <= d).all()


def test_fundamentals_asof_returns_one_row_per_ticker(small_bundle):
    got = small_bundle.fundamentals_asof(small_bundle.dates[-1])
    assert got.index.is_unique


def test_fundamentals_asof_empty_before_first_filing(small_bundle):
    early = small_bundle.dates[0]
    assert len(small_bundle.fundamentals_asof(early)) == 0


def test_slice_filters_fundamentals_by_availability(small_bundle):
    mid = small_bundle.dates[len(small_bundle.dates) // 2]
    sub = small_bundle.slice(None, mid)
    assert (pd.to_datetime(sub.fundamentals["available_from"]) <= mid).all()


def test_no_availability_violations_in_synthetic(small_bundle):
    assert check_monotone_availability(small_bundle.fundamentals) == []


def test_validate_bundle_clean_on_synthetic(small_bundle):
    rep = validate_bundle(small_bundle)
    assert rep.n_errors == 0
    assert rep.duplicate_timestamps == 0
    assert rep.n_assets == small_bundle.prices.shape[1]
    assert "Data Quality Report" in rep.render()


def test_validate_bundle_detects_injected_problems(small_bundle):
    bad = small_bundle.slice(None, None)
    bad.prices = bad.prices.copy()
    bad.prices.iloc[10, 0] = -5.0
    rep = validate_bundle(bad)
    assert any(i.category == "abnormal_price" for i in rep.issues)
    assert rep.n_errors >= 1


def test_validate_bundle_handles_empty():
    from finrisk.data.base import MarketDataBundle

    empty = MarketDataBundle(
        prices=pd.DataFrame(),
        volumes=pd.DataFrame(),
        benchmark=pd.Series(dtype=float),
        risk_free=pd.Series(dtype=float),
        fundamentals=pd.DataFrame(),
    )
    rep = validate_bundle(empty)
    assert rep.n_errors >= 1


def test_assert_no_lookahead_raises():
    df = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2024-01-10"]))
    with pytest.raises(AssertionError):
        assert_no_lookahead_in_frame(df, pd.Timestamp("2024-01-05"))
    assert_no_lookahead_in_frame(df, pd.Timestamp("2024-01-15"))  # no raise


def test_benchmark_aligned_to_prices(small_bundle):
    assert small_bundle.benchmark.index.equals(small_bundle.prices.index)
    assert small_bundle.risk_free.notna().all()


def test_returns_do_not_forward_fill(small_bundle):
    r = small_bundle.returns()
    assert len(r) == len(small_bundle.prices) - 1
