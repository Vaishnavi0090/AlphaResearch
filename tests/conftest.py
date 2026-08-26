"""Shared fixtures. Session-scoped so the dataset is built once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.config.settings import Settings, default_settings
from finrisk.data.market_data import align_to_calendar, load_bundle
from finrisk.data.synthetic import SyntheticProvider


@pytest.fixture(scope="session")
def small_settings() -> Settings:
    """Short, small config so the full suite runs in seconds."""
    s = default_settings()
    s.data.start = "2012-01-02"
    s.data.end = "2020-12-31"
    s.data.use_cache = False
    s.backtest.warmup = 504
    s.portfolio.top_n = 8
    return s


@pytest.fixture(scope="session")
def small_bundle(small_settings):
    provider = SyntheticProvider(seed=11, n_assets=18)
    return align_to_calendar(
        provider.fetch(None, small_settings.data.start, small_settings.data.end, "BENCH")
    )


@pytest.fixture(scope="session")
def full_settings() -> Settings:
    s = default_settings()
    s.data.use_cache = True
    return s


@pytest.fixture(scope="session")
def full_bundle(full_settings):
    return align_to_calendar(load_bundle(full_settings.data))


@pytest.fixture
def toy_returns() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=800)
    data = rng.normal(0.0004, 0.011, (800, 6))
    data[:, 0] *= 2.0
    return pd.DataFrame(data, index=idx, columns=[f"T{i}" for i in range(6)])
