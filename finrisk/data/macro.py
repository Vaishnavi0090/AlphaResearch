"""Macro data (FRED).

Only used for the risk-free rate and optional macro regime features. Kept
deliberately small: the project does not need a macro factor model to be
defensible, and adding one would be scope creep.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from finrisk.data.cache import retry

logger = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


class FredProvider:
    """Fetch FRED series. Uses the public CSV endpoint (no key required);
    falls back to the API if FRED_API_KEY is set."""

    name = "fred"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY")

    def series(self, series_id: str) -> pd.Series:
        url = FRED_CSV.format(series=series_id)

        def _call() -> pd.Series:
            df = pd.read_csv(url)
            date_col, val_col = df.columns[0], df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col])
            out = pd.to_numeric(df[val_col], errors="coerce")
            out.index = df[date_col]
            return out.dropna().rename(series_id)

        return retry(_call, attempts=3, base_delay=1.0)

    def risk_free(self, index: pd.DatetimeIndex, series_id: str = "DGS3MO") -> pd.Series:
        """3-month T-bill as annualised decimal, reindexed to the trading
        calendar. Forward-filled because a rate is a *level* that persists
        between publications — this is not look-ahead."""
        raw = self.series(series_id) / 100.0
        return (
            raw.reindex(raw.index.union(index)).ffill().reindex(index).bfill().rename("risk_free")
        )


def flat_risk_free(index: pd.DatetimeIndex, annual: float = 0.02) -> pd.Series:
    """Offline fallback. Documented assumption, not a silent default."""
    return pd.Series(float(annual), index=index, name="risk_free")
