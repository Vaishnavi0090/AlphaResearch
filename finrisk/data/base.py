"""Data-layer contracts.

A `MarketDataBundle` is the single object the rest of the system consumes.
Any provider (synthetic, yfinance, a vendor feed) must produce one.

Point-in-time discipline lives here: `fundamentals` carries an explicit
`available_from` column and consumers MUST filter on it rather than on the
fiscal period date. See docs/DATA_LEAKAGE_AUDIT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from finrisk.schemas import AssetMeta

FUNDAMENTAL_COLUMNS = [
    "ticker",
    "period_end",
    "available_from",
    "book_value_per_share",
    "eps",
    "ebitda_per_share",
    "free_cash_flow_per_share",
    "revenue_per_share",
    "total_debt_per_share",
    "total_equity_per_share",
    "net_income_per_share",
    "total_assets_per_share",
]


@dataclass
class MarketDataBundle:
    """All market data for a universe over a date range.

    Attributes
    ----------
    prices:
        Adjusted close prices. Index = DatetimeIndex (tz-naive, normalised to
        midnight), columns = tickers. Adjusted for splits/dividends.
    volumes:
        Share volume, same shape as prices.
    benchmark:
        Benchmark price level, indexed like `prices`.
    risk_free:
        Annualised risk-free rate as a decimal, indexed like `prices`.
    fundamentals:
        Long-format frame with FUNDAMENTAL_COLUMNS. One row per
        (ticker, period_end).
    meta:
        Per-ticker static metadata (sector etc.).
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    benchmark: pd.Series
    risk_free: pd.Series
    fundamentals: pd.DataFrame
    meta: dict[str, AssetMeta] = field(default_factory=dict)
    provider: str = "unknown"
    notes: list[str] = field(default_factory=list)
    # Only populated by the synthetic provider. Used exclusively by validation
    # tests to check that the engines recover a known effect; the factor,
    # regime, portfolio and backtest layers never read it.
    ground_truth: dict | None = None

    # -- convenience -------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)

    def returns(self, log: bool = False) -> pd.DataFrame:
        """Simple (or log) daily returns. First row dropped, not filled."""
        if log:
            import numpy as np

            return np.log(self.prices).diff().iloc[1:]
        return self.prices.pct_change().iloc[1:]

    def benchmark_returns(self) -> pd.Series:
        return self.benchmark.pct_change().iloc[1:]

    def sector_map(self) -> dict[str, str]:
        return {t: m.sector for t, m in self.meta.items()}

    def slice(self, start: date | str | None, end: date | str | None) -> MarketDataBundle:
        """Return a bundle truncated to [start, end]. Fundamentals are filtered
        on `available_from` so slicing can never reveal future filings."""
        px = self.prices.loc[start:end]
        idx_end = px.index.max() if len(px) else pd.Timestamp(end)
        fund = self.fundamentals
        if len(fund):
            fund = fund[pd.to_datetime(fund["available_from"]) <= idx_end]
        return MarketDataBundle(
            prices=px,
            volumes=self.volumes.loc[start:end],
            benchmark=self.benchmark.loc[start:end],
            risk_free=self.risk_free.loc[start:end],
            fundamentals=fund.reset_index(drop=True),
            meta=self.meta,
            provider=self.provider,
            notes=list(self.notes),
            ground_truth=self.ground_truth,
        )

    def fundamentals_asof(self, as_of: pd.Timestamp | date) -> pd.DataFrame:
        """Latest fundamental record per ticker that was *publicly available*
        on or before `as_of`. This is the only sanctioned accessor."""
        if not len(self.fundamentals):
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        ts = pd.Timestamp(as_of)
        f = self.fundamentals.copy()
        f["available_from"] = pd.to_datetime(f["available_from"])
        f = f[f["available_from"] <= ts]
        if not len(f):
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        f = f.sort_values(["ticker", "available_from", "period_end"])
        return f.groupby("ticker", as_index=False).tail(1).set_index("ticker")


@runtime_checkable
class DataProvider(Protocol):
    """Anything that can materialise a MarketDataBundle."""

    name: str

    def fetch(
        self,
        universe: list[str],
        start: str,
        end: str,
        benchmark: str,
    ) -> MarketDataBundle: ...
