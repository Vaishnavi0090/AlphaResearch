"""Market data ingestion.

Two providers:

* `SyntheticProvider` (finrisk.data.synthetic) — offline, seeded, reproducible.
* `YFinanceProvider`  — real adjusted prices from Yahoo Finance.

The normalisation helpers are pure functions so they can be unit-tested
without any network access, which is where most real ingestion bugs live
(timezones, duplicate bars, silent forward fills).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrisk.config.settings import DataConfig
from finrisk.data.base import FUNDAMENTAL_COLUMNS, MarketDataBundle
from finrisk.data.cache import DataCache, cache_key, retry
from finrisk.data.synthetic import SyntheticProvider
from finrisk.schemas import AssetMeta

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Pure normalisation helpers (unit-tested in tests/test_data.py)
# ----------------------------------------------------------------------


def normalize_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Make a price index safe: tz-naive, midnight-normalised, sorted, unique.

    Duplicate timestamps keep the LAST observation, which is the correct choice
    for vendor feeds that re-publish a corrected bar.
    """
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.normalize()
    out.index = idx
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def limited_ffill(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Forward-fill *bounded* gaps only.

    Unlimited ffill is the classic way to manufacture a dead stock that keeps
    trading at its last price forever, which flatters a backtest. We fill at
    most `limit` consecutive missing bars (exchange holidays / single-day
    outages) and leave longer gaps as NaN so downstream code drops the asset.
    """
    if limit <= 0:
        return df.copy()
    return df.ffill(limit=limit)


def drop_sparse_columns(df: pd.DataFrame, min_coverage: float = 0.60) -> pd.DataFrame:
    """Drop assets with insufficient history in the window."""
    cov = df.notna().mean()
    keep = cov[cov >= min_coverage].index
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        logger.warning("dropping sparse tickers %s", dropped)
    return df[list(keep)]


def flag_abnormal_returns(df: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
    """Locate |daily return| beyond `threshold` — candidate bad ticks or
    unadjusted corporate actions. Returned as a long frame, never auto-fixed:
    silently 'cleaning' real crash days is itself a bias."""
    r = df.pct_change()
    mask = r.abs() > threshold
    if not mask.to_numpy().any():
        return pd.DataFrame(columns=["date", "ticker", "return"])
    # dropna() is required: as of pandas 3.0 `stack()` retains NaN entries, so
    # without it every non-flagged cell would be reported as an anomaly.
    stacked = r.where(mask).stack().dropna().reset_index()
    stacked.columns = ["date", "ticker", "return"]
    return stacked


# ----------------------------------------------------------------------
# yfinance provider
# ----------------------------------------------------------------------


class YFinanceProvider:
    """Real market data via yfinance.

    Requires network access to Yahoo Finance. Not exercised in this sandbox
    (egress blocked) — the synthetic provider is used for reproducible runs.
    """

    name = "yfinance"

    def __init__(self, cache: DataCache | None = None, ffill_limit: int = 3) -> None:
        self.cache = cache
        self.ffill_limit = ffill_limit

    def _download(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "yfinance is required for provider='yfinance'. "
                "Install with: pip install yfinance"
            ) from exc

        def _call() -> pd.DataFrame:
            return yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,  # split/dividend adjusted
                progress=False,
                group_by="column",
                threads=True,
            )

        return retry(_call, attempts=4, base_delay=1.0)

    def fetch(
        self,
        universe: list[str],
        start: str,
        end: str,
        benchmark: str = "SPY",
    ) -> MarketDataBundle:
        tickers = list(dict.fromkeys([*universe, benchmark]))
        key = cache_key("yf", tickers, start, end)
        raw = (
            self.cache.frame(key, lambda: self._download(tickers, start, end))
            if self.cache
            else self._download(tickers, start, end)
        )

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        volume = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
        close = normalize_index(pd.DataFrame(close))
        volume = normalize_index(pd.DataFrame(volume))
        close = limited_ffill(close, self.ffill_limit)
        volume = volume.fillna(0.0)

        if benchmark not in close.columns:
            raise ValueError(f"benchmark {benchmark} missing from download")
        bench = close[benchmark].dropna()
        px = drop_sparse_columns(close.drop(columns=[benchmark]))
        vol = volume.reindex(columns=px.columns).reindex(index=px.index)

        rf = self._risk_free(px.index)
        meta = {t: AssetMeta(ticker=t, name=t, sector=self._sector(t)) for t in px.columns}

        return MarketDataBundle(
            prices=px,
            volumes=vol.fillna(0.0),
            benchmark=bench.reindex(px.index).ffill(),
            risk_free=rf,
            fundamentals=pd.DataFrame(columns=FUNDAMENTAL_COLUMNS),
            meta=meta,
            provider="yfinance",
            notes=[
                "yfinance adjusted closes (auto_adjust=True).",
                "Universe is a fixed current list => SURVIVORSHIP BIAS. "
                "See docs/DATA_LEAKAGE_AUDIT.md.",
                "Fundamentals must be loaded separately (finrisk.data.fundamentals).",
            ],
        )

    @staticmethod
    def _sector(ticker: str) -> str:
        """Sector lookup via yfinance metadata; 'Unknown' on failure.

        Note: yfinance returns the *current* sector, i.e. it is not
        point-in-time. Documented as a limitation rather than hidden.
        """
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).get_info()
            return str(info.get("sector") or "Unknown")
        except Exception:  # pragma: no cover - network dependent
            return "Unknown"

    @staticmethod
    def _risk_free(index: pd.DatetimeIndex) -> pd.Series:
        """Try FRED DGS3MO, else fall back to a flat assumption."""
        from finrisk.data.macro import FredProvider

        try:
            return FredProvider().risk_free(index)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("risk-free fetch failed (%s); using flat 2%%", exc)
            return pd.Series(0.02, index=index, name="risk_free")


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------


def load_bundle(cfg: DataConfig, n_assets: int | None = None) -> MarketDataBundle:
    """Materialise a bundle from config, honouring the cache.

    `n_assets` overrides `cfg.synthetic_assets` for callers that need a
    specific size without mutating config.
    """
    cache = DataCache(cfg.cache_dir, enabled=cfg.use_cache)

    if cfg.provider == "synthetic":
        n_assets = n_assets if n_assets is not None else cfg.synthetic_assets
        provider = SyntheticProvider(seed=cfg.synthetic_seed, n_assets=n_assets)
        key = cache_key("synthetic", cfg.synthetic_seed, n_assets, cfg.start, cfg.end)
        return cache.object(
            key,
            lambda: provider.fetch(cfg.universe or None, cfg.start, cfg.end, cfg.benchmark),
        )

    if cfg.provider == "yfinance":
        if not cfg.universe:
            raise ValueError("provider='yfinance' requires an explicit universe")
        return YFinanceProvider(cache=cache).fetch(cfg.universe, cfg.start, cfg.end, cfg.benchmark)

    raise ValueError(f"unknown provider {cfg.provider!r}")


def align_to_calendar(bundle: MarketDataBundle, min_assets: int = 5) -> MarketDataBundle:
    """Restrict the bundle to dates where the benchmark trades and at least
    `min_assets` assets have prices. Prevents partial-session artefacts."""
    px = bundle.prices
    ok = (px.notna().sum(axis=1) >= min_assets) & bundle.benchmark.reindex(px.index).notna()
    idx = px.index[ok.to_numpy()]
    return MarketDataBundle(
        prices=px.loc[idx],
        volumes=bundle.volumes.reindex(idx),
        benchmark=bundle.benchmark.reindex(idx),
        risk_free=bundle.risk_free.reindex(idx).ffill().bfill(),
        fundamentals=bundle.fundamentals,
        meta=bundle.meta,
        provider=bundle.provider,
        notes=[*bundle.notes, f"Calendar-aligned to {len(idx)} sessions."],
        ground_truth=bundle.ground_truth,
    )


def winsorize(s: pd.Series, q: float = 0.02) -> pd.Series:
    """Clip a cross-section at symmetric quantiles. Applied to factor values,
    not to returns."""
    if s.notna().sum() < 3 or q <= 0:
        return s
    lo, hi = s.quantile(q), s.quantile(1 - q)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return s
    return s.clip(lo, hi)
