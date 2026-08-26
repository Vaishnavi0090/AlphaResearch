"""Data quality validation (Phase 3).

Runs a battery of checks over a `MarketDataBundle` and produces a
`DataQualityReport`. Nothing is auto-repaired: the report tells you what is
wrong so you can decide, because automatic repair of financial data is how
silent bias gets introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.data.base import MarketDataBundle
from finrisk.schemas import DataQualityIssue, DataQualityReport

ABNORMAL_RETURN = 0.60
STALE_RUN = 10
MAX_MISSING_PCT = 20.0


def _stale_runs(s: pd.Series, min_run: int = STALE_RUN) -> int:
    """Longest run of identical consecutive prices (excluding NaN)."""
    v = s.dropna()
    if len(v) < 2:
        return 0
    changed = v.diff().ne(0.0)
    grp = changed.cumsum()
    longest = int(v.groupby(grp).size().max()) if len(v) else 0
    return longest if longest >= min_run else 0


def validate_bundle(bundle: MarketDataBundle) -> DataQualityReport:
    px = bundle.prices
    issues: list[DataQualityIssue] = []

    if px.empty:
        return DataQualityReport(
            n_assets=0,
            start=pd.Timestamp("1970-01-01").date(),
            end=pd.Timestamp("1970-01-01").date(),
            n_observations=0,
            missing_observations=0,
            duplicate_timestamps=0,
            issues=[
                DataQualityIssue(severity="error", category="empty", detail="price frame is empty")
            ],
        )

    idx = pd.DatetimeIndex(px.index)
    n_dup = int(idx.duplicated().sum())
    if n_dup:
        issues.append(
            DataQualityIssue(
                severity="error",
                category="duplicate_timestamps",
                detail=f"{n_dup} duplicated index entries",
            )
        )
    if not idx.is_monotonic_increasing:
        issues.append(
            DataQualityIssue(
                severity="error",
                category="index_order",
                detail="price index is not sorted ascending",
            )
        )
    if getattr(idx, "tz", None) is not None:
        issues.append(
            DataQualityIssue(
                severity="warning",
                category="timezone",
                detail=f"index is tz-aware ({idx.tz}); expected tz-naive",
            )
        )

    total_cells = int(px.size)
    n_missing = int(px.isna().to_numpy().sum())
    n_obs = total_cells - n_missing

    # -- per-asset checks ------------------------------------------------
    rets = px.pct_change()
    for tk in px.columns:
        col = px[tk]
        miss_pct = 100.0 * col.isna().mean()
        if miss_pct > MAX_MISSING_PCT:
            issues.append(
                DataQualityIssue(
                    severity="warning",
                    category="missing_values",
                    ticker=tk,
                    detail=f"{miss_pct:.1f}% of bars missing",
                )
            )
        nonpos = int((col <= 0).sum())
        if nonpos:
            issues.append(
                DataQualityIssue(
                    severity="error",
                    category="abnormal_price",
                    ticker=tk,
                    detail=f"{nonpos} non-positive prices",
                )
            )
        run = _stale_runs(col)
        if run:
            issues.append(
                DataQualityIssue(
                    severity="warning",
                    category="stale_price",
                    ticker=tk,
                    detail=f"price unchanged for {run} consecutive sessions",
                )
            )
        extreme = rets[tk].abs() > ABNORMAL_RETURN
        if extreme.any():
            worst = rets[tk].abs().max()
            issues.append(
                DataQualityIssue(
                    severity="warning",
                    category="abnormal_return",
                    ticker=tk,
                    detail=(
                        f"{int(extreme.sum())} bars with |return| > "
                        f"{ABNORMAL_RETURN:.0%} (max {worst:.1%}); check for "
                        "unadjusted corporate actions"
                    ),
                )
            )

    # -- frequency / calendar -------------------------------------------
    if len(idx) > 5:
        gaps = pd.Series(idx).diff().dt.days.dropna()
        median_gap = float(gaps.median())
        big = int((gaps > max(median_gap * 6, 10)).sum())
        if big:
            issues.append(
                DataQualityIssue(
                    severity="info",
                    category="calendar_gap",
                    detail=(
                        f"{big} gaps > {max(median_gap * 6, 10):.0f} days "
                        "(exchange closures or missing history)"
                    ),
                )
            )
        if median_gap > 4:
            issues.append(
                DataQualityIssue(
                    severity="warning",
                    category="frequency",
                    detail=f"median gap {median_gap:.1f}d — not daily data?",
                )
            )

    # -- alignment ------------------------------------------------------
    if not bundle.benchmark.index.equals(px.index):
        missing = len(px.index.difference(bundle.benchmark.index))
        issues.append(
            DataQualityIssue(
                severity="warning" if missing else "info",
                category="alignment",
                detail=f"benchmark missing {missing} of {len(px.index)} sessions",
            )
        )
    if not bundle.volumes.index.equals(px.index):
        issues.append(
            DataQualityIssue(
                severity="warning",
                category="alignment",
                detail="volume index differs from price index",
            )
        )
    if bundle.risk_free.isna().any():
        issues.append(
            DataQualityIssue(
                severity="warning",
                category="missing_values",
                detail=f"{int(bundle.risk_free.isna().sum())} missing risk-free values",
            )
        )

    # -- fundamentals ---------------------------------------------------
    fund = bundle.fundamentals
    if not len(fund):
        issues.append(
            DataQualityIssue(
                severity="warning",
                category="fundamentals",
                detail="no fundamental data — value/quality factors will be unavailable",
            )
        )
    else:
        pe = pd.to_datetime(fund["period_end"])
        af = pd.to_datetime(fund["available_from"])
        leak = int((af < pe).sum())
        if leak:
            issues.append(
                DataQualityIssue(
                    severity="error",
                    category="point_in_time",
                    detail=f"{leak} records available before their period end (LEAKAGE)",
                )
            )
        lag = (af - pe).dt.days
        issues.append(
            DataQualityIssue(
                severity="info",
                category="fundamentals",
                detail=(
                    f"{len(fund)} records, {fund['ticker'].nunique()} tickers, "
                    f"publication lag median {lag.median():.0f}d "
                    f"[{lag.min():.0f}-{lag.max():.0f}]"
                ),
            )
        )
        no_fund = sorted(set(px.columns) - set(fund["ticker"].unique()))
        if no_fund:
            issues.append(
                DataQualityIssue(
                    severity="warning",
                    category="fundamentals",
                    detail=f"{len(no_fund)} tickers lack fundamentals: {no_fund[:8]}",
                )
            )

    return DataQualityReport(
        n_assets=px.shape[1],
        start=idx.min().date(),
        end=idx.max().date(),
        n_observations=n_obs,
        missing_observations=n_missing,
        duplicate_timestamps=n_dup,
        issues=issues,
    )


def assert_no_lookahead_in_frame(values: pd.DataFrame, decision_date: pd.Timestamp) -> None:
    """Raise if a feature frame contains rows dated after the decision date.

    Called from the backtester on every rebalance — a cheap runtime assertion
    that turns a silent leakage bug into a loud failure.
    """
    if values.empty:
        return
    latest = pd.DatetimeIndex(values.index).max()
    if latest > pd.Timestamp(decision_date):
        raise AssertionError(
            f"look-ahead detected: feature data dated {latest.date()} used for a "
            f"decision on {pd.Timestamp(decision_date).date()}"
        )


def check_monotone_availability(fund: pd.DataFrame) -> list[str]:
    """Return human-readable violations of point-in-time discipline."""
    if not len(fund):
        return []
    pe = pd.to_datetime(fund["period_end"])
    af = pd.to_datetime(fund["available_from"])
    out = []
    bad = fund[af < pe]
    for _, row in bad.head(20).iterrows():
        out.append(
            f"{row['ticker']}: available_from {row['available_from']} < "
            f"period_end {row['period_end']}"
        )
    if np.isfinite((af - pe).dt.days.max()) and (af - pe).dt.days.max() > 400:
        out.append("publication lag exceeds 400 days for at least one record")
    return out
