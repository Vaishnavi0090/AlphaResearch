"""Fundamental data with point-in-time stamping.

The hard problem with fundamentals is not fetching them, it is knowing *when
they became public*. A fiscal quarter ending 2023-03-31 was not knowable on
2023-03-31; the 10-Q landed weeks later. Using the period-end date as the
availability date is the single most common source of inflated factor
backtests.

Every record therefore carries `available_from`, and `MarketDataBundle
.fundamentals_asof()` is the only sanctioned read path.
"""

from __future__ import annotations

import logging

import pandas as pd

from finrisk.data.base import FUNDAMENTAL_COLUMNS
from finrisk.data.cache import retry

logger = logging.getLogger(__name__)

EDGAR_FACTS = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"


def stamp_availability(
    df: pd.DataFrame,
    lag_days: int = 90,
    period_col: str = "period_end",
) -> pd.DataFrame:
    """Add `available_from = period_end + lag_days` where it is missing.

    A flat lag is a simplification. Real filing dates (EDGAR `filed`) are
    preferred and used when present; the flat lag is the conservative fallback.
    """
    out = df.copy()
    out[period_col] = pd.to_datetime(out[period_col])
    if "available_from" not in out.columns:
        out["available_from"] = out[period_col] + pd.Timedelta(days=lag_days)
    else:
        missing = out["available_from"].isna()
        out.loc[missing, "available_from"] = out.loc[missing, period_col] + pd.Timedelta(
            days=lag_days
        )
    out["available_from"] = pd.to_datetime(out["available_from"])
    # Guard: availability can never precede the period it reports on.
    bad = out["available_from"] < out[period_col]
    if bad.any():
        logger.warning("%d records had available_from < period_end; clamped", bad.sum())
        out.loc[bad, "available_from"] = out.loc[bad, period_col]
    return out


def derive_per_share(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the canonical per-share columns exist, filling with NaN when a
    source does not provide them. Missing is represented as missing — never 0."""
    out = df.copy()
    for col in FUNDAMENTAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[FUNDAMENTAL_COLUMNS]


class EdgarProvider:
    """SEC EDGAR XBRL company-concept reader.

    EDGAR exposes a `filed` date per fact, which is a genuine point-in-time
    availability stamp — strictly better than a flat lag. Requires a
    descriptive User-Agent per SEC policy.
    """

    name = "edgar"

    def __init__(self, user_agent: str = "finrisk-ai research contact@example.com"):
        self.user_agent = user_agent

    def concept(self, cik: int, tag: str) -> pd.DataFrame:
        import json
        import urllib.request

        url = EDGAR_FACTS.format(cik=cik, tag=tag)

        def _call() -> pd.DataFrame:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=20) as fh:
                payload = json.load(fh)
            rows = []
            for unit, facts in payload.get("units", {}).items():
                for f in facts:
                    rows.append(
                        {
                            "period_end": f.get("end"),
                            "available_from": f.get("filed"),
                            "value": f.get("val"),
                            "unit": unit,
                            "form": f.get("form"),
                            "fy": f.get("fy"),
                            "fp": f.get("fp"),
                        }
                    )
            return pd.DataFrame(rows)

        return retry(_call, attempts=3, base_delay=1.5)
