"""Factor definitions.

SIGN CONVENTION
---------------
Every factor is defined so that **higher = more attractive**. This matters:
raw realised volatility is "higher = worse", so the factor is stored as
`-volatility` and named `low_volatility`. Getting this wrong silently inverts a
strategy, so the convention is enforced in one place (`FACTOR_REGISTRY`) and
tested in `tests/test_factors.py`.

POINT-IN-TIME
-------------
Every function receives only data up to and including the decision date. None
of them may look at `t+1`. Price windows are taken from the tail of the
supplied frame; fundamentals arrive pre-filtered by
`MarketDataBundle.fundamentals_asof`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

FactorGroup = Literal["momentum", "value", "quality", "risk", "market"]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    group: FactorGroup
    description: str
    higher_is_better: bool = True
    requires_fundamentals: bool = False


FACTOR_REGISTRY: dict[str, FactorSpec] = {
    # ---- momentum ----
    "mom_1m": FactorSpec("mom_1m", "momentum", "Trailing 21-session total return"),
    "mom_3m": FactorSpec("mom_3m", "momentum", "Trailing 63-session total return"),
    "mom_6m": FactorSpec("mom_6m", "momentum", "Trailing 126-session total return"),
    "mom_12m_1m": FactorSpec(
        "mom_12m_1m",
        "momentum",
        "12-month return skipping the most recent month (short-term reversal control)",
    ),
    "px_to_ma200": FactorSpec("px_to_ma200", "momentum", "Price / 200-session moving average - 1"),
    "ma50_to_ma200": FactorSpec(
        "ma50_to_ma200", "momentum", "50-session MA / 200-session MA - 1 (trend slope)"
    ),
    # ---- value (all expressed as yields so higher = cheaper) ----
    "earnings_yield": FactorSpec(
        "earnings_yield",
        "value",
        "Trailing EPS / price (inverse P/E)",
        requires_fundamentals=True,
    ),
    "book_to_price": FactorSpec(
        "book_to_price",
        "value",
        "Book value per share / price (inverse P/B)",
        requires_fundamentals=True,
    ),
    "ebitda_to_ev": FactorSpec(
        "ebitda_to_ev",
        "value",
        "EBITDA per share / (price + net debt per share) (inverse EV/EBITDA)",
        requires_fundamentals=True,
    ),
    "fcf_yield": FactorSpec(
        "fcf_yield",
        "value",
        "Free cash flow per share / price",
        requires_fundamentals=True,
    ),
    # ---- quality ----
    "roe": FactorSpec("roe", "quality", "Net income / book equity", requires_fundamentals=True),
    "roa": FactorSpec("roa", "quality", "Net income / total assets", requires_fundamentals=True),
    "low_leverage": FactorSpec(
        "low_leverage",
        "quality",
        "Negated total debt / equity",
        requires_fundamentals=True,
    ),
    "earnings_stability": FactorSpec(
        "earnings_stability",
        "quality",
        "Negated coefficient of variation of trailing 8 quarterly EPS",
        requires_fundamentals=True,
    ),
    # ---- risk (low-risk anomaly: less risk scores higher) ----
    "low_volatility": FactorSpec(
        "low_volatility", "risk", "Negated annualised 63-session realised volatility"
    ),
    "low_downside_vol": FactorSpec(
        "low_downside_vol",
        "risk",
        "Negated annualised volatility of negative returns only",
    ),
    "small_drawdown": FactorSpec(
        "small_drawdown", "risk", "Negated trailing 252-session maximum drawdown"
    ),
    # ---- market ----
    "low_beta": FactorSpec("low_beta", "market", "Negated 252-session beta to benchmark"),
    "low_correlation": FactorSpec(
        "low_correlation",
        "market",
        "Negated 252-session correlation to benchmark (diversification proxy)",
    ),
    "liquidity": FactorSpec("liquidity", "market", "Log median 63-session dollar volume"),
}

GROUPS: tuple[FactorGroup, ...] = ("momentum", "value", "quality", "risk", "market")


def factors_in_group(group: FactorGroup) -> list[str]:
    return [n for n, s in FACTOR_REGISTRY.items() if s.group == group]


# ----------------------------------------------------------------------
# Price-based factors
# ----------------------------------------------------------------------


def _tail_return(prices: pd.DataFrame, window: int, skip: int = 0) -> pd.Series:
    """Total return over `window` sessions ending `skip` sessions ago."""
    need = window + skip + 1
    if len(prices) < need:
        return pd.Series(np.nan, index=prices.columns)
    end = prices.iloc[-1 - skip]
    start = prices.iloc[-1 - skip - window]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = end / start - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def momentum_factors(
    prices: pd.DataFrame, lookbacks: list[int], skip: int = 21
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    names = {21: "mom_1m", 63: "mom_3m", 126: "mom_6m"}
    for lb in lookbacks:
        if lb in names:
            out[names[lb]] = _tail_return(prices, lb)
    if 252 in lookbacks:
        out["mom_12m_1m"] = _tail_return(prices, 252 - skip, skip=skip)

    if len(prices) >= 200:
        ma200 = prices.iloc[-200:].mean()
        out["px_to_ma200"] = prices.iloc[-1] / ma200 - 1.0
        if len(prices) >= 200:
            ma50 = prices.iloc[-50:].mean()
            out["ma50_to_ma200"] = ma50 / ma200 - 1.0
    return out


def risk_factors(
    returns: pd.DataFrame, vol_window: int = 63, dd_window: int = 252
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    ann = np.sqrt(252.0)

    if len(returns) >= max(vol_window // 2, 10):
        win = returns.iloc[-vol_window:]
        out["low_volatility"] = -(win.std(ddof=1) * ann)
        neg = win.where(win < 0)
        # Downside deviation: sum of squared negatives / n (not / n_negatives),
        # the standard Sortino denominator.
        dd = np.sqrt((neg.pow(2).sum()) / max(len(win) - 1, 1)) * ann
        out["low_downside_vol"] = -dd

    if len(returns) >= max(dd_window // 2, 20):
        win = returns.iloc[-dd_window:]
        curve = (1.0 + win).cumprod()
        peak = curve.cummax()
        out["small_drawdown"] = (curve / peak - 1.0).min()  # already <= 0
    return out


def market_factors(
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    beta_window: int = 252,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    n = min(beta_window, len(returns), len(bench_returns))
    if n >= 40:
        r = returns.iloc[-n:]
        b = bench_returns.reindex(r.index)
        valid = b.notna()
        r, b = r[valid.to_numpy()], b[valid]
        bvar = float(b.var(ddof=1))
        if bvar > 0:
            cov = r.apply(lambda col: col.cov(b))
            out["low_beta"] = -(cov / bvar)
            out["low_correlation"] = -r.apply(lambda col: col.corr(b))

    if len(volumes) >= 20 and len(prices) >= 20:
        dollar = (volumes.iloc[-63:] * prices.iloc[-63:]).median()
        out["liquidity"] = np.log1p(dollar.clip(lower=0.0))
    return out


# ----------------------------------------------------------------------
# Fundamental factors
# ----------------------------------------------------------------------


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b.where(b.abs() > 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


def value_factors(fundamentals: pd.DataFrame, last_price: pd.Series) -> dict[str, pd.Series]:
    """`fundamentals` is indexed by ticker (output of fundamentals_asof)."""
    if not len(fundamentals):
        return {}
    f = fundamentals.reindex(last_price.index)
    px = last_price.astype(float)
    num = lambda c: pd.to_numeric(f[c], errors="coerce")  # noqa: E731

    ev = px + num("total_debt_per_share").fillna(0.0)
    return {
        "earnings_yield": _safe_div(num("eps"), px),
        "book_to_price": _safe_div(num("book_value_per_share"), px),
        "ebitda_to_ev": _safe_div(num("ebitda_per_share"), ev),
        "fcf_yield": _safe_div(num("free_cash_flow_per_share"), px),
    }


def quality_factors(
    fundamentals: pd.DataFrame,
    history: pd.DataFrame | None,
    tickers: pd.Index,
) -> dict[str, pd.Series]:
    """`history` is the full point-in-time fundamental history (already filtered
    to available_from <= as_of), used for earnings stability."""
    if not len(fundamentals):
        return {}
    f = fundamentals.reindex(tickers)
    num = lambda c: pd.to_numeric(f[c], errors="coerce")  # noqa: E731

    out = {
        "roe": _safe_div(num("net_income_per_share"), num("total_equity_per_share")),
        "roa": _safe_div(num("net_income_per_share"), num("total_assets_per_share")),
        "low_leverage": -_safe_div(num("total_debt_per_share"), num("total_equity_per_share")),
    }

    if history is not None and len(history):
        h = history.copy()
        h["eps"] = pd.to_numeric(h["eps"], errors="coerce")
        h = h.sort_values(["ticker", "available_from"])
        recent = h.groupby("ticker").tail(8)
        agg = recent.groupby("ticker")["eps"].agg(["mean", "std", "count"])
        cv = agg["std"] / agg["mean"].abs().where(agg["mean"].abs() > 1e-9)
        cv[agg["count"] < 4] = np.nan
        out["earnings_stability"] = -cv.reindex(tickers)
    return out
