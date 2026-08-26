"""Risk attribution (Phase 11).

Answers four separate questions that are easy to conflate:

1. **Which positions carry the risk?**  Euler decomposition on the covariance
   matrix (in `RiskEngine`).
2. **Which sectors carry the risk?**  Aggregation of (1).
3. **Which *factors* carry the risk?**  A time-series regression of portfolio
   returns on factor-mimicking portfolios. This is a different question from
   (1): a portfolio can be spread across 20 names and still be one big
   momentum bet.
4. **Which positions drive the tail?**  Average contribution to portfolio
   return conditional on the portfolio being in its worst 5% of days. Tail
   contribution is not proportional to volatility contribution, because
   correlations rise in a sell-off.

FACTOR MODEL
------------
Factor returns are built as long-short quintile spreads of each factor group's
score panel, plus the benchmark as the market factor. Variance is then
decomposed as

    Var(R_p) = b' Sigma_f b + Var(residual)

with `b` the OLS loadings. The residual share is reported explicitly — a model
that explains 20% of the variance should not present a confident factor
attribution, and hiding the residual is how that happens.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_factor_returns(
    group_panels: dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    quantile: float = 0.2,
) -> pd.DataFrame:
    """Long-short factor-mimicking portfolio returns.

    For each factor group: long the top `quantile` of scores, short the bottom,
    equal-weighted, rebalanced on each score date, weights lagged one session.
    """
    out: dict[str, pd.Series] = {}
    for name, panel in group_panels.items():
        if panel is None or panel.empty:
            continue
        w = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        for d in panel.index:
            row = panel.loc[d].dropna()
            k = int(max(round(len(row) * quantile), 1))
            if len(row) < 2 * k:
                continue
            order = row.sort_values(ascending=False)
            longs, shorts = order.index[:k], order.index[-k:]
            w.loc[d, longs] = 1.0 / k
            w.loc[d, shorts] = -1.0 / k
        daily = w.reindex(asset_returns.index).ffill().shift(1)
        cols = [c for c in daily.columns if c in asset_returns.columns]
        series = (daily[cols] * asset_returns[cols].fillna(0.0)).sum(axis=1)
        out[name] = series[daily[cols].abs().sum(axis=1) > 0]

    if benchmark_returns is not None:
        out["market"] = benchmark_returns

    return pd.DataFrame(out).dropna(how="all")


def factor_risk_decomposition(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, float | dict[str, float]]:
    """OLS variance decomposition of portfolio risk onto factors."""
    df = pd.concat([portfolio_returns.rename("_p"), factor_returns], axis=1).dropna()
    if len(df) < 60 or df.shape[1] < 2:
        return {"r_squared": 0.0, "factor_share": {}, "residual_share": 1.0}

    y = df["_p"].to_numpy()
    F = df.drop(columns=["_p"])
    X = np.column_stack([np.ones(len(F)), F.to_numpy()])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b = coef[1:]
    fitted = X @ coef
    resid = y - fitted

    sigma_f = np.cov(F.to_numpy(), rowvar=False, ddof=1)
    if sigma_f.ndim == 0:
        sigma_f = sigma_f.reshape(1, 1)

    total_var = float(np.var(y, ddof=1))
    systematic = float(b @ sigma_f @ b)
    resid_var = float(np.var(resid, ddof=1))
    denom = systematic + resid_var
    if denom <= 1e-18:
        return {"r_squared": 0.0, "factor_share": {}, "residual_share": 1.0}

    # Per-factor share via the marginal covariance contribution
    # b_i * (Sigma_f b)_i, which sums exactly to the systematic variance.
    contrib = b * (sigma_f @ b)
    share = {str(name): float(c / denom) for name, c in zip(F.columns, contrib, strict=True)}
    return {
        "r_squared": float(1.0 - resid_var / total_var) if total_var > 0 else 0.0,
        "loadings": {str(n): float(v) for n, v in zip(F.columns, b, strict=True)},
        "factor_share": share,
        "residual_share": float(resid_var / denom),
        "systematic_vol_ann": float(np.sqrt(max(systematic, 0.0) * periods_per_year)),
        "residual_vol_ann": float(np.sqrt(max(resid_var, 0.0) * periods_per_year)),
    }


def tail_contribution(
    weights: dict[str, float],
    asset_returns: pd.DataFrame,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Share of the portfolio's tail loss attributable to each position.

    Conditions on the portfolio's own worst days, then attributes the average
    loss on those days across positions. This is the CVaR analogue of an Euler
    decomposition and, like it, sums to 1.
    """
    active = {k: v for k, v in weights.items() if abs(v) > 1e-9}
    cols = [t for t in active if t in asset_returns.columns]
    if not cols:
        return {}
    w = pd.Series({t: active[t] for t in cols})
    R = asset_returns[cols].dropna()
    if len(R) < 40:
        return {}
    port = (R * w).sum(axis=1)
    cutoff = np.quantile(port.to_numpy(), 1.0 - confidence)
    tail = R[port <= cutoff]
    if not len(tail):
        return {}
    contrib = (tail * w).mean()
    total = contrib.sum()
    if abs(total) < 1e-14:
        return {}
    return {t: float(v / total) for t, v in contrib.items()}


def correlation_concentration(
    weights: dict[str, float], asset_returns: pd.DataFrame, threshold: float = 0.75
) -> list[dict[str, Any]]:
    """Weighted pairs whose correlation exceeds `threshold`.

    Two 15% positions correlated at 0.9 are effectively one 30% position — a
    concentration a per-name weight cap cannot see.
    """
    active = {k: v for k, v in weights.items() if abs(v) > 1e-9}
    cols = [t for t in active if t in asset_returns.columns]
    if len(cols) < 2:
        return []
    corr = asset_returns[cols].corr()
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = float(corr.loc[a, b])
            if np.isfinite(c) and c >= threshold:
                out.append(
                    {
                        "asset_a": a,
                        "asset_b": b,
                        "correlation": c,
                        "combined_weight": float(active[a] + active[b]),
                    }
                )
    return sorted(out, key=lambda d: -float(d["combined_weight"]))


def render_attribution(
    portfolio_vol: float,
    position_risk: dict[str, float],
    sector_risk: dict[str, float],
    factor_decomp: dict,
    top_n: int = 6,
) -> str:
    lines = [f"Portfolio Volatility: {portfolio_vol:.1%}", ""]
    lines.append("Risk Contribution (positions):")
    ranked = sorted(position_risk.items(), key=lambda kv: -kv[1])
    shown = ranked[:top_n]
    for t, v in shown:
        lines.append(f"  {t:<10}{v:>7.1%}")
    if len(ranked) > top_n:
        lines.append(f"  {'Other':<10}{sum(v for _, v in ranked[top_n:]):>7.1%}")

    if sector_risk:
        lines += ["", "Sector Risk:"]
        for s, v in sorted(sector_risk.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {s:<16}{v:>7.1%}")

    fs = factor_decomp.get("factor_share") or {}
    if fs:
        lines += ["", "Factor Risk:"]
        for f, v in sorted(fs.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"  {f:<16}{v:>7.1%}")
        lines.append(f"  {'Idiosyncratic':<16}{factor_decomp.get('residual_share', 0):>7.1%}")
        lines.append(f"  (regression R^2 = {factor_decomp.get('r_squared', 0):.2f})")
    return "\n".join(lines)
