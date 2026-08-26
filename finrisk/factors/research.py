"""Factor research (Phase 5).

Asks the prior question that a backtest cannot answer on its own: *does this
factor contain cross-sectional information at all?*

Two complementary lenses:

* **Information Coefficient (IC)** — the rank correlation between a factor
  score at time t and the realised forward return. It measures signal
  independently of any portfolio-construction choice, so a strong backtest with
  a zero IC is a red flag that the return came from position sizing or luck
  rather than from the signal.
* **Factor portfolios** — a tradable quintile spread, which measures whether
  the IC survives contact with concentration and turnover.

Spearman (rank) correlation is used rather than Pearson because factor scores
are ordinal by construction and a single outlier return would otherwise
dominate the estimate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrisk.risk import metrics as m
from finrisk.schemas import FactorPerformance

logger = logging.getLogger(__name__)


def forward_returns(
    prices: pd.DataFrame, dates: list[pd.Timestamp], horizon_days: int
) -> pd.DataFrame:
    """Realised return over the `horizon_days` sessions AFTER each date.

    Strictly forward-looking by design — this is the label, used only for
    research diagnostics, never as a model input.
    """
    idx = prices.index
    rows = {}
    for d in dates:
        d = pd.Timestamp(d)
        pos = idx.searchsorted(d)
        if pos >= len(idx):
            continue
        end = pos + horizon_days
        if end >= len(idx):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rows[d] = prices.iloc[end] / prices.iloc[pos] - 1.0
    return pd.DataFrame(rows).T.sort_index()


def information_coefficient(
    scores: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 8
) -> pd.Series:
    """Per-date Spearman IC between scores and forward returns."""
    out = {}
    for d in scores.index.intersection(fwd.index):
        s, f = scores.loc[d], fwd.loc[d]
        joint = pd.concat([s, f], axis=1).dropna()
        if len(joint) < min_names:
            continue
        ic = joint.iloc[:, 0].corr(joint.iloc[:, 1], method="spearman")
        if pd.notna(ic):
            out[d] = float(ic)
    return pd.Series(out).sort_index()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """IC mean, std, and IC-IR (the t-stat-like ratio quants actually quote).

    `ic_ir = mean(IC)/std(IC)`; multiplying by sqrt(n) gives the t-statistic
    for "is the mean IC different from zero".
    """
    if not len(ic):
        return {"mean_ic": 0.0, "std_ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "n": 0}
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    ir = mean / sd if sd > 1e-12 else 0.0
    return {
        "mean_ic": mean,
        "std_ic": sd,
        "ic_ir": float(ir),
        "t_stat": float(ir * np.sqrt(len(ic))),
        "hit_rate": float((ic > 0).mean()),
        "n": len(ic),
    }


def ic_decay(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 21, 63, 126, 252),
) -> dict[str, float]:
    """How fast the signal dies. A factor whose IC peaks at 5 days and vanishes
    by 21 is untradable after costs; one that persists to 126 days can be
    rebalanced monthly."""
    dates = list(scores.index)
    out = {}
    for h in horizons:
        fwd = forward_returns(prices, dates, h)
        out[f"{h}d"] = ic_summary(information_coefficient(scores, fwd))["mean_ic"]
    return out


# ----------------------------------------------------------------------
# Factor portfolios
# ----------------------------------------------------------------------


def quantile_portfolios(scores: pd.DataFrame, n_quantiles: int = 5) -> dict[int, pd.DataFrame]:
    """Assign each asset to a quantile bucket per date (1 = worst score)."""
    buckets: dict[int, pd.DataFrame] = {}
    labels = list(range(1, n_quantiles + 1))
    masks = {q: pd.DataFrame(False, index=scores.index, columns=scores.columns) for q in labels}
    for d in scores.index:
        row = scores.loc[d].dropna()
        if len(row) < n_quantiles * 2:
            continue
        try:
            q = pd.qcut(row.rank(method="first"), n_quantiles, labels=labels)
        except ValueError:
            continue
        for lab in labels:
            members = row.index[(q == lab).to_numpy()]
            masks[lab].loc[d, members] = True
    for lab in labels:
        buckets[lab] = masks[lab]
    return buckets


def _bucket_returns(mask: pd.DataFrame, asset_returns: pd.DataFrame) -> tuple[pd.Series, float]:
    """Equal-weight, held from each signal date to the next.

    Weights are shifted forward by one session so that the return earned on the
    signal date itself is excluded — otherwise the study would be trading on a
    price it used to build the signal.
    """
    w = mask.astype(float)
    w = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0)
    daily_w = w.reindex(asset_returns.index).ffill().shift(1)
    port = (daily_w * asset_returns.reindex(columns=daily_w.columns).fillna(0.0)).sum(axis=1)
    port = port[daily_w.notna().any(axis=1)]

    rebal = w.dropna(how="all")
    tos = []
    prev = None
    for d in rebal.index:
        cur = rebal.loc[d].fillna(0.0)
        tos.append(1.0 if prev is None else m.turnover(prev, cur))
        prev = cur
    return port, float(np.mean(tos)) if tos else 0.0


def study_factor(
    factor_name: str,
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    n_quantiles: int = 5,
    horizon_days: int = 21,
    risk_free_annual: float = 0.0,
) -> FactorPerformance:
    """Full single-factor study: long top quintile, long-short spread, IC."""
    asset_returns = prices.pct_change().iloc[1:]
    buckets = quantile_portfolios(scores, n_quantiles)
    if not buckets or buckets[n_quantiles].to_numpy().sum() == 0:
        raise ValueError(f"insufficient cross-section to study {factor_name}")

    long_ret, long_to = _bucket_returns(buckets[n_quantiles], asset_returns)
    short_ret, _ = _bucket_returns(buckets[1], asset_returns)
    ls = (long_ret - short_ret).dropna()

    fwd = forward_returns(prices, list(scores.index), horizon_days)
    ic = information_coefficient(scores, fwd)
    ic_stats = ic_summary(ic)

    return FactorPerformance(
        factor=factor_name,
        start=long_ret.index.min().date(),
        end=long_ret.index.max().date(),
        n_rebalances=len(scores),
        long_ann_return=m.cagr(long_ret),
        long_short_ann_return=m.cagr(ls) if len(ls) > 20 else None,
        ann_volatility=m.annualized_volatility(long_ret),
        sharpe=m.sharpe_ratio(long_ret, risk_free_annual),
        max_drawdown=m.max_drawdown(long_ret),
        turnover=long_to,
        mean_ic=ic_stats["mean_ic"],
        ic_ir=ic_stats["ic_ir"],
        hit_rate=ic_stats.get("hit_rate", 0.0),
        ic_by_horizon=ic_decay(scores, prices),
    )


def factor_correlation(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average cross-sectional correlation between factor score panels.

    Two factors with 0.9 correlation are one factor with extra turnover; this is
    what justifies grouping rather than stacking every signal into a composite.
    """
    names = list(panels)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i:]:
            pa, pb = panels[a], panels[b]
            common_d = pa.index.intersection(pb.index)
            vals = []
            for d in common_d:
                j = pd.concat([pa.loc[d], pb.loc[d]], axis=1).dropna()
                if len(j) >= 8:
                    c = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
                    if pd.notna(c):
                        vals.append(c)
            v = float(np.mean(vals)) if vals else np.nan
            out.loc[a, b] = out.loc[b, a] = v
    return out


def render_factor_report(perfs: list[FactorPerformance]) -> str:
    lines = ["Factor Research Report", "=" * 96]
    hdr = (
        f"{'Factor':<14}{'LongCAGR':>10}{'L/S CAGR':>10}{'Vol':>8}"
        f"{'Sharpe':>8}{'MaxDD':>9}{'Turn':>7}{'MeanIC':>9}{'IC-IR':>8}{'IC>0':>7}"
    )
    lines += [hdr, "-" * 96]
    for p in perfs:
        ls = (
            f"{p.long_short_ann_return:>9.2%}"
            if p.long_short_ann_return is not None
            else "       n/a"
        )
        lines.append(
            f"{p.factor:<14}{p.long_ann_return:>9.2%} {ls} "
            f"{p.ann_volatility:>7.1%}{p.sharpe:>8.2f}{p.max_drawdown:>9.1%}"
            f"{p.turnover:>7.1%}{p.mean_ic:>9.3f}"
            f"{(p.ic_ir if p.ic_ir is not None else 0):>8.2f}{p.hit_rate:>7.0%}"
        )
    lines.append("-" * 96)
    lines.append(
        "IC = Spearman rank correlation of score vs 21d forward return. "
        "IC-IR = mean(IC)/std(IC)."
    )
    if perfs and perfs[0].ic_by_horizon:
        lines += ["", "IC decay by horizon (mean IC):", "-" * 60]
        hs = list(perfs[0].ic_by_horizon)
        lines.append(f"{'Factor':<14}" + "".join(f"{h:>9}" for h in hs))
        for p in perfs:
            lines.append(
                f"{p.factor:<14}" + "".join(f"{p.ic_by_horizon.get(h, 0):>9.3f}" for h in hs)
            )
    return "\n".join(lines)
