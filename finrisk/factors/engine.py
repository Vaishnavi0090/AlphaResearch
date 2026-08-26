"""Factor engine (Phase 4).

Produces, for a single decision date, a point-in-time cross-section of
normalised factor scores plus group and composite scores.

NORMALISATION
-------------
Two-step, and the order matters:

1. **Winsorize** the raw cross-section at symmetric quantiles. Financial factor
   distributions are fat-tailed — one company with a near-zero book value
   produces a book-to-price of 400 and would dominate a z-score.
2. **Z-score** the winsorized cross-section.

Cross-sectional (not time-series) normalisation is used deliberately: the
question a stock selector answers is "which asset is more attractive *than its
peers today*", which makes the comparison immune to market-wide level drift.

MISSING DATA
------------
A missing factor becomes a z-score of 0 — i.e. the cross-sectional mean, the
neutral value. It is *not* dropped (that would silently shrink the universe)
and *not* imputed with an optimistic value. Each asset's `coverage` records the
fraction of factors that were actually observed, so the optimiser and the
report can discount thin-evidence names.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrisk.config.settings import FactorConfig
from finrisk.data.base import MarketDataBundle
from finrisk.factors.definitions import (
    FACTOR_REGISTRY,
    GROUPS,
    factors_in_group,
    market_factors,
    momentum_factors,
    quality_factors,
    risk_factors,
    value_factors,
)
from finrisk.schemas import FactorScore

logger = logging.getLogger(__name__)


def winsorize_cross_section(s: pd.Series, q: float = 0.02) -> pd.Series:
    if q <= 0 or s.notna().sum() < 5:
        return s
    lo, hi = s.quantile(q), s.quantile(1.0 - q)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return s
    return s.clip(lo, hi)


def zscore_cross_section(s: pd.Series) -> pd.Series:
    """Z-score with a degenerate-spread guard."""
    v = s.astype(float)
    mu = v.mean(skipna=True)
    sd = v.std(ddof=1, skipna=True)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=v.index).where(v.notna())
    return (v - mu) / sd


class FactorEngine:
    """Computes point-in-time factor cross-sections."""

    def __init__(self, config: FactorConfig | None = None) -> None:
        self.cfg = config or FactorConfig()

    # ------------------------------------------------------------------
    def raw_panel(self, bundle: MarketDataBundle, as_of: pd.Timestamp) -> pd.DataFrame:
        """Raw (un-normalised) factor values for every asset, as of `as_of`.

        Only data with index <= as_of is touched. Assets with no price on
        `as_of` are excluded from the cross-section entirely.
        """
        as_of = pd.Timestamp(as_of)
        px_all = bundle.prices.loc[:as_of]
        if px_all.empty:
            return pd.DataFrame()

        tradable = px_all.iloc[-1].dropna().index
        if len(tradable) < self.cfg.min_cross_section:
            logger.warning(
                "cross-section of %d < min %d on %s",
                len(tradable),
                self.cfg.min_cross_section,
                as_of.date(),
            )
        px = px_all[tradable]
        # Require a usable price history for return-based factors.
        px = px.loc[:, px.notna().sum() >= 60]
        if px.empty:
            return pd.DataFrame()

        rets = px.pct_change().iloc[1:]
        bench_r = bundle.benchmark.loc[:as_of].pct_change().iloc[1:]
        vols = bundle.volumes.loc[:as_of, px.columns]

        values: dict[str, pd.Series] = {}
        values.update(
            momentum_factors(px, self.cfg.momentum_lookbacks, self.cfg.momentum_skip_days)
        )
        values.update(risk_factors(rets, self.cfg.volatility_lookback, self.cfg.drawdown_lookback))
        values.update(market_factors(rets, bench_r, px, vols, self.cfg.beta_lookback))

        fund_asof = bundle.fundamentals_asof(as_of)
        if len(fund_asof):
            last_px = px.iloc[-1]
            values.update(value_factors(fund_asof, last_px))
            hist = bundle.fundamentals
            if len(hist):
                hist = hist[pd.to_datetime(hist["available_from"]) <= as_of]
            values.update(quality_factors(fund_asof, hist, px.columns))

        panel = pd.DataFrame({k: v.reindex(px.columns).astype(float) for k, v in values.items()})
        # Keep registry column order for stable reports.
        ordered = [c for c in FACTOR_REGISTRY if c in panel.columns]
        return panel[ordered]

    # ------------------------------------------------------------------
    def normalize(self, panel: pd.DataFrame) -> pd.DataFrame:
        if panel.empty:
            return panel
        out = {}
        for col in panel.columns:
            w = winsorize_cross_section(panel[col], self.cfg.winsorize_quantile)
            out[col] = zscore_cross_section(w)
        return pd.DataFrame(out, index=panel.index)

    # ------------------------------------------------------------------
    def group_scores(self, normed: pd.DataFrame) -> pd.DataFrame:
        """Average the available z-scores within each factor group.

        Averaging (rather than summing) means a group with 2 observed factors is
        comparable to one with 6, so partial fundamental coverage does not
        mechanically shrink the value score toward zero.
        """
        cols = {}
        for g in GROUPS:
            members = [f for f in factors_in_group(g) if f in normed.columns]
            if members:
                cols[g] = normed[members].mean(axis=1, skipna=True)
        gs = pd.DataFrame(cols, index=normed.index)
        # Re-standardise so composite weights mean what they say.
        for c in gs.columns:
            gs[c] = zscore_cross_section(gs[c])
        return gs

    # ------------------------------------------------------------------
    def composite(self, groups: pd.DataFrame) -> pd.Series:
        weights = self.cfg.normalised_weights()
        used = {g: w for g, w in weights.items() if g in groups.columns}
        if not used:
            return pd.Series(0.0, index=groups.index)
        total = sum(used.values())
        acc = pd.Series(0.0, index=groups.index)
        for g, w in used.items():
            acc = acc + (w / total) * groups[g].fillna(0.0)
        return acc

    # ------------------------------------------------------------------
    def compute(
        self, bundle: MarketDataBundle, as_of: pd.Timestamp
    ) -> tuple[dict[str, FactorScore], pd.DataFrame]:
        """Full pipeline for one date.

        Returns
        -------
        (scores, diagnostics) where `diagnostics` is a frame with normalised
        factors, group scores, composite and rank — handy for reports and for
        the LLM layer, which reads numbers from here instead of inventing them.
        """
        as_of = pd.Timestamp(as_of)
        panel = self.raw_panel(bundle, as_of)
        if panel.empty:
            return {}, pd.DataFrame()

        normed = self.normalize(panel)
        groups = self.group_scores(normed)
        comp = self.composite(groups)
        coverage = panel.notna().mean(axis=1)

        rank = comp.rank(ascending=False, method="min").astype(int)
        n = len(comp)

        scores: dict[str, FactorScore] = {}
        for tk in panel.index:
            scores[tk] = FactorScore(
                ticker=str(tk),
                as_of=as_of.date(),
                raw_values={k: float(v) for k, v in panel.loc[tk].items() if pd.notna(v)},
                normalized={k: float(v) for k, v in normed.loc[tk].items() if pd.notna(v)},
                group_scores={k: float(v) for k, v in groups.loc[tk].items() if pd.notna(v)},
                composite=float(comp.loc[tk]),
                rank=int(rank.loc[tk]),
                n_universe=n,
                coverage=float(coverage.loc[tk]),
            )

        diag = normed.copy()
        for g in groups.columns:
            diag[f"grp_{g}"] = groups[g]
        diag["composite"] = comp
        diag["rank"] = rank
        diag["coverage"] = coverage
        return scores, diag.sort_values("composite", ascending=False)

    # ------------------------------------------------------------------
    def composite_series(self, bundle: MarketDataBundle, dates: list[pd.Timestamp]) -> pd.DataFrame:
        """Composite score panel over many dates (rows=date, cols=ticker)."""
        rows = {}
        for d in dates:
            _, diag = self.compute(bundle, d)
            if not diag.empty:
                rows[pd.Timestamp(d)] = diag["composite"]
        return pd.DataFrame(rows).T.sort_index()

    def group_series(
        self, bundle: MarketDataBundle, dates: list[pd.Timestamp], group: str
    ) -> pd.DataFrame:
        rows = {}
        col = f"grp_{group}"
        for d in dates:
            _, diag = self.compute(bundle, d)
            if not diag.empty and col in diag.columns:
                rows[pd.Timestamp(d)] = diag[col]
        return pd.DataFrame(rows).T.sort_index()
