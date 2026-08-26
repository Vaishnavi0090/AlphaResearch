"""Strategy definitions and the shared research context.

A `Strategy` is a pure function from a `DecisionContext` to target weights. All
strategies see exactly the same context object, so the ablation ladder
(Phase 24) isolates one component at a time instead of comparing accidentally
different pipelines.

The `ResearchContext` exists for a practical reason: factor panels and HMM
refits are the expensive part of a run, and every strategy in an ablation study
needs the *same* ones. Computing them once and sharing them makes a 6-strategy
ablation roughly 6x cheaper, and — more importantly — guarantees that
strategy B and strategy C are reading identical signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from finrisk.alpha.engine import AlphaEngine
from finrisk.config.settings import Settings
from finrisk.data.base import MarketDataBundle
from finrisk.factors.engine import FactorEngine
from finrisk.portfolio.optimizer import OptimizationInputs, PortfolioOptimizer
from finrisk.regimes.detector import RegimeDetector
from finrisk.risk import metrics as m
from finrisk.schemas import AlphaSignal, FactorScore, PortfolioWeights, RegimeState

logger = logging.getLogger(__name__)


# ======================================================================
# Shared context
# ======================================================================


@dataclass
class ResearchContext:
    """Point-in-time signals precomputed at every rebalance date.

    Every entry is built using only data at or before its own date. The
    dictionary is keyed by decision date, and strategies may only look up the
    key they are handed — enforced by `DecisionContext`.
    """

    dates: list[pd.Timestamp]
    factor_scores: dict[pd.Timestamp, dict[str, FactorScore]] = field(default_factory=dict)
    factor_diag: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
    regimes: dict[pd.Timestamp, RegimeState] = field(default_factory=dict)
    ic_estimates: dict[pd.Timestamp, float] = field(default_factory=dict)
    composite_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    group_panels: dict[str, pd.DataFrame] = field(default_factory=dict)


def build_research_context(
    bundle: MarketDataBundle,
    dates: list[pd.Timestamp],
    settings: Settings,
    estimate_ic: bool = True,
    ic_refresh_every: int = 12,
) -> ResearchContext:
    """Precompute factor scores, regimes and rolling IC estimates.

    The IC used to scale alpha is re-estimated from an *expanding* window of
    past rebalance dates, refreshed annually. It never uses a date at or after
    the decision date, which is what keeps the alpha scale out-of-sample.
    """
    fe = FactorEngine(settings.factors)
    rd = RegimeDetector(settings.regimes)
    rd.prepare(bundle.benchmark)
    ae = AlphaEngine(fe)

    ctx = ResearchContext(dates=list(dates))
    comp_rows: dict[pd.Timestamp, pd.Series] = {}
    group_rows: dict[str, dict[pd.Timestamp, pd.Series]] = {}

    ic_current = ae.default_ic
    for i, d in enumerate(dates):
        d = pd.Timestamp(d)
        scores, diag = fe.compute(bundle, d)
        if not scores:
            continue
        ctx.factor_scores[d] = scores
        ctx.factor_diag[d] = diag
        ctx.regimes[d] = rd.detect(bundle.benchmark, d)
        comp_rows[d] = diag["composite"]
        for col in [c for c in diag.columns if c.startswith("grp_")]:
            group_rows.setdefault(col[4:], {})[d] = diag[col]

        if estimate_ic and i >= 12 and i % ic_refresh_every == 0:
            # Strictly past dates only, and a forward horizon that has fully
            # elapsed before `d` (drop the last 2 dates to be safe).
            past = list(dates[: max(i - 2, 0)])
            if len(past) >= 12:
                past_panel = pd.DataFrame(
                    {p: comp_rows[p] for p in past if p in comp_rows}
                ).T.sort_index()
                if len(past_panel) >= 12:
                    from finrisk.factors.research import (
                        forward_returns,
                        information_coefficient,
                    )

                    fwd = forward_returns(bundle.prices.loc[:d], list(past_panel.index), 21)
                    ic = information_coefficient(past_panel, fwd)
                    if len(ic) >= 6:
                        ic_current = float(np.clip(ic.mean(), 0.0, 0.15))
        ctx.ic_estimates[d] = ic_current

    ctx.composite_panel = pd.DataFrame(comp_rows).T.sort_index()
    ctx.group_panels = {g: pd.DataFrame(rows).T.sort_index() for g, rows in group_rows.items()}
    return ctx


@dataclass
class DecisionContext:
    """Everything a strategy may see at one decision date. Nothing else."""

    as_of: pd.Timestamp
    bundle: MarketDataBundle  # already sliced to <= as_of
    settings: Settings
    factor_scores: dict[str, FactorScore]
    regime: RegimeState | None
    ic_estimate: float
    prev_weights: dict[str, float]
    trailing_returns: pd.DataFrame  # asset returns, <= as_of
    benchmark_returns: pd.Series  # <= as_of
    risk_free: float


# ======================================================================
# Strategy protocol
# ======================================================================


class Strategy(Protocol):
    name: str
    description: str

    def decide(self, ctx: DecisionContext) -> PortfolioWeights: ...


def _covariance(ctx: DecisionContext, tickers: list[str]) -> pd.DataFrame:
    cfg = ctx.settings.portfolio
    hist = ctx.trailing_returns[tickers].tail(cfg.covariance_lookback).dropna(axis=1, how="all")
    cov = m.estimate_covariance(hist, cfg.covariance_estimator, annualize=True)
    return cov.reindex(index=tickers, columns=tickers).fillna(0.0)


def _scenarios(ctx: DecisionContext, tickers: list[str], n: int = 504) -> np.ndarray:
    return ctx.trailing_returns[tickers].tail(n).dropna().to_numpy(dtype=float)


def _build_inputs(
    ctx: DecisionContext,
    tickers: list[str],
    expected: pd.Series,
) -> OptimizationInputs:
    cov = _covariance(ctx, tickers)
    return OptimizationInputs(
        tickers=tickers,
        expected_returns=expected.reindex(tickers).fillna(0.0).to_numpy(float),
        cov=cov.to_numpy(float),
        scenario_returns=_scenarios(ctx, tickers),
        sectors=ctx.bundle.sector_map(),
        risk_free=ctx.risk_free,
        prev_weights=ctx.prev_weights,
    )


# ======================================================================
# Concrete strategies (the ablation ladder)
# ======================================================================


class BuyHoldBenchmark:
    name = "benchmark_buy_hold"
    description = "Hold the benchmark index. The bar every strategy must clear."

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        return PortfolioWeights(
            as_of=ctx.as_of.date(),
            method="benchmark",
            weights={"__BENCHMARK__": 1.0},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe=0.0,
        )


class EqualWeightUniverse:
    """A: equal weight over the whole investable universe. No signal at all."""

    name = "A_equal_weight_universe"
    description = "1/N over every tradable asset. Zero estimation error."

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        tickers = sorted(ctx.factor_scores)
        if not tickers:
            return _empty(ctx)
        opt = PortfolioOptimizer(
            ctx.settings.portfolio.model_copy(update={"max_weight": 1.0, "target_volatility": None})
        )
        inp = _build_inputs(ctx, tickers, pd.Series(0.0, index=tickers))
        return opt.equal_weight(inp, ctx.as_of)


class FactorEqualWeight:
    """B: factor selection, equal weight. Isolates signal from sizing."""

    name = "B_factor_equal_weight"
    description = "Top-N by composite factor score, equal weighted."

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        top = _top_n(ctx)
        if not top:
            return _empty(ctx)
        opt = PortfolioOptimizer(
            ctx.settings.portfolio.model_copy(update={"target_volatility": None})
        )
        inp = _build_inputs(ctx, top, pd.Series(0.0, index=top))
        return opt.equal_weight(inp, ctx.as_of)


class FactorRegimeEqualWeight:
    """C: B plus the regime overlay. Isolates the value of the regime model."""

    name = "C_factor_regime_equal_weight"
    description = "Top-N factor, equal weight, gross exposure scaled by regime."

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        top = _top_n(ctx)
        if not top:
            return _empty(ctx)
        opt = PortfolioOptimizer(
            ctx.settings.portfolio.model_copy(update={"target_volatility": None})
        )
        inp = _build_inputs(ctx, top, pd.Series(0.0, index=top))
        pw = opt.equal_weight(inp, ctx.as_of)
        mult = RegimeDetector().risk_multiplier(ctx.regime) if ctx.regime is not None else 1.0
        return _scale(pw, mult, "C_factor_regime_equal_weight")


class FactorMeanVariance:
    """D: B plus mean-variance sizing. Isolates the value of optimisation."""

    name = "D_factor_mean_variance"
    description = "Top-N factor, mean-variance optimised, fully invested (no vol target)."

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        top = _top_n(ctx)
        if not top:
            return _empty(ctx)
        expected = _alpha(ctx, top, use_regime=False)
        # Fully invested (budget equality). Without it, the mean-variance
        # objective w'mu - (lam/2) w'Sigma w with realistic alphas (~1% p.a.)
        # and lam=5 is optimised by holding almost entirely cash, which is a
        # correct solution to the stated problem but a useless comparison for
        # the ablation: it would measure "cash vs equities", not "optimisation
        # vs equal weight". The textbook MVO formulation carries sum(w)=1.
        opt = PortfolioOptimizer(
            ctx.settings.portfolio.model_copy(
                update={"target_volatility": None, "cash_allowed": False}
            )
        )
        return opt.mean_variance(_build_inputs(ctx, top, expected), ctx.as_of)


class FactorRiskParity:
    name = "D2_factor_risk_parity"
    description = "Top-N factor, equal risk contribution. Uses no return forecast."

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        top = _top_n(ctx)
        if not top:
            return _empty(ctx)
        opt = PortfolioOptimizer(
            ctx.settings.portfolio.model_copy(
                update={"target_volatility": None, "cash_allowed": False}
            )
        )
        return opt.risk_parity(_build_inputs(ctx, top, pd.Series(0.0, index=top)), ctx.as_of)


class FactorRiskConstrained:
    """E: D plus the volatility ceiling. Isolates the value of the risk limit."""

    name = "E_factor_optimized_risk_constrained"
    description = "Top-N factor, maximise alpha subject to a volatility target."

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        top = _top_n(ctx)
        if not top:
            return _empty(ctx)
        expected = _alpha(ctx, top, use_regime=False)
        opt = PortfolioOptimizer(ctx.settings.portfolio)
        return opt.risk_constrained(_build_inputs(ctx, top, expected), ctx.as_of)


class FinriskFull:
    """F: the full pipeline — factors + regime + risk-constrained optimisation,
    with a CVaR limit enforced when one is configured."""

    name = "F_finrisk_full"
    description = (
        "Factors -> regime-scaled alpha -> risk-constrained (or CVaR-constrained) "
        "optimisation -> regime exposure overlay."
    )

    def decide(self, ctx: DecisionContext) -> PortfolioWeights:
        top = _top_n(ctx)
        if not top:
            return _empty(ctx)
        expected = _alpha(ctx, top, use_regime=True)
        cfg = ctx.settings.portfolio
        opt = PortfolioOptimizer(cfg)
        inp = _build_inputs(ctx, top, expected)
        pw = (
            opt.cvar_constrained(inp, ctx.as_of)
            if cfg.max_cvar_95 is not None
            else opt.risk_constrained(inp, ctx.as_of)
        )
        mult = RegimeDetector().risk_multiplier(ctx.regime) if ctx.regime is not None else 1.0
        return _scale(pw, mult, "F_finrisk_full")


# ======================================================================
# Helpers
# ======================================================================


def _top_n(ctx: DecisionContext) -> list[str]:
    """Top-N by composite score, deterministic tie-break, tradable only."""
    scores = ctx.factor_scores
    if not scores:
        return []
    avail = set(ctx.trailing_returns.columns)
    ranked = sorted(
        (s for s in scores.values() if s.ticker in avail),
        key=lambda s: (-s.composite, s.ticker),
    )
    return [s.ticker for s in ranked[: ctx.settings.portfolio.top_n]]


def _alpha(ctx: DecisionContext, tickers: list[str], use_regime: bool) -> pd.Series:
    ae = AlphaEngine(FactorEngine(ctx.settings.factors))
    sub = {t: ctx.factor_scores[t] for t in tickers if t in ctx.factor_scores}
    signals: dict[str, AlphaSignal] = ae.generate(
        ctx.bundle,
        ctx.as_of,
        scores=sub,
        regime=ctx.regime,
        ic=ctx.ic_estimate,
        use_regime=use_regime,
    )
    return AlphaEngine.to_expected_returns(signals).reindex(tickers).fillna(0.0)


def _scale(pw: PortfolioWeights, mult: float, method: str) -> PortfolioWeights:
    """Scale gross exposure, moving the remainder to cash."""
    mult = float(np.clip(mult, 0.0, 1.0))
    w = {k: v * mult for k, v in pw.weights.items()}
    return pw.model_copy(
        update={
            "weights": w,
            "method": method,
            "cash": float(max(1.0 - sum(w.values()), 0.0)),
            "expected_return": pw.expected_return * mult,
            "expected_volatility": pw.expected_volatility * mult,
            "diagnostics": {**pw.diagnostics, "regime_multiplier": mult},
        }
    )


def _empty(ctx: DecisionContext) -> PortfolioWeights:
    return PortfolioWeights(
        as_of=ctx.as_of.date(),
        method="cash",
        weights={},
        expected_return=0.0,
        expected_volatility=0.0,
        sharpe=0.0,
        cash=1.0,
        optimization_status="no_investable_universe",
    )


ABLATION_LADDER: list[type] = [
    EqualWeightUniverse,
    FactorEqualWeight,
    FactorRegimeEqualWeight,
    FactorMeanVariance,
    FactorRiskParity,
    FactorRiskConstrained,
    FinriskFull,
]


def all_strategies() -> dict[str, Strategy]:
    out: dict[str, Strategy] = {}
    for cls in [BuyHoldBenchmark, *ABLATION_LADDER]:
        inst = cls()
        out[inst.name] = inst
    return out
