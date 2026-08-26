"""Agent tools (Phase 17).

Every tool here wraps a **deterministic** engine. Inputs and outputs are
Pydantic models, so an agent's call is validated before it reaches the
quantitative core and its result is a typed object rather than prose.

This is the mechanism behind Phase 20: the agent layer can *request* a
calculation but cannot *perform* one. If an agent wants a CVaR number it must
call `calculate_cvar`, and the number it receives is the same number the
backtester would compute. There is no path by which a language model's token
predictions become a portfolio weight.

Tools are plain callables plus a schema, deliberately not tied to LangChain's
`BaseTool`. `to_langchain_tools()` adapts them when LangChain is installed, and
`tests/test_tools.py` exercises them with no LLM dependency at all.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from finrisk.alpha.engine import AlphaEngine
from finrisk.config.settings import Settings
from finrisk.data.base import MarketDataBundle
from finrisk.factors.engine import FactorEngine
from finrisk.portfolio.optimizer import OptimizationInputs, PortfolioOptimizer
from finrisk.rag.retriever import DocumentRetriever
from finrisk.regimes.detector import RegimeDetector
from finrisk.risk import attribution as attr
from finrisk.risk import metrics as m
from finrisk.risk import stress as stress_mod
from finrisk.risk.engine import RiskEngine

logger = logging.getLogger(__name__)


# ======================================================================
# Shared context
# ======================================================================


@dataclass
class ToolContext:
    """Everything the tools need. Constructed once per session."""

    bundle: MarketDataBundle
    settings: Settings
    retriever: DocumentRetriever | None = None
    as_of: pd.Timestamp | None = None

    def date(self, as_of: str | None = None) -> pd.Timestamp:
        """Resolve a requested date to an actual trading session <= it.

        Snapping to the previous session (rather than erroring) means an agent
        asking about a weekend gets the last real close, and — critically — can
        never obtain data from *after* the date it asked about.
        """
        target = pd.Timestamp(as_of) if as_of else (self.as_of or self.bundle.dates[-1])
        idx = self.bundle.dates
        pos = idx.searchsorted(target, side="right") - 1
        if pos < 0:
            raise ValueError(f"no trading session on or before {target.date()}")
        return idx[pos]

    def trailing_returns(self, as_of: pd.Timestamp, lookback: int = 504) -> pd.DataFrame:
        return self.bundle.prices.loc[:as_of].pct_change().iloc[1:].tail(lookback)


# ======================================================================
# Schemas
# ======================================================================


class MarketDataInput(BaseModel):
    tickers: list[str] = Field(default_factory=list, description="Empty = whole universe")
    as_of: str | None = Field(None, description="ISO date; defaults to latest session")
    lookback_days: int = Field(252, ge=5, le=5000)


class FundamentalsInput(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    as_of: str | None = None


class FactorScoreInput(BaseModel):
    as_of: str | None = None
    # Ceiling covers a full universe request (used by /factors/{ticker}, which
    # needs the whole cross-section to locate one name's rank).
    top_n: int = Field(10, ge=1, le=2000)


class RegimeInput(BaseModel):
    as_of: str | None = None


class PortfolioInput(BaseModel):
    weights: dict[str, float] = Field(..., description="ticker -> weight")
    as_of: str | None = None
    lookback_days: int = Field(504, ge=60, le=5000)


class VarInput(PortfolioInput):
    confidence: float = Field(0.95, gt=0.5, lt=1.0)


class OptimizeInput(BaseModel):
    method: str = Field("risk_constrained")
    tickers: list[str] = Field(default_factory=list, description="Empty = top-N by factor score")
    as_of: str | None = None
    max_weight: float | None = None
    target_volatility: float | None = None
    max_cvar_95: float | None = None
    use_regime: bool = True


class BacktestInput(BaseModel):
    strategy: str = Field("F_finrisk_full")
    rebalance_frequency: str = Field("M", pattern="^[MQW]$")


class RetrieveInput(BaseModel):
    query: str = Field(..., min_length=3)
    ticker: str | None = None
    doc_type: str | None = None
    top_k: int = Field(5, ge=1, le=20)
    # Narrow to a filing section (e.g. "Risk Factors") and/or pin the retrieval
    # path. Both default to None so existing callers are unaffected.
    section: str | None = None
    mode: str | None = Field(None, pattern="^(bm25|dense|hybrid)$")


class StressInput(PortfolioInput):
    include_historical: bool = True


# ======================================================================
# Tool implementations
# ======================================================================


def get_market_data(ctx: ToolContext, inp: MarketDataInput) -> dict[str, Any]:
    """Prices, returns and summary statistics for a set of tickers."""
    d = ctx.date(inp.as_of)
    px = ctx.bundle.prices.loc[:d]
    cols = [t for t in (inp.tickers or px.columns) if t in px.columns]
    if not cols:
        return {"error": "no valid tickers", "requested": inp.tickers}
    px = px[cols].tail(inp.lookback_days)
    rets = px.pct_change().iloc[1:]
    return {
        "as_of": str(d.date()),
        "n_sessions": len(px),
        "tickers": cols,
        "last_price": {t: round(float(px[t].iloc[-1]), 4) for t in cols},
        "total_return": {t: round(float(px[t].iloc[-1] / px[t].iloc[0] - 1), 6) for t in cols},
        "ann_volatility": {t: round(float(m.annualized_volatility(rets[t])), 6) for t in cols},
        "benchmark_total_return": round(
            float(
                ctx.bundle.benchmark.loc[:d].iloc[-1]
                / ctx.bundle.benchmark.loc[:d].tail(inp.lookback_days).iloc[0]
                - 1
            ),
            6,
        ),
        "provider": ctx.bundle.provider,
    }


def get_fundamentals(ctx: ToolContext, inp: FundamentalsInput) -> dict[str, Any]:
    """Latest point-in-time fundamentals. Never returns unpublished filings."""
    d = ctx.date(inp.as_of)
    f = ctx.bundle.fundamentals_asof(d)
    if not len(f):
        return {"as_of": str(d.date()), "records": {}, "note": "no fundamentals available"}
    tickers = [t for t in (inp.tickers or f.index) if t in f.index]
    out = {}
    for t in tickers:
        row = f.loc[t]
        eps = float(row["eps"])
        eq = float(row["total_equity_per_share"])
        out[t] = {
            "period_end": str(row["period_end"]),
            "available_from": str(row["available_from"]),
            "eps": round(eps, 4),
            "book_value_per_share": round(float(row["book_value_per_share"]), 4),
            "free_cash_flow_per_share": round(float(row["free_cash_flow_per_share"]), 4),
            "debt_to_equity": round(
                float(row["total_debt_per_share"]) / eq if abs(eq) > 1e-9 else float("nan"), 4
            ),
            "roe": round(eps / eq if abs(eq) > 1e-9 else float("nan"), 4),
        }
    return {"as_of": str(d.date()), "records": out}


def calculate_factor_scores(ctx: ToolContext, inp: FactorScoreInput) -> dict[str, Any]:
    """Point-in-time factor cross-section with group and composite scores."""
    d = ctx.date(inp.as_of)
    fe = FactorEngine(ctx.settings.factors)
    scores, diag = fe.compute(ctx.bundle, d)
    if not scores:
        return {"error": "no factor cross-section available", "as_of": str(d.date())}
    top = diag.head(inp.top_n)
    return {
        "as_of": str(d.date()),
        "n_universe": len(scores),
        "composite_weights": fe.cfg.normalised_weights(),
        "ranking": [
            {
                "ticker": str(t),
                "rank": int(row["rank"]),
                "composite": round(float(row["composite"]), 4),
                "coverage": round(float(row["coverage"]), 3),
                **{c[4:]: round(float(row[c]), 4) for c in diag.columns if c.startswith("grp_")},
            }
            for t, row in top.iterrows()
        ],
    }


def detect_market_regime(ctx: ToolContext, inp: RegimeInput) -> dict[str, Any]:
    """HMM regime state using filtered (point-in-time) probabilities."""
    d = ctx.date(inp.as_of)
    rd = RegimeDetector(ctx.settings.regimes)
    rd.prepare(ctx.bundle.benchmark)
    st = rd.detect(ctx.bundle.benchmark, d)
    return {
        "as_of": str(st.as_of),
        "regime": st.regime,
        "confidence": round(st.confidence, 4),
        "state_probabilities": {k: round(v, 4) for k, v in st.state_probabilities.items()},
        "transition_probabilities": {
            k: round(v, 4) for k, v in st.transition_probabilities.items()
        },
        "expected_duration_days": (
            round(st.expected_duration_days, 1) if st.expected_duration_days else None
        ),
        "features": {k: round(v, 5) for k, v in st.features.items()},
        "risk_multiplier": round(rd.risk_multiplier(st), 4),
        "inference": "filtered (forward-only) — no future information used",
    }


def calculate_risk(ctx: ToolContext, inp: PortfolioInput) -> dict[str, Any]:
    """Full risk report for a supplied weight vector."""
    d = ctx.date(inp.as_of)
    rets = ctx.trailing_returns(d, inp.lookback_days)
    bench = ctx.bundle.benchmark.loc[:d].pct_change().iloc[1:].tail(inp.lookback_days)
    rep = RiskEngine().analyze_portfolio(
        weights=inp.weights,
        asset_returns=rets,
        benchmark_returns=bench,
        sectors=ctx.bundle.sector_map(),
        risk_free_annual=ctx.bundle.risk_free.loc[:d],
        as_of=d,
    )
    return {
        "as_of": str(rep.as_of),
        "n_observations": rep.n_observations,
        "ann_volatility": round(rep.ann_volatility, 6),
        "model_volatility": (
            None if rep.model_volatility is None else round(rep.model_volatility, 6)
        ),
        "ann_return": round(rep.ann_return, 6),
        "sharpe": round(rep.sharpe, 4),
        "sortino": round(rep.sortino, 4),
        "max_drawdown": round(rep.max_drawdown, 6),
        "calmar": round(rep.calmar, 4),
        "beta": None if rep.beta is None else round(rep.beta, 4),
        "var_95_hist": round(rep.var_95_hist, 6),
        "var_95_param": round(rep.var_95_param, 6),
        "cvar_95": round(rep.cvar_95, 6),
        "cvar_99": round(rep.cvar_99, 6),
        "effective_n": round(rep.effective_n, 3),
        "mean_correlation": (
            None if rep.mean_correlation is None else round(rep.mean_correlation, 4)
        ),
        "sector_risk": {k: round(v, 4) for k, v in rep.sector_risk.items()},
        "concentration_flags": RiskEngine().risk_budget_check(rep),
    }


def calculate_var(ctx: ToolContext, inp: VarInput) -> dict[str, Any]:
    """Historical, parametric and Cornish-Fisher VaR side by side."""
    d = ctx.date(inp.as_of)
    rets = ctx.trailing_returns(d, inp.lookback_days)
    port = m.portfolio_returns_series(pd.Series(inp.weights), rets)
    if len(port) < 20:
        return {"error": "insufficient history", "n_observations": len(port)}
    hist = m.historical_var(port, inp.confidence)
    param = m.parametric_var(port, inp.confidence)
    return {
        "as_of": str(d.date()),
        "confidence": inp.confidence,
        "horizon": "1 day",
        "var_historical": round(hist, 6),
        "var_parametric_gaussian": round(param, 6),
        "var_cornish_fisher": round(m.cornish_fisher_var(port, inp.confidence), 6),
        "gaussian_understates_tail": bool(param > hist),
        "skew": round(float(port.skew()), 4),
        "excess_kurtosis": round(float(port.kurtosis()), 4),
        "note": "negative values denote losses",
    }


def calculate_cvar(ctx: ToolContext, inp: VarInput) -> dict[str, Any]:
    """CVaR (expected shortfall) plus per-position tail attribution."""
    d = ctx.date(inp.as_of)
    rets = ctx.trailing_returns(d, inp.lookback_days)
    port = m.portfolio_returns_series(pd.Series(inp.weights), rets)
    if len(port) < 20:
        return {"error": "insufficient history", "n_observations": len(port)}
    return {
        "as_of": str(d.date()),
        "confidence": inp.confidence,
        "cvar": round(m.conditional_var(port, inp.confidence), 6),
        "var": round(m.historical_var(port, inp.confidence), 6),
        "worst_day": round(float(port.min()), 6),
        "tail_contribution": {
            k: round(v, 4)
            for k, v in sorted(
                attr.tail_contribution(inp.weights, rets, inp.confidence).items(),
                key=lambda kv: -kv[1],
            )[:10]
        },
    }


def calculate_risk_contribution(ctx: ToolContext, inp: PortfolioInput) -> dict[str, Any]:
    """Euler risk decomposition: marginal, component and percentage."""
    d = ctx.date(inp.as_of)
    rets = ctx.trailing_returns(d, inp.lookback_days)
    rep = RiskEngine().analyze_portfolio(
        inp.weights, rets, sectors=ctx.bundle.sector_map(), as_of=d
    )
    total = sum(c.component_contribution for c in rep.contributions)
    model_vol = rep.model_volatility or 0.0
    return {
        "as_of": str(d.date()),
        # Model volatility (from the shrunk covariance) is what the Euler
        # decomposition sums to. Realised volatility is reported alongside it
        # so the estimation gap is visible rather than hidden.
        "model_volatility": round(model_vol, 6),
        "realised_volatility": round(rep.ann_volatility, 6),
        "sum_of_components": round(float(total), 6),
        "euler_identity_holds": bool(abs(total - model_vol) < 1e-8 + 1e-6 * max(model_vol, 1.0)),
        "contributions": [
            {
                "ticker": c.ticker,
                "weight": round(c.weight, 5),
                "marginal": round(c.marginal_contribution, 6),
                "component": round(c.component_contribution, 6),
                "pct_of_risk": round(c.pct_of_risk, 5),
            }
            for c in sorted(rep.contributions, key=lambda x: -x.pct_of_risk)
        ],
        "sector_risk": {k: round(v, 4) for k, v in rep.sector_risk.items()},
        "correlation_clusters": attr.correlation_concentration(inp.weights, rets),
    }


def optimize_portfolio(ctx: ToolContext, inp: OptimizeInput) -> dict[str, Any]:
    """Run the deterministic optimiser. The ONLY way weights get produced."""
    d = ctx.date(inp.as_of)
    fe = FactorEngine(ctx.settings.factors)
    scores, diag = fe.compute(ctx.bundle, d)
    if not scores:
        return {"error": "no factor cross-section", "as_of": str(d.date())}

    cfg = ctx.settings.portfolio.model_copy(
        update={
            k: v
            for k, v in {
                "max_weight": inp.max_weight,
                "target_volatility": inp.target_volatility,
                "max_cvar_95": inp.max_cvar_95,
            }.items()
            if v is not None
        }
    )
    rets = ctx.trailing_returns(d, cfg.covariance_lookback)
    universe = [t for t in (inp.tickers or list(diag.index[: cfg.top_n])) if t in rets.columns]
    if not universe:
        return {"error": "no investable tickers", "as_of": str(d.date())}

    regime = None
    if inp.use_regime:
        rd = RegimeDetector(ctx.settings.regimes)
        rd.prepare(ctx.bundle.benchmark)
        regime = rd.detect(ctx.bundle.benchmark, d)

    ae = AlphaEngine(fe)
    signals = ae.generate(
        ctx.bundle,
        d,
        scores={t: scores[t] for t in universe if t in scores},
        regime=regime,
        use_regime=inp.use_regime,
    )
    expected = AlphaEngine.to_expected_returns(signals).reindex(universe).fillna(0.0)

    cov = m.estimate_covariance(rets[universe], cfg.covariance_estimator, annualize=True)
    opt = PortfolioOptimizer(cfg)
    oi = OptimizationInputs(
        tickers=universe,
        expected_returns=expected.to_numpy(float),
        cov=cov.reindex(index=universe, columns=universe).fillna(0.0).to_numpy(float),
        scenario_returns=rets[universe].dropna().to_numpy(float),
        sectors=ctx.bundle.sector_map(),
        risk_free=float(ctx.bundle.risk_free.loc[:d].iloc[-1]),
    )
    pw = opt.optimize(oi, d, inp.method)
    return {
        "as_of": str(pw.as_of),
        "method": pw.method,
        "status": pw.optimization_status,
        "weights": {k: round(v, 5) for k, v in pw.active().items()},
        "cash": round(pw.cash, 5),
        # Per-asset alpha forecasts (Grinold-Kahn), surfaced so the
        # explainability layer can report each position's expected-return
        # contribution rather than only its weight.
        "alphas": {t: round(float(v), 6) for t, v in expected.items() if t in pw.active()},
        "expected_return_ann": round(pw.expected_return, 6),
        "expected_volatility_ann": round(pw.expected_volatility, 6),
        "sharpe_ex_ante": round(pw.sharpe, 4),
        "risk_metrics": {k: round(v, 6) for k, v in pw.risk_metrics.items()},
        "constraints_satisfied": pw.constraints.feasible if pw.constraints else None,
        "violations": pw.constraints.violations if pw.constraints else [],
        "regime": None if regime is None else regime.regime,
    }


def run_backtest(ctx: ToolContext, inp: BacktestInput) -> dict[str, Any]:
    """Walk-forward backtest of a named strategy."""
    from finrisk.backtesting.engine import WalkForwardBacktester
    from finrisk.backtesting.strategies import all_strategies

    strategies = all_strategies()
    if inp.strategy not in strategies:
        return {"error": f"unknown strategy {inp.strategy}", "available": list(strategies)}
    settings = ctx.settings.model_copy(deep=True)
    settings.backtest.rebalance_frequency = inp.rebalance_frequency  # type: ignore[assignment]
    bt = WalkForwardBacktester(settings)
    res = bt.run(ctx.bundle, strategies[inp.strategy])
    mt = res.metrics
    return {
        "strategy": res.strategy,
        "config_digest": res.config_digest,
        "period": f"{mt.start} to {mt.end}",
        "n_rebalances": res.n_rebalances,
        "cagr": round(mt.cagr, 6),
        "ann_volatility": round(mt.ann_volatility, 6),
        "sharpe": round(mt.sharpe, 4),
        "sortino": round(mt.sortino, 4),
        "max_drawdown": round(mt.max_drawdown, 6),
        "calmar": round(mt.calmar, 4),
        "cvar_95": round(mt.cvar_95, 6),
        "avg_turnover": round(mt.avg_turnover, 5),
        "total_cost_drag": round(mt.total_cost_drag, 6),
        "beta": None if mt.beta is None else round(mt.beta, 4),
        "information_ratio": (
            None if mt.information_ratio is None else round(mt.information_ratio, 4)
        ),
        "warnings": res.warnings,
    }


def retrieve_financial_documents(ctx: ToolContext, inp: RetrieveInput) -> dict[str, Any]:
    """Retrieve document evidence with citations."""
    if ctx.retriever is None or ctx.retriever.n_chunks == 0:
        return {"error": "no document index available", "citations": []}
    cits = ctx.retriever.retrieve(
        inp.query,
        inp.top_k,
        inp.ticker,
        inp.doc_type,
        section=inp.section,
        mode=inp.mode,
    )
    return {
        "query": inp.query,
        "n_results": len(cits),
        "citations": [c.model_dump() for c in cits],
        "corpus": {
            "documents": ctx.retriever.n_documents,
            "chunks": ctx.retriever.n_chunks,
            "mode": ctx.retriever.cfg.retriever,
            "encoder": ctx.retriever.encoder_name,
        },
    }


def stress_test(ctx: ToolContext, inp: StressInput) -> dict[str, Any]:
    """Historical replay and hypothetical shock scenarios."""
    d = ctx.date(inp.as_of)
    rets = ctx.trailing_returns(d, inp.lookback_days)
    bench = ctx.bundle.benchmark.loc[:d].pct_change().iloc[1:].tail(inp.lookback_days)
    scen = stress_mod.run_stress_suite(
        inp.weights, rets, bench, ctx.bundle.sector_map(), inp.include_historical
    )
    return {
        "as_of": str(d.date()),
        "scenarios": [
            {
                "name": s.name,
                "kind": s.kind,
                "description": s.description,
                "portfolio_return": round(s.portfolio_return, 6),
                "benchmark_return": (
                    None if s.benchmark_return is None else round(s.benchmark_return, 6)
                ),
                "max_drawdown": round(s.max_drawdown, 6),
                "cvar_95": round(s.cvar_95, 6),
                "worst_contributors": {k: round(v, 6) for k, v in s.worst_contributors.items()},
            }
            for s in scen
        ],
        "note": "conditional scenarios, not forecasts",
    }


def analyze_portfolio(ctx: ToolContext, inp: PortfolioInput) -> dict[str, Any]:
    """Composite view: risk + factor exposures + attribution. The tool the
    report agent uses when it needs the whole picture at once."""
    d = ctx.date(inp.as_of)
    rets = ctx.trailing_returns(d, inp.lookback_days)
    bench = ctx.bundle.benchmark.loc[:d].pct_change().iloc[1:].tail(inp.lookback_days)

    fe = FactorEngine(ctx.settings.factors)
    _, diag = fe.compute(ctx.bundle, d)
    exposures: dict[str, float] = {}
    if not diag.empty:
        w = pd.Series(inp.weights)
        for col in [c for c in diag.columns if c.startswith("grp_")]:
            common = [t for t in w.index if t in diag.index]
            if common:
                exposures[col[4:]] = round(float((w[common] * diag.loc[common, col]).sum()), 4)

    risk = calculate_risk(
        ctx, PortfolioInput(weights=inp.weights, as_of=inp.as_of, lookback_days=inp.lookback_days)
    )
    contrib = calculate_risk_contribution(
        ctx, PortfolioInput(weights=inp.weights, as_of=inp.as_of, lookback_days=inp.lookback_days)
    )

    # Factor risk attribution is a TIME-SERIES regression, so it needs factor
    # returns over the whole lookback. Building the factor-mimicking portfolios
    # from a single date's scores yields one weight row, hence a degenerate
    # regression and a meaningless R^2 of zero. Score panels are therefore
    # rebuilt on a monthly grid across the lookback window.
    panel_dates = _monthly_grid(rets.index)
    group_panels = _group_panels(fe, ctx, panel_dates)
    port_ret = m.portfolio_returns_series(pd.Series(inp.weights), rets)
    factor_rets = attr.build_factor_returns(group_panels, rets, bench)
    decomp: dict[str, Any] = attr.factor_risk_decomposition(port_ret, factor_rets)

    return {
        "as_of": str(d.date()),
        "n_positions": len([v for v in inp.weights.values() if abs(v) > 1e-9]),
        "gross_exposure": round(float(sum(abs(v) for v in inp.weights.values())), 5),
        "risk": risk,
        "factor_exposures": exposures,
        "factor_risk_decomposition": {
            "r_squared": round(float(decomp.get("r_squared", 0.0)), 4),
            "factor_share": {k: round(v, 4) for k, v in (decomp.get("factor_share") or {}).items()},
            "residual_share": round(float(decomp.get("residual_share", 1.0)), 4),
        },
        "top_risk_contributors": contrib["contributions"][:5],
    }


def _monthly_grid(index: pd.DatetimeIndex, max_points: int = 30) -> list[pd.Timestamp]:
    """Last session of each month within `index`, capped for cost control."""
    if not len(index):
        return []
    s = pd.Series(index, index=index)
    dates = s.groupby([index.year, index.month]).last().tolist()
    dates = [pd.Timestamp(d) for d in sorted(dates)]
    return dates[-max_points:]


def _group_panels(
    fe: FactorEngine, ctx: ToolContext, dates: list[pd.Timestamp]
) -> dict[str, pd.DataFrame]:
    """Group-score panels (rows=date, cols=ticker) over a set of dates."""
    rows: dict[str, dict[pd.Timestamp, pd.Series]] = {}
    for d in dates:
        _, diag = fe.compute(ctx.bundle, d)
        if diag.empty:
            continue
        for col in [c for c in diag.columns if c.startswith("grp_")]:
            rows.setdefault(col[4:], {})[d] = diag[col]
    return {g: pd.DataFrame(v).T.sort_index() for g, v in rows.items() if v}


# ======================================================================
# Registry
# ======================================================================


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], dict[str, Any]]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec("get_market_data", get_market_data.__doc__ or "", MarketDataInput, get_market_data),
    ToolSpec(
        "get_fundamentals", get_fundamentals.__doc__ or "", FundamentalsInput, get_fundamentals
    ),
    ToolSpec(
        "calculate_factor_scores",
        calculate_factor_scores.__doc__ or "",
        FactorScoreInput,
        calculate_factor_scores,
    ),
    ToolSpec(
        "detect_market_regime",
        detect_market_regime.__doc__ or "",
        RegimeInput,
        detect_market_regime,
    ),
    ToolSpec("calculate_risk", calculate_risk.__doc__ or "", PortfolioInput, calculate_risk),
    ToolSpec("calculate_var", calculate_var.__doc__ or "", VarInput, calculate_var),
    ToolSpec("calculate_cvar", calculate_cvar.__doc__ or "", VarInput, calculate_cvar),
    ToolSpec(
        "calculate_risk_contribution",
        calculate_risk_contribution.__doc__ or "",
        PortfolioInput,
        calculate_risk_contribution,
    ),
    ToolSpec(
        "optimize_portfolio", optimize_portfolio.__doc__ or "", OptimizeInput, optimize_portfolio
    ),
    ToolSpec("run_backtest", run_backtest.__doc__ or "", BacktestInput, run_backtest),
    ToolSpec(
        "retrieve_financial_documents",
        retrieve_financial_documents.__doc__ or "",
        RetrieveInput,
        retrieve_financial_documents,
    ),
    ToolSpec("stress_test", stress_test.__doc__ or "", StressInput, stress_test),
    ToolSpec(
        "analyze_portfolio", analyze_portfolio.__doc__ or "", PortfolioInput, analyze_portfolio
    ),
]

TOOLS: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}


class ToolRegistry:
    """Validated dispatch with call accounting (feeds Phase 30 observability)."""

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self.calls: list[dict[str, Any]] = []

    def names(self) -> list[str]:
        return list(TOOLS)

    def schema(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description.strip().split("\n")[0],
                "input_schema": s.input_model.model_json_schema(),
            }
            for s in TOOL_SPECS
        ]

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        import time

        spec = TOOLS.get(name)
        if spec is None:
            # Recorded, not silently dropped: an agent repeatedly calling a
            # tool that does not exist is exactly the kind of failure the
            # observability layer needs to surface.
            self.calls.append({"tool": name, "ok": False, "error": "unknown_tool"})
            return {"error": f"unknown tool {name!r}", "available": self.names()}
        t0 = time.perf_counter()
        try:
            payload = spec.input_model.model_validate(kwargs)
        except Exception as exc:
            self.calls.append({"tool": name, "ok": False, "error": "validation"})
            return {"error": "input validation failed", "detail": str(exc)}
        try:
            result = spec.fn(self.ctx, payload)
            ok = "error" not in result
        except Exception as exc:
            logger.exception("tool %s failed", name)
            result, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
        self.calls.append(
            {
                "tool": name,
                "ok": ok,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        )
        return result

    def stats(self) -> dict[str, Any]:
        if not self.calls:
            return {"n_calls": 0}
        return {
            "n_calls": len(self.calls),
            "n_failed": sum(1 for c in self.calls if not c["ok"]),
            "total_latency_ms": round(sum(float(c.get("latency_ms", 0.0)) for c in self.calls), 2),
            # Plain Python ints: pandas value_counts yields numpy.int64, which
            # is not JSON-serialisable and breaks the API response layer.
            "by_tool": {
                str(k): int(v)
                for k, v in pd.Series([c["tool"] for c in self.calls]).value_counts().items()
            },
        }


def to_langchain_tools(registry: ToolRegistry) -> list[Any]:
    """Adapt to LangChain `StructuredTool`s when LangChain is installed."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:  # pragma: no cover - optional dep
        logger.warning("langchain_core not installed; returning raw specs")
        return []
    out = []
    for spec in TOOL_SPECS:

        def _make(name: str):
            def _run(**kwargs: Any) -> dict[str, Any]:
                return registry.call(name, **kwargs)

            return _run

        out.append(
            StructuredTool.from_function(
                func=_make(spec.name),
                name=spec.name,
                description=spec.description.strip().split("\n")[0],
                args_schema=spec.input_model,
            )
        )
    return out
