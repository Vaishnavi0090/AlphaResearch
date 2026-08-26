"""FastAPI application (Phase 28).

The API is deliberately thin: every endpoint validates input, calls a core
module, and serialises the result. No business logic lives here, which is why
the same behaviour is reachable from `scripts/`, the tests and the dashboard
without duplication.

Security posture (Phase 32):
* No endpoint accepts code, a file path, or a shell command.
* Universe and ticker inputs are validated against the loaded bundle, so a
  request cannot reach outside the configured dataset.
* Optimiser/backtest parameters are bounded by Pydantic (`ge`/`le`), which
  prevents a request from pinning the process with an absurd lookback.
* Expensive endpoints are explicit about cost and the heavy state (bundle,
  document index) is built once at startup, not per request.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from finrisk.agents.graph import run_research
from finrisk.api.observability import Telemetry
from finrisk.config.settings import Settings, default_settings
from finrisk.data.documents import load_corpus
from finrisk.data.market_data import align_to_calendar, load_bundle
from finrisk.data.validation import validate_bundle
from finrisk.rag.retriever import DocumentRetriever
from finrisk.tools.finance_tools import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)


class AppState:
    """Process-wide state, initialised once on startup."""

    settings: Settings
    bundle: Any = None
    retriever: DocumentRetriever | None = None
    telemetry: Telemetry

    @classmethod
    def init(cls, settings: Settings | None = None) -> None:
        cls.settings = settings or default_settings()
        cls.telemetry = Telemetry()
        with cls.telemetry.stage("startup.load_bundle"):
            cls.bundle = align_to_calendar(load_bundle(cls.settings.data))
        with cls.telemetry.stage("startup.index_documents"):
            docs = load_corpus(cls.settings.rag.docs_dir)
            cls.retriever = DocumentRetriever(cls.settings.rag)
            if docs:
                cls.retriever.add_documents(docs)
        logger.info(
            "loaded %d assets, %d document chunks",
            len(cls.bundle.tickers),
            cls.retriever.n_chunks,
        )

    @classmethod
    def registry(cls) -> ToolRegistry:
        return ToolRegistry(ToolContext(cls.bundle, cls.settings, cls.retriever))


@asynccontextmanager
async def lifespan(app: FastAPI):
    AppState.init()
    yield


app = FastAPI(
    title="FINRISK AI",
    version="1.0.0",
    description=(
        "Agentic quantitative portfolio and risk intelligence platform. "
        "Research and decision-support only — not investment advice."
    ),
    lifespan=lifespan,
)


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    method: str = Field("risk_constrained")
    as_of: str | None = None
    tickers: list[str] = Field(default_factory=list, max_length=200)
    top_n: int | None = Field(None, ge=1, le=100)
    max_weight: float | None = Field(None, gt=0.0, le=1.0)
    target_volatility: float | None = Field(None, gt=0.0, le=1.0)
    max_cvar_95: float | None = Field(None, gt=0.0, le=0.5)
    use_regime: bool = True


class PortfolioRequest(BaseModel):
    weights: dict[str, float] = Field(..., max_length=200)
    as_of: str | None = None
    lookback_days: int = Field(504, ge=60, le=3000)


class BacktestRequest(BaseModel):
    strategy: str = Field("F_finrisk_full")
    rebalance_frequency: str = Field("M", pattern="^[MQW]$")


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    as_of: str | None = None
    method: str = "risk_constrained"
    include_backtest: bool = False
    top_n: int = Field(10, ge=1, le=50)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    ticker: str | None = Field(None, max_length=12)
    top_k: int = Field(5, ge=1, le=20)
    section: str | None = Field(None, max_length=64)
    mode: str | None = Field(None, pattern="^(bm25|dense|hybrid)$")


def _validate_tickers(tickers: list[str]) -> list[str]:
    """Reject anything not in the loaded universe."""
    known = set(AppState.bundle.tickers)
    bad = [t for t in tickers if t not in known]
    if bad:
        raise HTTPException(422, f"unknown tickers: {bad[:10]}")
    return tickers


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    ready = AppState.bundle is not None
    return {
        "status": "ok" if ready else "initialising",
        "provider": getattr(AppState.bundle, "provider", None),
        "n_assets": len(AppState.bundle.tickers) if ready else 0,
        "date_range": (
            [str(AppState.bundle.dates.min().date()), str(AppState.bundle.dates.max().date())]
            if ready
            else None
        ),
        "document_chunks": AppState.retriever.n_chunks if AppState.retriever else 0,
        "llm_provider": AppState.settings.agents.llm_provider,
        "anthropic_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.get("/assets")
def assets() -> dict[str, Any]:
    b = AppState.bundle
    return {
        "n_assets": len(b.tickers),
        "benchmark": AppState.settings.data.benchmark,
        "assets": [
            (
                {"ticker": t, "name": b.meta[t].name, "sector": b.meta[t].sector}
                if t in b.meta
                else {"ticker": t, "name": t, "sector": "Unknown"}
            )
            for t in b.tickers
        ],
    }


@app.get("/data/quality")
def data_quality() -> dict[str, Any]:
    rep = validate_bundle(AppState.bundle)
    return {**rep.model_dump(mode="json"), "rendered": rep.render()}


@app.get("/factors")
def factors(
    as_of: str | None = Query(None), top_n: int = Query(20, ge=1, le=200)
) -> dict[str, Any]:
    return AppState.registry().call("calculate_factor_scores", as_of=as_of, top_n=top_n)


@app.get("/factors/{ticker}")
def factor_detail(ticker: str, as_of: str | None = Query(None)) -> dict[str, Any]:
    _validate_tickers([ticker])
    n = len(AppState.bundle.tickers)
    res = AppState.registry().call("calculate_factor_scores", as_of=as_of, top_n=n)
    if "error" in res:
        raise HTTPException(404, res["error"])
    row = next((r for r in res["ranking"] if r["ticker"] == ticker), None)
    if row is None:
        raise HTTPException(404, f"{ticker} not in the cross-section on {res['as_of']}")
    fund = AppState.registry().call("get_fundamentals", tickers=[ticker], as_of=as_of)
    return {
        "as_of": res["as_of"],
        "factor_scores": row,
        "n_universe": res["n_universe"],
        "fundamentals": fund.get("records", {}).get(ticker),
    }


@app.get("/regime")
def regime(as_of: str | None = Query(None)) -> dict[str, Any]:
    return AppState.registry().call("detect_market_regime", as_of=as_of)


@app.post("/portfolio/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    if req.tickers:
        _validate_tickers(req.tickers)
    settings = AppState.settings
    if req.top_n is not None:
        settings = settings.model_copy(deep=True)
        settings.portfolio.top_n = req.top_n
    reg = ToolRegistry(ToolContext(AppState.bundle, settings, AppState.retriever))
    res = reg.call(
        "optimize_portfolio",
        method=req.method,
        tickers=req.tickers,
        as_of=req.as_of,
        max_weight=req.max_weight,
        target_volatility=req.target_volatility,
        max_cvar_95=req.max_cvar_95,
        use_regime=req.use_regime,
    )
    if "error" in res:
        raise HTTPException(422, res["error"])
    return res


@app.post("/risk")
def risk(req: PortfolioRequest) -> dict[str, Any]:
    _validate_tickers(list(req.weights))
    return AppState.registry().call(
        "calculate_risk",
        weights=req.weights,
        as_of=req.as_of,
        lookback_days=req.lookback_days,
    )


@app.post("/risk/attribution")
def risk_attribution(req: PortfolioRequest) -> dict[str, Any]:
    _validate_tickers(list(req.weights))
    return AppState.registry().call(
        "calculate_risk_contribution",
        weights=req.weights,
        as_of=req.as_of,
        lookback_days=req.lookback_days,
    )


@app.post("/risk/stress")
def stress(req: PortfolioRequest) -> dict[str, Any]:
    _validate_tickers(list(req.weights))
    return AppState.registry().call(
        "stress_test",
        weights=req.weights,
        as_of=req.as_of,
        lookback_days=req.lookback_days,
    )


@app.post("/analyze")
def analyze(req: PortfolioRequest) -> dict[str, Any]:
    _validate_tickers(list(req.weights))
    return AppState.registry().call(
        "analyze_portfolio",
        weights=req.weights,
        as_of=req.as_of,
        lookback_days=req.lookback_days,
    )


@app.post("/backtest")
def backtest(req: BacktestRequest) -> dict[str, Any]:
    """Expensive: a full walk-forward run takes tens of seconds."""
    res = AppState.registry().call(
        "run_backtest", strategy=req.strategy, rebalance_frequency=req.rebalance_frequency
    )
    if "error" in res:
        raise HTTPException(422, res)
    return res


@app.post("/research")
def research(req: ResearchRequest) -> dict[str, Any]:
    """Run the agent graph and return the validated report."""
    reg = AppState.registry()
    report, state = run_research(
        reg,
        question=req.question,
        as_of=req.as_of,
        options={"method": req.method, "top_n": req.top_n},
        include_backtest=req.include_backtest,
    )
    return {
        "report": report.model_dump(mode="json"),
        "trace": state.get("trace", []),
        "errors": state.get("errors", []),
        "tool_stats": reg.stats(),
    }


@app.post("/documents/retrieve")
def retrieve(req: RetrieveRequest) -> dict[str, Any]:
    return AppState.registry().call(
        "retrieve_financial_documents",
        query=req.query,
        ticker=req.ticker,
        top_k=req.top_k,
        section=req.section,
        mode=req.mode,
    )


class NeutralRequest(BaseModel):
    factor: str = Field("value", pattern="^(composite|momentum|value|quality|risk|market)$")
    as_of: str | None = None
    gross_exposure: float = Field(2.0, gt=0.0, le=4.0)
    max_weight: float | None = Field(None, gt=0.0, le=1.0)


@app.post("/factors/neutral")
def factor_neutral(req: NeutralRequest) -> dict[str, Any]:
    """Sector- and beta-neutral portfolio for one factor, with the realised
    exposures that evidence the neutrality claim."""

    from finrisk.factors.engine import FactorEngine
    from finrisk.factors.neutralization import (
        build_neutral_portfolio,
        estimate_betas,
        exposure_report,
    )

    ctx = AppState.registry().ctx
    d = ctx.date(req.as_of)
    _, diag = FactorEngine(AppState.settings.factors).compute(AppState.bundle, d)
    if diag.empty:
        raise HTTPException(422, f"no factor cross-section on {d.date()}")
    column = "composite" if req.factor == "composite" else f"grp_{req.factor}"
    if column not in diag.columns:
        raise HTTPException(422, f"factor '{req.factor}' unavailable on {d.date()}")

    scores = diag[column].dropna()
    sectors = AppState.bundle.sector_map()
    rets = ctx.trailing_returns(d, 252)
    betas = estimate_betas(
        rets[[c for c in scores.index if c in rets.columns]],
        AppState.bundle.benchmark.loc[:d].pct_change().tail(252),
    )
    weights = build_neutral_portfolio(scores, sectors, betas, req.gross_exposure, req.max_weight)
    if weights.empty:
        raise HTTPException(422, "insufficient cross-section to build a neutral book")
    return {
        "as_of": str(d.date()),
        "factor": req.factor,
        "weights": {k: round(float(v), 6) for k, v in weights.items() if abs(v) > 1e-9},
        "exposures": {
            k: round(float(v), 8) for k, v in exposure_report(weights, sectors, betas).items()
        },
        "note": (
            "Exposures are the evidence for neutrality; they are not exactly zero "
            "because betas are estimated with error."
        ),
    }


@app.get("/rag/diagnostics")
def rag_diagnostics() -> dict[str, Any]:
    """Which retriever and encoder are actually active."""
    if AppState.retriever is None:
        raise HTTPException(503, "document index not initialised")
    return AppState.retriever.diagnostics()


@app.get("/strategies")
def strategies() -> dict[str, Any]:
    from finrisk.backtesting.strategies import all_strategies

    return {
        "strategies": [
            {"name": s.name, "description": s.description} for s in all_strategies().values()
        ]
    }


@app.get("/tools")
def tools() -> dict[str, Any]:
    return {"tools": AppState.registry().schema()}


@app.get("/config")
def config() -> dict[str, Any]:
    """Settings actually in force. No secrets are stored in Settings."""
    return AppState.settings.model_dump(mode="json")


@app.get("/observability")
def observability() -> dict[str, Any]:
    return AppState.telemetry.summary()
