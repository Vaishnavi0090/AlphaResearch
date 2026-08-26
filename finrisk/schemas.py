"""Typed domain objects passed between FINRISK AI layers.

These are the contracts. Every layer consumes and emits these, which is what
makes it possible to hand quantitative output to an LLM *as structured data*
instead of letting the LLM invent numbers.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------


class AssetMeta(BaseModel):
    ticker: str
    name: str = ""
    sector: str = "Unknown"


class DataQualityIssue(BaseModel):
    severity: Literal["info", "warning", "error"]
    category: str
    ticker: str | None = None
    detail: str


class DataQualityReport(BaseModel):
    n_assets: int
    start: date
    end: date
    n_observations: int
    missing_observations: int
    duplicate_timestamps: int
    issues: list[DataQualityIssue] = Field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def missing_pct(self) -> float:
        denom = self.n_observations + self.missing_observations
        return 100.0 * self.missing_observations / denom if denom else 0.0

    def render(self) -> str:
        lines = [
            "Data Quality Report",
            "=" * 60,
            f"Assets:                  {self.n_assets}",
            f"Date range:              {self.start} -> {self.end}",
            f"Observations:            {self.n_observations:,}",
            f"Missing observations:    {self.missing_observations:,} " f"({self.missing_pct:.2f}%)",
            f"Duplicate timestamps:    {self.duplicate_timestamps}",
            f"Issues:                  {len(self.issues)} " f"({self.n_errors} errors)",
        ]
        if self.issues:
            lines.append("-" * 60)
            for issue in self.issues[:40]:
                tag = f"[{issue.severity.upper()}]"
                who = f" {issue.ticker}" if issue.ticker else ""
                lines.append(f"{tag:<10}{issue.category}{who}: {issue.detail}")
            if len(self.issues) > 40:
                lines.append(f"... {len(self.issues) - 40} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Factor layer
# --------------------------------------------------------------------------


class FactorScore(BaseModel):
    """Point-in-time factor evidence for a single asset."""

    ticker: str
    as_of: date
    raw_values: dict[str, float] = Field(default_factory=dict)
    normalized: dict[str, float] = Field(default_factory=dict)
    group_scores: dict[str, float] = Field(default_factory=dict)
    composite: float = 0.0
    rank: int | None = None
    n_universe: int | None = None
    coverage: float = 1.0

    @property
    def percentile(self) -> float | None:
        if self.rank is None or not self.n_universe:
            return None
        return 100.0 * (1.0 - (self.rank - 1) / max(self.n_universe - 1, 1))


class FactorPerformance(BaseModel):
    """Result of a single-factor research study (Phase 5)."""

    factor: str
    start: date
    end: date
    n_rebalances: int
    long_ann_return: float
    long_short_ann_return: float | None
    ann_volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    mean_ic: float
    ic_ir: float | None
    hit_rate: float
    ic_by_horizon: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Regime layer
# --------------------------------------------------------------------------


class RegimeState(BaseModel):
    as_of: date
    regime: str
    state_id: int
    confidence: float
    features: dict[str, float] = Field(default_factory=dict)
    state_probabilities: dict[str, float] = Field(default_factory=dict)
    transition_probabilities: dict[str, float] = Field(default_factory=dict)
    expected_duration_days: float | None = None


# --------------------------------------------------------------------------
# Alpha layer
# --------------------------------------------------------------------------


class AlphaSignal(BaseModel):
    ticker: str
    as_of: date
    score: float
    direction: Literal["long", "short", "flat"]
    confidence: float
    factor_contributions: dict[str, float] = Field(default_factory=dict)
    regime_context: str | None = None
    regime_adjustment: float = 1.0

    @model_validator(mode="after")
    def _clip(self) -> AlphaSignal:
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))
        return self


# --------------------------------------------------------------------------
# Portfolio layer
# --------------------------------------------------------------------------


class ConstraintReport(BaseModel):
    max_weight: float
    min_weight: float
    max_sector_weight: float | None = None
    target_volatility: float | None = None
    max_cvar_95: float | None = None
    max_leverage: float = 1.0
    violations: list[str] = Field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.violations


class PortfolioWeights(BaseModel):
    as_of: date
    method: str
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe: float
    cash: float = 0.0
    optimization_status: str = "ok"
    constraints: ConstraintReport | None = None
    risk_metrics: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    def active(self, threshold: float = 1e-6) -> dict[str, float]:
        return {k: v for k, v in self.weights.items() if abs(v) > threshold}


# --------------------------------------------------------------------------
# Risk layer
# --------------------------------------------------------------------------


class RiskContribution(BaseModel):
    ticker: str
    weight: float
    marginal_contribution: float
    component_contribution: float
    pct_of_risk: float


class RiskReport(BaseModel):
    as_of: date
    n_observations: int
    ann_volatility: float
    ann_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    downside_deviation: float
    # Volatility implied by the covariance MODEL (sqrt(w'Sigma w)). Reported
    # separately from `ann_volatility`, which is the realised sample standard
    # deviation of the portfolio return stream. They differ because the model
    # uses a shrunk covariance estimate; the Euler risk decomposition sums to
    # THIS number, not to the realised one.
    model_volatility: float | None = None
    beta: float | None = None
    tracking_error: float | None = None
    var_95_hist: float = 0.0
    var_99_hist: float = 0.0
    var_95_param: float = 0.0
    var_99_param: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    herfindahl: float = 0.0
    effective_n: float = 0.0
    mean_correlation: float | None = None
    contributions: list[RiskContribution] = Field(default_factory=list)
    sector_risk: dict[str, float] = Field(default_factory=dict)
    factor_risk: dict[str, float] = Field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"Risk Report — {self.as_of}",
            "=" * 52,
            f"Ann. return          {self.ann_return:>9.2%}",
            f"Ann. volatility      {self.ann_volatility:>9.2%}",
            f"Sharpe               {self.sharpe:>9.2f}",
            f"Sortino              {self.sortino:>9.2f}",
            f"Max drawdown         {self.max_drawdown:>9.2%}",
            f"Calmar               {self.calmar:>9.2f}",
            f"VaR 95% (hist, 1d)   {self.var_95_hist:>9.2%}",
            f"VaR 99% (hist, 1d)   {self.var_99_hist:>9.2%}",
            f"CVaR 95% (1d)        {self.cvar_95:>9.2%}",
            f"CVaR 99% (1d)        {self.cvar_99:>9.2%}",
            f"Effective N          {self.effective_n:>9.2f}",
        ]
        if self.beta is not None:
            lines.append(f"Beta vs benchmark    {self.beta:>9.2f}")
        if self.contributions:
            lines += ["-" * 52, "Risk contribution:"]
            for c in sorted(self.contributions, key=lambda x: -x.pct_of_risk)[:10]:
                lines.append(f"  {c.ticker:<8} w={c.weight:>6.2%}  " f"risk={c.pct_of_risk:>6.1%}")
        if self.factor_risk:
            lines += ["-" * 52, "Factor risk share:"]
            for k, v in sorted(self.factor_risk.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {k:<14} {v:>6.1%}")
        return "\n".join(lines)


class StressScenario(BaseModel):
    name: str
    kind: Literal["historical", "hypothetical"]
    description: str
    portfolio_return: float
    benchmark_return: float | None = None
    max_drawdown: float
    var_95: float
    cvar_95: float
    worst_contributors: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Backtest layer
# --------------------------------------------------------------------------


class RebalanceEvent(BaseModel):
    decision_date: date
    execution_date: date
    method: str
    weights: dict[str, float]
    turnover: float
    cost: float
    regime: str | None = None
    optimization_status: str = "ok"


class PerformanceMetrics(BaseModel):
    start: date
    end: date
    n_days: int
    total_return: float
    cagr: float
    ann_return: float
    ann_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    var_95: float
    cvar_95: float
    hit_rate: float
    avg_turnover: float
    total_cost_drag: float
    beta: float | None = None
    information_ratio: float | None = None


class BacktestResult(BaseModel):
    strategy: str
    config_digest: str
    metrics: PerformanceMetrics
    n_rebalances: int
    events: list[RebalanceEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # equity curve stored as ISO date -> value for JSON round-tripping
    equity_curve: dict[str, float] = Field(default_factory=dict)
    benchmark_curve: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Research / LLM structured output
# --------------------------------------------------------------------------


class Citation(BaseModel):
    """A traceable pointer to indexed text.

    Metadata beyond the excerpt exists so a claim can be audited back to a
    specific section of a specific filing on a specific date — 'the 10-K said
    so' is not traceability if you cannot say which part of which filing.
    """

    doc_id: str
    source: str
    chunk_id: str
    score: float
    excerpt: str
    ticker: str | None = None
    doc_type: str | None = None
    section: str | None = None
    page: int | None = None
    filing_date: str | None = None
    fiscal_period: str | None = None
    retrieval_mode: str | None = None


class ResearchConclusion(BaseModel):
    """Structured LLM output. Numbers must be copied from quantitative_evidence,
    never invented — enforced by the critic in finrisk/agents/critic.py."""

    thesis: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    quantitative_evidence: dict[str, float] = Field(default_factory=dict)
    risk_factors: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    sources: list[Citation] = Field(default_factory=list)


class PositionExplanation(BaseModel):
    """Why one position is held, at the level of detail an IC would demand.

    Every numeric field is computed by a deterministic engine and copied here.
    The LLM writes the prose fields, and the critic checks the numbers it
    quotes against these values — see finrisk/agents/critic.py.
    """

    ticker: str
    weight: float
    drivers_positive: list[str] = Field(default_factory=list)
    drivers_negative: list[str] = Field(default_factory=list)
    risk_note: str = ""
    regime_note: str = ""
    factor_contributions: dict[str, float] = Field(default_factory=dict)
    pct_of_portfolio_risk: float = 0.0
    evidence: list[Citation] = Field(default_factory=list)

    # --- allocation vs contribution -------------------------------------
    # Expected annual return this name is forecast to add: alpha_i * w_i.
    alpha_contribution: float = 0.0
    # The raw Grinold-Kahn forecast for the asset itself, before sizing.
    alpha_forecast: float = 0.0
    # Share of the portfolio's total forecast alpha.
    pct_of_portfolio_alpha: float = 0.0
    # Risk share minus weight share. Positive means the position consumes more
    # risk budget than its size suggests — the number that decides trims.
    risk_vs_weight_gap: float = 0.0

    # --- diversification -------------------------------------------------
    # Standalone volatility of the asset.
    standalone_volatility: float = 0.0
    # Marginal contribution to portfolio volatility (dSigma/dw_i).
    marginal_risk: float = 0.0
    # 1 - (marginal risk / standalone vol). Near 1 means correlations absorb
    # most of this asset's risk; near 0 means it adds risk almost one-for-one.
    diversification_benefit: float = 0.0
    # Mean correlation with the rest of the book.
    avg_correlation_to_book: float = 0.0
    # Share of the worst 5% of days attributable to this position.
    pct_of_tail_loss: float = 0.0

    # --- evidence and confidence -----------------------------------------
    contradicting_evidence: list[str] = Field(default_factory=list)
    # 0-1, derived from factor coverage, signal agreement across groups, and
    # documentary support. Never LLM-assigned.
    confidence: float = 0.0
    confidence_basis: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    """A falsifiable claim with the test that would settle it.

    The `falsifier` field is the point: a "hypothesis" with no stated way to
    be wrong is a narrative. Requiring one up front stops the research loop
    from drifting into justification after the fact.
    """

    statement: str
    rationale: str = ""
    # What observation would REFUTE this. Required, not optional.
    falsifier: str
    # Tool calls that would produce that observation.
    test_plan: list[str] = Field(default_factory=list)
    prior_confidence: float = Field(0.5, ge=0.0, le=1.0)
    # Set only after the test runs.
    verdict: Literal["supported", "refuted", "inconclusive", "untested"] = "untested"
    evidence: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class ResearchPlan(BaseModel):
    """What the agent intends to do, recorded BEFORE it does it.

    Writing the plan first makes the difference between a research process and
    a post-hoc story visible: the plan can be compared against what was
    actually run and what was actually concluded.
    """

    question: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    planned_tools: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    as_of: date | None = None


class RiskSummary(BaseModel):
    """Structured risk view, so the report never depends on prose for a number."""

    as_of: date
    ann_volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    effective_n: float
    beta: float | None = None
    top_risk_contributors: dict[str, float] = Field(default_factory=dict)
    tail_contributors: dict[str, float] = Field(default_factory=dict)
    sector_risk: dict[str, float] = Field(default_factory=dict)
    breached_limits: list[str] = Field(default_factory=list)
    concentration_flags: list[str] = Field(default_factory=list)
    stress_worst_case: dict[str, float] = Field(default_factory=dict)


class CommitteeOpinion(BaseModel):
    role: Literal["bull", "bear", "risk", "quant"]
    stance: Literal["support", "oppose", "conditional", "neutral"]
    argument: str
    cited_metrics: dict[str, float] = Field(default_factory=dict)
    concerns: list[str] = Field(default_factory=list)


class CommitteeVerdict(BaseModel):
    as_of: date
    opinions: list[CommitteeOpinion]
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    unresolved_uncertainty: list[str] = Field(default_factory=list)
    recommended_action: str = "no_change"
    rationale: str = ""


class CriticFinding(BaseModel):
    severity: Literal["info", "warning", "error"]
    check: str
    detail: str


class ValidationReport(BaseModel):
    passed: bool
    findings: list[CriticFinding] = Field(default_factory=list)

    @property
    def errors(self) -> list[CriticFinding]:
        return [f for f in self.findings if f.severity == "error"]


class FinalReport(BaseModel):
    generated_at: datetime
    as_of: date
    universe: list[str]
    regime: RegimeState | None = None
    portfolio: PortfolioWeights | None = None
    risk: RiskReport | None = None
    explanations: list[PositionExplanation] = Field(default_factory=list)
    plan: ResearchPlan | None = None
    risk_summary: RiskSummary | None = None
    research: ResearchConclusion | None = None
    committee: CommitteeVerdict | None = None
    validation: ValidationReport | None = None
    backtest: BacktestResult | None = None
    disclaimer: str = (
        "Research and decision-support output only. Not investment advice. "
        "Backtested and simulated results do not indicate future returns."
    )
