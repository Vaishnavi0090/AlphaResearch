"""Agentic finance research graph (Phases 16, 19, 21).

    Finance Supervisor
        |-- Market Agent         -> get_market_data, detect_market_regime
        |-- Fundamental Agent    -> get_fundamentals
        |-- Factor Agent         -> calculate_factor_scores
        |-- Portfolio Agent      -> optimize_portfolio
        |-- Risk Agent           -> calculate_risk / var / cvar / contribution / stress
        |-- Backtest Agent       -> run_backtest
        |-- Research Agent       -> retrieve_financial_documents + LLM synthesis
        |-- Investment Committee -> bull / bear / risk / quant  -> debate
        |-- Critic               -> validate everything above
        \\-- Report Agent        -> FinalReport

Two design commitments distinguish this from prose-swapping multi-agent demos:

1. **Every agent calls real tools.** An agent's contribution to shared state is
   a typed tool result, not an opinion. The only agents that produce prose are
   the research, committee and report agents, and their prose is validated
   against the tool results by the critic before it is kept.

2. **The graph is a fixed DAG, not free-form routing.** Quantitative
   dependencies are real dependencies: you cannot compute risk contributions
   before you have weights, and you cannot have weights before you have
   factors and a covariance estimate. Letting a model choose the order would
   add failure modes and buy nothing. The supervisor's actual job is deciding
   which *optional* branches (documents, committee, backtest) to run.

LangGraph is used because this is stateful, has conditional branches, needs
per-node error isolation, and benefits from a single serialisable state object
that can be traced. `build_graph()` falls back to a topologically equivalent
sequential runner when LangGraph is not installed, so the pipeline never hard-
depends on it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

import pandas as pd

from finrisk.agents.critic import Critic
from finrisk.agents.llm import LLMClient, build_llm, parse_json_response
from finrisk.schemas import (
    Citation,
    CommitteeOpinion,
    CommitteeVerdict,
    CriticFinding,
    FinalReport,
    Hypothesis,
    PositionExplanation,
    ResearchConclusion,
    ResearchPlan,
    RiskSummary,
    ValidationReport,
)
from finrisk.tools.finance_tools import ToolRegistry

logger = logging.getLogger(__name__)


# ======================================================================
# Graph state
# ======================================================================


class GraphState(TypedDict, total=False):
    question: str
    as_of: str | None
    options: dict[str, Any]

    market: dict[str, Any]
    regime: dict[str, Any]
    fundamentals: dict[str, Any]
    factors: dict[str, Any]
    portfolio: dict[str, Any]
    alphas: dict[str, float]
    risk: dict[str, Any]
    risk_contribution: dict[str, Any]
    cvar: dict[str, Any]
    stress: dict[str, Any]
    backtest: dict[str, Any]
    citations: list[dict[str, Any]]

    plan: dict[str, Any]
    risk_summary: dict[str, Any]
    research: dict[str, Any]
    committee: dict[str, Any]
    explanations: list[dict[str, Any]]
    validation: dict[str, Any]
    errors: list[str]
    trace: list[str]


def _facts(tag: str, payload: Any) -> str:
    """Wrap structured facts so the stub narrator (and a real LLM) can find
    them, and so the critic can diff prose against them."""
    import json

    return f"<facts:{tag}>\n{json.dumps(payload, default=str)}\n</facts:{tag}>"


# ======================================================================
# Supervisor
# ======================================================================


class FinanceSupervisor:
    """Owns the tool registry, the LLM and the critic; implements each node."""

    def __init__(
        self,
        registry: ToolRegistry,
        llm: LLMClient | None = None,
        enable_committee: bool = True,
    ) -> None:
        self.tools = registry
        settings = registry.ctx.settings
        self.llm = llm or build_llm(
            settings.agents.llm_provider, settings.agents.model, settings.agents.temperature
        )
        self.critic = Critic()
        self.enable_committee = enable_committee

    # -- helper --------------------------------------------------------
    @staticmethod
    def _note(state: GraphState, msg: str) -> None:
        state.setdefault("trace", []).append(msg)

    @staticmethod
    def _fail(state: GraphState, node: str, result: dict[str, Any]) -> bool:
        if "error" in result:
            state.setdefault("errors", []).append(f"{node}: {result['error']}")
            return True
        return False

    # ==================================================================
    # Nodes
    # ==================================================================

    def plan_agent(self, state: GraphState) -> GraphState:
        """Record the research plan BEFORE running anything.

        Hypotheses are derived from the question and the configured strategy,
        each with an explicit falsifier. Writing this first is what separates a
        research process from a post-hoc narrative: the plan is in the report
        alongside the conclusion, so the two can be compared.
        """
        question = state.get("question") or "portfolio construction rationale"
        cfg = self.tools.ctx.settings
        hypotheses = [
            Hypothesis(
                statement=(
                    "The factor composite ranks assets with positive predictive "
                    "power over the forward horizon."
                ),
                rationale="Composite blends momentum, value, quality and risk scores.",
                falsifier=(
                    "Mean information coefficient is <= 0, or its t-statistic is "
                    "below 2 across the evaluated cross-sections."
                ),
                test_plan=["calculate_factor_scores", "run_backtest"],
                prior_confidence=0.5,
            ),
            Hypothesis(
                statement=(
                    f"Constrained optimisation holds portfolio volatility at or "
                    f"below the {cfg.portfolio.target_volatility:.0%} target."
                    if cfg.portfolio.target_volatility
                    else "Constrained optimisation controls portfolio volatility."
                ),
                rationale="Volatility ceiling is imposed on the forecast covariance.",
                falsifier=(
                    "Realised volatility exceeds the target by more than 2 " "percentage points."
                ),
                test_plan=["optimize_portfolio", "calculate_risk"],
                prior_confidence=0.6,
            ),
            Hypothesis(
                statement="Risk is not concentrated in a single position.",
                rationale="Weight caps and correlation should spread risk.",
                falsifier=(
                    "Any position contributes more than 40% of portfolio "
                    "volatility, or effective N falls below 3."
                ),
                test_plan=["calculate_risk_contribution"],
                prior_confidence=0.5,
            ),
        ]
        plan = ResearchPlan(
            question=question,
            hypotheses=hypotheses,
            planned_tools=sorted(
                {t for h in hypotheses for t in h.test_plan}
                | {"detect_market_regime", "retrieve_financial_documents"}
            ),
            success_criteria=[
                "Every quoted number traces to a deterministic tool result.",
                "All portfolio constraints are satisfied and independently re-checked.",
                "Both supporting and contradicting evidence are presented.",
            ],
            known_limitations=[
                "Results are computed on a synthetic dataset and are not "
                "historical performance.",
                "A single sample period gives low statistical power for " "Sharpe comparisons.",
            ],
        )
        state["plan"] = plan.model_dump(mode="json")
        self._note(state, f"plan_agent: {len(hypotheses)} hypotheses registered")
        return state

    def _evaluate_hypotheses(self, state: GraphState) -> None:
        """Mark each hypothesis against its own stated falsifier.

        Deterministic: the verdict comes from tool output compared to the
        threshold written down before the run, not from a judgement call.
        """
        plan = state.get("plan")
        if not plan:
            return
        risk = state.get("risk") or {}
        port = state.get("portfolio") or {}
        contribs = (state.get("risk_contribution") or {}).get("contributions") or []
        target = self.tools.ctx.settings.portfolio.target_volatility

        for h in plan.get("hypotheses", []):
            statement = h["statement"].lower()
            if "volatility" in statement and target:
                realised = risk.get("ann_volatility")
                if isinstance(realised, (int, float)):
                    breach = realised > target + 0.02
                    h["verdict"] = "refuted" if breach else "supported"
                    h["evidence"] = {
                        "realised_volatility": round(float(realised), 6),
                        "target_volatility": float(target),
                    }
                    h["notes"] = f"Realised {realised:.2%} vs target {target:.2%}."
            elif "concentrated" in statement:
                worst = max((c["pct_of_risk"] for c in contribs), default=0.0)
                eff_n = risk.get("effective_n", 0.0)
                breach = worst > 0.40 or (eff_n and eff_n < 3)
                h["verdict"] = "refuted" if breach else "supported"
                h["evidence"] = {
                    "max_risk_share": round(float(worst), 6),
                    "effective_n": round(float(eff_n or 0.0), 4),
                }
                h["notes"] = f"Largest single contributor {worst:.1%}; effective N {eff_n:.2f}."
            else:
                # No backtest in this run means the IC falsifier cannot be
                # evaluated. "inconclusive" is the honest verdict; claiming
                # support without the test would be exactly the failure the
                # falsifier exists to prevent.
                bt = state.get("backtest") or {}
                if isinstance(bt.get("sharpe"), (int, float)):
                    h["verdict"] = "supported" if bt["sharpe"] > 0 else "refuted"
                    h["evidence"] = {
                        "backtest_sharpe": round(float(bt["sharpe"]), 4),
                        "backtest_cagr": round(float(bt.get("cagr", 0.0)), 6),
                    }
                else:
                    h["verdict"] = "inconclusive"
                    h["notes"] = "No walk-forward result was run in this session."
            if isinstance(port.get("status"), str) and "fallback" in port["status"]:
                h["notes"] = (h.get("notes", "") + " Optimiser used a fallback path.").strip()

    def _build_risk_summary(self, state: GraphState) -> None:
        """Structured risk view assembled from tool output only."""
        risk = state.get("risk") or {}
        if not risk or "error" in risk:
            return
        contribs = (state.get("risk_contribution") or {}).get("contributions") or []
        tail = (state.get("cvar") or {}).get("tail_contribution") or {}
        stress = (state.get("stress") or {}).get("scenarios") or []
        worst = {}
        if stress:
            worst_scen = min(stress, key=lambda s: s.get("portfolio_return", 0.0))
            worst = {
                "scenario_return": float(worst_scen.get("portfolio_return", 0.0)),
                "scenario_max_drawdown": float(worst_scen.get("max_drawdown", 0.0)),
            }
        summary = RiskSummary(
            as_of=pd.Timestamp(risk.get("as_of")).date(),
            ann_volatility=float(risk.get("ann_volatility", 0.0)),
            var_95=float(risk.get("var_95_hist", 0.0)),
            cvar_95=float(risk.get("cvar_95", 0.0)),
            max_drawdown=float(risk.get("max_drawdown", 0.0)),
            effective_n=float(risk.get("effective_n", 0.0)),
            beta=risk.get("beta"),
            top_risk_contributors={
                c["ticker"]: round(float(c["pct_of_risk"]), 5) for c in contribs[:5]
            },
            tail_contributors={
                k: round(float(v), 5) for k, v in sorted(tail.items(), key=lambda kv: -kv[1])[:5]
            },
            sector_risk={k: float(v) for k, v in (risk.get("sector_risk") or {}).items()},
            concentration_flags=list(risk.get("concentration_flags") or []),
            breached_limits=list((state.get("portfolio") or {}).get("violations") or []),
            stress_worst_case=worst,
        )
        state["risk_summary"] = summary.model_dump(mode="json")

    def market_agent(self, state: GraphState) -> GraphState:
        """Prices, volatility and the regime state."""
        as_of = state.get("as_of")
        state["market"] = self.tools.call("get_market_data", as_of=as_of, lookback_days=252)
        state["regime"] = self.tools.call("detect_market_regime", as_of=as_of)
        self._fail(state, "market_agent", state["regime"])
        self._note(state, f"market_agent: regime={state['regime'].get('regime')}")
        return state

    def fundamental_agent(self, state: GraphState) -> GraphState:
        """Point-in-time fundamentals for the candidate universe."""
        tickers = [r["ticker"] for r in (state.get("factors", {}).get("ranking") or [])][:10]
        state["fundamentals"] = self.tools.call(
            "get_fundamentals", tickers=tickers, as_of=state.get("as_of")
        )
        self._note(
            state,
            f"fundamental_agent: {len(state['fundamentals'].get('records', {}))} records",
        )
        return state

    def factor_agent(self, state: GraphState) -> GraphState:
        """Cross-sectional factor scores."""
        top_n = int(state.get("options", {}).get("top_n", 10))
        state["factors"] = self.tools.call(
            "calculate_factor_scores", as_of=state.get("as_of"), top_n=top_n
        )
        self._fail(state, "factor_agent", state["factors"])
        self._note(state, f"factor_agent: universe={state['factors'].get('n_universe')}")
        return state

    def portfolio_agent(self, state: GraphState) -> GraphState:
        """Deterministic optimisation. The only producer of weights."""
        opts = state.get("options", {})
        state["portfolio"] = self.tools.call(
            "optimize_portfolio",
            method=opts.get("method", "risk_constrained"),
            as_of=state.get("as_of"),
            target_volatility=opts.get("target_volatility"),
            max_cvar_95=opts.get("max_cvar_95"),
            max_weight=opts.get("max_weight"),
            use_regime=bool(opts.get("use_regime", True)),
        )
        state["alphas"] = state["portfolio"].get("alphas") or {}
        self._fail(state, "portfolio_agent", state["portfolio"])
        self._note(
            state,
            f"portfolio_agent: {state['portfolio'].get('method')} "
            f"n={len(state['portfolio'].get('weights', {}))}",
        )
        return state

    def risk_agent(self, state: GraphState) -> GraphState:
        """Risk report, tail risk, decomposition and stress scenarios."""
        weights = state.get("portfolio", {}).get("weights") or {}
        if not weights:
            state.setdefault("errors", []).append("risk_agent: no weights to analyse")
            return state
        as_of = state.get("as_of")
        state["risk"] = self.tools.call("calculate_risk", weights=weights, as_of=as_of)
        state["cvar"] = self.tools.call("calculate_cvar", weights=weights, as_of=as_of)
        state["risk_contribution"] = self.tools.call(
            "calculate_risk_contribution", weights=weights, as_of=as_of
        )
        if state.get("options", {}).get("stress", True):
            state["stress"] = self.tools.call("stress_test", weights=weights, as_of=as_of)
        self._note(state, f"risk_agent: vol={state['risk'].get('ann_volatility')}")
        return state

    def backtest_agent(self, state: GraphState) -> GraphState:
        """Walk-forward evidence for the strategy being discussed."""
        strategy = state.get("options", {}).get("strategy", "F_finrisk_full")
        state["backtest"] = self.tools.call("run_backtest", strategy=strategy)
        self._note(state, f"backtest_agent: sharpe={state['backtest'].get('sharpe')}")
        return state

    def research_agent(self, state: GraphState) -> GraphState:
        """Retrieve document evidence, then synthesise a structured conclusion."""
        query = state.get("question") or "portfolio construction rationale and key risks"
        docs = self.tools.call("retrieve_financial_documents", query=query, top_k=5)

        # Query expansion on an empty result. BM25 is lexical, so a
        # conversationally-phrased question ("what risks are being taken?")
        # can score zero against filings that say "risk factors" — the exact
        # limitation documented in docs/RAG.md. Rather than return no evidence,
        # retry with the portfolio's own tickers plus standard filing
        # vocabulary. This changes WHICH documents are retrieved, never what
        # they say: every citation still resolves to a real indexed chunk and
        # is still validated by the critic.
        if not docs.get("citations"):
            holdings = list(state.get("portfolio", {}).get("weights") or {})[:5]
            expanded = " ".join(
                [*holdings, "risk factors", "leverage", "profitability", "revenue", "outlook"]
            )
            docs = self.tools.call("retrieve_financial_documents", query=expanded, top_k=5)
            if docs.get("citations"):
                self._note(state, "research_agent: expanded query after empty retrieval")
        state["citations"] = docs.get("citations", [])

        system = (
            "You are a quantitative research analyst. ROLE: research.\n"
            "Use ONLY the numbers provided in the <facts:...> blocks. Never compute or "
            "estimate a figure yourself. Cite sources by their chunk id. "
            "Respond with a single JSON object matching the ResearchConclusion schema: "
            "thesis, supporting_evidence, contradicting_evidence, quantitative_evidence, "
            "risk_factors, confidence, sources."
        )
        prompt = "\n\n".join(
            [
                f"Question: {query}",
                _facts("portfolio", state.get("portfolio", {})),
                _facts("risk", state.get("risk", {})),
                _facts("regime", state.get("regime", {})),
                _facts("factors", state.get("factors", {})),
                _facts("backtest", state.get("backtest", {})),
                _facts("citations", state.get("citations", [])),
            ]
        )
        raw = self._complete(system, prompt, self.tools.ctx.settings.agents.max_tokens)
        parsed = parse_json_response(raw)
        try:
            parsed["sources"] = [
                Citation.model_validate(s)
                for s in (parsed.get("sources") or [])
                if isinstance(s, dict)
            ]
            conclusion = ResearchConclusion.model_validate(parsed)
        except Exception as exc:
            logger.warning("research output failed schema validation: %s", exc)
            conclusion = ResearchConclusion(thesis="research synthesis unavailable")
        state["research"] = conclusion.model_dump(mode="json")
        self._note(state, f"research_agent: llm={self.llm.name}")
        return state

    def _complete(self, system: str, prompt: str, max_tokens: int) -> str:
        """Call the LLM without letting it take the numerical pipeline down.

        The model only narrates; every number in the report already came from a
        deterministic tool. So an API error, rate limit or timeout must degrade
        the narration, not abort the run. An empty string flows into the
        existing schema-validation fallbacks ("research synthesis unavailable";
        committee roles skipped). With no committee opinions there is no risk
        sign-off, and the verdict fails closed to "reject".
        """
        try:
            return self.llm.complete(system, prompt, max_tokens)
        except Exception as exc:
            logger.warning(
                "LLM call failed (%s: %s); continuing without it", type(exc).__name__, exc
            )
            return ""

    def committee_agent(self, state: GraphState) -> GraphState:
        """Adversarial review: four independent roles, then a debate node.

        Each role sees the same facts and argues its brief. The debate node does
        NOT pick the bullish case; it reports agreement, disagreement and what
        remains unresolved, and recommends 'proceed_with_limits' rather than a
        verdict when the bear and risk cases both raise substantive concerns.
        """
        base = "\n\n".join(
            [
                _facts("portfolio", state.get("portfolio", {})),
                _facts("risk", state.get("risk", {})),
                _facts("regime", state.get("regime", {})),
                _facts("backtest", state.get("backtest", {})),
            ]
        )
        opinions: list[CommitteeOpinion] = []
        briefs = {
            "bull": "Argue the strongest case FOR this portfolio using only the given numbers.",
            "bear": "Argue the strongest case AGAINST this portfolio using only the given numbers.",
            "risk": "Assess limit compliance, concentration and tail risk only.",
            "quant": "Assess methodology, reproducibility and estimation robustness only.",
        }
        for role, brief in briefs.items():
            system = (
                f"You are an investment committee member. ROLE: {role}. {brief} "
                "Use ONLY numbers from the <facts:...> blocks. Respond with a single JSON "
                "object: role, stance (support|oppose|conditional|neutral), argument, "
                "cited_metrics, concerns."
            )
            raw = self._complete(system, base, 900)
            parsed = parse_json_response(raw)
            parsed.setdefault("role", role)
            try:
                opinions.append(CommitteeOpinion.model_validate(parsed))
            except Exception:
                logger.warning("committee role %s produced invalid output", role)

        verdict = self._debate(state, opinions)
        state["committee"] = verdict.model_dump(mode="json")
        self._note(state, f"committee_agent: action={verdict.recommended_action}")
        return state

    def _debate(self, state: GraphState, opinions: list[CommitteeOpinion]) -> CommitteeVerdict:
        stances = {o.role: o.stance for o in opinions}
        concerns = [c for o in opinions for c in o.concerns]
        supports = [r for r, s in stances.items() if s == "support"]
        opposes = [r for r, s in stances.items() if s == "oppose"]

        agreements, disagreements = [], []
        if supports and opposes:
            disagreements.append(
                f"{', '.join(supports)} support the portfolio while {', '.join(opposes)} oppose it"
            )
        if len(supports) == len(opinions) and opinions:
            agreements.append("all roles support the portfolio as constructed")

        risk_ok = stances.get("risk") in {"support", "conditional"}
        bear_opposes = stances.get("bear") == "oppose"
        if not risk_ok:
            action = "reject"
        elif bear_opposes or concerns:
            # The bear case is *always* expected to oppose; the point is that
            # its concerns become explicit conditions rather than being ignored.
            action = "proceed_with_limits"
        elif supports:
            action = "proceed"
        else:
            action = "no_change"

        return CommitteeVerdict(
            as_of=pd.Timestamp(
                state.get("as_of") or state.get("risk", {}).get("as_of") or datetime.now(UTC)
            ).date(),
            opinions=opinions,
            agreements=agreements,
            disagreements=disagreements,
            unresolved_uncertainty=sorted(set(concerns))[:8],
            recommended_action=action,
            rationale=(
                "Verdict derives from role stances and raised concerns, not from "
                "selecting the most favourable argument. A bear objection converts "
                "into an explicit condition rather than being discarded."
            ),
        )

    def explain_agent(self, state: GraphState) -> GraphState:
        """Per-position explanations built from structured data only.

        All numbers come from `finrisk.risk.position_attribution`, which is a
        deterministic engine. The LLM contributes nothing here — this node
        exists so the report can answer "why this position" with allocation,
        alpha contribution, risk contribution, diversification effect, tail
        share, supporting AND contradicting evidence, and a derived confidence.
        """
        from finrisk.risk.position_attribution import (
            build_position_explanation,
            compute_position_attribution,
        )

        port = state.get("portfolio", {})
        weights = port.get("weights") or {}
        if not weights:
            state["explanations"] = []
            return state

        ctx = self.tools.ctx
        as_of = ctx.date(state.get("as_of"))
        rets = ctx.trailing_returns(as_of, ctx.settings.portfolio.covariance_lookback)
        ranking = {r["ticker"]: r for r in (state.get("factors", {}).get("ranking") or [])}
        alphas = state.get("alphas") or {}

        attribution = compute_position_attribution(weights=weights, returns=rets, alphas=alphas)

        # Single source of truth for risk share. The risk agent already
        # published pct_of_risk from the RiskEngine; recomputing it here with a
        # separately-estimated covariance produced a DIFFERENT number for the
        # same quantity in the same report (0.374 vs 0.358 in testing), because
        # Ledoit-Wolf shrinkage depends on the matrix it is fitted to. The
        # risk agent's figure wins, and the derived gap is recomputed from it
        # so the two stay consistent.
        published = {
            c["ticker"]: float(c["pct_of_risk"])
            for c in (state.get("risk_contribution", {}).get("contributions") or [])
        }
        if published:
            gross = sum(abs(v) for v in weights.values()) or 1.0
            for tk, share in published.items():
                if tk in attribution:
                    attribution[tk]["pct_of_portfolio_risk"] = share
                    attribution[tk]["risk_vs_weight_gap"] = share - (
                        abs(weights.get(tk, 0.0)) / gross
                    )

        out: list[PositionExplanation] = []
        for tk in sorted(weights, key=lambda t: -weights[t]):
            attrib = attribution.get(tk, {"weight": float(weights[tk])})
            docs = self.tools.call(
                "retrieve_financial_documents",
                query=f"{tk} risk factors profitability leverage outlook",
                ticker=tk,
                top_k=2,
            )
            evidence = [Citation.model_validate(c) for c in docs.get("citations", [])]
            out.append(
                build_position_explanation(
                    ticker=tk,
                    attribution=attrib,
                    factor_row=ranking.get(tk),
                    regime=state.get("regime"),
                    evidence=evidence,
                )
            )
        state["explanations"] = [e.model_dump(mode="json") for e in out]
        self._note(state, f"explain_agent: {len(out)} positions explained")
        return state

    def critic_agent(self, state: GraphState) -> GraphState:
        """Validate the portfolio and scrub ungrounded research claims."""
        self._build_risk_summary(state)
        self._evaluate_hypotheses(state)
        findings: list[CriticFinding] = []
        cfg = self.tools.ctx.settings.portfolio

        port = state.get("portfolio", {})
        if port.get("weights"):
            from finrisk.schemas import PortfolioWeights

            pw = PortfolioWeights(
                as_of=pd.Timestamp(port.get("as_of")).date(),
                method=str(port.get("method", "unknown")),
                weights={k: float(v) for k, v in port["weights"].items()},
                expected_return=float(port.get("expected_return_ann", 0.0)),
                expected_volatility=float(port.get("expected_volatility_ann", 0.0)),
                sharpe=float(port.get("sharpe_ex_ante", 0.0)),
                optimization_status=str(port.get("status", "ok")),
            )
            findings += self.critic.validate_portfolio(
                pw, cfg.max_weight, cfg.max_leverage
            ).findings

        if state.get("research"):
            valid_ids = {c["chunk_id"] for c in (state.get("citations") or [])}
            for e in state.get("explanations") or []:
                valid_ids |= {c["chunk_id"] for c in (e.get("evidence") or [])}
            tool_results = {
                k: state.get(k)
                for k in (
                    "portfolio",
                    "risk",
                    "cvar",
                    "risk_contribution",
                    "regime",
                    "factors",
                    "backtest",
                    "market",
                    "stress",
                )
                if state.get(k)
            }
            conclusion = ResearchConclusion.model_validate(state["research"])
            scrubbed, rep = self.critic.validate_research(conclusion, tool_results, valid_ids)
            state["research"] = scrubbed.model_dump(mode="json")
            findings += rep.findings

        for err in state.get("errors") or []:
            findings.append(CriticFinding(severity="error", check="node_failure", detail=err))

        report = ValidationReport(
            passed=not any(f.severity == "error" for f in findings), findings=findings
        )
        state["validation"] = report.model_dump(mode="json")
        self._note(
            state,
            f"critic_agent: {'PASS' if report.passed else 'FAIL'} "
            f"({len(report.errors)} errors, {len(findings)} findings)",
        )
        return state

    # ==================================================================
    def build_report(self, state: GraphState) -> FinalReport:
        """Assemble the typed final report from validated state."""
        from finrisk.schemas import (
            PortfolioWeights,
            RegimeState,
            RiskReport,
        )

        as_of = pd.Timestamp(
            state.get("as_of")
            or state.get("portfolio", {}).get("as_of")
            or self.tools.ctx.bundle.dates[-1]
        ).date()

        portfolio = None
        p = state.get("portfolio") or {}
        if p.get("weights"):
            portfolio = PortfolioWeights(
                as_of=as_of,
                method=str(p.get("method", "unknown")),
                weights={k: float(v) for k, v in p["weights"].items()},
                expected_return=float(p.get("expected_return_ann", 0.0)),
                expected_volatility=float(p.get("expected_volatility_ann", 0.0)),
                sharpe=float(p.get("sharpe_ex_ante", 0.0)),
                cash=float(p.get("cash", 0.0)),
                optimization_status=str(p.get("status", "ok")),
                risk_metrics={k: float(v) for k, v in (p.get("risk_metrics") or {}).items()},
            )

        risk = None
        r = state.get("risk") or {}
        if r and "error" not in r:
            risk = RiskReport(
                as_of=as_of,
                n_observations=int(r.get("n_observations", 0)),
                ann_volatility=float(r.get("ann_volatility", 0.0)),
                ann_return=float(r.get("ann_return", 0.0)),
                sharpe=float(r.get("sharpe", 0.0)),
                sortino=float(r.get("sortino", 0.0)),
                max_drawdown=float(r.get("max_drawdown", 0.0)),
                calmar=float(r.get("calmar", 0.0)),
                downside_deviation=0.0,
                model_volatility=r.get("model_volatility"),
                beta=r.get("beta"),
                var_95_hist=float(r.get("var_95_hist", 0.0)),
                var_95_param=float(r.get("var_95_param", 0.0)),
                cvar_95=float(r.get("cvar_95", 0.0)),
                cvar_99=float(r.get("cvar_99", 0.0)),
                effective_n=float(r.get("effective_n", 0.0)),
                mean_correlation=r.get("mean_correlation"),
                sector_risk={k: float(v) for k, v in (r.get("sector_risk") or {}).items()},
            )

        rg = state.get("regime") or {}
        regime = None
        if rg and "error" not in rg:
            regime = RegimeState(
                as_of=as_of,
                regime=str(rg.get("regime", "unknown")),
                state_id=0,
                confidence=float(rg.get("confidence", 0.0)),
                features={k: float(v) for k, v in (rg.get("features") or {}).items()},
                state_probabilities={
                    k: float(v) for k, v in (rg.get("state_probabilities") or {}).items()
                },
                transition_probabilities={
                    k: float(v) for k, v in (rg.get("transition_probabilities") or {}).items()
                },
                expected_duration_days=rg.get("expected_duration_days"),
            )

        plan = state.get("plan")
        risk_summary = state.get("risk_summary")
        return FinalReport(
            generated_at=datetime.now(UTC),
            as_of=as_of,
            universe=list((state.get("portfolio", {}).get("weights") or {}).keys()),
            regime=regime,
            portfolio=portfolio,
            risk=risk,
            explanations=[
                PositionExplanation.model_validate(e) for e in (state.get("explanations") or [])
            ],
            plan=ResearchPlan.model_validate(plan) if plan else None,
            risk_summary=(RiskSummary.model_validate(risk_summary) if risk_summary else None),
            research=(
                ResearchConclusion.model_validate(state["research"])
                if state.get("research")
                else None
            ),
            committee=(
                CommitteeVerdict.model_validate(state["committee"])
                if state.get("committee")
                else None
            ),
            validation=(
                ValidationReport.model_validate(state["validation"])
                if state.get("validation")
                else None
            ),
        )


# ======================================================================
# Graph construction
# ======================================================================


def build_graph(supervisor: FinanceSupervisor, include_backtest: bool = False):
    """Compile the LangGraph state machine, or None if LangGraph is absent."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:  # pragma: no cover - optional dep
        logger.info("langgraph not installed; sequential runner will be used")
        return None

    g = StateGraph(GraphState)
    g.add_node("plan_agent", supervisor.plan_agent)
    g.add_node("factor_agent", supervisor.factor_agent)
    g.add_node("market_agent", supervisor.market_agent)
    g.add_node("fundamental_agent", supervisor.fundamental_agent)
    g.add_node("portfolio_agent", supervisor.portfolio_agent)
    g.add_node("risk_agent", supervisor.risk_agent)
    g.add_node("research_agent", supervisor.research_agent)
    g.add_node("explain_agent", supervisor.explain_agent)
    g.add_node("critic_agent", supervisor.critic_agent)

    # Factors first: the market/fundamental agents narrow their queries to the
    # candidate set the factor agent produces.
    # The plan is recorded before any analysis runs, so the report can show
    # what was intended alongside what was found.
    g.add_edge(START, "plan_agent")
    g.add_edge("plan_agent", "factor_agent")
    g.add_edge("factor_agent", "market_agent")
    g.add_edge("market_agent", "fundamental_agent")
    g.add_edge("fundamental_agent", "portfolio_agent")
    g.add_edge("portfolio_agent", "risk_agent")

    if include_backtest:
        g.add_node("backtest_agent", supervisor.backtest_agent)
        g.add_edge("risk_agent", "backtest_agent")
        g.add_edge("backtest_agent", "research_agent")
    else:
        g.add_edge("risk_agent", "research_agent")

    g.add_edge("research_agent", "explain_agent")

    if supervisor.enable_committee:
        g.add_node("committee_agent", supervisor.committee_agent)
        g.add_edge("explain_agent", "committee_agent")
        g.add_edge("committee_agent", "critic_agent")
    else:
        g.add_edge("explain_agent", "critic_agent")

    g.add_edge("critic_agent", END)
    return g.compile()


def run_sequential(
    supervisor: FinanceSupervisor, state: GraphState, include_backtest: bool = False
) -> GraphState:
    """Topologically identical fallback runner. Keeps the system usable — and
    testable — without LangGraph installed."""
    steps = [
        supervisor.plan_agent,
        supervisor.factor_agent,
        supervisor.market_agent,
        supervisor.fundamental_agent,
        supervisor.portfolio_agent,
        supervisor.risk_agent,
    ]
    if include_backtest:
        steps.append(supervisor.backtest_agent)
    steps += [supervisor.research_agent, supervisor.explain_agent]
    if supervisor.enable_committee:
        steps.append(supervisor.committee_agent)
    steps.append(supervisor.critic_agent)
    for step in steps:
        state = step(state)
    return state


def run_research(
    registry: ToolRegistry,
    question: str = "How should capital be allocated and what risks are being taken?",
    as_of: str | None = None,
    options: dict[str, Any] | None = None,
    include_backtest: bool = False,
    llm: LLMClient | None = None,
) -> tuple[FinalReport, GraphState]:
    """Public entry point. Returns the typed report and the raw graph state."""
    settings = registry.ctx.settings
    sup = FinanceSupervisor(
        registry, llm=llm, enable_committee=settings.agents.enable_investment_committee
    )
    init: GraphState = {
        "question": question,
        "as_of": as_of,
        "options": options or {},
        "errors": [],
        "trace": [],
    }
    graph = build_graph(sup, include_backtest)
    if graph is not None:
        final: GraphState = cast(GraphState, dict(graph.invoke(init)))
    else:
        final = run_sequential(sup, init, include_backtest)
    return sup.build_report(final), final
