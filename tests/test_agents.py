"""Agent layer: graph wiring, structured output, and the LLM/quant barrier.

The most important tests here are the critic tests: they demonstrate that a
fabricated number cannot survive into a report. That is the project's central
safety claim, so it is tested by *injecting* fabrications rather than by
inspecting well-behaved output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.agents.critic import Critic, render_validation
from finrisk.agents.graph import (
    FinanceSupervisor,
    build_graph,
    run_research,
    run_sequential,
)
from finrisk.agents.llm import StubLLM, parse_json_response
from finrisk.schemas import Citation, PortfolioWeights, ResearchConclusion
from finrisk.tools.finance_tools import ToolContext, ToolRegistry


@pytest.fixture(scope="module")
def registry(small_bundle, small_settings):
    from finrisk.config.settings import RAGConfig
    from finrisk.data.documents import FinancialDocument
    from finrisk.rag.retriever import DocumentRetriever

    r = DocumentRetriever(RAGConfig(top_k=3))
    r.add_documents(
        [
            FinancialDocument(
                doc_id="dz",
                source="SY00_10K_2019.txt",
                ticker="SY00",
                doc_type="10-K",
                text=(
                    "Return on equity was 14 percent. Total debt per share was 3.10 "
                    "against equity of 9.00. Risk factors include commodity price "
                    "volatility and rate shocks."
                ),
            )
        ]
    )
    return ToolRegistry(ToolContext(small_bundle, small_settings, r))


@pytest.fixture(scope="module")
def as_of(small_bundle):
    return str(small_bundle.dates[-1].date())


# ---------------------------------------------------------------- LLM plumbing
def test_stub_llm_is_deterministic():
    llm = StubLLM()
    a = llm.complete("ROLE: research", '<facts:risk>{"cvar_95": -0.02}</facts:risk>')
    b = llm.complete("ROLE: research", '<facts:risk>{"cvar_95": -0.02}</facts:risk>')
    assert a == b


def test_parse_json_handles_fenced_and_prefixed():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Sure! {"a": 2} hope that helps') == {"a": 2}
    assert parse_json_response("not json at all") == {}


def test_parse_json_handles_nested_braces():
    assert parse_json_response('{"a": {"b": 1}}') == {"a": {"b": 1}}


# ---------------------------------------------------------------- critic
def test_critic_removes_fabricated_metric():
    """THE core safety test: a number the tools never produced is stripped."""
    tool_results = {"risk": {"cvar_95": -0.0187, "ann_volatility": 0.1386}}
    conclusion = ResearchConclusion(
        thesis="Portfolio is sound.",
        quantitative_evidence={
            "cvar_95": -0.0187,  # real
            "secret_alpha": 0.4200,  # fabricated
        },
    )
    scrubbed, report = Critic().validate_research(conclusion, tool_results)
    assert "cvar_95" in scrubbed.quantitative_evidence
    assert "secret_alpha" not in scrubbed.quantitative_evidence
    assert not report.passed
    assert any(f.check == "numeric_grounding" for f in report.errors)


def test_critic_accepts_grounded_metrics():
    tool_results = {"risk": {"cvar_95": -0.0187}}
    c = ResearchConclusion(
        thesis="ok",
        quantitative_evidence={"cvar_95": -0.0187},
        contradicting_evidence=["estimation error"],
    )
    scrubbed, report = Critic().validate_research(c, tool_results)
    assert scrubbed.quantitative_evidence == {"cvar_95": -0.0187}
    assert not report.errors


def test_critic_removes_unbacked_citation():
    c = ResearchConclusion(
        thesis="t",
        sources=[
            Citation(
                doc_id="x",
                source="fake_10K.txt",
                chunk_id="ghost:99",
                score=1.0,
                excerpt="invented",
            ),
        ],
    )
    scrubbed, report = Critic().validate_research(c, {}, valid_chunk_ids={"real:1"})
    assert scrubbed.sources == []
    assert any(f.check == "citation_validity" for f in report.errors)


def test_critic_keeps_valid_citation():
    cit = Citation(doc_id="x", source="a.txt", chunk_id="real:1", score=1.0, excerpt="e")
    c = ResearchConclusion(thesis="t", sources=[cit])
    scrubbed, _ = Critic().validate_research(c, {}, valid_chunk_ids={"real:1"})
    assert len(scrubbed.sources) == 1


def test_critic_flags_ungrounded_percentage_in_prose():
    tool_results = {"risk": {"cvar_95": -0.0187}}
    c = ResearchConclusion(
        thesis="The strategy outperformed by 340%.",
        contradicting_evidence=["noise"],
    )
    _, report = Critic().validate_research(c, tool_results)
    assert any(f.check == "prose_numeric_grounding" for f in report.findings)


def test_critic_flags_one_sided_and_overconfident():
    c = ResearchConclusion(
        thesis="t", supporting_evidence=["a"], contradicting_evidence=[], confidence=0.95
    )
    _, report = Critic().validate_research(c, {})
    checks = {f.check for f in report.findings}
    assert "evidence_balance" in checks
    assert "overconfidence" in checks


def test_critic_rejects_weight_cap_breach():
    pw = PortfolioWeights(
        as_of=pd.Timestamp("2024-01-31").date(),
        method="m",
        weights={"A": 0.5, "B": 0.5},
        expected_return=0.1,
        expected_volatility=0.1,
        sharpe=1.0,
    )
    rep = Critic().validate_portfolio(pw, max_weight=0.20)
    assert not rep.passed
    assert any(f.check == "max_weight" for f in rep.errors)


def test_critic_rejects_negative_weight_and_leverage():
    pw = PortfolioWeights(
        as_of=pd.Timestamp("2024-01-31").date(),
        method="m",
        weights={"A": -0.3, "B": 1.2},
        expected_return=0.1,
        expected_volatility=0.1,
        sharpe=1.0,
    )
    rep = Critic().validate_portfolio(pw, max_weight=1.5, max_leverage=1.0)
    checks = {f.check for f in rep.errors}
    assert "negative_weight" in checks
    assert "leverage" in checks


def test_critic_accepts_compliant_portfolio():
    pw = PortfolioWeights(
        as_of=pd.Timestamp("2024-01-31").date(),
        method="m",
        weights={"A": 0.2, "B": 0.2, "C": 0.2},
        expected_return=0.1,
        expected_volatility=0.1,
        sharpe=1.0,
    )
    assert Critic().validate_portfolio(pw, max_weight=0.20).passed


def test_critic_flags_optimizer_fallback():
    pw = PortfolioWeights(
        as_of=pd.Timestamp("2024-01-31").date(),
        method="m",
        weights={"A": 0.1},
        expected_return=0.0,
        expected_volatility=0.1,
        sharpe=0.0,
        optimization_status="fallback_equal_weight:singular",
    )
    rep = Critic().validate_portfolio(pw, 0.2, require_status_ok=True)
    assert not rep.passed
    assert any(f.check == "optimizer_status" for f in rep.errors)


def test_critic_detects_leakage():
    rep = Critic.validate_no_leakage("2024-01-31", ["2024-02-15"])
    assert not rep.passed
    assert rep.errors[0].check == "look_ahead"
    assert Critic.validate_no_leakage("2024-01-31", ["2024-01-15"]).passed


def test_render_validation_outputs_text():
    rep = Critic.validate_no_leakage("2024-01-31", ["2024-02-15"])
    assert "FAILED" in render_validation(rep)
    assert "PASSED" in render_validation(Critic.validate_no_leakage("2024-01-31", []))


# ---------------------------------------------------------------- graph
def test_graph_compiles_with_langgraph(registry):
    sup = FinanceSupervisor(registry, llm=StubLLM())
    graph = build_graph(sup)
    assert graph is not None, "langgraph is installed, so the graph must compile"


def test_sequential_runner_matches_graph_nodes(registry, as_of):
    sup = FinanceSupervisor(registry, llm=StubLLM(), enable_committee=True)
    state = run_sequential(
        sup, {"question": "q", "as_of": as_of, "options": {}, "errors": [], "trace": []}
    )
    # A node may emit more than one trace note (e.g. research_agent logs a
    # query expansion), so collapse consecutive repeats before comparing order.
    names: list[str] = []
    for t in state["trace"]:
        node = t.split(":")[0]
        if not names or names[-1] != node:
            names.append(node)
    assert names == [
        "plan_agent",
        "factor_agent",
        "market_agent",
        "fundamental_agent",
        "portfolio_agent",
        "risk_agent",
        "research_agent",
        "explain_agent",
        "committee_agent",
        "critic_agent",
    ]


def test_run_research_produces_validated_report(registry, as_of):
    report, state = run_research(registry, "Why these positions?", as_of=as_of)
    assert report.portfolio is not None and report.portfolio.weights
    assert report.risk is not None
    assert report.regime is not None
    assert report.research is not None
    assert report.validation is not None and report.validation.passed, [
        f.detail for f in (report.validation.findings if report.validation else [])
    ]
    assert not state.get("errors")


def test_report_numbers_match_tool_output(registry, as_of):
    """No number in the research conclusion may be absent from the tool results."""
    report, state = run_research(registry, "risk profile", as_of=as_of)
    risk = state["risk"]
    ev = report.research.quantitative_evidence
    assert ev
    assert ev["cvar_95"] == pytest.approx(risk["cvar_95"])
    assert ev["realised_volatility"] == pytest.approx(risk["ann_volatility"])


def test_committee_is_adversarial(registry, as_of):
    report, _ = run_research(registry, "assess this portfolio", as_of=as_of)
    c = report.committee
    assert c is not None
    roles = {o.role for o in c.opinions}
    assert roles == {"bull", "bear", "risk", "quant"}
    stances = {o.stance for o in c.opinions}
    assert len(stances) > 1, "committee produced a unanimous view — not adversarial"
    assert c.recommended_action in {"proceed", "proceed_with_limits", "reject", "no_change"}


def test_committee_does_not_simply_follow_the_bull(registry, as_of):
    report, _ = run_research(registry, "assess", as_of=as_of)
    c = report.committee
    bear = next(o for o in c.opinions if o.role == "bear")
    assert bear.stance == "oppose"
    # A bear objection must become an explicit condition, not be discarded.
    assert c.recommended_action != "proceed"
    assert c.unresolved_uncertainty


def test_explanations_are_grounded_in_structured_data(registry, as_of):
    report, state = run_research(registry, "explain positions", as_of=as_of)
    assert report.explanations
    contrib = {c["ticker"]: c["pct_of_risk"] for c in state["risk_contribution"]["contributions"]}
    for e in report.explanations:
        assert e.ticker in report.portfolio.weights
        assert e.pct_of_portfolio_risk == pytest.approx(contrib[e.ticker], abs=1e-6)
        assert e.factor_contributions


def test_llm_cannot_change_weights(registry, as_of):
    """The barrier test: the weights in the report must equal the optimiser's
    output exactly, regardless of what the narrative says."""
    report, state = run_research(registry, "recommend 90% in one name", as_of=as_of)
    assert report.portfolio.weights == {
        k: pytest.approx(v) for k, v in state["portfolio"]["weights"].items()
    }
    assert (
        max(report.portfolio.weights.values()) <= registry.ctx.settings.portfolio.max_weight + 1e-6
    )


def test_committee_never_cites_absent_metrics(registry, as_of):
    """Regression: the bear previously emitted 'Maximum drawdown of n/a' when
    no backtest had run — a sentence that reads like evidence but carries
    none. No opinion may contain a placeholder value."""
    report, _ = run_research(registry, "assess", as_of=as_of, include_backtest=False)
    for o in report.committee.opinions:
        assert "n/a" not in o.argument.lower(), f"{o.role}: {o.argument}"
        assert "nan" not in o.argument.lower(), f"{o.role}: {o.argument}"


def test_research_expands_query_when_retrieval_is_empty(registry, as_of):
    """BM25 is lexical, so a conversational question can score zero against
    filings. The agent retries with holdings plus filing vocabulary rather
    than reporting no evidence — and every citation still resolves."""
    report, state = run_research(
        registry, "Why these positions, and what risks are being taken?", as_of=as_of
    )
    valid = {c["chunk_id"] for c in state["citations"]}
    assert report.research.sources, "expansion produced no citations"
    for c in report.research.sources:
        assert c.chunk_id in valid


def test_graph_survives_a_failing_node(small_bundle, small_settings, as_of):
    """A tool failure must degrade the report, not crash the run."""
    reg = ToolRegistry(ToolContext(small_bundle, small_settings, None))
    report, _ = run_research(reg, "no documents available", as_of=as_of)
    assert report.portfolio is not None  # still produced
    assert report.research is not None
    assert report.research.sources == []  # no index, so no citations


class _ExplodingLLM:
    """Stands in for an API outage, rate limit or timeout."""

    name = "exploding"

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        raise RuntimeError("simulated API outage")


def test_llm_outage_degrades_narration_not_numbers(registry, as_of):
    """The model only narrates, so losing it must not abort the run or move a number."""
    baseline, _ = run_research(registry, "assess", as_of=as_of)
    report, _ = run_research(registry, "assess", as_of=as_of, llm=_ExplodingLLM())
    assert report.research.thesis == "research synthesis unavailable"
    assert report.portfolio.weights == baseline.portfolio.weights
    assert report.risk == baseline.risk
    # No committee opinions means no risk sign-off, so the verdict fails closed.
    assert report.committee.recommended_action == "reject"


def test_accepts_kwarg_reads_the_signature():
    from finrisk.agents.llm import _accepts_kwarg

    def explicit(*, temperature: float = 0.0): ...

    def catch_all(**kwargs): ...

    def without(*, max_tokens: int): ...

    assert _accepts_kwarg(explicit, "temperature")
    assert _accepts_kwarg(catch_all, "temperature")
    assert not _accepts_kwarg(without, "temperature")


def _fake_anthropic_response():
    from types import SimpleNamespace

    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )


def test_anthropic_backend_retries_without_temperature(monkeypatch):
    """Newer SDKs/models reject `temperature`; the backend must adapt, not crash."""
    pytest.importorskip("anthropic")
    from types import SimpleNamespace

    from finrisk.agents.llm import AnthropicLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = AnthropicLLM()
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(dict(kwargs))
        if "temperature" in kwargs:
            raise TypeError("create() got an unexpected keyword argument 'temperature'")
        return _fake_anthropic_response()

    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    llm._send_temperature = True

    assert llm.complete("system", "prompt") == "ok"
    assert "temperature" in seen[0] and "temperature" not in seen[1]
    assert llm._send_temperature is False  # remembered: no wasted retry next time
    llm.complete("system", "prompt")
    assert "temperature" not in seen[-1] and len(seen) == 3


def test_anthropic_backend_does_not_swallow_unrelated_errors(monkeypatch):
    pytest.importorskip("anthropic")
    from types import SimpleNamespace

    from finrisk.agents.llm import AnthropicLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = AnthropicLLM()

    def create(**kwargs):
        raise RuntimeError("connection reset")

    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    llm._send_temperature = True
    with pytest.raises(RuntimeError, match="connection reset"):
        llm.complete("system", "prompt")


# ------------------------------------------------------- structured research
def test_plan_is_recorded_before_analysis(registry, as_of):
    """The plan must exist in the report, and must run first — a plan written
    after the results is a narrative, not a plan."""
    report, state = run_research(registry, "why these positions", as_of=as_of)
    assert report.plan is not None
    assert state["trace"][0].startswith("plan_agent")
    assert report.plan.hypotheses
    assert report.plan.success_criteria and report.plan.known_limitations


def test_every_hypothesis_has_a_falsifier(registry, as_of):
    """A hypothesis with no stated way to be wrong is unfalsifiable."""
    report, _ = run_research(registry, "assess", as_of=as_of)
    for h in report.plan.hypotheses:
        assert h.falsifier.strip(), h.statement
        assert h.test_plan


def test_hypotheses_are_graded_against_their_own_thresholds(registry, as_of):
    report, state = run_research(registry, "assess", as_of=as_of)
    graded = [h for h in report.plan.hypotheses if h.verdict != "untested"]
    assert graded, "hypotheses must be evaluated after the run"
    vol = next((h for h in graded if "volatility" in h.statement.lower()), None)
    assert vol is not None
    assert vol.evidence.get("realised_volatility") == pytest.approx(
        state["risk"]["ann_volatility"], abs=1e-6
    )


def test_untested_hypothesis_is_inconclusive_not_supported(registry, as_of):
    """Without a backtest, the predictive-power claim cannot be checked.
    Claiming support anyway is exactly what the falsifier exists to prevent."""
    report, _ = run_research(registry, "assess", as_of=as_of, include_backtest=False)
    ic = next(h for h in report.plan.hypotheses if "predictive power" in h.statement)
    assert ic.verdict == "inconclusive"


def test_risk_summary_matches_tool_output(registry, as_of):
    report, state = run_research(registry, "risk profile", as_of=as_of)
    rs = report.risk_summary
    assert rs is not None
    assert rs.ann_volatility == pytest.approx(state["risk"]["ann_volatility"])
    assert rs.cvar_95 == pytest.approx(state["risk"]["cvar_95"])
    assert rs.effective_n == pytest.approx(state["risk"]["effective_n"])
    assert rs.top_risk_contributors


def test_explanations_include_alpha_and_diversification(registry, as_of):
    """Explainability must answer contribution, not just allocation."""
    report, _ = run_research(registry, "explain positions", as_of=as_of)
    assert report.explanations
    for e in report.explanations:
        assert e.confidence_basis, e.ticker
        assert 0.0 <= e.confidence <= 1.0
        assert np.isfinite(e.diversification_benefit)
        assert np.isfinite(e.alpha_contribution)
    # At least one position should carry a real alpha forecast and tail share.
    assert any(abs(e.alpha_forecast) > 0 for e in report.explanations)
    assert any(e.pct_of_tail_loss > 0 for e in report.explanations)


def test_risk_share_is_consistent_across_the_report(registry, as_of):
    """Regression: the same quantity appeared twice with different values
    (0.374 vs 0.358) because two engines fitted covariance to different
    subsets. One report, one number."""
    report, state = run_research(registry, "explain", as_of=as_of)
    published = {c["ticker"]: c["pct_of_risk"] for c in state["risk_contribution"]["contributions"]}
    for e in report.explanations:
        if e.ticker in published:
            assert e.pct_of_portfolio_risk == pytest.approx(published[e.ticker], abs=1e-9)
