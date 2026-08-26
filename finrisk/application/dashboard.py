"""Streamlit dashboard (Phase 29).

Deliberately thin. The brief says not to spend the project on the frontend, and
the quantitative engine is the deliverable — so this is a viewer over the same
core modules the API and scripts use, with no logic of its own.

Run with:
    streamlit run finrisk/application/dashboard.py
"""

from __future__ import annotations

import pandas as pd

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional dep
    raise SystemExit(
        "streamlit is not installed. Install with: pip install streamlit\n"
        "The API (uvicorn finrisk.api.app:app) exposes the same functionality."
    ) from exc

from finrisk.agents.graph import run_research
from finrisk.config.settings import default_settings
from finrisk.data.documents import load_corpus
from finrisk.data.market_data import align_to_calendar, load_bundle
from finrisk.data.validation import validate_bundle
from finrisk.factors.engine import FactorEngine
from finrisk.rag.retriever import DocumentRetriever
from finrisk.tools.finance_tools import ToolContext, ToolRegistry

st.set_page_config(page_title="FINRISK AI", layout="wide")


@st.cache_resource(show_spinner="Loading dataset…")
def _load():
    settings = default_settings()
    bundle = align_to_calendar(load_bundle(settings.data))
    retriever = DocumentRetriever(settings.rag)
    docs = load_corpus(settings.rag.docs_dir)
    if docs:
        retriever.add_documents(docs)
    return settings, bundle, retriever


settings, bundle, retriever = _load()

st.title("FINRISK AI")
st.caption(
    "Agentic quantitative portfolio & risk intelligence. "
    "Research and decision support only — not investment advice."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuration")
    dates = bundle.dates
    as_of = st.select_slider(
        "As-of date",
        options=[d.date() for d in dates[settings.backtest.warmup :: 21]],
        value=dates[-1].date(),
    )
    method = st.selectbox(
        "Construction method",
        [
            "risk_constrained",
            "cvar_constrained",
            "max_sharpe",
            "mean_variance",
            "risk_parity",
            "equal_weight",
        ],
    )
    top_n = st.slider("Universe size (top-N by score)", 3, 30, settings.portfolio.top_n)
    max_w = st.slider("Max position weight", 0.05, 1.0, settings.portfolio.max_weight, 0.05)
    target_vol = st.slider("Target volatility", 0.05, 0.40, 0.15, 0.01)
    use_regime = st.checkbox("Apply regime overlay", value=True)
    st.divider()
    st.caption(f"Provider: `{bundle.provider}` · {len(bundle.tickers)} assets")
    if bundle.provider == "synthetic":
        st.warning(
            "Synthetic dataset. Figures validate the pipeline; they are NOT "
            "historical performance.",
            icon="⚠️",
        )

cfg = settings.model_copy(deep=True)
cfg.portfolio.top_n = top_n
registry = ToolRegistry(ToolContext(bundle, cfg, retriever, pd.Timestamp(as_of)))
iso = str(as_of)

# ---------------------------------------------------------------- compute
port = registry.call(
    "optimize_portfolio",
    method=method,
    as_of=iso,
    max_weight=max_w,
    target_volatility=target_vol,
    use_regime=use_regime,
)
if "error" in port:
    st.error(port["error"])
    st.stop()
weights = port["weights"]
risk = registry.call("calculate_risk", weights=weights, as_of=iso)
regime = registry.call("detect_market_regime", as_of=iso)
contrib = registry.call("calculate_risk_contribution", weights=weights, as_of=iso)
factors = registry.call("calculate_factor_scores", as_of=iso, top_n=top_n)

# ---------------------------------------------------------------- headline
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Positions", len(weights))
c2.metric("Ex-ante vol", f"{port['expected_volatility_ann']:.2%}")
c3.metric("CVaR 95% (1d)", f"{risk['cvar_95']:.2%}")
c4.metric("Effective N", f"{risk['effective_n']:.1f}")
c5.metric("Regime", regime.get("regime", "n/a"), f"conf {regime.get('confidence', 0):.2f}")

tabs = st.tabs(
    [
        "Allocation",
        "Factors",
        "Neutralisation",
        "Risk",
        "Attribution",
        "Backtest",
        "Regime",
        "Stress",
        "Research",
        "Data quality",
    ]
)

with tabs[0]:
    st.subheader("Allocation")
    wf = (
        pd.Series(weights, name="weight")
        .sort_values(ascending=False)
        .to_frame()
        .assign(sector=lambda d: [bundle.sector_map().get(t, "Unknown") for t in d.index])
    )
    st.bar_chart(wf["weight"])
    st.dataframe(wf.style.format({"weight": "{:.2%}"}), width="stretch")
    st.caption(
        f"Cash {port['cash']:.2%} · status `{port['status']}` · "
        f"constraints satisfied: {port['constraints_satisfied']}"
    )
    if port.get("violations"):
        st.error("Constraint violations: " + "; ".join(port["violations"]))

with tabs[1]:
    st.subheader("Factor cross-section")
    st.dataframe(pd.DataFrame(factors["ranking"]), width="stretch")
    st.caption(f"Composite weights: {factors['composite_weights']}")

with tabs[2]:
    st.subheader("Factor neutralisation")
    st.caption(
        "Does the factor premium survive removing sector and beta exposure, or "
        "was it those exposures all along?"
    )
    factor_choice = st.selectbox("Factor", ["value", "quality", "momentum", "risk", "composite"])
    gross = st.slider("Gross exposure", 0.5, 4.0, 2.0, 0.5)
    cap = st.slider("Max absolute weight", 0.02, 0.50, 0.10, 0.01)
    from finrisk.factors.neutralization import (
        build_neutral_portfolio,
        estimate_betas,
        exposure_report,
    )

    col = "composite" if factor_choice == "composite" else f"grp_{factor_choice}"
    _, diag_panel = FactorEngine(cfg.factors).compute(bundle, pd.Timestamp(as_of))
    if diag_panel.empty or col not in diag_panel.columns:
        st.warning(f"Factor '{factor_choice}' unavailable on {as_of}.")
    else:
        scores = diag_panel[col].dropna()
        hist = bundle.prices.loc[:as_of].pct_change().tail(252)
        betas_est = estimate_betas(
            hist[[c for c in scores.index if c in hist.columns]],
            bundle.benchmark.loc[:as_of].pct_change().tail(252),
        )
        neutral_w = build_neutral_portfolio(scores, bundle.sector_map(), betas_est, gross, cap)
        if neutral_w.empty:
            st.warning("Insufficient cross-section to build a neutral book.")
        else:
            exposures = exposure_report(neutral_w, bundle.sector_map(), betas_est)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Net exposure", f"{exposures['net_exposure']:.2e}")
            e2.metric("Portfolio beta", f"{exposures['portfolio_beta']:.2e}")
            e3.metric("Max sector", f"{exposures['max_abs_sector_exposure']:.2e}")
            e4.metric("Long / short", f"{exposures['n_long']} / {exposures['n_short']}")
            st.bar_chart(neutral_w.sort_values(ascending=False))
            st.caption(
                "These exposures ARE the evidence for neutrality. They are not "
                "exactly zero because betas are estimated with error."
            )

with tabs[3]:
    st.subheader("Risk metrics")
    keys = [
        "ann_volatility",
        "model_volatility",
        "ann_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "beta",
        "var_95_hist",
        "var_95_param",
        "cvar_95",
        "cvar_99",
        "effective_n",
        "mean_correlation",
    ]
    st.dataframe(
        pd.Series({k: risk.get(k) for k in keys}, name="value").to_frame(),
        width="stretch",
    )
    if risk.get("concentration_flags"):
        for f in risk["concentration_flags"]:
            st.warning(f)
    st.caption(
        "Historical and parametric VaR are shown together: a large gap means "
        "returns are non-Gaussian and the Gaussian figure understates the tail."
    )

with tabs[4]:
    st.subheader("Risk attribution")
    cdf = pd.DataFrame(contrib["contributions"]).set_index("ticker")
    st.bar_chart(cdf["pct_of_risk"])
    st.dataframe(cdf, width="stretch")
    if contrib.get("sector_risk"):
        st.write("**Sector risk share**")
        st.dataframe(pd.Series(contrib["sector_risk"], name="share").to_frame(), width="stretch")
    st.caption(
        f"Euler identity holds: {contrib['euler_identity_holds']} "
        f"(components sum to model volatility {contrib['model_volatility']:.4f})"
    )

with tabs[5]:
    st.subheader("Walk-forward backtest")
    strategy = st.selectbox(
        "Strategy",
        [
            "F_finrisk_full",
            "E_factor_optimized_risk_constrained",
            "D_factor_mean_variance",
            "D2_factor_risk_parity",
            "C_factor_regime_equal_weight",
            "B_factor_equal_weight",
            "A_equal_weight_universe",
            "benchmark_buy_hold",
        ],
    )
    if st.button("Run backtest (slow)"):
        with st.spinner("Running walk-forward evaluation…"):
            res = registry.call("run_backtest", strategy=strategy)
        if "error" in res:
            st.error(res["error"])
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR", f"{res['cagr']:.2%}")
            k2.metric("Sharpe", f"{res['sharpe']:.2f}")
            k3.metric("Max drawdown", f"{res['max_drawdown']:.2%}")
            k4.metric("Avg turnover", f"{res['avg_turnover']:.2%}")
            st.json(res)

with tabs[6]:
    st.subheader("Market regime")
    st.json(regime)
    st.caption(
        "Inference is FILTERED (forward-only). Smoothed probabilities would use "
        "future data and are excluded from any decision path."
    )

with tabs[7]:
    st.subheader("Stress scenarios")
    stress = registry.call("stress_test", weights=weights, as_of=iso)
    st.dataframe(pd.DataFrame(stress["scenarios"]), width="stretch")
    st.caption(stress.get("note", ""))

with tabs[8]:
    st.subheader("Agentic research")
    question = st.text_input("Question", "Why these positions, and what risks are being taken?")
    if st.button("Run research graph"):
        with st.spinner("Running agent graph…"):
            report, state = run_research(
                registry,
                question=question,
                as_of=iso,
                options={"method": method, "top_n": top_n},
            )
        st.write("**Trace**")
        st.code("\n".join(state.get("trace", [])))
        if report.research:
            st.write("**Thesis**")
            st.info(report.research.thesis)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.write("**Supporting**")
                for e in report.research.supporting_evidence:
                    st.write(f"- {e}")
            with cc2:
                st.write("**Contradicting**")
                for e in report.research.contradicting_evidence:
                    st.write(f"- {e}")
            st.write("**Risk factors**")
            for r in report.research.risk_factors:
                st.write(f"- {r}")
            st.write("**Sources**")
            for c in report.research.sources:
                st.caption(f"`{c.chunk_id}` {c.source} — {c.excerpt[:180]}…")
        if report.committee:
            st.write("**Investment committee**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"role": o.role, "stance": o.stance, "argument": o.argument}
                        for o in report.committee.opinions
                    ]
                ),
                width="stretch",
            )
            st.write(f"Recommended action: `{report.committee.recommended_action}`")
        if report.validation:
            (st.success if report.validation.passed else st.error)(
                f"Critic validation: {'PASSED' if report.validation.passed else 'FAILED'} "
                f"({len(report.validation.findings)} findings)"
            )
            for f in report.validation.findings:
                st.caption(f"[{f.severity}] {f.check}: {f.detail}")
        if report.plan:
            st.write("**Research plan (recorded before analysis)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "hypothesis": h.statement,
                            "falsifier": h.falsifier,
                            "verdict": h.verdict,
                            "evidence": str(h.evidence),
                        }
                        for h in report.plan.hypotheses
                    ]
                ),
                width="stretch",
            )
        if report.explanations:
            from finrisk.risk.position_attribution import render_explanations

            st.write("**Position attribution**")
            st.code(render_explanations(report.explanations))
        st.write("**Position explanations**")
        for pos in report.explanations:
            with st.expander(f"{pos.ticker} — {pos.weight:.2%}"):
                st.write("Positive drivers:", pos.drivers_positive or "none")
                st.write("Negative drivers:", pos.drivers_negative or "none")
                st.write(pos.risk_note)
                st.caption(pos.regime_note)

with tabs[9]:
    st.subheader("Data quality")
    rep = validate_bundle(bundle)
    st.code(rep.render())

st.divider()
st.caption(
    "FINRISK AI · quantitative research and decision-support platform. "
    "Backtested and simulated results do not indicate future returns. "
    "Not investment advice."
)
