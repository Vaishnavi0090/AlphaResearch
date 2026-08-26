#!/usr/bin/env python3
"""Single reproducible entry point (Phase 35).

    python scripts/run_all.py --quick    # ~2 min   smoke test of every stage
    python scripts/run_all.py            # ~6 min   default: suite + ablation + audit
    python scripts/run_all.py --full     # ~15 min  adds robustness + factor study

Writes to:
    evaluation/results/{results,benchmark_suite,ablation,robustness}.{json,csv}
    evaluation/reports/REPORT.md
    evaluation/reports/{LEAKAGE_AUDIT,FACTOR_STUDY,AGENT_REPORT}.txt
    evaluation/results/observability.json

Every stage prints its own numbers so a reader can verify the docs against a
live run rather than trusting the committed files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from finrisk.agents.critic import render_validation  # noqa: E402
from finrisk.agents.graph import run_research  # noqa: E402
from finrisk.api.observability import Telemetry  # noqa: E402
from finrisk.config.settings import Settings, default_settings  # noqa: E402
from finrisk.data.documents import load_corpus  # noqa: E402
from finrisk.data.market_data import align_to_calendar, load_bundle  # noqa: E402
from finrisk.data.validation import validate_bundle  # noqa: E402
from finrisk.evaluation.leakage import run_full_audit  # noqa: E402
from finrisk.evaluation.runner import (  # noqa: E402
    ContextCache,
    render_ablation,
    render_robustness,
    run_ablation,
    run_benchmark_suite,
    run_robustness,
    save_combined,
    save_run,
    statistical_note,
    write_report,
)
from finrisk.factors.engine import FactorEngine  # noqa: E402
from finrisk.factors.research import (  # noqa: E402
    factor_correlation,
    render_factor_report,
    study_factor,
)
from finrisk.rag.corpus import generate_corpus  # noqa: E402
from finrisk.rag.retriever import DocumentRetriever  # noqa: E402
from finrisk.regimes.detector import RegimeDetector  # noqa: E402
from finrisk.tools.finance_tools import ToolContext, ToolRegistry  # noqa: E402

RESULTS = ROOT / "evaluation" / "results"
REPORTS = ROOT / "evaluation" / "reports"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description="FINRISK AI end-to-end pipeline")
    ap.add_argument("--quick", action="store_true", help="short window, fewer studies")
    ap.add_argument("--full", action="store_true", help="add robustness + factor study")
    ap.add_argument("--no-agent", action="store_true", help="skip the agent graph")
    ap.add_argument("--seed", type=int, default=None, help="override synthetic seed")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    telemetry = Telemetry()
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    settings: Settings = default_settings()
    if args.seed is not None:
        settings.data.synthetic_seed = args.seed
    if args.quick:
        settings.data.start = "2014-01-02"
        settings.data.end = "2020-12-31"
        settings.data.synthetic_assets = 20

    # ---------------------------------------------------------------- data
    banner("1. DATA")
    with telemetry.stage("load_bundle"):
        bundle = align_to_calendar(load_bundle(settings.data))
    print(
        f"provider={bundle.provider}  assets={len(bundle.tickers)}  "
        f"sessions={len(bundle.dates)}  "
        f"{bundle.dates.min().date()} -> {bundle.dates.max().date()}"
    )
    for note in bundle.notes:
        print(f"  note: {note}")

    with telemetry.stage("validate_bundle"):
        quality = validate_bundle(bundle)
    print(quality.render())

    # ---------------------------------------------------------------- documents
    banner("2. DOCUMENT CORPUS")
    with telemetry.stage("build_corpus"):
        written = generate_corpus(
            bundle,
            settings.rag.docs_dir,
            n_tickers=len(bundle.tickers),
            clean=True,
        )
        docs = load_corpus(settings.rag.docs_dir)
        retriever = DocumentRetriever(settings.rag)
        if docs:
            retriever.add_documents(docs)
    print(
        f"generated {len(written)} documents; indexed {retriever.n_documents} "
        f"documents -> {retriever.n_chunks} chunks"
    )
    demo = retriever.retrieve("return on equity and leverage", top_k=3)
    for c in demo:
        print(f"  [{c.score:.2f}] {c.source} :: {c.excerpt[:90]}...")

    # ---------------------------------------------------------------- regime
    banner("3. REGIME DETECTION")
    with telemetry.stage("regime_path"):
        rd = RegimeDetector(settings.regimes)
        rd.prepare(bundle.benchmark)
        month_ends = (
            pd.Series(bundle.dates).groupby([bundle.dates.year, bundle.dates.month]).last().tolist()
        )
        month_ends = [d for d in month_ends if d >= bundle.dates[settings.backtest.warmup]]
        path = rd.history(bundle.benchmark, month_ends)
        stab = rd.stability(path)
    print(
        f"periods={stab['n_periods']:.0f}  switch_rate={stab['switch_rate']:.2f}  "
        f"mean_run={stab['mean_run_length']:.2f} months  "
        f"mean_confidence={stab['mean_confidence']:.3f}"
    )
    print(
        "regime shares: "
        + ", ".join(f"{k[6:]}={v:.1%}" for k, v in sorted(stab.items()) if k.startswith("share_"))
    )
    if bundle.ground_truth is not None:
        truth = bundle.ground_truth.get("market_regime")
        if truth is not None:
            aligned = pd.DataFrame(
                {
                    "detected": pd.Series({pd.Timestamp(p.as_of): p.regime for p in path}),
                    "true": truth.reindex([pd.Timestamp(p.as_of) for p in path]).values,
                }
            ).dropna()
            hit_bear = (
                aligned.loc[aligned["true"] == "bear_high_vol", "detected"] == "risk_off_high_vol"
            ).mean()
            hit_bull = (
                aligned.loc[aligned["true"] == "bull_low_vol", "detected"] == "risk_on_low_vol"
            ).mean()
            print(
                f"vs ground truth: P(risk_off | true bear)={hit_bear:.3f}  "
                f"P(risk_on | true bull)={hit_bull:.3f}"
            )

    # ---------------------------------------------------------------- factors
    if args.full:
        banner("4. FACTOR RESEARCH")
        with telemetry.stage("factor_study"):
            fe = FactorEngine(settings.factors)
            groups: dict[str, dict] = {}
            comp_rows: dict = {}
            for d in month_ends:
                _, diag = fe.compute(bundle, d)
                if diag.empty:
                    continue
                comp_rows[d] = diag["composite"]
                for col in [c for c in diag.columns if c.startswith("grp_")]:
                    groups.setdefault(col[4:], {})[d] = diag[col]
            gp = {k: pd.DataFrame(v).T.sort_index() for k, v in groups.items()}
            panels = {"composite": pd.DataFrame(comp_rows).T.sort_index()}
            panels.update(gp)
            perfs = [study_factor(name, panel, bundle.prices) for name, panel in panels.items()]
            report_txt = render_factor_report(perfs)
            corr = factor_correlation(panels)
        print(report_txt)
        print("\nFactor group correlations (mean cross-sectional):")
        print(corr.round(3).to_string())
        (REPORTS / "FACTOR_STUDY.txt").write_text(
            report_txt + "\n\nCorrelations\n" + corr.round(4).to_string()
        )

        # Factor neutralisation: does the premium survive removing sector and
        # beta exposure, or was it those exposures all along?
        with telemetry.stage("neutralisation"):
            from finrisk.factors.neutralization import (
                neutral_vs_raw,
                render_neutrality_report,
            )

            neutral_tables = {}
            for name in ("value", "quality", "momentum"):
                if name not in gp:
                    continue
                neutral_tables[name] = neutral_vs_raw(
                    gp[name], bundle.prices, bundle.sector_map(), bundle.benchmark
                )
        for name, table in neutral_tables.items():
            print()
            print(f"### {name.upper()}")
            print(render_neutrality_report(table))
        if neutral_tables:
            combined = pd.concat([t.assign(factor=n) for n, t in neutral_tables.items()])
            combined.to_csv(RESULTS / "neutralization.csv")
            (REPORTS / "NEUTRALIZATION.txt").write_text(
                "\n\n".join(
                    f"### {n.upper()}\n{render_neutrality_report(t)}"
                    for n, t in neutral_tables.items()
                )
            )

        # Retrieval ablation: measure whether hybrid actually beats BM25 here.
        banner("4b. RETRIEVAL ABLATION")
        with telemetry.stage("retrieval_ablation"):
            from finrisk.evaluation.rag_eval import (
                build_fact_probes,
                build_paraphrase_probes,
                render_retrieval_report,
                run_retrieval_ablation,
            )

            probes = build_fact_probes(bundle, 12) + build_paraphrase_probes(bundle, 12)
            retrieval = run_retrieval_ablation(retriever, probes, k=3)
        print(f"probes: {len(probes)} (gold labels derived from the generator)")
        print(render_retrieval_report(retrieval, retriever.encoder_name))
        retrieval.to_csv(RESULTS / "retrieval_ablation.csv")
        (REPORTS / "RETRIEVAL_ABLATION.txt").write_text(
            render_retrieval_report(retrieval, retriever.encoder_name)
        )

    # ---------------------------------------------------------------- backtests
    banner("5. STRATEGY COMPARISON")
    with telemetry.stage("benchmark_suite"):
        suite, suite_results = run_benchmark_suite(bundle, settings, return_results=True)
    save_run(suite, RESULTS)
    df = suite.frame()
    print(
        df[
            [
                "strategy",
                "cagr",
                "ann_volatility",
                "sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
                "cvar_95",
                "beta",
                "avg_turnover",
                "total_cost_drag",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:,.4f}")
    )

    banner("6. ABLATION STUDY")
    with telemetry.stage("ablation"):
        ablation = run_ablation(bundle, settings)
    save_run(ablation, RESULTS)
    print(render_ablation(ablation))

    runs = {"benchmark_suite": suite, "ablation": ablation}

    if args.full:
        banner("7. ROBUSTNESS")
        with telemetry.stage("robustness"):
            robustness = run_robustness(bundle, settings, cache=ContextCache(bundle))
        save_run(robustness, RESULTS)
        print(render_robustness(robustness))
        runs["robustness"] = robustness

    # ---------------------------------------------------------------- leakage
    banner("8. DATA LEAKAGE AUDIT")
    with telemetry.stage("leakage_audit"):
        audit = run_full_audit(bundle, settings, include_placebo=not args.quick)
    text = audit.render()
    print(text)
    (REPORTS / "LEAKAGE_AUDIT.txt").write_text(text)
    (RESULTS / "leakage_audit.json").write_text(
        json.dumps(
            [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "detail": c.detail,
                    "evidence": c.evidence,
                }
                for c in audit.checks
            ],
            indent=2,
            default=str,
        )
    )

    # ---------------------------------------------------------------- agent
    if not args.no_agent:
        banner("9. AGENTIC RESEARCH")
        with telemetry.stage("agent_graph"):
            registry = ToolRegistry(ToolContext(bundle, settings, retriever))
            report, state = run_research(
                registry,
                "Why these positions, and what risks are being taken?",
                as_of=str(bundle.dates[-1].date()),
            )
        lines = ["Trace:", *[f"  {t}" for t in state.get("trace", [])], ""]
        if report.portfolio:
            lines.append(
                f"Portfolio: {report.portfolio.method}, "
                f"{len(report.portfolio.weights)} positions, "
                f"ex-ante vol {report.portfolio.expected_volatility:.2%}"
            )
            lines += [
                f"  {t:<8}{w:>7.2%}"
                for t, w in sorted(report.portfolio.weights.items(), key=lambda kv: -kv[1])
            ]
        if report.research:
            lines += ["", f"Thesis: {report.research.thesis}", "", "Supporting:"]
            lines += [f"  - {e}" for e in report.research.supporting_evidence]
            lines += ["Contradicting:"]
            lines += [f"  - {e}" for e in report.research.contradicting_evidence]
            lines += ["Risk factors:"]
            lines += [f"  - {r}" for r in report.research.risk_factors]
            lines += [
                "",
                "Grounded metrics: "
                + json.dumps(report.research.quantitative_evidence, default=str),
                f"Citations: {len(report.research.sources)}",
            ]
        if report.committee:
            lines += ["", "Investment committee:"]
            lines += [
                f"  {o.role:<6}{o.stance:<12}{o.argument[:110]}" for o in report.committee.opinions
            ]
            lines += [f"  -> action: {report.committee.recommended_action}"]
        if report.validation:
            lines += ["", render_validation(report.validation)]
        text = "\n".join(lines)
        print(text)
        (REPORTS / "AGENT_REPORT.txt").write_text(text)
        (RESULTS / "agent_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, default=str)
        )
        telemetry.record_tools(registry.stats())

    # ---------------------------------------------------------------- output
    banner("10. ARTEFACTS")
    paths = save_combined(list(runs.values()), RESULTS)
    report_path = write_report(runs, REPORTS)
    note = statistical_note(suite_results, "benchmark_buy_hold")
    print(note)
    (REPORTS / "STATISTICAL_NOTE.txt").write_text(note)
    telemetry.save(RESULTS / "observability.json")
    print(telemetry.render())
    print()
    for p in (
        paths["json"],
        paths["csv"],
        report_path,
        REPORTS / "LEAKAGE_AUDIT.txt",
        RESULTS / "observability.json",
    ):
        print(f"  wrote {p.relative_to(ROOT)}")

    if not audit.passed:
        print("\nLEAKAGE AUDIT FAILED — results are not trustworthy.")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
