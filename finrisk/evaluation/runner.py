"""Reproducible evaluation framework (Phases 14, 23-25).

Three studies, all sharing one dataset, one calendar, one cost model and one
precomputed signal context — so a difference between rows is a difference
between the things being compared and nothing else.

* `run_benchmark_suite` — every strategy on identical assumptions (Phase 14/23).
* `run_ablation`        — the A-F component ladder (Phase 24).
* `run_robustness`      — sensitivity to costs, rebalance frequency, factor
                          weights, constraints, and sub-period (Phase 25).

Outputs land in `evaluation/results/` as `results.json` + `results.csv`, and in
`evaluation/reports/REPORT.md`. Every row carries the config digest that
produced it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finrisk.backtesting.engine import (
    WalkForwardBacktester,
    render_comparison,
)
from finrisk.backtesting.strategies import ResearchContext, all_strategies
from finrisk.config.settings import Settings
from finrisk.data.base import MarketDataBundle
from finrisk.schemas import BacktestResult

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "results"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "reports"


@dataclass
class EvaluationRun:
    name: str
    started_at: str
    dataset: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _dataset_meta(bundle: MarketDataBundle, settings: Settings) -> dict[str, Any]:
    return {
        "provider": bundle.provider,
        "n_assets": len(bundle.tickers),
        "start": str(bundle.dates.min().date()),
        "end": str(bundle.dates.max().date()),
        "n_sessions": len(bundle.dates),
        "benchmark": settings.data.benchmark,
        "synthetic_seed": settings.data.synthetic_seed,
        "notes": list(bundle.notes),
    }


def _row(res: BacktestResult, label: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    mt = res.metrics
    row = {
        "label": label,
        "strategy": res.strategy,
        "config_digest": res.config_digest,
        "start": str(mt.start),
        "end": str(mt.end),
        "n_days": mt.n_days,
        "n_rebalances": res.n_rebalances,
        "total_return": mt.total_return,
        "cagr": mt.cagr,
        "ann_return": mt.ann_return,
        "ann_volatility": mt.ann_volatility,
        "sharpe": mt.sharpe,
        "sortino": mt.sortino,
        "max_drawdown": mt.max_drawdown,
        "calmar": mt.calmar,
        "var_95": mt.var_95,
        "cvar_95": mt.cvar_95,
        "hit_rate": mt.hit_rate,
        "avg_turnover": mt.avg_turnover,
        "total_cost_drag": mt.total_cost_drag,
        "beta": mt.beta,
        "information_ratio": mt.information_ratio,
        "warnings": "; ".join(res.warnings),
    }
    if extra:
        row.update(extra)
    return row


# ======================================================================
# Studies
# ======================================================================


def run_benchmark_suite(
    bundle: MarketDataBundle,
    settings: Settings,
    strategies: list[str] | None = None,
    return_results: bool = False,
) -> EvaluationRun | tuple[EvaluationRun, list[BacktestResult]]:
    """All strategies, identical assumptions (Phases 14 and 23).

    With `return_results=True` the raw `BacktestResult` objects come back too,
    so significance testing can use the full return series rather than the
    summarised row.
    """
    t0 = time.perf_counter()
    run = EvaluationRun(
        name="benchmark_suite",
        started_at=pd.Timestamp.utcnow().isoformat(),
        dataset=_dataset_meta(bundle, settings),
    )
    bt = WalkForwardBacktester(settings)
    ctx = bt.prepare_context(bundle)
    run.notes.append(f"{len(ctx.dates)} rebalance dates; shared signal context")

    available = all_strategies()
    names = strategies or list(available)
    results: list[BacktestResult] = []
    for name in names:
        if name not in available:
            run.notes.append(f"skipped unknown strategy {name}")
            continue
        res = bt.run(bundle, available[name], ctx)
        results.append(res)
        run.rows.append(_row(res, label=name))
        logger.info("%s: CAGR %.2f%% Sharpe %.2f", name, res.metrics.cagr * 100, res.metrics.sharpe)
    run.elapsed_seconds = time.perf_counter() - t0
    return (run, results) if return_results else run


def run_ablation(bundle: MarketDataBundle, settings: Settings) -> EvaluationRun:
    """The A-F ladder, plus the incremental delta of each component."""
    ladder = [
        ("A_equal_weight_universe", "A: no signal, no optimisation"),
        ("B_factor_equal_weight", "B: + factor selection"),
        ("C_factor_regime_equal_weight", "C: + regime overlay"),
        ("D_factor_mean_variance", "D: + mean-variance optimisation"),
        ("D2_factor_risk_parity", "D2: + risk-parity optimisation"),
        ("E_factor_optimized_risk_constrained", "E: + volatility constraint"),
        ("F_finrisk_full", "F: full pipeline"),
    ]
    t0 = time.perf_counter()
    run = EvaluationRun(
        name="ablation",
        started_at=pd.Timestamp.utcnow().isoformat(),
        dataset=_dataset_meta(bundle, settings),
    )
    bt = WalkForwardBacktester(settings)
    ctx = bt.prepare_context(bundle)
    strategies = all_strategies()

    prev: dict[str, float] | None = None
    for name, desc in ladder:
        res = bt.run(bundle, strategies[name], ctx)
        mt = res.metrics
        extra: dict[str, Any] = {"description": desc}
        if prev is not None:
            extra.update(
                {
                    "delta_cagr": mt.cagr - prev["cagr"],
                    "delta_sharpe": mt.sharpe - prev["sharpe"],
                    "delta_vol": mt.ann_volatility - prev["vol"],
                    "delta_maxdd": mt.max_drawdown - prev["maxdd"],
                    "delta_cvar": mt.cvar_95 - prev["cvar"],
                    "delta_turnover": mt.avg_turnover - prev["turnover"],
                }
            )
        run.rows.append(_row(res, label=name, extra=extra))
        prev = {
            "cagr": mt.cagr,
            "sharpe": mt.sharpe,
            "vol": mt.ann_volatility,
            "maxdd": mt.max_drawdown,
            "cvar": mt.cvar_95,
            "turnover": mt.avg_turnover,
        }
    run.notes.append(
        "Deltas are versus the PREVIOUS rung, so each row isolates one component. "
        "D2 is an alternative to D, not a further step; its delta is vs D."
    )
    run.elapsed_seconds = time.perf_counter() - t0
    return run


class ContextCache:
    """Reuse precomputed signal contexts across sweeps.

    Building a context (factor panels + HMM refits over ~156 rebalance dates)
    dominates run time. It depends ONLY on the factor config, the regime config,
    the rebalance calendar and the data — not on costs, constraints or the
    optimiser. So a cost sweep and a constraint sweep can share one context,
    which cuts the robustness study from ~20 rebuilds to 10 and, more
    importantly, guarantees those rows differ *only* in the swept parameter.
    """

    def __init__(self, bundle: MarketDataBundle) -> None:
        self.bundle = bundle
        self._cache: dict[str, ResearchContext] = {}
        self.builds = 0
        self.hits = 0

    @staticmethod
    def key(settings: Settings) -> str:
        import hashlib

        payload = json.dumps(
            {
                "factors": settings.factors.model_dump(mode="json"),
                "regimes": settings.regimes.model_dump(mode="json"),
                "freq": settings.backtest.rebalance_frequency,
                "warmup": settings.backtest.warmup,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, settings: Settings) -> ResearchContext:
        k = self.key(settings)
        if k in self._cache:
            self.hits += 1
            return self._cache[k]
        ctx = WalkForwardBacktester(settings).prepare_context(self.bundle)
        self._cache[k] = ctx
        self.builds += 1
        return ctx


def run_robustness(
    bundle: MarketDataBundle,
    settings: Settings,
    strategy: str = "F_finrisk_full",
    cache: ContextCache | None = None,
) -> EvaluationRun:
    """Sensitivity sweeps. A result that survives only one parameter set is not
    a result."""
    t0 = time.perf_counter()
    run = EvaluationRun(
        name="robustness",
        started_at=pd.Timestamp.utcnow().isoformat(),
        dataset=_dataset_meta(bundle, settings),
    )
    strat = all_strategies()[strategy]
    cache = cache or ContextCache(bundle)

    def _go(s: Settings, label: str, sweep: str, value: Any) -> None:
        bt = WalkForwardBacktester(s)
        res = bt.run(bundle, strat, cache.get(s))
        run.rows.append(_row(res, label=label, extra={"sweep": sweep, "value": value}))

    # --- transaction costs (context-invariant) ------------------------
    for bps in (0.0, 5.0, 10.0, 20.0, 40.0):
        s = settings.model_copy(deep=True)
        s.backtest.costs.commission_bps = bps / 2
        s.backtest.costs.spread_bps = bps / 2
        s.backtest.costs.slippage_bps = 0.0
        _go(s, f"cost_{bps:g}bps", "transaction_cost", bps)

    # --- rebalance frequency (changes the calendar -> new context) ----
    for freq in ("W", "M", "Q"):
        s = settings.model_copy(deep=True)
        s.backtest.rebalance_frequency = freq
        _go(s, f"rebalance_{freq}", "rebalance_frequency", freq)

    # --- factor composite weights (new context each) ------------------
    schemes = {
        "baseline_30_25_25_20": {"momentum": 0.30, "value": 0.25, "quality": 0.25, "risk": 0.20},
        "equal_all": {"momentum": 0.25, "value": 0.25, "quality": 0.25, "risk": 0.25},
        "momentum_tilt": {"momentum": 0.55, "value": 0.15, "quality": 0.15, "risk": 0.15},
        "value_tilt": {"momentum": 0.15, "value": 0.55, "quality": 0.15, "risk": 0.15},
        "quality_tilt": {"momentum": 0.15, "value": 0.15, "quality": 0.55, "risk": 0.15},
        "defensive": {"momentum": 0.15, "value": 0.20, "quality": 0.25, "risk": 0.40},
    }
    for name, w in schemes.items():
        s = settings.model_copy(deep=True)
        s.factors.composite_weights = w
        _go(s, f"weights_{name}", "factor_weights", json.dumps(w))

    # --- portfolio constraints (context-invariant) --------------------
    for top_n, max_w, vol in (
        (5, 0.35, 0.15),
        (10, 0.20, 0.15),
        (20, 0.12, 0.15),
        (10, 0.20, 0.10),
        (10, 0.20, 0.25),
    ):
        s = settings.model_copy(deep=True)
        s.portfolio.top_n = top_n
        s.portfolio.max_weight = max_w
        s.portfolio.target_volatility = vol
        _go(
            s,
            f"cons_n{top_n}_w{max_w:g}_v{vol:g}",
            "constraints",
            f"top_n={top_n},max_w={max_w},vol={vol}",
        )

    # --- sub-periods (different data -> own contexts) -----------------
    for label, sub in _subperiods(bundle, settings).items():
        try:
            bt = WalkForwardBacktester(settings)
            res = bt.run(sub, strat)
            run.rows.append(
                _row(res, label=f"period_{label}", extra={"sweep": "subperiod", "value": label})
            )
        except Exception as exc:
            run.notes.append(f"subperiod {label} skipped: {exc}")

    run.notes.append(f"signal contexts: {cache.builds} built, {cache.hits} reused")
    run.elapsed_seconds = time.perf_counter() - t0
    return run


def _subperiods(bundle: MarketDataBundle, settings: Settings) -> dict[str, MarketDataBundle]:
    """Split the sample by realised benchmark regime.

    Splits are defined on the *benchmark's own realised* return and volatility
    over each calendar year, which is ex-post by construction. That is
    legitimate here because these are descriptive conditioning labels for
    reporting where the strategy struggles — they are never fed to the strategy.
    """
    bench = bundle.benchmark
    yearly = bench.resample("YE").last().pct_change().dropna()
    vol = bench.pct_change().resample("YE").std() * np.sqrt(252)
    out: dict[str, MarketDataBundle] = {}
    warm = settings.backtest.warmup

    bull_years = [y.year for y, v in yearly.items() if v > 0.05]
    bear_years = [y.year for y, v in yearly.items() if v < -0.05]
    hi_vol_years = [y.year for y, v in vol.items() if v > float(vol.median())]

    for label, years in (("bull", bull_years), ("bear", bear_years), ("high_vol", hi_vol_years)):
        if not years:
            continue
        # Contiguous span covering those years, plus warm-up before it.
        lo, hi = min(years), max(years)
        start_idx = max(bundle.dates.searchsorted(pd.Timestamp(f"{lo}-01-01")) - warm, 0)
        start = bundle.dates[start_idx]
        end = pd.Timestamp(f"{hi}-12-31")
        sub = bundle.slice(start, end)
        if len(sub.dates) > warm + 260:
            out[label] = sub

    # Chronological halves — the cleanest out-of-sample split available.
    mid = bundle.dates[len(bundle.dates) // 2]
    first = bundle.slice(bundle.dates[0], mid)
    second_start = bundle.dates[max(len(bundle.dates) // 2 - warm, 0)]
    second = bundle.slice(second_start, bundle.dates[-1])
    if len(first.dates) > warm + 260:
        out["first_half"] = first
    if len(second.dates) > warm + 260:
        out["second_half"] = second
    return out


# ======================================================================
# Persistence and reporting
# ======================================================================


def save_run(run: EvaluationRun, results_dir: Path | None = None) -> dict[str, Path]:
    d = results_dir or RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    jp = d / f"{run.name}.json"
    cp = d / f"{run.name}.csv"
    jp.write_text(json.dumps(asdict(run), indent=2, default=str))
    run.frame().to_csv(cp, index=False)
    return {"json": jp, "csv": cp}


def save_combined(runs: list[EvaluationRun], results_dir: Path | None = None) -> dict[str, Path]:
    d = results_dir or RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    combined = {r.name: asdict(r) for r in runs}
    jp = d / "results.json"
    jp.write_text(json.dumps(combined, indent=2, default=str))
    frames = []
    for r in runs:
        f = r.frame()
        if len(f):
            f = f.assign(study=r.name)
            frames.append(f)
    cp = d / "results.csv"
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(cp, index=False)
    return {"json": jp, "csv": cp}


def render_ablation(run: EvaluationRun) -> str:
    df = run.frame()
    if df.empty:
        return "Ablation produced no rows."
    lines = ["Ablation Study", "=" * 112]
    lines.append(
        f"{'Rung':<38}{'CAGR':>8}{'dCAGR':>8}{'Vol':>8}{'Sharpe':>8}{'dShrp':>8}"
        f"{'MaxDD':>9}{'CVaR95':>9}{'Turn':>8}"
    )
    lines.append("-" * 112)
    for _, r in df.iterrows():
        dc = (
            f"{r.get('delta_cagr', float('nan')):>7.2%}"
            if pd.notna(r.get("delta_cagr"))
            else "      —"
        )
        ds = (
            f"{r.get('delta_sharpe', float('nan')):>7.2f}"
            if pd.notna(r.get("delta_sharpe"))
            else "      —"
        )
        lines.append(
            f"{r['label']:<38}{r['cagr']:>7.2%}{dc}{r['ann_volatility']:>8.1%}"
            f"{r['sharpe']:>8.2f}{ds}{r['max_drawdown']:>9.1%}"
            f"{r['cvar_95']:>9.2%}{r['avg_turnover']:>8.1%}"
        )
    lines.append("-" * 112)
    lines += ["", *run.notes]
    return "\n".join(lines)


def render_robustness(run: EvaluationRun) -> str:
    df = run.frame()
    if df.empty:
        return "Robustness produced no rows."
    lines = ["Robustness / Sensitivity Analysis", "=" * 100]
    for sweep, grp in df.groupby("sweep", sort=False):
        lines += ["", f"[{sweep}]", "-" * 100]
        lines.append(
            f"{'Setting':<40}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}"
            f"{'Turn':>8}{'Cost':>9}"
        )
        for _, r in grp.iterrows():
            lines.append(
                f"{r['label']!s:<40}{r['cagr']:>8.2%}{r['ann_volatility']:>8.1%}"
                f"{r['sharpe']:>8.2f}{r['max_drawdown']:>9.1%}"
                f"{r['avg_turnover']:>8.1%}{r['total_cost_drag']:>9.2%}"
            )
        sh = grp["sharpe"]
        lines.append(
            f"{'-> Sharpe range':<40}{sh.min():>8.2f} to {sh.max():.2f} "
            f"(spread {sh.max() - sh.min():.2f})"
        )
    lines += ["", *run.notes]
    return "\n".join(lines)


def write_report(runs: dict[str, EvaluationRun], reports_dir: Path | None = None) -> Path:
    d = reports_dir or REPORTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    parts: list[str] = ["# FINRISK AI — Evaluation Report", ""]

    any_run = next(iter(runs.values()))
    ds = any_run.dataset
    parts += [
        "## Dataset",
        "",
        f"- Provider: `{ds['provider']}`",
        f"- Assets: {ds['n_assets']}",
        f"- Period: {ds['start']} to {ds['end']} ({ds['n_sessions']} sessions)",
        f"- Benchmark: `{ds['benchmark']}`",
        "",
    ]
    if ds.get("notes"):
        parts += ["**Data notes:**", ""]
        parts += [f"- {n}" for n in ds["notes"]]
        parts.append("")

    if "benchmark_suite" in runs:
        parts += [
            "## Strategy comparison",
            "",
            "```",
            render_comparison(_suite_frame(runs["benchmark_suite"])),
            "```",
            "",
        ]
    if "ablation" in runs:
        parts += ["## Ablation study", "", "```", render_ablation(runs["ablation"]), "```", ""]
    if "robustness" in runs:
        parts += [
            "## Robustness",
            "",
            "```",
            render_robustness(runs["robustness"]),
            "```",
            "",
        ]

    parts += [
        "## Reproduction",
        "",
        "```bash",
        "python scripts/run_all.py --full",
        "```",
        "",
        "Every row carries the `config_digest` of the settings that produced it.",
        "",
    ]
    p = d / "REPORT.md"
    p.write_text("\n".join(parts))
    return p


def _suite_frame(run: EvaluationRun) -> pd.DataFrame:
    df = run.frame()
    ren = {
        "cagr": "CAGR",
        "ann_return": "AnnRet",
        "ann_volatility": "AnnVol",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "max_drawdown": "MaxDD",
        "calmar": "Calmar",
        "var_95": "VaR95",
        "cvar_95": "CVaR95",
        "beta": "Beta",
        "information_ratio": "InfoRatio",
        "avg_turnover": "AvgTurnover",
        "total_cost_drag": "CostDrag",
        "n_rebalances": "Rebalances",
    }
    return df.rename(columns=ren).set_index("strategy")[list(ren.values())]


def statistical_note(results: list[BacktestResult], benchmark_name: str) -> str:
    """Honest significance framing.

    A Sharpe difference over ~13 years of monthly rebalancing is a *very* noisy
    statistic. The standard error of an annualised Sharpe is roughly
    sqrt((1 + S^2/2)/T) with T in years, so anything under ~0.3 of difference is
    not distinguishable from noise at this sample size. This function states
    that explicitly rather than letting a table imply significance.
    """
    bench = next((r for r in results if r.strategy == benchmark_name), None)
    if bench is None:
        return "Benchmark not present; no comparison made."
    lines = [
        f"Benchmark: {benchmark_name} (Sharpe {bench.metrics.sharpe:.2f}, "
        f"CAGR {bench.metrics.cagr:.2%})",
        "",
    ]
    for r in results:
        if r.strategy == benchmark_name:
            continue
        years = max(r.metrics.n_days / 252.0, 1e-9)
        s = r.metrics.sharpe
        se = float(np.sqrt((1.0 + 0.5 * s**2) / years))
        diff = s - bench.metrics.sharpe
        t_stat = diff / max(se * np.sqrt(2.0), 1e-12)
        verdict = (
            "not distinguishable from the benchmark"
            if abs(t_stat) < 1.96
            else "difference exceeds two standard errors"
        )
        lines.append(
            f"{r.strategy:<38} Sharpe {s:>5.2f}  diff {diff:>+5.2f}  "
            f"approx t={t_stat:>+5.2f}  -> {verdict}"
        )
    lines += [
        "",
        "Standard error of an annualised Sharpe approximated as "
        "sqrt((1 + S^2/2)/T) with T in years; the pairwise comparison assumes "
        "independence, which overstates precision because both series share "
        "market exposure. Treat these t-statistics as an upper bound on "
        "confidence, not a formal test.",
    ]
    return "\n".join(lines)
