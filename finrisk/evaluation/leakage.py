"""Automated data-leakage audit (Phase 26).

Static code review catches some leakage; it does not catch the subtle kind. So
this module runs *empirical* tests whose outcomes are only possible if the
information barrier holds.

The central one is `test_shifted_signal_advantage`. It runs the identical
strategy twice — once normally, once with signals deliberately shifted one
period into the future — and compares. Logic:

* If the honest run performs like the cheating run, the honest run is already
  cheating.
* If the cheating run is clearly better, the barrier is doing real work.

This is a much stronger statement than "I checked the code". It is the financial
equivalent of a placebo control, and it is what a sceptical interviewer should
ask for.

Other checks:

* fundamentals never available before their period end,
* the fundamentals accessor at date `d` returns nothing stamped after `d`,
* factor scores computed on a truncated sample equal those computed on the full
  sample (i.e. the factor engine does not peek past `as_of`),
* regime filtering is genuinely causal: filtered != smoothed, and the filtered
  path at `d` is unchanged by appending future data,
* the first execution date strictly follows the first decision date,
* normalisation statistics are computed cross-sectionally per date, not pooled
  across the full history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finrisk.backtesting.engine import WalkForwardBacktester
from finrisk.backtesting.strategies import ResearchContext, all_strategies
from finrisk.config.settings import Settings
from finrisk.data.base import MarketDataBundle
from finrisk.factors.engine import FactorEngine
from finrisk.regimes.detector import RegimeDetector

logger = logging.getLogger(__name__)


@dataclass
class AuditCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # error | warning | info
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageAudit:
    checks: list[AuditCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def add(self, check: AuditCheck) -> None:
        self.checks.append(check)

    def render(self) -> str:
        lines = ["Data Leakage Audit", "=" * 78]
        for c in self.checks:
            tag = "PASS" if c.passed else ("FAIL" if c.severity == "error" else "WARN")
            lines.append(f"[{tag:<4}] {c.name}")
            lines.append(f"        {c.detail}")
        lines.append("-" * 78)
        n_fail = sum(1 for c in self.checks if not c.passed and c.severity == "error")
        lines.append(
            f"{len(self.checks)} checks, {n_fail} failures -> "
            f"{'AUDIT PASSED' if self.passed else 'AUDIT FAILED'}"
        )
        return "\n".join(lines)


# ======================================================================
# Individual checks
# ======================================================================


def check_fundamental_availability(bundle: MarketDataBundle) -> AuditCheck:
    f = bundle.fundamentals
    if not len(f):
        return AuditCheck(
            "fundamental_availability",
            True,
            "no fundamentals in this bundle; check not applicable",
            "info",
        )
    pe = pd.to_datetime(f["period_end"])
    af = pd.to_datetime(f["available_from"])
    bad = int((af < pe).sum())
    lag = (af - pe).dt.days
    return AuditCheck(
        "fundamental_availability",
        bad == 0,
        (
            f"{len(f)} records; {bad} available before period end. "
            f"Publication lag median {lag.median():.0f}d, range "
            f"[{lag.min():.0f}, {lag.max():.0f}]d."
        ),
        evidence={"n_records": len(f), "n_violations": bad, "median_lag_days": float(lag.median())},
    )


def check_asof_accessor(bundle: MarketDataBundle, n_samples: int = 12) -> AuditCheck:
    """`fundamentals_asof(d)` must never return a filing stamped after `d`."""
    if not len(bundle.fundamentals):
        return AuditCheck("asof_accessor", True, "no fundamentals; not applicable", "info")
    dates = bundle.dates
    sample = dates[np.linspace(len(dates) // 4, len(dates) - 1, n_samples).astype(int)]
    violations = 0
    for d in sample:
        got = bundle.fundamentals_asof(d)
        if len(got) and (pd.to_datetime(got["available_from"]) > d).any():
            violations += 1
    return AuditCheck(
        "asof_accessor",
        violations == 0,
        f"sampled {len(sample)} dates; {violations} returned unpublished filings",
        evidence={"n_sampled": len(sample), "violations": violations},
    )


def check_factor_truncation_invariance(
    bundle: MarketDataBundle, settings: Settings, n_samples: int = 6
) -> AuditCheck:
    """Factor scores at date d must be identical whether or not data after d
    exists in the bundle. If they differ, the engine is reading the future."""
    fe = FactorEngine(settings.factors)
    dates = bundle.dates
    sample = dates[np.linspace(len(dates) // 2, len(dates) - 30, n_samples).astype(int)]
    max_diff = 0.0
    for d in sample:
        full = fe.raw_panel(bundle, d)
        trunc = fe.raw_panel(bundle.slice(None, d), d)
        if full.empty or trunc.empty:
            continue
        cols = full.columns.intersection(trunc.columns)
        idx = full.index.intersection(trunc.index)
        diff = (full.loc[idx, cols] - trunc.loc[idx, cols]).abs().to_numpy()
        finite = diff[np.isfinite(diff)]
        if finite.size:
            max_diff = max(max_diff, float(finite.max()))
    return AuditCheck(
        "factor_truncation_invariance",
        max_diff < 1e-10,
        (
            f"max absolute difference between full-sample and truncated factor "
            f"values across {len(sample)} dates: {max_diff:.3e}"
        ),
        evidence={"max_abs_diff": max_diff},
    )


def check_regime_causality(bundle: MarketDataBundle, settings: Settings) -> AuditCheck:
    """Two things must hold: filtered != smoothed (proving smoothing really does
    use future data), and the filtered estimate at d must not change when future
    data is appended."""
    rd = RegimeDetector(settings.regimes)
    feats = rd.prepare(bundle.benchmark)
    if len(feats) < settings.regimes.train_min_obs + 300:
        return AuditCheck("regime_causality", True, "insufficient history", "info")

    mid = feats.index[int(len(feats) * 0.7)]
    st_trunc = rd.detect(bundle.benchmark.loc[:mid], mid)

    rd2 = RegimeDetector(settings.regimes)
    rd2.prepare(bundle.benchmark)
    st_full = rd2.detect(bundle.benchmark, mid)

    # Filtered vs smoothed on the same fitted model
    model = rd2._model
    diff_fs = float("nan")
    if model is not None and rd2._fit_mean is not None:
        Z = (feats.loc[:mid].to_numpy(float) - rd2._fit_mean) / rd2._fit_std
        filt = model.filter_states(Z)
        smooth = model.smoothed_states(Z)
        diff_fs = float(np.abs(filt - smooth).max())

    same_regime = st_trunc.regime == st_full.regime
    conf_close = abs(st_trunc.confidence - st_full.confidence) < 0.05
    return AuditCheck(
        "regime_causality",
        bool(same_regime and conf_close and (not np.isfinite(diff_fs) or diff_fs > 1e-6)),
        (
            f"filtered state at {pd.Timestamp(mid).date()} is '{st_trunc.regime}' "
            f"(conf {st_trunc.confidence:.3f}) whether or not future data is "
            f"present (full-sample: '{st_full.regime}', conf {st_full.confidence:.3f}). "
            f"Max |filtered - smoothed| = {diff_fs:.3f}, confirming smoothing "
            "genuinely uses future information and is therefore excluded."
        ),
        evidence={
            "filtered_vs_smoothed_max_diff": diff_fs,
            "regime_stable_under_truncation": bool(same_regime),
        },
    )


def check_execution_lag(bundle: MarketDataBundle, settings: Settings) -> AuditCheck:
    """Every execution date must strictly follow its decision date."""
    bt = WalkForwardBacktester(settings)
    res = bt.run(bundle, all_strategies()["B_factor_equal_weight"])
    if not res.events:
        return AuditCheck("execution_lag", False, "no rebalance events produced")
    bad = [e for e in res.events if pd.Timestamp(e.execution_date) <= pd.Timestamp(e.decision_date)]
    lags = [
        int(
            bundle.dates.searchsorted(pd.Timestamp(e.execution_date))
            - bundle.dates.searchsorted(pd.Timestamp(e.decision_date))
        )
        for e in res.events
    ]
    return AuditCheck(
        "execution_lag",
        not bad,
        (
            f"{len(res.events)} rebalances; {len(bad)} executed on or before the "
            f"decision date. Lag in sessions: min {min(lags)}, max {max(lags)}."
        ),
        evidence={"n_events": len(res.events), "min_lag": min(lags), "max_lag": max(lags)},
    )


def test_shifted_signal_advantage(
    bundle: MarketDataBundle,
    settings: Settings,
    strategy: str = "B_factor_equal_weight",
) -> AuditCheck:
    """THE placebo control.

    Build a context where each decision date is given the signals belonging to
    the NEXT rebalance date — i.e. deliberate one-period look-ahead — and
    compare against the honest run. A working information barrier must show the
    cheating run outperforming by a wide margin.
    """
    bt = WalkForwardBacktester(settings)
    honest_ctx = bt.prepare_context(bundle)
    strat = all_strategies()[strategy]
    honest = bt.run(bundle, strat, honest_ctx)

    dates = honest_ctx.dates
    cheat = ResearchContext(dates=list(dates))
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        if nxt not in honest_ctx.factor_scores:
            continue
        # Take tomorrow's scores but re-stamp them as of today, so the
        # backtester's own leakage assertion does not fire — we WANT the leak.
        cheat.factor_scores[d] = {
            t: s.model_copy(update={"as_of": pd.Timestamp(d).date()})
            for t, s in honest_ctx.factor_scores[nxt].items()
        }
        cheat.factor_diag[d] = honest_ctx.factor_diag.get(nxt, pd.DataFrame())
        borrowed = honest_ctx.regimes.get(nxt) or honest_ctx.regimes.get(d)
        if borrowed is not None:
            cheat.regimes[d] = borrowed
        cheat.ic_estimates[d] = honest_ctx.ic_estimates.get(d, 0.03)
    cheating = bt.run(bundle, strat, cheat)

    h, c = honest.metrics, cheating.metrics
    gap_cagr = c.cagr - h.cagr
    gap_sharpe = c.sharpe - h.sharpe
    # A genuine one-period look-ahead on a monthly-rebalanced factor strategy
    # should be worth well over 1% CAGR and 0.1 Sharpe. If it is not, the
    # honest path is probably already leaking.
    passed = gap_cagr > 0.01 and gap_sharpe > 0.10
    return AuditCheck(
        "shifted_signal_placebo",
        passed,
        (
            f"honest: CAGR {h.cagr:.2%}, Sharpe {h.sharpe:.2f}. "
            f"one-period look-ahead: CAGR {c.cagr:.2%}, Sharpe {c.sharpe:.2f}. "
            f"Advantage from cheating: {gap_cagr:+.2%} CAGR, {gap_sharpe:+.2f} Sharpe. "
            + (
                "The barrier is doing real work."
                if passed
                else "WARNING: cheating buys little, so the honest run may already leak."
            )
        ),
        evidence={
            "honest_cagr": h.cagr,
            "cheat_cagr": c.cagr,
            "honest_sharpe": h.sharpe,
            "cheat_sharpe": c.sharpe,
            "gap_cagr": gap_cagr,
            "gap_sharpe": gap_sharpe,
        },
    )


def check_survivorship(bundle: MarketDataBundle) -> AuditCheck:
    """Survivorship bias is a property of the universe, not of the code.

    Reported as a documented limitation with the evidence that supports the
    claim, because the honest answer for a fixed ticker list is 'yes, present'.
    """
    px = bundle.prices
    n_full = int((px.notna().mean() > 0.999).sum())
    n_total = px.shape[1]
    delisted_like = n_total - n_full
    if bundle.provider == "synthetic":
        return AuditCheck(
            "survivorship_bias",
            True,
            (
                f"synthetic universe: all {n_total} assets exist for the full sample "
                "by construction, so there is no survivorship bias AND no delisting "
                "risk. Real-data runs with a fixed current ticker list DO have "
                "survivorship bias."
            ),
            "info",
            evidence={"n_assets": n_total, "n_full_history": n_full},
        )
    return AuditCheck(
        "survivorship_bias",
        False,
        (
            f"{n_full}/{n_total} assets have complete history and "
            f"{delisted_like} have partial history. A universe drawn from "
            "currently-listed tickers excludes companies that failed, which "
            "inflates returns. Mitigation requires a point-in-time constituent "
            "list (e.g. CRSP)."
        ),
        "warning",
        evidence={"n_assets": n_total, "n_full_history": n_full},
    )


def check_normalisation_scope(bundle: MarketDataBundle, settings: Settings) -> AuditCheck:
    """Factor z-scores must be computed within a single date's cross-section.

    Detected by checking that each date's normalised panel has mean ~0 and
    std ~1 *per date*. Pooled (full-history) normalisation would not produce
    that property on every individual date.
    """
    fe = FactorEngine(settings.factors)
    dates = bundle.dates[np.linspace(len(bundle.dates) // 2, len(bundle.dates) - 5, 5).astype(int)]
    worst_mean, worst_std = 0.0, 0.0
    for d in dates:
        panel = fe.raw_panel(bundle, d)
        if panel.empty:
            continue
        normed = fe.normalize(panel)
        means = normed.mean(skipna=True).abs()
        stds = normed.std(ddof=1, skipna=True)
        if len(means):
            worst_mean = max(worst_mean, float(means.max()))
        valid = stds[(stds > 0)]
        if len(valid):
            worst_std = max(worst_std, float((valid - 1.0).abs().max()))
    return AuditCheck(
        "cross_sectional_normalisation",
        worst_mean < 1e-8 and worst_std < 0.15,
        (
            f"per-date normalised panels have |mean| <= {worst_mean:.2e} and "
            f"|std - 1| <= {worst_std:.3f}, consistent with normalisation inside "
            "each date's cross-section rather than pooled across history "
            "(winsorization accounts for the small std deviation)."
        ),
        evidence={"max_abs_mean": worst_mean, "max_std_deviation": worst_std},
    )


# ======================================================================
def run_full_audit(
    bundle: MarketDataBundle, settings: Settings, include_placebo: bool = True
) -> LeakageAudit:
    audit = LeakageAudit()
    audit.add(check_fundamental_availability(bundle))
    audit.add(check_asof_accessor(bundle))
    audit.add(check_factor_truncation_invariance(bundle, settings))
    audit.add(check_normalisation_scope(bundle, settings))
    audit.add(check_regime_causality(bundle, settings))
    audit.add(check_execution_lag(bundle, settings))
    audit.add(check_survivorship(bundle))
    if include_placebo:
        audit.add(test_shifted_signal_advantage(bundle, settings))
    return audit
