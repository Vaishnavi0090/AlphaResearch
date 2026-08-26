"""Critic / validation agent (Phases 18-20).

The enforcement point for the project's central safety claim: an LLM may
describe the quantitative output but may not invent it.

Four checks:

1. **Numeric grounding.** Every number in `quantitative_evidence` must match a
   value present in the tool results, to tolerance. A figure the model produced
   on its own is removed and recorded as an error.
2. **Citation validity.** Every citation must resolve to a chunk that actually
   exists in the index. A plausible-looking `10-K, page 42` that no document
   backs is a fabrication.
3. **Numbers in prose.** Percentages appearing in the narrative are checked
   against the grounded set. This catches the common failure where the
   structured fields are clean but the paragraph invents "outperformed by 340
   basis points".
4. **Constraint and leakage integrity.** Weights are re-checked against the
   configured limits, and any decision whose signal timestamp post-dates the
   decision date is a hard error.

Findings are `error` when they indicate fabrication or a limit breach, and
`warning` when they indicate a weaker claim than the evidence supports.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from finrisk.schemas import (
    Citation,
    CriticFinding,
    PortfolioWeights,
    ResearchConclusion,
    ValidationReport,
)

logger = logging.getLogger(__name__)

# Relative tolerance for matching a quoted number to a computed one. Generous
# enough to allow reasonable rounding in prose, tight enough that a different
# number is a different number.
REL_TOL = 0.02
ABS_TOL = 5e-4

_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def _collect_numbers(obj: Any, out: list[float] | None = None) -> list[float]:
    """Recursively harvest every numeric leaf from a tool-result structure."""
    if out is None:
        out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    return out


def _matches(value: float, pool: list[float]) -> bool:
    return any(abs(value - p) <= max(ABS_TOL, REL_TOL * abs(p)) for p in pool)


class Critic:
    """Validates agent output against the deterministic record."""

    def __init__(self, rel_tol: float = REL_TOL) -> None:
        self.rel_tol = rel_tol

    # ------------------------------------------------------------------
    def validate_research(
        self,
        conclusion: ResearchConclusion,
        tool_results: dict[str, Any],
        valid_chunk_ids: set[str] | None = None,
    ) -> tuple[ResearchConclusion, ValidationReport]:
        """Return a *scrubbed* conclusion plus the findings.

        Scrubbing rather than rejecting: a report with the ungrounded claims
        removed is more useful than no report, and the removals are recorded so
        the failure is visible.
        """
        findings: list[CriticFinding] = []
        pool = _collect_numbers(tool_results)

        # 1. numeric grounding of structured evidence
        clean_evidence: dict[str, float] = {}
        for key, val in (conclusion.quantitative_evidence or {}).items():
            if not isinstance(val, (int, float)):
                continue
            if _matches(float(val), pool):
                clean_evidence[key] = float(val)
            else:
                findings.append(
                    CriticFinding(
                        severity="error",
                        check="numeric_grounding",
                        detail=(
                            f"quantitative_evidence['{key}']={val} does not match any "
                            "value computed by the tools; removed"
                        ),
                    )
                )

        # 2. citation validity
        clean_sources: list[Citation] = []
        for c in conclusion.sources or []:
            if valid_chunk_ids is None or c.chunk_id in valid_chunk_ids:
                clean_sources.append(c)
            else:
                findings.append(
                    CriticFinding(
                        severity="error",
                        check="citation_validity",
                        detail=f"citation {c.chunk_id} ({c.source}) is not in the index; removed",
                    )
                )

        # 3. numbers embedded in prose
        prose = " ".join(
            [
                conclusion.thesis,
                *conclusion.supporting_evidence,
                *conclusion.contradicting_evidence,
                *conclusion.risk_factors,
            ]
        )
        grounded = pool + [v * 100.0 for v in pool] + list(clean_evidence.values())
        for match in _PCT_RE.finditer(prose):
            val = float(match.group(1))
            if not (_matches(val, grounded) or _matches(val / 100.0, pool)):
                findings.append(
                    CriticFinding(
                        severity="warning",
                        check="prose_numeric_grounding",
                        detail=f"'{match.group(0)}' in the narrative is not traceable to a tool result",
                    )
                )

        # 4. evidence balance — a one-sided conclusion is a red flag, not an error
        if conclusion.supporting_evidence and not conclusion.contradicting_evidence:
            findings.append(
                CriticFinding(
                    severity="warning",
                    check="evidence_balance",
                    detail="no contradicting evidence presented; conclusion may be one-sided",
                )
            )
        if conclusion.confidence > 0.8:
            findings.append(
                CriticFinding(
                    severity="warning",
                    check="overconfidence",
                    detail=(
                        f"stated confidence {conclusion.confidence:.2f} is high for a "
                        "single-sample backtest"
                    ),
                )
            )

        scrubbed = conclusion.model_copy(
            update={"quantitative_evidence": clean_evidence, "sources": clean_sources}
        )
        return scrubbed, ValidationReport(
            passed=not any(f.severity == "error" for f in findings), findings=findings
        )

    # ------------------------------------------------------------------
    def validate_portfolio(
        self,
        pw: PortfolioWeights,
        max_weight: float,
        max_leverage: float = 1.0,
        require_status_ok: bool = False,
    ) -> ValidationReport:
        """Independent re-check of the weight vector.

        Deliberately does NOT trust `pw.constraints`: the optimiser reports its
        own compliance, and a second, independent check is what makes that
        report worth anything.
        """
        findings: list[CriticFinding] = []
        weights = pw.weights or {}

        if not weights:
            findings.append(
                CriticFinding(
                    severity="warning",
                    check="empty_portfolio",
                    detail="portfolio holds no positions",
                )
            )
        for t, w in weights.items():
            if w < -1e-8:
                findings.append(
                    CriticFinding(
                        severity="error",
                        check="negative_weight",
                        detail=f"{t}={w:.6f} but the mandate is long-only",
                    )
                )
            if w > max_weight + 1e-6:
                findings.append(
                    CriticFinding(
                        severity="error",
                        check="max_weight",
                        detail=f"{t}={w:.4f} exceeds cap {max_weight:.4f}",
                    )
                )
        gross = sum(abs(w) for w in weights.values())
        if gross > max_leverage + 1e-6:
            findings.append(
                CriticFinding(
                    severity="error",
                    check="leverage",
                    detail=f"gross exposure {gross:.4f} exceeds {max_leverage:.4f}",
                )
            )
        if "fallback" in pw.optimization_status:
            findings.append(
                CriticFinding(
                    severity="error" if require_status_ok else "warning",
                    check="optimizer_status",
                    detail=f"optimiser did not converge: {pw.optimization_status}",
                )
            )
        if pw.constraints and pw.constraints.violations:
            for v in pw.constraints.violations:
                findings.append(
                    CriticFinding(severity="error", check="constraint_violation", detail=v)
                )
        return ValidationReport(
            passed=not any(f.severity == "error" for f in findings), findings=findings
        )

    # ------------------------------------------------------------------
    @staticmethod
    def validate_no_leakage(decision_date: Any, signal_dates: list[Any]) -> ValidationReport:
        """Hard check that no signal post-dates the decision."""
        import pandas as pd

        d = pd.Timestamp(decision_date)
        bad = [str(s) for s in signal_dates if pd.Timestamp(s) > d]
        findings = [
            CriticFinding(
                severity="error",
                check="look_ahead",
                detail=f"signal dated {s} used for a decision on {d.date()}",
            )
            for s in bad[:5]
        ]
        return ValidationReport(passed=not findings, findings=findings)


def render_validation(report: ValidationReport) -> str:
    if not report.findings:
        return "Validation: PASSED (no findings)"
    lines = [f"Validation: {'PASSED' if report.passed else 'FAILED'}"]
    for f in report.findings:
        lines.append(f"  [{f.severity.upper():<7}] {f.check}: {f.detail}")
    return "\n".join(lines)
