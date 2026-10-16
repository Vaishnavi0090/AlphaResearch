"""Per-position attribution: allocation, alpha, risk, diversification, confidence.

Answers, for each holding, the questions an investment committee actually asks:

* How much of the portfolio's *expected return* does this name carry, as
  opposed to how much of the portfolio it *is*?
* Does it consume more risk budget than its weight suggests?
* Is it diversifying, or is it adding risk almost one-for-one?
* How much of the worst days does it own?
* How confident are we, and on what basis?

Every number here is computed from covariance, factor scores and alpha
forecasts. The LLM never produces any of them — it narrates them, and the
critic checks its narration against these values.

Confidence is deliberately *derived* rather than asserted. An LLM asked "how
confident are you" produces a number with no referent; this one is a weighted
blend of factor coverage, cross-factor agreement, documentary support and
diversification, and `confidence_basis` records which component drove it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from finrisk.risk import metrics as m
from finrisk.schemas import Citation, PositionExplanation

logger = logging.getLogger(__name__)


def _safe(x: float, default: float = 0.0) -> float:
    return float(x) if np.isfinite(x) else default


def compute_position_attribution(
    weights: dict[str, float],
    returns: pd.DataFrame,
    factor_diag: pd.DataFrame | None = None,
    alphas: dict[str, float] | None = None,
    cov: pd.DataFrame | None = None,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Numeric attribution for every held position.

    Returns ticker -> metrics. Kept separate from prose generation so it can be
    tested numerically and reused by the API without an agent in the loop.
    """
    active = {t: float(w) for t, w in weights.items() if abs(w) > 1e-9}
    if not active:
        return {}
    tickers = [t for t in active if t in returns.columns]
    if not tickers:
        return {}

    w = np.array([active[t] for t in tickers], dtype=float)
    R = returns[tickers].dropna(how="all")
    if cov is not None:
        C = cov.reindex(index=tickers, columns=tickers).to_numpy(float)
    else:
        C = m.estimate_covariance(R, "ledoit_wolf", annualize=True).to_numpy(float)
    C = m.nearest_psd(C)

    port_vol = m.portfolio_volatility(w, C)
    mcr = m.marginal_risk_contribution(w, C)
    ccr = m.component_risk_contribution(w, C)
    pct_risk = m.percent_risk_contribution(w, C)
    standalone = np.sqrt(np.clip(np.diag(C), 0.0, None))
    corr = m.correlation_from_cov(C)

    # Tail attribution: share of the loss on the worst days.
    port_ret = m.portfolio_returns_series(pd.Series(active), R)
    tail_share: dict[str, float] = {}
    if len(port_ret) >= 20:
        cutoff = float(np.quantile(port_ret, 1.0 - confidence_level))
        bad = port_ret[port_ret <= cutoff].index
        if len(bad):
            contrib = R.loc[bad].mul(pd.Series(active)).sum()
            total = float(contrib.sum())
            if abs(total) > 1e-12:
                tail_share = {t: float(contrib.get(t, 0.0) / total) for t in tickers}

    total_alpha = 0.0
    if alphas:
        total_alpha = sum(alphas.get(t, 0.0) * active.get(t, 0.0) for t in tickers)

    out: dict[str, dict[str, float]] = {}
    for i, t in enumerate(tickers):
        alpha_i = float((alphas or {}).get(t, 0.0))
        alpha_contrib = alpha_i * w[i]
        # Average correlation to the REST of the book, weight-weighted, so a
        # tiny position does not distort the picture.
        others = [j for j in range(len(tickers)) if j != i]
        if others:
            ow = np.abs(w[others])
            denom = ow.sum()
            avg_corr = float((corr[i, others] * ow).sum()) / float(denom) if denom > 1e-12 else 0.0
        else:
            avg_corr = 0.0

        # Diversification benefit: how much of this asset's standalone risk is
        # absorbed by correlation with the rest. 1 = fully absorbed.
        div = 1.0 - (mcr[i] / standalone[i]) if standalone[i] > 1e-12 else 0.0

        out[t] = {
            "weight": float(w[i]),
            "alpha_forecast": alpha_i,
            "alpha_contribution": _safe(alpha_contrib),
            "pct_of_portfolio_alpha": _safe(
                alpha_contrib / total_alpha if abs(total_alpha) > 1e-12 else 0.0
            ),
            "pct_of_portfolio_risk": _safe(pct_risk[i]),
            "risk_vs_weight_gap": _safe(pct_risk[i] - w[i] / max(w.sum(), 1e-12)),
            "component_risk": _safe(ccr[i]),
            "marginal_risk": _safe(mcr[i]),
            "standalone_volatility": _safe(standalone[i]),
            "diversification_benefit": _safe(float(np.clip(div, -1.0, 1.0))),
            "avg_correlation_to_book": _safe(avg_corr),
            "pct_of_tail_loss": _safe(tail_share.get(t, 0.0)),
            "portfolio_volatility": _safe(port_vol),
        }
    return out


def compute_confidence(
    ticker: str,
    factor_row: dict[str, float] | None,
    attribution: dict[str, float],
    n_citations: int,
) -> tuple[float, list[str]]:
    """Derive a 0-1 confidence score and say what drove it.

    Four components, equally weighted:

    * **coverage** — fraction of factors with actual data for this name.
    * **agreement** — do the factor groups point the same way? A name that is
      cheap but deteriorating is a weaker call than one strong everywhere.
    * **evidence** — is there documentary support, capped at 3 citations.
    * **diversification** — a position that adds risk one-for-one is a
      lower-confidence holding than one the book absorbs.

    Returns the score and the human-readable basis, so a low number can be
    interrogated rather than merely displayed.
    """
    basis: list[str] = []
    row = factor_row or {}

    coverage = float(row.get("coverage", 0.0) or 0.0)
    basis.append(f"factor coverage {coverage:.0%}")

    groups = {
        k: float(v)
        for k, v in row.items()
        if k in ("momentum", "value", "quality", "risk", "market") and isinstance(v, (int, float))
    }
    if len(groups) >= 2:
        vals = np.array(list(groups.values()))
        share_positive = float((vals > 0).mean())
        agreement = abs(2.0 * share_positive - 1.0)  # 1 = unanimous
        basis.append(f"{int(share_positive * len(vals))}/{len(vals)} factor groups agree")
    else:
        agreement = 0.0
        basis.append("insufficient factor groups to assess agreement")

    evidence = min(n_citations, 3) / 3.0
    basis.append(f"{n_citations} supporting citation(s)")

    div = float(np.clip(attribution.get("diversification_benefit", 0.0), 0.0, 1.0))
    basis.append(f"diversification benefit {div:.0%}")

    score = float(np.clip(0.25 * (coverage + agreement + evidence + div), 0.0, 1.0))
    return score, basis


def build_position_explanation(
    ticker: str,
    attribution: dict[str, float],
    factor_row: dict[str, float] | None,
    regime: dict[str, Any] | None,
    evidence: list[Citation],
    contradicting: list[str] | None = None,
) -> PositionExplanation:
    """Assemble the full explanation from precomputed numbers."""
    row = factor_row or {}
    groups = {
        k: float(v)
        for k, v in row.items()
        if k in ("momentum", "value", "quality", "risk", "market") and isinstance(v, (int, float))
    }
    w = attribution.get("weight", 0.0)
    pct_risk = attribution.get("pct_of_portfolio_risk", 0.0)
    gap = attribution.get("risk_vs_weight_gap", 0.0)

    pos = [
        f"{name} score {val:+.2f} (cross-sectional z)"
        for name, val in sorted(groups.items(), key=lambda kv: -kv[1])
        if val > 0.25
    ]
    neg = [
        f"{name} score {val:+.2f} (cross-sectional z)"
        for name, val in sorted(groups.items(), key=lambda kv: kv[1])
        if val < -0.25
    ]

    risk_note = f"Contributes {pct_risk:.1%} of portfolio volatility on a {w:.1%} weight"
    if attribution.get("pct_of_tail_loss"):
        risk_note += f"; {attribution['pct_of_tail_loss']:.1%} of loss in the worst 5% of days"
    risk_note += "."
    if gap > 0.03:
        risk_note += (
            f" Risk share exceeds weight share by {gap:+.1%} — this position "
            "consumes more risk budget than its size implies."
        )
    elif gap < -0.03:
        risk_note += f" Risk share is {abs(gap):.1%} below weight share."

    div = attribution.get("diversification_benefit", 0.0)
    corr = attribution.get("avg_correlation_to_book", 0.0)
    div_note = (
        f"Standalone volatility {attribution.get('standalone_volatility', 0.0):.1%}, "
        f"marginal contribution {attribution.get('marginal_risk', 0.0):.1%}: "
        f"correlation absorbs {div:.0%} of its standalone risk "
        f"(average correlation to the rest of the book {corr:+.2f})."
    )

    contra = list(contradicting or [])
    if neg and pos:
        contra.append(
            "Signal is mixed across factor groups: "
            + ", ".join(n.split(" score")[0] for n in neg)
            + " oppose the thesis."
        )
    if gap > 0.05:
        contra.append(
            f"Risk contribution ({pct_risk:.1%}) materially exceeds allocation ({w:.1%})."
        )
    if div < 0.15:
        contra.append("Position adds risk close to one-for-one; it is not diversifying.")
    if not evidence:
        contra.append("No supporting document evidence was retrieved for this name.")

    confidence, basis = compute_confidence(ticker, row, attribution, len(evidence))

    return PositionExplanation(
        ticker=ticker,
        weight=w,
        drivers_positive=pos,
        drivers_negative=neg,
        risk_note=risk_note + " " + div_note,
        regime_note=(
            f"Regime {(regime or {}).get('regime', 'unknown')} "
            f"(confidence {float((regime or {}).get('confidence', 0.0)):.2f}), "
            f"exposure multiplier {float((regime or {}).get('risk_multiplier', 1.0)):.2f}"
        ),
        factor_contributions=groups,
        pct_of_portfolio_risk=pct_risk,
        evidence=evidence,
        alpha_contribution=attribution.get("alpha_contribution", 0.0),
        alpha_forecast=attribution.get("alpha_forecast", 0.0),
        pct_of_portfolio_alpha=attribution.get("pct_of_portfolio_alpha", 0.0),
        risk_vs_weight_gap=gap,
        standalone_volatility=attribution.get("standalone_volatility", 0.0),
        marginal_risk=attribution.get("marginal_risk", 0.0),
        diversification_benefit=div,
        avg_correlation_to_book=corr,
        pct_of_tail_loss=attribution.get("pct_of_tail_loss", 0.0),
        contradicting_evidence=contra,
        confidence=confidence,
        confidence_basis=basis,
    )


def render_explanations(explanations: list[PositionExplanation]) -> str:
    """Text table of allocation vs contribution — the view that shows whether
    what you own matches what is driving the portfolio."""
    if not explanations:
        return "No positions to explain."
    lines = [
        "Position Attribution",
        "=" * 108,
        f"{'Ticker':<10}{'Weight':>9}{'Alpha':>10}{'%Alpha':>9}{'%Risk':>9}"
        f"{'Gap':>9}{'Divers':>9}{'%Tail':>9}{'Conf':>8}{'Cites':>7}",
        "-" * 108,
    ]
    for e in sorted(explanations, key=lambda x: -x.weight):
        lines.append(
            f"{e.ticker:<10}{e.weight:>8.2%}{e.alpha_forecast:>10.2%}"
            f"{e.pct_of_portfolio_alpha:>9.1%}{e.pct_of_portfolio_risk:>9.1%}"
            f"{e.risk_vs_weight_gap:>+9.1%}{e.diversification_benefit:>9.0%}"
            f"{e.pct_of_tail_loss:>9.1%}{e.confidence:>8.2f}{len(e.evidence):>7}"
        )
    lines += [
        "-" * 108,
        "Gap = risk share minus weight share; positive means the position uses more",
        "risk budget than its size. Divers = share of standalone volatility absorbed",
        "by correlation. Conf is derived from coverage, factor agreement, evidence",
        "and diversification — never assigned by the language model.",
    ]
    return "\n".join(lines)
