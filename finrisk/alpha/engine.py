"""Alpha signal engine (Phase 7).

Turns a cross-section of normalised factor scores into expected excess returns
that a mean-variance optimiser can consume.

THE KEY PROBLEM
---------------
A composite factor score is a z-score — dimensionless. An optimiser needs
expected returns in units of return per year. The naive fix ("treat the z-score
as a percentage") is arbitrary and quietly determines how aggressive the
portfolio is.

Instead we use the **Grinold-Kahn forecasting rule**:

    alpha_i = IC * sigma_i * z_i

where `IC` is the realised information coefficient of the signal, `sigma_i` is
the asset's forecast volatility, and `z_i` is the normalised score. The
intuition: a signal with no skill (IC=0) produces no alpha, a volatile asset
gets a proportionally larger forecast because there is more return to forecast,
and the score only says *where* the asset sits in the cross-section.

Crucially, `IC` is estimated from the **training window only** — data strictly
before the decision date — and is clipped to a sane range. This is what stops
the alpha scale from being a free parameter fitted to the test period.

WHAT AN LLM MAY NOT DO
----------------------
Nothing in this module consults a language model. Alpha is deterministic and
reproducible from prices and fundamentals alone. See PHASE 20 / docs/
DESIGN_DECISIONS.md for why that separation is load-bearing.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
import pandas as pd

from finrisk.config.settings import FactorConfig
from finrisk.data.base import MarketDataBundle
from finrisk.factors.engine import FactorEngine
from finrisk.factors.research import forward_returns, information_coefficient
from finrisk.schemas import AlphaSignal, FactorScore, RegimeState

logger = logging.getLogger(__name__)

# Bounds on the estimated IC. An IC above ~0.15 on monthly equity
# cross-sections is implausible and almost always an artefact of a short
# estimation sample; clipping prevents one lucky window from producing an
# enormous alpha forecast that the optimiser then levers into.
IC_FLOOR = 0.0
IC_CEILING = 0.15
DEFAULT_IC = 0.03


class AlphaEngine:
    """Deterministic alpha forecasts from factor scores and regime state."""

    def __init__(
        self,
        factor_engine: FactorEngine | None = None,
        default_ic: float = DEFAULT_IC,
        long_only: bool = True,
    ) -> None:
        self.factors = factor_engine or FactorEngine(FactorConfig())
        self.default_ic = float(default_ic)
        self.long_only = long_only

    # ------------------------------------------------------------------
    def estimate_ic(
        self,
        bundle: MarketDataBundle,
        train_dates: list[pd.Timestamp],
        horizon_days: int = 21,
    ) -> float:
        """Realised IC of the composite score over the TRAINING window.

        Returns `default_ic` when there is not enough history to estimate it —
        a deliberately conservative fallback rather than an optimistic one.
        """
        if len(train_dates) < 8:
            return self.default_ic
        scores = self.factors.composite_series(bundle, train_dates)
        if scores.empty:
            return self.default_ic
        fwd = forward_returns(bundle.prices, list(scores.index), horizon_days)
        ic = information_coefficient(scores, fwd)
        if len(ic) < 6:
            return self.default_ic
        est = float(ic.mean())
        return float(np.clip(est, IC_FLOOR, IC_CEILING))

    # ------------------------------------------------------------------
    @staticmethod
    def forecast_volatility(
        bundle: MarketDataBundle, as_of: pd.Timestamp, window: int = 126
    ) -> pd.Series:
        """Annualised realised volatility over a trailing window."""
        rets = bundle.prices.loc[:as_of].pct_change().iloc[1:]
        if len(rets) < 20:
            return pd.Series(0.20, index=bundle.prices.columns)
        vol = rets.iloc[-window:].std(ddof=1) * np.sqrt(252.0)
        return vol.replace(0.0, np.nan).fillna(vol.median()).fillna(0.20)

    # ------------------------------------------------------------------
    def generate(
        self,
        bundle: MarketDataBundle,
        as_of: pd.Timestamp,
        scores: dict[str, FactorScore] | None = None,
        regime: RegimeState | None = None,
        ic: float | None = None,
        use_regime: bool = True,
    ) -> dict[str, AlphaSignal]:
        as_of = pd.Timestamp(as_of)
        if scores is None:
            scores, _ = self.factors.compute(bundle, as_of)
        if not scores:
            return {}

        ic_eff = self.default_ic if ic is None else float(np.clip(ic, IC_FLOOR, IC_CEILING))
        vol = self.forecast_volatility(bundle, as_of)

        # Regime scaling. Attenuates the *magnitude* of the forecast in hostile
        # regimes rather than flipping its sign: a regime model has no view on
        # the cross-section, only on how much conviction is warranted.
        adj = 1.0
        regime_name = None
        if use_regime and regime is not None and regime.regime != "unknown":
            from finrisk.regimes.detector import RegimeDetector

            adj = RegimeDetector().risk_multiplier(regime)
            regime_name = regime.regime

        comp = pd.Series({t: s.composite for t, s in scores.items()})
        # Confidence blends signal strength (|z| relative to the cross-section)
        # with data coverage, so a strong score built on 40% of the factors is
        # not treated like a strong score built on all of them.
        spread = comp.abs().max() or 1.0

        out: dict[str, AlphaSignal] = {}
        for tk, sc in scores.items():
            z = float(sc.composite)
            sigma = float(vol.get(tk, 0.20))
            alpha = ic_eff * sigma * z * adj

            contributions: dict[str, float] = {}
            weights = self.factors.cfg.normalised_weights()
            for grp, gz in sc.group_scores.items():
                w = weights.get(grp, 0.0)
                contributions[grp] = float(ic_eff * sigma * w * gz * adj)

            direction = "flat"
            if z > 1e-9:
                direction = "long"
            elif z < -1e-9:
                direction = "flat" if self.long_only else "short"

            out[tk] = AlphaSignal(
                ticker=tk,
                as_of=as_of.date(),
                score=float(alpha),
                direction=cast(Literal["long", "short", "flat"], direction),
                confidence=float(min(abs(z) / spread, 1.0) * sc.coverage),
                factor_contributions=contributions,
                regime_context=regime_name,
                regime_adjustment=float(adj),
            )
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def to_expected_returns(signals: dict[str, AlphaSignal]) -> pd.Series:
        """Alpha forecasts as an annualised expected-excess-return vector."""
        return pd.Series({t: s.score for t, s in signals.items()}).sort_index()

    @staticmethod
    def select(
        signals: dict[str, AlphaSignal], top_n: int, min_confidence: float = 0.0
    ) -> list[str]:
        """Top-N by alpha score, filtered on confidence.

        Ties are broken by ticker so selection is deterministic — otherwise a
        backtest is not reproducible when two assets score identically.
        """
        eligible = [
            s for s in signals.values() if s.confidence >= min_confidence and s.direction == "long"
        ]
        eligible.sort(key=lambda s: (-s.score, s.ticker))
        return [s.ticker for s in eligible[:top_n]]
