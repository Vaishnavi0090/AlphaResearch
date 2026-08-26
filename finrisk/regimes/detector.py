"""Market regime detector (Phase 6).

Wraps `GaussianHMM` in the point-in-time discipline a backtest needs:

* Parameters are re-estimated on an **expanding window** ending at the decision
  date, at most once every `refit_every` sessions (annually by default). Refits
  are the expensive part and monthly refitting changes nothing material.
* State probabilities come from **filtered** (forward-only) inference.
* Feature standardisation uses training-window moments only. Standardising with
  full-sample mean/std is a subtle and very common leak.
* State labels are assigned from the fitted emission means, so a state's name
  is derived from its statistical character, not from the order EM happened to
  converge in.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrisk.config.settings import RegimeConfig
from finrisk.regimes.hmm import GaussianHMM
from finrisk.schemas import RegimeState

logger = logging.getLogger(__name__)

# Risk multipliers applied to gross equity exposure by regime. Deliberately
# mild: this is a risk-management overlay, not a market-timing bet.
REGIME_RISK_MULTIPLIER = {
    "risk_on_low_vol": 1.00,
    "neutral": 0.85,
    "risk_off_high_vol": 0.60,
    "unknown": 0.85,
}


def build_regime_features(benchmark: pd.Series, features: list[str] | None = None) -> pd.DataFrame:
    """Regime features from the benchmark only.

    Kept to benchmark-level trend and volatility on purpose: regime is a
    property of the market, and feeding it cross-sectional data would blur the
    boundary with the alpha model.
    """
    features = features or ["ret_21d", "vol_21d"]
    r = benchmark.pct_change()
    out = pd.DataFrame(index=benchmark.index)
    if "ret_21d" in features:
        out["ret_21d"] = benchmark.pct_change(21)
    if "ret_63d" in features:
        out["ret_63d"] = benchmark.pct_change(63)
    if "vol_21d" in features:
        out["vol_21d"] = r.rolling(21).std(ddof=1) * np.sqrt(252)
    if "vol_63d" in features:
        out["vol_63d"] = r.rolling(63).std(ddof=1) * np.sqrt(252)
    if "drawdown" in features:
        out["drawdown"] = benchmark / benchmark.cummax() - 1.0
    if "trend" in features:
        out["trend"] = benchmark / benchmark.rolling(200).mean() - 1.0
    return out[features].dropna()


def label_states(means: np.ndarray, feature_names: list[str]) -> dict[int, str]:
    """Map state index -> semantic name using the fitted emission means.

    Ranking is on (standardised trend - standardised volatility): the state with
    the best trend and lowest volatility is risk-on, the worst is risk-off.
    """
    ret_cols = [
        i for i, f in enumerate(feature_names) if f.startswith("ret") or f in ("trend", "drawdown")
    ]
    vol_cols = [i for i, f in enumerate(feature_names) if f.startswith("vol")]
    ret_score = means[:, ret_cols].mean(axis=1) if ret_cols else np.zeros(len(means))
    vol_score = means[:, vol_cols].mean(axis=1) if vol_cols else np.zeros(len(means))
    composite = ret_score - vol_score

    order = np.argsort(composite)  # worst -> best
    k = len(order)
    names: dict[int, str] = {}
    if k == 1:
        names[int(order[0])] = "neutral"
    elif k == 2:
        names[int(order[0])] = "risk_off_high_vol"
        names[int(order[1])] = "risk_on_low_vol"
    else:
        names[int(order[0])] = "risk_off_high_vol"
        names[int(order[-1])] = "risk_on_low_vol"
        for i in order[1:-1]:
            names[int(i)] = "neutral"
    return names


class RegimeDetector:
    """Point-in-time regime inference with cached expanding-window refits."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.cfg = config or RegimeConfig()
        self._model: GaussianHMM | None = None
        self._labels: dict[int, str] = {}
        self._fit_mean: np.ndarray | None = None
        self._fit_std: np.ndarray | None = None
        self._last_fit_pos: int = -(10**9)
        self._features: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def prepare(self, benchmark: pd.Series) -> pd.DataFrame:
        self._features = build_regime_features(benchmark, self.cfg.features)
        return self._features

    # ------------------------------------------------------------------
    def _maybe_refit(self, train: pd.DataFrame, pos: int) -> bool:
        """Refit if we have moved `refit_every` sessions since the last fit."""
        if self._model is not None and (pos - self._last_fit_pos) < self.cfg.refit_every:
            return False
        X = train.to_numpy(dtype=float)
        self._fit_mean = X.mean(axis=0)
        self._fit_std = np.where(X.std(axis=0, ddof=1) > 1e-12, X.std(axis=0, ddof=1), 1.0)
        Z = (X - self._fit_mean) / self._fit_std
        model = GaussianHMM(
            n_states=self.cfg.n_states,
            max_iter=self.cfg.max_iter,
            tol=self.cfg.tol,
            seed=self.cfg.seed,
        )
        model.fit(Z)
        self._model = model
        assert model.params_ is not None
        self._labels = label_states(model.params_.means, list(train.columns))
        self._last_fit_pos = pos
        logger.debug(
            "HMM refit at pos=%d (%s), ll=%.1f, iters=%d",
            pos,
            train.index[-1].date(),
            model.params_.log_likelihood,
            model.params_.n_iter,
        )
        return True

    # ------------------------------------------------------------------
    def detect(self, benchmark: pd.Series, as_of: pd.Timestamp) -> RegimeState:
        """Infer the regime at `as_of` using only data up to `as_of`."""
        as_of = pd.Timestamp(as_of)
        if self._features is None:
            self.prepare(benchmark)
        assert self._features is not None
        train = self._features.loc[:as_of]

        if len(train) < self.cfg.train_min_obs:
            return RegimeState(
                as_of=as_of.date(),
                regime="unknown",
                state_id=-1,
                confidence=0.0,
                features={},
                state_probabilities={},
                transition_probabilities={},
            )

        pos = len(train)
        self._maybe_refit(train, pos)
        assert self._model is not None and self._fit_mean is not None

        Z = (train.to_numpy(dtype=float) - self._fit_mean) / self._fit_std
        probs = self._model.filter_states(Z)[-1]
        sid = int(np.argmax(probs))
        name = self._labels.get(sid, "neutral")

        params = self._model.params_
        assert params is not None
        trans = params.transition[sid]
        durations = params.expected_durations()

        return RegimeState(
            as_of=as_of.date(),
            regime=name,
            state_id=sid,
            confidence=float(probs[sid]),
            features={k: float(v) for k, v in train.iloc[-1].items()},
            state_probabilities={
                self._labels.get(i, f"state_{i}"): float(p) for i, p in enumerate(probs)
            },
            transition_probabilities={
                self._labels.get(j, f"state_{j}"): float(trans[j]) for j in range(len(trans))
            },
            expected_duration_days=float(durations[sid]),
        )

    # ------------------------------------------------------------------
    def risk_multiplier(self, state: RegimeState, floor: float = 0.5) -> float:
        """Probability-weighted exposure multiplier.

        Blending across the posterior rather than switching on the argmax means
        a 51/49 call does not produce a discontinuous jump in exposure — it
        halves the turnover a hard switch would cause.
        """
        if not state.state_probabilities:
            return REGIME_RISK_MULTIPLIER["unknown"]
        mult = sum(
            REGIME_RISK_MULTIPLIER.get(name, 0.85) * p
            for name, p in state.state_probabilities.items()
        )
        return float(np.clip(mult, floor, 1.0))

    # ------------------------------------------------------------------
    def history(self, benchmark: pd.Series, dates: list[pd.Timestamp]) -> pd.DataFrame:
        """Point-in-time regime path over many dates (for reports/plots)."""
        rows = []
        for d in dates:
            st = self.detect(benchmark, d)
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "regime": st.regime,
                    "confidence": st.confidence,
                    "risk_multiplier": self.risk_multiplier(st),
                    **{f"p_{k}": v for k, v in st.state_probabilities.items()},
                }
            )
        return pd.DataFrame(rows).set_index("date")

    # ------------------------------------------------------------------
    def stability(self, path: pd.DataFrame) -> dict[str, float]:
        """Regime stability diagnostics.

        A detector that flips every month is describing noise. `switch_rate`
        and `mean_run_length` are how you tell.
        """
        if path.empty or "regime" not in path:
            return {}
        reg = path["regime"]
        switches = int((reg != reg.shift(1)).iloc[1:].sum())
        runs = (reg != reg.shift(1)).cumsum()
        run_lengths = reg.groupby(runs).size()
        return {
            "n_periods": float(len(reg)),
            "n_switches": float(switches),
            "switch_rate": float(switches / max(len(reg) - 1, 1)),
            "mean_run_length": float(run_lengths.mean()),
            "median_run_length": float(run_lengths.median()),
            "mean_confidence": float(path["confidence"].mean()),
            **{f"share_{r}": float((reg == r).mean()) for r in sorted(reg.unique())},
        }
