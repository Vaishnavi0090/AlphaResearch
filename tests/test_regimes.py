"""HMM correctness and the causality guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.regimes.detector import (
    REGIME_RISK_MULTIPLIER,
    RegimeDetector,
    build_regime_features,
    label_states,
)
from finrisk.regimes.hmm import GaussianHMM, logsumexp


def test_logsumexp_matches_naive():
    a = np.array([1.0, 2.0, 3.0])
    assert logsumexp(a) == pytest.approx(np.log(np.exp(a).sum()))


def test_logsumexp_is_stable_for_large_values():
    a = np.array([1000.0, 1001.0])
    out = logsumexp(a)
    assert np.isfinite(out)
    assert out == pytest.approx(1001.0 + np.log(1 + np.exp(-1.0)))


def test_logsumexp_axis_and_keepdims():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert logsumexp(a, axis=1).shape == (2,)
    assert logsumexp(a, axis=1, keepdims=True).shape == (2, 1)


@pytest.fixture(scope="module")
def separable_data():
    """Two clearly separated regimes with long persistent runs."""
    rng = np.random.default_rng(0)
    blocks = []
    truth = []
    for i in range(12):
        n = 200
        if i % 2 == 0:
            blocks.append(rng.normal([0.02, 0.10], [0.01, 0.01], (n, 2)))
            truth += [0] * n
        else:
            blocks.append(rng.normal([-0.03, 0.35], [0.01, 0.02], (n, 2)))
            truth += [1] * n
    return np.vstack(blocks), np.array(truth)


def test_hmm_fits_and_converges(separable_data):
    X, _ = separable_data
    hmm = GaussianHMM(n_states=2, seed=0).fit(X)
    assert hmm.params_ is not None
    assert hmm.params_.converged
    assert np.isfinite(hmm.params_.log_likelihood)


def test_hmm_recovers_separable_states(separable_data):
    X, truth = separable_data
    hmm = GaussianHMM(n_states=2, seed=0).fit(X)
    pred = hmm.smoothed_states(X).argmax(axis=1)
    acc = max((pred == truth).mean(), (pred != truth).mean())
    assert acc > 0.95, f"state recovery accuracy only {acc:.3f}"


def test_transition_matrix_rows_sum_to_one(separable_data):
    X, _ = separable_data
    hmm = GaussianHMM(n_states=3, seed=0).fit(X)
    assert np.allclose(hmm.params_.transition.sum(axis=1), 1.0)


def test_filtered_probabilities_are_a_distribution(separable_data):
    X, _ = separable_data
    hmm = GaussianHMM(n_states=3, seed=0).fit(X)
    p = hmm.filter_states(X)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_em_log_likelihood_is_monotonically_non_decreasing(separable_data):
    """EM's defining guarantee. Note we do NOT assert that more states gives a
    higher likelihood: with deterministic quantile initialisation, a larger
    model can converge to a worse local optimum, so that would be an invalid
    assumption rather than a property of the algorithm."""
    X, _ = separable_data
    hist = GaussianHMM(3, seed=0).fit(X).params_.ll_history
    assert len(hist) >= 2
    diffs = np.diff(np.array(hist))
    assert (diffs >= -1e-6).all(), f"EM likelihood decreased: min step {diffs.min()}"


def test_bic_penalises_extra_states(separable_data):
    X, _ = separable_data
    b2 = GaussianHMM(2, seed=0).fit(X).bic(X)
    b8 = GaussianHMM(8, seed=0).fit(X).bic(X)
    assert b2 < b8


def test_expected_durations_are_positive(separable_data):
    X, _ = separable_data
    hmm = GaussianHMM(2, seed=0).fit(X)
    assert (hmm.params_.expected_durations() > 1.0).all()


def test_hmm_is_deterministic(separable_data):
    X, _ = separable_data
    a = GaussianHMM(3, seed=0).fit(X).params_
    b = GaussianHMM(3, seed=0).fit(X).params_
    assert np.allclose(a.means, b.means)
    assert np.allclose(a.transition, b.transition)


@pytest.fixture(scope="module")
def overlapping_data():
    """Noisy, OVERLAPPING regimes so inference is genuinely uncertain.

    The separable fixture is too easy: when the posterior is ~1.0 everywhere,
    filtered and smoothed estimates coincide numerically and the causality
    contrast becomes untestable. Real markets look like this fixture.
    """
    rng = np.random.default_rng(3)
    blocks = []
    for i in range(10):
        n = 150
        centre = [0.004, 0.16] if i % 2 == 0 else [-0.004, 0.22]
        blocks.append(rng.normal(centre, [0.05, 0.06], (n, 2)))
    return np.vstack(blocks)


def test_filtered_differs_from_smoothed(overlapping_data):
    """If these were equal, filtering would be secretly using future data."""
    hmm = GaussianHMM(2, seed=0).fit(overlapping_data)
    diff = np.abs(hmm.filter_states(overlapping_data) - hmm.smoothed_states(overlapping_data)).max()
    assert diff > 1e-3, f"filtered and smoothed are identical (max diff {diff:.2e})"


def test_filtering_is_causal(overlapping_data):
    """The filtered estimate at time t must not change when future
    observations are appended. This is THE look-ahead guarantee."""
    hmm = GaussianHMM(2, seed=0).fit(overlapping_data)
    t = 900
    assert np.allclose(
        hmm.filter_states(overlapping_data[: t + 1])[t],
        hmm.filter_states(overlapping_data)[t],
    )


def test_smoothing_is_not_causal(overlapping_data):
    """Contrast case: smoothed estimates DO change, proving they use the
    future — which is why they are banned from the decision path."""
    hmm = GaussianHMM(2, seed=0).fit(overlapping_data)
    t = 900
    a = hmm.smoothed_states(overlapping_data[: t + 1])[t]
    b = hmm.smoothed_states(overlapping_data)[t]
    assert np.abs(a - b).max() > 1e-6


def test_fit_rejects_short_and_nonfinite():
    with pytest.raises(ValueError):
        GaussianHMM(3).fit(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        GaussianHMM(2).fit(np.full((200, 2), np.nan))
    with pytest.raises(ValueError):
        GaussianHMM(2).fit(np.zeros(100))


def test_filter_before_fit_raises():
    with pytest.raises(RuntimeError):
        GaussianHMM(2).filter_states(np.zeros((10, 2)))


# ---------------------------------------------------------------- detector
def test_build_regime_features_shape(small_bundle):
    f = build_regime_features(small_bundle.benchmark, ["ret_21d", "vol_21d"])
    assert list(f.columns) == ["ret_21d", "vol_21d"]
    assert f.notna().all().all()


def test_label_states_orders_by_risk():
    means = np.array([[-0.05, 0.9], [0.01, 0.0], [0.06, -0.8]])
    labels = label_states(means, ["ret_21d", "vol_21d"])
    assert labels[0] == "risk_off_high_vol"
    assert labels[2] == "risk_on_low_vol"
    assert labels[1] == "neutral"


def test_detect_returns_unknown_before_warmup(small_bundle, small_settings):
    rd = RegimeDetector(small_settings.regimes)
    rd.prepare(small_bundle.benchmark)
    st = rd.detect(small_bundle.benchmark, small_bundle.dates[50])
    assert st.regime == "unknown"
    assert st.confidence == 0.0


def test_detect_produces_valid_state(small_bundle, small_settings):
    rd = RegimeDetector(small_settings.regimes)
    rd.prepare(small_bundle.benchmark)
    st = rd.detect(small_bundle.benchmark, small_bundle.dates[-1])
    assert st.regime in REGIME_RISK_MULTIPLIER
    assert 0.0 <= st.confidence <= 1.0
    assert sum(st.state_probabilities.values()) == pytest.approx(1.0)
    assert sum(st.transition_probabilities.values()) == pytest.approx(1.0)


def test_risk_multiplier_is_bounded_and_blended(small_bundle, small_settings):
    rd = RegimeDetector(small_settings.regimes)
    rd.prepare(small_bundle.benchmark)
    st = rd.detect(small_bundle.benchmark, small_bundle.dates[-1])
    mult = rd.risk_multiplier(st)
    assert 0.5 <= mult <= 1.0


def test_risk_multiplier_lower_in_risk_off():
    from finrisk.schemas import RegimeState

    rd = RegimeDetector()
    off = RegimeState(
        as_of=pd.Timestamp("2020-03-20").date(),
        regime="risk_off_high_vol",
        state_id=0,
        confidence=0.95,
        state_probabilities={"risk_off_high_vol": 0.95, "risk_on_low_vol": 0.05},
    )
    on = RegimeState(
        as_of=pd.Timestamp("2021-03-20").date(),
        regime="risk_on_low_vol",
        state_id=1,
        confidence=0.95,
        state_probabilities={"risk_on_low_vol": 0.95, "risk_off_high_vol": 0.05},
    )
    assert rd.risk_multiplier(off) < rd.risk_multiplier(on)


def test_detection_is_stable_under_truncation(small_bundle, small_settings):
    """Same date, with and without future data available."""
    d = small_bundle.dates[-200]
    a = RegimeDetector(small_settings.regimes)
    a.prepare(small_bundle.benchmark)
    st_full = a.detect(small_bundle.benchmark, d)
    b = RegimeDetector(small_settings.regimes)
    trunc = small_bundle.benchmark.loc[:d]
    b.prepare(trunc)
    st_trunc = b.detect(trunc, d)
    assert st_full.regime == st_trunc.regime


def test_stability_diagnostics(small_bundle, small_settings):
    rd = RegimeDetector(small_settings.regimes)
    rd.prepare(small_bundle.benchmark)
    dates = list(small_bundle.dates[600::63])
    path = rd.history(small_bundle.benchmark, dates)
    stats = rd.stability(path)
    assert 0.0 <= stats["switch_rate"] <= 1.0
    assert stats["mean_run_length"] >= 1.0
    assert stats["n_periods"] == len(dates)
