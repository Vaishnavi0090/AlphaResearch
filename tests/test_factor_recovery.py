"""Ground-truth recovery tests.

These are only possible because the synthetic DGP exposes the latent states it
used to generate returns. They answer a question a real dataset cannot:
*does the factor engine measure the thing it claims to measure?*

A backtest that makes money proves nothing about that — it conflates signal,
sizing and luck. Correlating a computed factor against the latent state it is
supposed to proxy is a direct test of construction correctness.

They also document a genuine NEGATIVE result: momentum is correctly implemented
but has almost no power in this DGP, because trailing return is a low
signal-to-noise proxy for the latent AR(1) drift. See docs/RESEARCH_REPORT.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrisk.data.market_data import align_to_calendar
from finrisk.data.synthetic import TRUE_PREMIA_ANN, SyntheticProvider
from finrisk.factors.engine import FactorEngine
from finrisk.factors.research import forward_returns, information_coefficient


@pytest.fixture(scope="module")
def recovery_setup():
    bundle = align_to_calendar(
        SyntheticProvider(seed=7, n_assets=30).fetch(None, "2010-01-04", "2022-12-31")
    )
    from finrisk.config.settings import FactorConfig

    fe = FactorEngine(FactorConfig())
    dates = pd.Series(bundle.dates).groupby([bundle.dates.year, bundle.dates.month]).last().tolist()
    dates = [d for d in dates if d >= bundle.dates[504]]
    panels = {}
    groups: dict[str, dict] = {}
    for d in dates:
        raw = fe.raw_panel(bundle, d)
        if raw.empty:
            continue
        panels[d] = raw
        _, diag = fe.compute(bundle, d)
        for c in [x for x in diag.columns if x.startswith("grp_")]:
            groups.setdefault(c[4:], {})[d] = diag[c]
    group_panels = {k: pd.DataFrame(v).T.sort_index() for k, v in groups.items()}
    return bundle, panels, group_panels, dates


def _mean_corr(panels, factor, truth_frame):
    vals = []
    for d, panel in panels.items():
        if factor not in panel.columns:
            continue
        j = pd.concat([panel[factor], truth_frame.loc[d]], axis=1).dropna()
        if len(j) > 5:
            vals.append(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
    return float(np.nanmean(vals)) if vals else float("nan")


def test_ground_truth_is_exposed(recovery_setup):
    bundle, *_ = recovery_setup
    gt = bundle.ground_truth
    assert gt is not None
    assert set(gt) >= {"trend_state", "value_state", "quality_state", "true_premia_ann"}
    assert gt["true_premia_ann"] == TRUE_PREMIA_ANN


def test_book_to_price_recovers_latent_value_state(recovery_setup):
    """The DGP maps the latent value state into reported book value, so
    book-to-price must be a strong monotone proxy for it."""
    bundle, panels, _, _ = recovery_setup
    corr = _mean_corr(panels, "book_to_price", bundle.ground_truth["value_state"])
    assert corr > 0.5, f"book_to_price recovers value state at only rho={corr:.3f}"


def test_roe_recovers_latent_quality_state(recovery_setup):
    bundle, panels, _, _ = recovery_setup
    corr = _mean_corr(panels, "roe", bundle.ground_truth["quality_state"])
    assert corr > 0.5, f"roe recovers quality state at only rho={corr:.3f}"


def test_low_beta_factor_recovers_true_betas(recovery_setup):
    """`low_beta` is stored negated, so it must be NEGATIVELY correlated with
    the true beta. This is the sign-convention check that matters most."""
    bundle, panels, _, _ = recovery_setup
    true_betas = bundle.ground_truth["betas"]
    vals = []
    for panel in panels.values():
        if "low_beta" not in panel.columns:
            continue
        j = pd.concat([panel["low_beta"], true_betas.rename("true")], axis=1).dropna()
        vals.append(j["low_beta"].corr(j["true"], method="spearman"))
    corr = float(np.nanmean(vals))
    assert corr < -0.7, f"low_beta should oppose true beta, got rho={corr:.3f}"


def test_value_group_has_positive_information_coefficient(recovery_setup):
    """The value premium is injected, so the value group must show positive IC."""
    bundle, _, group_panels, _ = recovery_setup
    panel = group_panels["value"]
    fwd = forward_returns(bundle.prices, list(panel.index), 126)
    ic = information_coefficient(panel, fwd)
    assert ic.mean() > 0.02, f"value group mean IC only {ic.mean():.4f}"


def test_quality_group_has_positive_information_coefficient(recovery_setup):
    bundle, _, group_panels, _ = recovery_setup
    panel = group_panels["quality"]
    fwd = forward_returns(bundle.prices, list(panel.index), 126)
    ic = information_coefficient(panel, fwd)
    assert ic.mean() > 0.01, f"quality group mean IC only {ic.mean():.4f}"


def test_latent_trend_state_does_predict_returns(recovery_setup):
    """Establishes that the trend premium really is present in the data, which
    is what makes the next test a statement about the FACTOR rather than about
    the dataset."""
    bundle, _, _, dates = recovery_setup
    trend = bundle.ground_truth["trend_state"]
    fwd = forward_returns(bundle.prices, dates, 63)
    ics = []
    for d in fwd.index:
        j = pd.concat([trend.loc[d], fwd.loc[d]], axis=1).dropna()
        ics.append(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
    assert float(np.nanmean(ics)) > 0.01


def test_momentum_lookback_decay_matches_state_half_life(recovery_setup):
    """DOCUMENTED NEGATIVE RESULT — momentum does not work in this DGP.

    Two facts, both asserted here:

    1. SHORT lookbacks (1m, 3m) correlate POSITIVELY with the latent trend
       state, confirming the implementation and sign convention are correct.
       The correlation is small (< 0.10) because over 21 sessions the injected
       alpha contributes ~0.25% of return against ~6% of idiosyncratic noise —
       trailing return is a ~96%-noise proxy for the drift.

    2. The 12m-minus-1m lookback is UNCORRELATED or slightly NEGATIVELY
       correlated with the current state. This is the expected consequence of
       an AR(1) trend with a ~46-session half-life: a 231-session window
       averages over roughly five half-lives, so it mostly measures states that
       have already reverted, and a high trailing return implies the state has
       since fallen.

    Conclusion: classical 12-1 momentum is mis-specified for this DGP. The
    correct lookback here is 1-3 months. Asserting this prevents a later change
    from silently converting a genuine negative result into a false positive.
    """
    bundle, panels, _, _ = recovery_setup
    trend = bundle.ground_truth["trend_state"]
    corrs = {f: _mean_corr(panels, f, trend) for f in ("mom_1m", "mom_3m", "mom_6m", "mom_12m_1m")}
    # Short lookbacks: correct sign, small magnitude.
    assert corrs["mom_1m"] > 0, f"sign error in mom_1m: {corrs}"
    assert corrs["mom_3m"] > 0, f"sign error in mom_3m: {corrs}"
    assert max(corrs.values()) < 0.20, f"unexpectedly strong momentum: {corrs}"
    # Long lookback decays to ~zero and may go negative through mean reversion.
    assert corrs["mom_12m_1m"] < corrs["mom_3m"], f"no lookback decay: {corrs}"
    assert corrs["mom_12m_1m"] < 0.02, f"12-1 momentum should be ~0 here: {corrs}"
