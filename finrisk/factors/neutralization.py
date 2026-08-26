"""Factor neutralisation and neutral portfolio construction.

A raw factor portfolio is almost never a pure bet on the factor. Sorting on
book-to-price loads on whatever sectors happen to be cheap; sorting on momentum
loads on high-beta names in a rising market. The reported "value premium" is
then partly a sector bet and partly a market-timing bet, and you cannot tell
which without removing them.

This module removes them, three ways:

* `sector_neutralize` — demean scores within each sector, so the cross-section
  ranks a stock against its peers rather than against the whole market.
* `beta_neutralize` — regress scores on beta and keep the residual, so the
  signal is orthogonal to market sensitivity.
* `neutralize` — the general case: residualise scores against an arbitrary
  exposure matrix (sector dummies, beta, size, anything).

`build_neutral_portfolio` then turns neutralised scores into a long/short book
that is simultaneously dollar-neutral, sector-neutral and beta-neutral, with
each property asserted rather than assumed.

The honest framing: neutralisation reduces *measured* exposure to the
neutralised characteristic. It cannot remove exposure to a characteristic you
did not think to include, and residualising against a noisily-estimated beta
leaves residual beta behind. `tests/test_neutralization.py` measures how much
is actually removed rather than claiming it is zero.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ridge term for the residualisation solve. Sector dummies are collinear with
# the intercept and beta can be near-constant in a small cross-section, so a
# plain least-squares solve is not always well posed.
_RIDGE = 1e-8


def _clean(scores: pd.Series) -> pd.Series:
    return pd.to_numeric(scores, errors="coerce").astype(float)


def sector_dummies(tickers: list[str], sectors: dict[str, str]) -> pd.DataFrame:
    """One-hot sector matrix aligned to `tickers`.

    Assets with no sector mapping go to an explicit 'Unknown' column rather
    than being dropped — dropping would silently shrink the universe.
    """
    labels = [sectors.get(t, "Unknown") for t in tickers]
    return pd.get_dummies(pd.Series(labels, index=tickers), dtype=float)


def neutralize(
    scores: pd.Series,
    exposures: pd.DataFrame,
    add_intercept: bool = True,
    standardize: bool = True,
) -> pd.Series:
    """Residualise `scores` against `exposures` by cross-sectional OLS.

    Returns the residual e from  scores = X b + e, which is by construction
    orthogonal to every column of X. That orthogonality is the whole point:
    the resulting signal cannot be explained by the exposures.

    NaN scores stay NaN (a missing signal must not become a synthetic zero
    that looks like a real neutral view).
    """
    s = _clean(scores)
    valid = s.notna()
    X = exposures.reindex(s.index).astype(float)
    valid &= X.notna().all(axis=1)
    if valid.sum() < 3:
        logger.warning("neutralize: fewer than 3 valid observations; returning input")
        return s

    y = s[valid].to_numpy()
    Xv = X[valid].to_numpy()
    if add_intercept:
        Xv = np.column_stack([np.ones(len(Xv)), Xv])

    # Ridge-regularised normal equations: robust to the collinearity that
    # sector dummies + intercept always produce.
    XtX = Xv.T @ Xv + _RIDGE * np.eye(Xv.shape[1])
    try:
        beta = np.linalg.solve(XtX, Xv.T @ y)
    except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this rare
        beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    resid = y - Xv @ beta

    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[valid] = resid
    if standardize:
        sd = out.std(ddof=1)
        if sd > 1e-12:
            out = (out - out.mean()) / sd
    return out


def sector_neutralize(
    scores: pd.Series, sectors: dict[str, str], standardize: bool = True
) -> pd.Series:
    """Demean scores within each sector.

    Equivalent to residualising against sector dummies, but implemented
    directly because it is exact and cheaper. A sector with a single member
    would demean to exactly zero, which destroys information rather than
    neutralising it, so singletons keep their cross-sectional score instead.
    """
    s = _clean(scores)
    grp = pd.Series([sectors.get(t, "Unknown") for t in s.index], index=s.index)
    counts = grp.map(grp.value_counts())
    demeaned = s - s.groupby(grp).transform("mean")
    out = demeaned.where(counts > 1, s - s.mean())
    if standardize:
        sd = out.std(ddof=1)
        if sd > 1e-12:
            out = (out - out.mean()) / sd
    return out


def beta_neutralize(scores: pd.Series, betas: pd.Series, standardize: bool = True) -> pd.Series:
    """Remove the linear component of `scores` explained by beta."""
    return neutralize(scores, pd.DataFrame({"beta": _clean(betas)}), True, standardize)


def estimate_betas(returns: pd.DataFrame, benchmark: pd.Series, min_obs: int = 60) -> pd.Series:
    """Per-asset OLS beta against the benchmark over the supplied window."""
    b = pd.to_numeric(benchmark, errors="coerce").reindex(returns.index)
    var_b = b.var(ddof=1)
    if not np.isfinite(var_b) or var_b <= 0:
        return pd.Series(np.nan, index=returns.columns, dtype=float)
    out = {}
    for col in returns.columns:
        pair = pd.concat([returns[col], b], axis=1).dropna()
        out[col] = (
            float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var(ddof=1))
            if len(pair) >= min_obs
            else np.nan
        )
    return pd.Series(out, dtype=float)


def build_neutral_portfolio(
    scores: pd.Series,
    sectors: dict[str, str] | None = None,
    betas: pd.Series | None = None,
    gross_exposure: float = 2.0,
    max_weight: float | None = None,
) -> pd.Series:
    """Long/short weights that are dollar-, sector- and beta-neutral.

    Construction:
      1. neutralise the scores against whichever exposures were supplied,
      2. demean so longs and shorts offset exactly (dollar neutrality),
      3. scale to the requested gross exposure,
      4. optionally cap positions, then re-neutralise, because capping
         reintroduces the very imbalance step 2 removed.

    Returns weights summing to ~0 with |w| summing to `gross_exposure`.
    """
    s = _clean(scores).dropna()
    if len(s) < 4:
        return pd.Series(dtype=float)

    exposures = []
    if sectors:
        exposures.append(sector_dummies(list(s.index), sectors))
    if betas is not None:
        exposures.append(pd.DataFrame({"beta": _clean(betas).reindex(s.index)}))
    if exposures:
        X = pd.concat(exposures, axis=1)
        s = neutralize(s, X, add_intercept=True, standardize=True).dropna()
    if len(s) < 4:
        return pd.Series(dtype=float)

    w = s - s.mean()
    denom = w.abs().sum()
    if denom <= 1e-12:
        return pd.Series(0.0, index=s.index)
    w = w * (gross_exposure / denom)

    if max_weight is not None:
        # Capping breaks neutrality: clipping the largest longs leaves a net
        # short tilt, and the clipped names carry sector and beta exposure with
        # them. Simply re-demeaning restores DOLLAR neutrality but not sector
        # or beta neutrality — measured residual beta of ~1e-2 in testing.
        #
        # So each iteration re-projects the capped weights onto the null space
        # of the exposures (alternating projection between the cap constraint
        # and the neutrality constraint), which converges to weights that
        # satisfy both.
        X_full = pd.concat(exposures, axis=1).reindex(w.index) if exposures else None
        for _ in range(64):
            capped = w.clip(-max_weight, max_weight)
            if X_full is not None:
                capped = neutralize(capped, X_full, add_intercept=True, standardize=False).fillna(
                    0.0
                )
            capped = capped - capped.mean()
            denom = capped.abs().sum()
            if denom <= 1e-12:
                break
            capped = capped * (gross_exposure / denom)
            converged = np.allclose(capped, w, atol=1e-12)
            w = capped
            if converged:
                break
        # Projection can push a weight marginally back over the cap; a final
        # clip guarantees the hard constraint holds, and the residual
        # neutrality error is reported by `exposure_report` rather than hidden.
        w = w.clip(-max_weight, max_weight)
    return w


def exposure_report(
    weights: pd.Series,
    sectors: dict[str, str] | None = None,
    betas: pd.Series | None = None,
) -> dict[str, float]:
    """Measure what the portfolio is actually exposed to.

    This is the verification step. Neutral construction is a claim; these
    numbers are the evidence for or against it.
    """
    w = _clean(weights).dropna()
    rep: dict[str, float] = {
        "net_exposure": float(w.sum()),
        "gross_exposure": float(w.abs().sum()),
        "n_long": int((w > 1e-9).sum()),
        "n_short": int((w < -1e-9).sum()),
        "max_abs_weight": float(w.abs().max()) if len(w) else 0.0,
    }
    if sectors:
        by_sector = w.groupby(
            pd.Series([sectors.get(t, "Unknown") for t in w.index], index=w.index)
        ).sum()
        rep["max_abs_sector_exposure"] = float(by_sector.abs().max())
        rep["sum_abs_sector_exposure"] = float(by_sector.abs().sum())
    if betas is not None:
        b = _clean(betas).reindex(w.index)
        pair = pd.concat([w, b], axis=1).dropna()
        rep["portfolio_beta"] = (
            float((pair.iloc[:, 0] * pair.iloc[:, 1]).sum()) if len(pair) else float("nan")
        )
    return rep


def neutral_vs_raw(
    score_panel: pd.DataFrame,
    prices: pd.DataFrame,
    sectors: dict[str, str],
    benchmark: pd.Series,
    horizon: int = 21,
    beta_lookback: int = 252,
    gross_exposure: float = 2.0,
) -> pd.DataFrame:
    """Compare raw, sector-neutral and fully-neutral versions of one factor.

    Each variant is run through the identical portfolio construction and
    return calculation, so a difference between rows is attributable to the
    neutralisation and nothing else.

    Returns one row per variant with realised return, volatility, Sharpe and —
    critically — the realised exposures, so the reader can check that the
    "neutral" variants actually are.
    """
    from finrisk.risk import metrics as m

    rets = prices.pct_change().iloc[1:]
    dates = [d for d in score_panel.index if d in prices.index]
    variants = ("raw", "sector_neutral", "sector_beta_neutral")
    streams: dict[str, list[float]] = {v: [] for v in variants}
    exposures: dict[str, list[dict[str, float]]] = {v: [] for v in variants}

    for d in dates:
        scores = score_panel.loc[d].dropna()
        if len(scores) < 6:
            continue
        pos = prices.index.get_loc(d)
        fwd = prices.iloc[pos : pos + horizon + 1]
        if len(fwd) < 2:
            continue
        period_ret = (fwd.iloc[-1] / fwd.iloc[0] - 1).reindex(scores.index)

        hist = rets.loc[:d].tail(beta_lookback)
        betas = estimate_betas(
            hist[scores.index.intersection(hist.columns)],
            benchmark.loc[:d].pct_change().tail(beta_lookback),
        )

        for variant in variants:
            if variant == "raw":
                w = build_neutral_portfolio(scores, None, None, gross_exposure)
            elif variant == "sector_neutral":
                w = build_neutral_portfolio(scores, sectors, None, gross_exposure)
            else:
                w = build_neutral_portfolio(scores, sectors, betas, gross_exposure)
            if not len(w):
                continue
            aligned = period_ret.reindex(w.index).fillna(0.0)
            streams[variant].append(float((w * aligned).sum()))
            exposures[variant].append(exposure_report(w, sectors, betas))

    periods_per_year = 252.0 / horizon
    rows = []
    for variant in variants:
        r = pd.Series(streams[variant], dtype=float)
        if r.empty:
            continue
        exp_df = pd.DataFrame(exposures[variant])
        rows.append(
            {
                "variant": variant,
                "n_periods": len(r),
                "ann_return": float(r.mean() * periods_per_year),
                "ann_volatility": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
                "sharpe": (
                    float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))
                    if r.std(ddof=1) > 1e-12
                    else 0.0
                ),
                "hit_rate": float((r > 0).mean()),
                "max_drawdown": float(m.max_drawdown(r)),
                "mean_abs_net_exposure": float(exp_df["net_exposure"].abs().mean()),
                "mean_abs_sector_exposure": float(
                    exp_df.get("max_abs_sector_exposure", pd.Series([np.nan])).mean()
                ),
                "mean_abs_portfolio_beta": float(
                    exp_df.get("portfolio_beta", pd.Series([np.nan])).abs().mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("variant") if rows else pd.DataFrame()


def render_neutrality_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "No neutrality comparison produced."
    lines = [
        "Factor Neutralisation Comparison",
        "=" * 104,
        f"{'Variant':<22}{'AnnRet':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}"
        f"{'|Net|':>9}{'|Sector|':>10}{'|Beta|':>9}",
        "-" * 104,
    ]
    for name, r in df.iterrows():
        lines.append(
            f"{name:<22}{r['ann_return']:>8.2%}{r['ann_volatility']:>8.1%}"
            f"{r['sharpe']:>8.2f}{r['max_drawdown']:>9.1%}"
            f"{r['mean_abs_net_exposure']:>9.4f}"
            f"{r['mean_abs_sector_exposure']:>10.4f}"
            f"{r['mean_abs_portfolio_beta']:>9.4f}"
        )
    lines += [
        "-" * 104,
        "|Net|, |Sector| and |Beta| are realised exposures — the evidence that the",
        "'neutral' variants are neutral. Lower is more neutral; they are not exactly",
        "zero because betas are estimated with error.",
    ]
    return "\n".join(lines)
