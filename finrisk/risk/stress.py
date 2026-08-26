"""Portfolio stress testing (Phase 22).

Two families of scenario:

* **Historical replay.** Hold today's weights fixed and apply the *actual*
  returns from a named historical window. Captures the correlation structure of
  a real crisis, which no parametric model reproduces.
* **Hypothetical shocks.** Analytically transform the covariance/return
  assumptions: a volatility multiplier, a uniform correlation floor, a beta-
  driven market drop, a single-sector drawdown.

The correlation-shock scenario matters most. In a crisis, cross-asset
correlations converge toward one, so diversification evaporates precisely when
it is needed. A portfolio optimised on placid-period correlations can look
well-diversified and still lose like a single concentrated bet.

None of this forecasts anything. These are conditional statements of the form
"if returns behaved as they did in window X, this portfolio would have lost Y".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrisk.risk import metrics as m
from finrisk.schemas import StressScenario

logger = logging.getLogger(__name__)

# Real US-equity crisis windows, for use with real (not synthetic) data.
HISTORICAL_WINDOWS: dict[str, tuple[str, str, str]] = {
    "gfc_2008": ("2008-09-01", "2009-03-09", "Global financial crisis collapse"),
    "flash_crash_2010": ("2010-04-23", "2010-07-02", "2010 correction / flash crash"),
    "taper_tantrum_2013": ("2013-05-22", "2013-06-24", "Rate-shock repricing"),
    "vol_spike_2018": ("2018-01-26", "2018-02-08", "Volatility spike (VIX complex)"),
    "q4_2018": ("2018-10-01", "2018-12-24", "Q4 2018 growth scare"),
    "covid_2020": ("2020-02-19", "2020-03-23", "COVID-19 crash"),
    "rates_2022": ("2022-01-03", "2022-10-12", "2022 inflation / rate shock"),
}


def _portfolio_series(weights: dict[str, float], rets: pd.DataFrame) -> pd.Series:
    cols = [t for t in weights if t in rets.columns]
    if not cols:
        return pd.Series(dtype=float)
    w = pd.Series({t: weights[t] for t in cols})
    return (rets[cols].fillna(0.0) * w).sum(axis=1)


def historical_replay(
    weights: dict[str, float],
    asset_returns: pd.DataFrame,
    start: str,
    end: str,
    name: str,
    description: str = "",
    benchmark_returns: pd.Series | None = None,
) -> StressScenario | None:
    """Apply a historical window's realised returns to current weights."""
    window = asset_returns.loc[start:end]
    if len(window) < 5:
        logger.debug("window %s has %d rows; skipping", name, len(window))
        return None
    port = _portfolio_series(weights, window)
    if not len(port):
        return None
    contrib = (
        window[[t for t in weights if t in window.columns]].fillna(0.0)
        * pd.Series({t: weights[t] for t in weights if t in window.columns})
    ).sum()
    worst = contrib.nsmallest(5).to_dict()
    bench = None
    if benchmark_returns is not None:
        bw = benchmark_returns.loc[start:end]
        bench = float((1 + bw).prod() - 1) if len(bw) else None

    return StressScenario(
        name=name,
        kind="historical",
        description=description or f"{start} to {end}",
        portfolio_return=float((1 + port).prod() - 1),
        benchmark_return=bench,
        max_drawdown=m.max_drawdown(port),
        var_95=m.historical_var(port, 0.95),
        cvar_95=m.conditional_var(port, 0.95),
        worst_contributors={k: float(v) for k, v in worst.items()},
    )


def volatility_shock(
    weights: dict[str, float],
    asset_returns: pd.DataFrame,
    multiplier: float = 2.0,
) -> StressScenario:
    """Scale all return deviations by `multiplier`, holding means fixed.

    Preserves the correlation structure and inflates only dispersion, isolating
    the effect of a pure volatility regime change.
    """
    cols = [t for t in weights if t in asset_returns.columns]
    R = asset_returns[cols].dropna()
    shocked = R.mean() + (R - R.mean()) * multiplier
    port = _portfolio_series(weights, shocked)
    return StressScenario(
        name=f"vol_shock_{multiplier:g}x",
        kind="hypothetical",
        description=f"All return deviations scaled {multiplier:g}x (correlations unchanged)",
        portfolio_return=float((1 + port).prod() - 1),
        max_drawdown=m.max_drawdown(port),
        var_95=m.historical_var(port, 0.95),
        cvar_95=m.conditional_var(port, 0.95),
    )


def correlation_shock(
    weights: dict[str, float],
    asset_returns: pd.DataFrame,
    target_correlation: float = 0.85,
    seed: int = 0,
) -> StressScenario:
    """Raise every pairwise correlation to at least `target_correlation`.

    Implemented by shrinking the empirical correlation matrix toward an
    all-ones matrix, projecting back to PSD, and resampling returns from the
    shocked Gaussian. This is where "well-diversified" portfolios break.
    """
    cols = [t for t in weights if t in asset_returns.columns]
    R = asset_returns[cols].dropna()
    if R.shape[1] < 2 or len(R) < 60:
        return StressScenario(
            name="correlation_shock",
            kind="hypothetical",
            description="insufficient data",
            portfolio_return=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
        )
    sd = R.std(ddof=1).to_numpy()
    corr = R.corr().to_numpy()
    ones = np.ones_like(corr)
    np.fill_diagonal(ones, 1.0)
    # Shrink only where existing correlation is below the target.
    lam = float(np.clip(target_correlation, 0.0, 0.99))
    shocked_corr = np.maximum(corr, lam * ones)
    np.fill_diagonal(shocked_corr, 1.0)
    cov = m.nearest_psd(shocked_corr * np.outer(sd, sd))

    rng = np.random.default_rng(seed)
    sim = rng.multivariate_normal(R.mean().to_numpy(), cov, size=len(R))
    sim_df = pd.DataFrame(sim, index=R.index, columns=R.columns)
    port = _portfolio_series(weights, sim_df)
    return StressScenario(
        name=f"correlation_shock_{lam:g}",
        kind="hypothetical",
        description=(
            f"Pairwise correlations floored at {lam:g}, marginal volatilities "
            "unchanged; returns resampled from the shocked covariance"
        ),
        portfolio_return=float((1 + port).prod() - 1),
        max_drawdown=m.max_drawdown(port),
        var_95=m.historical_var(port, 0.95),
        cvar_95=m.conditional_var(port, 0.95),
    )


def market_shock(
    weights: dict[str, float],
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    shock: float = -0.20,
) -> StressScenario:
    """Instantaneous beta-driven market move.

    Loss = sum_i w_i * beta_i * shock. First-order only: it ignores the
    convexity and correlation shifts a real 20% drop would bring, so it should
    be read as a floor on the damage, not an estimate of it.
    """
    cols = [t for t in weights if t in asset_returns.columns]
    betas = {t: m.beta(asset_returns[t].dropna(), benchmark_returns) for t in cols}
    contrib = {
        t: float(weights[t] * (betas[t] if np.isfinite(betas[t]) else 1.0) * shock) for t in cols
    }
    total = float(sum(contrib.values()))
    return StressScenario(
        name=f"market_shock_{shock:+.0%}",
        kind="hypothetical",
        description=f"Benchmark moves {shock:+.0%}; assets respond via beta (first-order)",
        portfolio_return=total,
        benchmark_return=shock,
        max_drawdown=min(total, 0.0),
        var_95=total,
        cvar_95=total,
        worst_contributors=dict(sorted(contrib.items(), key=lambda kv: kv[1])[:5]),
    )


def sector_shock(
    weights: dict[str, float],
    sectors: dict[str, str],
    sector: str,
    shock: float = -0.30,
) -> StressScenario:
    """Single-sector drawdown, other sectors unaffected."""
    contrib = {t: float(w * shock) for t, w in weights.items() if sectors.get(t) == sector}
    total = float(sum(contrib.values()))
    return StressScenario(
        name=f"sector_shock_{sector}_{shock:+.0%}",
        kind="hypothetical",
        description=f"{sector} falls {shock:+.0%}, other sectors flat",
        portfolio_return=total,
        max_drawdown=min(total, 0.0),
        var_95=total,
        cvar_95=total,
        worst_contributors=dict(sorted(contrib.items(), key=lambda kv: kv[1])[:5]),
    )


def run_stress_suite(
    weights: dict[str, float],
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    sectors: dict[str, str] | None = None,
    historical: bool = True,
) -> list[StressScenario]:
    """Full suite. Historical windows are skipped when the sample does not
    cover them (e.g. synthetic data has no 2008)."""
    out: list[StressScenario] = []

    if historical:
        for name, (s, e, desc) in HISTORICAL_WINDOWS.items():
            sc = historical_replay(weights, asset_returns, s, e, name, desc, benchmark_returns)
            if sc is not None:
                out.append(sc)

    # Worst realised drawdown window in the available sample — always
    # applicable, including on synthetic data.
    port = _portfolio_series(weights, asset_returns)
    if len(port) > 60:
        dd = m.drawdown_series(port)
        trough = dd.idxmin()
        peak = port.loc[:trough].pipe(lambda s: (1 + s).cumprod()).idxmax()
        sc = historical_replay(
            weights,
            asset_returns,
            str(peak.date()),
            str(trough.date()),
            "worst_realised_drawdown",
            "Deepest peak-to-trough window present in the sample",
            benchmark_returns,
        )
        if sc is not None:
            out.append(sc)

    out.append(volatility_shock(weights, asset_returns, 2.0))
    out.append(correlation_shock(weights, asset_returns, 0.85))
    if benchmark_returns is not None:
        out.append(market_shock(weights, asset_returns, benchmark_returns, -0.20))
    if sectors:
        exposures: dict[str, float] = {}
        for t, w in weights.items():
            exposures[sectors.get(t, "Unknown")] = exposures.get(sectors.get(t, "Unknown"), 0.0) + w
        if exposures:
            worst = max(exposures, key=lambda k: exposures[k])
            out.append(sector_shock(weights, sectors, worst, -0.30))
    return out


def render_stress(scenarios: list[StressScenario]) -> str:
    lines = ["Stress Test Results", "=" * 84]
    lines.append(f"{'Scenario':<34}{'Return':>10}{'Bench':>10}{'MaxDD':>10}{'CVaR95':>10}")
    lines.append("-" * 84)
    for s in scenarios:
        bench = f"{s.benchmark_return:>9.1%}" if s.benchmark_return is not None else "      n/a"
        lines.append(
            f"{s.name:<34}{s.portfolio_return:>9.1%} {bench}"
            f"{s.max_drawdown:>10.1%}{s.cvar_95:>10.2%}"
        )
    lines.append("-" * 84)
    lines.append(
        "Conditional statements about historical/hypothetical return paths. " "Not forecasts."
    )
    return "\n".join(lines)
