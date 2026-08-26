"""Risk engine (Phase 10).

Assembles a full `RiskReport` for a portfolio: performance ratios, tail risk by
two independent methods, concentration, and an Euler risk decomposition.

Design notes
------------
* Historical **and** parametric VaR are always reported together. When they
  disagree materially the return distribution is non-Gaussian, which is
  information rather than an inconvenience.
* Risk decomposition uses the covariance matrix (analytic Euler
  decomposition), which sums exactly to portfolio volatility — a property
  asserted in the tests. Approximating it by re-running the portfolio without
  each asset does not add up and invites double-counting.
* Monte Carlo VaR is deliberately absent (Phase 10 guidance): with 15 years of
  daily scenarios, resampling adds simulation error without adding information
  the historical distribution does not already carry.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrisk.risk import metrics as m
from finrisk.schemas import RiskContribution, RiskReport

logger = logging.getLogger(__name__)


class RiskEngine:
    """Computes risk reports for weight vectors and return streams."""

    def __init__(self, periods_per_year: int = 252) -> None:
        self.ppy = periods_per_year

    # ------------------------------------------------------------------
    def analyze_returns(
        self,
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        risk_free_annual: pd.Series | float = 0.0,
        as_of: pd.Timestamp | None = None,
    ) -> RiskReport:
        """Risk report from a realised return stream alone."""
        r = returns.dropna()
        stamp = (
            pd.Timestamp(as_of)
            if as_of is not None
            else (r.index[-1] if len(r) else pd.Timestamp("1970-01-01"))
        )
        if len(r) < 5:
            return RiskReport(
                as_of=stamp.date(),
                n_observations=len(r),
                ann_volatility=0.0,
                ann_return=0.0,
                sharpe=0.0,
                sortino=0.0,
                max_drawdown=0.0,
                calmar=0.0,
                downside_deviation=0.0,
            )

        b = None if benchmark is None else benchmark.reindex(r.index)
        return RiskReport(
            as_of=stamp.date(),
            n_observations=len(r),
            ann_volatility=m.annualized_volatility(r, self.ppy),
            ann_return=m.annualized_return(r, self.ppy),
            sharpe=m.sharpe_ratio(r, risk_free_annual, self.ppy),
            sortino=m.sortino_ratio(r, risk_free_annual, self.ppy),
            max_drawdown=m.max_drawdown(r),
            calmar=m.calmar_ratio(r, self.ppy),
            downside_deviation=m.downside_deviation(r, 0.0, self.ppy),
            beta=None if b is None else m.beta(r, b),
            tracking_error=None if b is None else m.tracking_error(r, b, self.ppy),
            var_95_hist=m.historical_var(r, 0.95),
            var_99_hist=m.historical_var(r, 0.99),
            var_95_param=m.parametric_var(r, 0.95),
            var_99_param=m.parametric_var(r, 0.99),
            cvar_95=m.conditional_var(r, 0.95),
            cvar_99=m.conditional_var(r, 0.99),
        )

    # ------------------------------------------------------------------
    def analyze_portfolio(
        self,
        weights: dict[str, float],
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
        sectors: dict[str, str] | None = None,
        risk_free_annual: pd.Series | float = 0.0,
        as_of: pd.Timestamp | None = None,
        cov: pd.DataFrame | None = None,
    ) -> RiskReport:
        """Full report including position-level risk decomposition.

        `asset_returns` must contain only history available at `as_of`; the
        caller is responsible for slicing, and the backtester asserts it.
        """
        active = {k: v for k, v in weights.items() if abs(v) > 1e-9}
        tickers = [t for t in active if t in asset_returns.columns]
        if not tickers:
            return self.analyze_returns(
                pd.Series(dtype=float), benchmark_returns, risk_free_annual, as_of
            )

        w = np.array([active[t] for t in tickers], dtype=float)
        rets = asset_returns[tickers].dropna(how="all")
        port_ret = m.portfolio_returns_series(pd.Series(active), rets)

        report = self.analyze_returns(port_ret, benchmark_returns, risk_free_annual, as_of)

        cov_df = (
            cov.loc[tickers, tickers]
            if cov is not None
            else m.estimate_covariance(rets, "ledoit_wolf", self.ppy)
        )
        C = m.nearest_psd(cov_df.to_numpy())

        mrc = m.marginal_risk_contribution(w, C)
        crc = m.component_risk_contribution(w, C)
        pct = m.percent_risk_contribution(w, C)

        report.contributions = [
            RiskContribution(
                ticker=t,
                weight=float(w[i]),
                marginal_contribution=float(mrc[i]),
                component_contribution=float(crc[i]),
                pct_of_risk=float(pct[i]),
            )
            for i, t in enumerate(tickers)
        ]
        report.model_volatility = m.portfolio_volatility(w, C)
        report.herfindahl = m.herfindahl(w)
        report.effective_n = m.effective_n(w)
        report.mean_correlation = m.mean_pairwise_correlation(C)

        if sectors:
            agg: dict[str, float] = {}
            for i, t in enumerate(tickers):
                s = sectors.get(t, "Unknown")
                agg[s] = agg.get(s, 0.0) + float(pct[i])
            report.sector_risk = dict(sorted(agg.items(), key=lambda kv: -kv[1]))
        return report

    # ------------------------------------------------------------------
    def rolling_risk(self, returns: pd.Series, window: int = 126) -> pd.DataFrame:
        """Rolling volatility, VaR and CVaR — the time-series view of risk that
        a single full-sample number hides."""
        r = returns.dropna()
        out = pd.DataFrame(index=r.index)
        out["vol_ann"] = r.rolling(window).std(ddof=1) * np.sqrt(self.ppy)
        out["var_95"] = r.rolling(window).quantile(0.05)
        out["cvar_95"] = r.rolling(window).apply(
            lambda x: x[x <= np.quantile(x, 0.05)].mean(), raw=True
        )
        out["drawdown"] = m.drawdown_series(r)
        return out

    # ------------------------------------------------------------------
    def risk_budget_check(
        self, report: RiskReport, max_single_name_risk: float = 0.35
    ) -> list[str]:
        """Flag concentration that a weight cap alone does not catch.

        A 20% weight cap does not stop one high-volatility name contributing
        half the portfolio's risk, which is why the check is on risk share
        rather than weight.
        """
        out = []
        for c in report.contributions:
            if c.pct_of_risk > max_single_name_risk:
                out.append(
                    f"{c.ticker} contributes {c.pct_of_risk:.1%} of portfolio risk "
                    f"on a {c.weight:.1%} weight (limit {max_single_name_risk:.0%})"
                )
        if report.effective_n and report.effective_n < 5:
            out.append(
                f"effective number of positions is {report.effective_n:.1f} "
                "— portfolio is concentrated"
            )
        return out
