"""Walk-forward backtesting engine (Phases 12-14).

THE PROTOCOL
------------
For each rebalance date `d`:

    1. Slice the bundle to <= d.                (information set)
    2. Read precomputed signals keyed at d.     (assert nothing later exists)
    3. Strategy proposes target weights.
    4. Execute at the close of d + execution_lag_days.
    5. Hold, letting weights drift with prices, until the next rebalance.

Three things this gets right that naive backtests get wrong:

* **Execution lag.** A signal computed from the close of day `d` cannot be
  traded at that same close. `execution_lag_days >= 1` is validated in config,
  so the loophole cannot be opened by accident.
* **Weight drift.** Between rebalances, positions are NOT held at constant
  weight. A winner grows into a larger share of the portfolio. Assuming
  constant weights implies free continuous rebalancing and both understates
  turnover and silently adds a mean-reversion bonus.
* **Costs on realised trades.** Cost is charged on the difference between the
  target and the *drifted* portfolio, not the previous target — which is the
  amount actually traded.

Cash earns the risk-free rate. Turnover is one-way
(`0.5 * sum|dw|`); cost is charged on the full traded notional (`sum|dw|`),
so the reported turnover and cost are internally consistent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finrisk.backtesting.strategies import (
    DecisionContext,
    ResearchContext,
    Strategy,
    build_research_context,
)
from finrisk.config.settings import Settings
from finrisk.data.base import MarketDataBundle
from finrisk.risk import metrics as m
from finrisk.schemas import (
    BacktestResult,
    PerformanceMetrics,
    RebalanceEvent,
)

logger = logging.getLogger(__name__)

BENCH_KEY = "__BENCHMARK__"


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------


def rebalance_dates(
    index: pd.DatetimeIndex, frequency: str = "M", warmup: int = 504
) -> list[pd.Timestamp]:
    """Last trading session of each period, after the warm-up.

    The warm-up exists because factors need up to 252 sessions of history and
    the covariance estimate needs another 252. Starting earlier would silently
    produce decisions from partially-formed signals.
    """
    if len(index) <= warmup + 5:
        raise ValueError(f"need > {warmup + 5} sessions for warmup={warmup}, got {len(index)}")
    usable = index[warmup:]
    s = pd.Series(usable, index=usable)
    if frequency == "M":
        key = [usable.year, usable.month]
    elif frequency == "Q":
        key = [usable.year, usable.quarter]
    elif frequency == "W":
        iso = usable.isocalendar()
        key = [iso["year"].to_numpy(), iso["week"].to_numpy()]
    else:
        raise ValueError(f"unsupported frequency {frequency!r}")
    dates = s.groupby(key).last().tolist()
    # Drop the final date: there would be no forward period to hold it.
    return sorted(pd.Timestamp(d) for d in dates)[:-1]


def config_digest(settings: Settings, strategy_name: str) -> str:
    payload = json.dumps(
        {"strategy": strategy_name, "settings": settings.model_dump(mode="json")},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


@dataclass
class _State:
    """Mutable portfolio state during the walk."""

    weights: dict[str, float] = field(default_factory=dict)
    cash: float = 1.0
    equity: float = 1.0

    def gross(self) -> float:
        return float(sum(self.weights.values()))


class WalkForwardBacktester:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cost_rate = settings.backtest.costs.cost_per_unit_turnover

    # ------------------------------------------------------------------
    def prepare_context(
        self, bundle: MarketDataBundle, dates: list[pd.Timestamp] | None = None
    ) -> ResearchContext:
        """Build (or reuse) the shared point-in-time signal cache."""
        dates = dates or rebalance_dates(
            bundle.dates,
            self.settings.backtest.rebalance_frequency,
            self.settings.backtest.warmup,
        )
        return build_research_context(bundle, dates, self.settings)

    # ------------------------------------------------------------------
    def run(
        self,
        bundle: MarketDataBundle,
        strategy: Strategy,
        context: ResearchContext | None = None,
    ) -> BacktestResult:
        bt = self.settings.backtest
        index = bundle.dates
        dates = (
            list(context.dates)
            if context is not None
            else rebalance_dates(index, bt.rebalance_frequency, bt.warmup)
        )
        if context is None:
            context = self.prepare_context(bundle, dates)

        asset_rets = bundle.prices.pct_change()
        bench_rets = bundle.benchmark.pct_change()
        rf_daily = bundle.risk_free / 252.0

        # Benchmark path is a special case: no signals, no costs.
        if getattr(strategy, "name", "") == "benchmark_buy_hold":
            return self._benchmark_result(strategy, bundle, dates, bench_rets)

        state = _State(weights={}, cash=1.0, equity=1.0)
        events: list[RebalanceEvent] = []
        warnings: list[str] = []
        equity: dict[pd.Timestamp, float] = {}

        start_pos = index.searchsorted(dates[0])
        exec_map: dict[int, pd.Timestamp] = {}
        for d in dates:
            p = index.searchsorted(d) + bt.execution_lag_days
            if p < len(index):
                exec_map[p] = d

        fallbacks = 0
        for pos in range(start_pos, len(index)):
            t = index[pos]

            # ---- 1. accrue the day's return on yesterday's holdings ----
            if state.weights or state.cash > 0:
                day_r = asset_rets.iloc[pos]
                inv_ret = sum(
                    w * float(day_r.get(tk, 0.0) if pd.notna(day_r.get(tk, np.nan)) else 0.0)
                    for tk, w in state.weights.items()
                )
                cash_ret = state.cash * float(rf_daily.iloc[pos] or 0.0)
                port_ret = inv_ret + cash_ret
                state.equity *= 1.0 + port_ret
                # ---- weight drift ----
                if abs(1.0 + port_ret) > 1e-12:
                    scale = 1.0 / (1.0 + port_ret)
                    state.weights = {
                        tk: w
                        * (
                            1.0
                            + float(day_r.get(tk, 0.0) if pd.notna(day_r.get(tk, np.nan)) else 0.0)
                        )
                        * scale
                        for tk, w in state.weights.items()
                    }
                    state.cash = state.cash * (1.0 + float(rf_daily.iloc[pos] or 0.0)) * scale
            equity[t] = state.equity

            # ---- 2. rebalance if today is an execution date ----
            if pos in exec_map:
                decision_date = exec_map[pos]
                target = self._decide(
                    bundle,
                    strategy,
                    context,
                    decision_date,
                    state,
                    asset_rets,
                    bench_rets,
                    rf_daily,
                )
                if target is None:
                    continue
                if "fallback" in target.optimization_status:
                    fallbacks += 1

                tgt = {k: v for k, v in target.weights.items() if abs(v) > 1e-8}
                traded = _traded_notional(state.weights, tgt)
                one_way = 0.5 * traded
                cost = traded * self.cost_rate

                state.equity *= 1.0 - cost
                state.weights = tgt
                state.cash = max(1.0 - sum(tgt.values()), 0.0)
                equity[t] = state.equity

                events.append(
                    RebalanceEvent(
                        decision_date=decision_date.date(),
                        execution_date=t.date(),
                        method=target.method,
                        weights=tgt,
                        turnover=one_way,
                        cost=cost,
                        regime=(
                            context.regimes[decision_date].regime
                            if decision_date in context.regimes
                            else None
                        ),
                        optimization_status=target.optimization_status,
                    )
                )

        if fallbacks:
            warnings.append(
                f"{fallbacks}/{len(events)} rebalances used the equal-weight " "optimiser fallback"
            )
        if not events:
            warnings.append("no rebalances executed")

        curve = pd.Series(equity).sort_index()
        returns = curve.pct_change().dropna()
        bench_curve = (1 + bench_rets.reindex(curve.index).fillna(0.0)).cumprod()

        metrics = self._metrics(
            returns,
            bench_rets.reindex(returns.index),
            events,
            bundle.risk_free.reindex(returns.index),
        )
        return BacktestResult(
            strategy=getattr(strategy, "name", "unnamed"),
            config_digest=config_digest(self.settings, getattr(strategy, "name", "")),
            metrics=metrics,
            n_rebalances=len(events),
            events=events,
            warnings=warnings,
            equity_curve={str(k.date()): float(v) for k, v in curve.items()},
            benchmark_curve={str(k.date()): float(v) for k, v in bench_curve.items()},
        )

    # ------------------------------------------------------------------
    def _decide(
        self,
        bundle: MarketDataBundle,
        strategy: Strategy,
        context: ResearchContext,
        d: pd.Timestamp,
        state: _State,
        asset_rets: pd.DataFrame,
        bench_rets: pd.Series,
        rf_daily: pd.Series,
    ):
        scores = context.factor_scores.get(d)
        if not scores:
            return None

        # Runtime leakage assertion: every score must be stamped at d.
        bad = [s.ticker for s in scores.values() if pd.Timestamp(s.as_of) > d]
        if bad:
            raise AssertionError(f"look-ahead: factor scores for {bad[:3]} dated after {d.date()}")

        trailing = asset_rets.loc[:d].iloc[1:]
        ctx = DecisionContext(
            as_of=d,
            bundle=bundle.slice(None, d),
            settings=self.settings,
            factor_scores=scores,
            regime=context.regimes.get(d),
            ic_estimate=context.ic_estimates.get(d, 0.03),
            prev_weights=dict(state.weights),
            trailing_returns=trailing,
            benchmark_returns=bench_rets.loc[:d].iloc[1:],
            risk_free=float(bundle.risk_free.loc[:d].iloc[-1]),
        )
        return strategy.decide(ctx)

    # ------------------------------------------------------------------
    def _benchmark_result(
        self,
        strategy: Strategy,
        bundle: MarketDataBundle,
        dates: list[pd.Timestamp],
        bench_rets: pd.Series,
    ) -> BacktestResult:
        start = dates[0]
        r = bench_rets.loc[start:].dropna()
        curve = (1 + r).cumprod()
        metrics = self._metrics(r, r, [], bundle.risk_free.reindex(r.index))
        return BacktestResult(
            strategy=strategy.name,
            config_digest=config_digest(self.settings, strategy.name),
            metrics=metrics,
            n_rebalances=0,
            events=[],
            warnings=["benchmark buy-and-hold: no trading, no costs"],
            equity_curve={str(k.date()): float(v) for k, v in curve.items()},
            benchmark_curve={str(k.date()): float(v) for k, v in curve.items()},
        )

    # ------------------------------------------------------------------
    def _metrics(
        self,
        returns: pd.Series,
        bench: pd.Series,
        events: list[RebalanceEvent],
        rf: pd.Series,
    ) -> PerformanceMetrics:
        r = returns.dropna()
        if len(r) < 5:
            zero = pd.Timestamp("1970-01-01").date()
            return PerformanceMetrics(
                start=zero,
                end=zero,
                n_days=0,
                total_return=0.0,
                cagr=0.0,
                ann_return=0.0,
                ann_volatility=0.0,
                sharpe=0.0,
                sortino=0.0,
                max_drawdown=0.0,
                calmar=0.0,
                var_95=0.0,
                cvar_95=0.0,
                hit_rate=0.0,
                avg_turnover=0.0,
                total_cost_drag=0.0,
            )
        turns = [e.turnover for e in events]
        costs = [e.cost for e in events]
        b = bench.reindex(r.index)
        return PerformanceMetrics(
            start=r.index.min().date(),
            end=r.index.max().date(),
            n_days=len(r),
            total_return=m.total_return(r),
            cagr=m.cagr(r),
            ann_return=m.annualized_return(r),
            ann_volatility=m.annualized_volatility(r),
            sharpe=m.sharpe_ratio(r, rf),
            sortino=m.sortino_ratio(r, rf),
            max_drawdown=m.max_drawdown(r),
            calmar=m.calmar_ratio(r),
            var_95=m.historical_var(r, 0.95),
            cvar_95=m.conditional_var(r, 0.95),
            hit_rate=m.hit_rate(r),
            avg_turnover=float(np.mean(turns)) if turns else 0.0,
            total_cost_drag=float(np.sum(costs)),
            beta=m.beta(r, b) if b.notna().sum() > 10 else None,
            information_ratio=(m.information_ratio(r, b) if b.notna().sum() > 10 else None),
        )


def _traded_notional(prev: dict[str, float], new: dict[str, float]) -> float:
    """sum |w_new - w_prev| across the union of holdings."""
    keys = set(prev) | set(new)
    return float(sum(abs(new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))


def summarize_results(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        mt = res.metrics
        rows.append(
            {
                "strategy": res.strategy,
                "CAGR": mt.cagr,
                "AnnRet": mt.ann_return,
                "AnnVol": mt.ann_volatility,
                "Sharpe": mt.sharpe,
                "Sortino": mt.sortino,
                "MaxDD": mt.max_drawdown,
                "Calmar": mt.calmar,
                "VaR95": mt.var_95,
                "CVaR95": mt.cvar_95,
                "Beta": mt.beta,
                "InfoRatio": mt.information_ratio,
                "AvgTurnover": mt.avg_turnover,
                "CostDrag": mt.total_cost_drag,
                "Rebalances": res.n_rebalances,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def render_comparison(df: pd.DataFrame) -> str:
    lines = ["Strategy Comparison", "=" * 118]
    hdr = (
        f"{'Strategy':<36}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'Sortino':>9}"
        f"{'MaxDD':>9}{'Calmar':>8}{'CVaR95':>9}{'Beta':>7}{'Turn':>7}{'Cost':>8}"
    )
    lines += [hdr, "-" * 118]
    for name, row in df.iterrows():
        beta = f"{row['Beta']:>6.2f}" if pd.notna(row["Beta"]) else "   n/a"
        lines.append(
            f"{name!s:<36}{row['CAGR']:>7.2%}{row['AnnVol']:>8.1%}"
            f"{row['Sharpe']:>8.2f}{row['Sortino']:>9.2f}{row['MaxDD']:>9.1%}"
            f"{row['Calmar']:>8.2f}{row['CVaR95']:>9.2%} {beta}"
            f"{row['AvgTurnover']:>7.1%}{row['CostDrag']:>8.2%}"
        )
    lines.append("-" * 118)
    lines.append(
        "Turn = mean one-way turnover per rebalance. Cost = cumulative "
        "transaction-cost drag over the full period."
    )
    return "\n".join(lines)
