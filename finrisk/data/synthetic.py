"""Deterministic synthetic market simulator.

WHY THIS EXISTS
---------------
Two reasons, both methodological rather than cosmetic:

1. *Reproducibility.* A backtest whose inputs change every time a vendor
   restates history is not reproducible. The synthetic provider gives a fixed,
   seeded dataset so every number in `evaluation/` can be regenerated exactly,
   including in CI with no network access.

2. *Ground truth.* On real data you can never verify that a factor engine
   actually measures what it claims — you only observe that a backtest made
   money, which conflates signal, luck and bugs. Here the data-generating
   process (DGP) injects *known* value/quality/trend premia, so the factor
   engine can be checked for recovery of a known effect
   (`tests/test_factor_recovery.py`). If the engine cannot find a premium that
   is provably present, the engine is broken.

WHAT IT IS NOT
--------------
It is not a claim about real markets, and results measured on it are NOT
historical performance. See docs/LIMITATIONS.md.

DATA-GENERATING PROCESS
-----------------------
Market regime  s_t : 3-state Markov chain (bull / chop / bear-highvol)
Market return  m_t ~ N(mu_s, sigma_s)
Sector return  f_kt ~ N(0, sigma_sector)
Asset return   r_it = a_it + beta_i * m_t + gamma_i * f_k(i),t + eps_it

with a genuine cross-sectional premium

    a_it = k_trend * d_it + k_value * v_it + k_quality * q_it

d_it : AR(1) latent drift  -> induces return autocorrelation => momentum works
v_it : latent book-to-price -> mapped into reported book value => value works
q_it : latent profitability -> mapped into reported EPS       => quality works

Fundamentals are reported quarterly and stamped with a publication lag, so a
naive implementation that reads them on the fiscal period-end date will show
inflated performance — which is exactly the bug the leakage audit hunts for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.data.base import FUNDAMENTAL_COLUMNS, MarketDataBundle
from finrisk.schemas import AssetMeta

SECTORS = [
    "Technology",
    "Financials",
    "Healthcare",
    "Industrials",
    "Consumer",
    "Energy",
]

REGIME_NAMES = ["bull_low_vol", "chop", "bear_high_vol"]

# Annualised drift / vol per market regime, then converted to daily.
_REGIME_DRIFT_ANN = np.array([0.14, 0.02, -0.18])
_REGIME_VOL_ANN = np.array([0.11, 0.16, 0.34])

# Persistent regimes: mean duration ~ 1/(1-p_ii) trading days.
_TRANSITION = np.array(
    [
        [0.990, 0.008, 0.002],
        [0.020, 0.965, 0.015],
        [0.010, 0.040, 0.950],
    ]
)

# True premia (annualised excess return per 1 sd of the latent state).
TRUE_PREMIA_ANN = {"trend": 0.030, "value": 0.045, "quality": 0.035}


class SyntheticProvider:
    """Seeded synthetic market. Same seed => byte-identical data."""

    name = "synthetic"

    def __init__(self, seed: int = 7, n_assets: int = 40) -> None:
        self.seed = int(seed)
        self.n_assets = int(n_assets)

    # ------------------------------------------------------------------
    def default_universe(self) -> list[str]:
        return [f"SY{i:02d}" for i in range(self.n_assets)]

    # ------------------------------------------------------------------
    def fetch(
        self,
        universe: list[str] | None = None,
        start: str = "2010-01-04",
        end: str = "2024-12-31",
        benchmark: str = "BENCH",
    ) -> MarketDataBundle:
        rng = np.random.default_rng(self.seed)
        tickers = list(universe) if universe else self.default_universe()
        n = len(tickers)

        # Business-day calendar (a real provider would use an exchange calendar;
        # `pandas_market_calendars` is the drop-in upgrade, see DATA_PIPELINE.md)
        dates = pd.bdate_range(start=start, end=end)
        t_len = len(dates)
        if t_len < 300:
            raise ValueError("synthetic history too short; need >= 300 bdays")

        # -- static characteristics ------------------------------------
        sectors = [SECTORS[i % len(SECTORS)] for i in range(n)]
        betas = np.clip(rng.normal(1.0, 0.28, n), 0.35, 1.9)
        sector_load = np.clip(rng.normal(0.9, 0.2, n), 0.3, 1.6)
        idio_vol_ann = np.clip(rng.normal(0.22, 0.07, n), 0.08, 0.55)
        shares = rng.uniform(0.5, 6.0, n) * 1e9

        # -- market regime path ----------------------------------------
        states = np.empty(t_len, dtype=int)
        states[0] = 0
        for t in range(1, t_len):
            states[t] = rng.choice(3, p=_TRANSITION[states[t - 1]])

        drift_d = _REGIME_DRIFT_ANN[states] / 252.0
        vol_d = _REGIME_VOL_ANN[states] / np.sqrt(252.0)
        market = rng.normal(drift_d, vol_d)

        # -- sector factors --------------------------------------------
        sector_ret = rng.normal(0.0, 0.10 / np.sqrt(252.0), (t_len, len(SECTORS)))
        sec_idx = np.array([SECTORS.index(s) for s in sectors])

        # -- latent premium states -------------------------------------
        # AR(1) trend state: phi 0.985 -> half-life ~46 bdays
        trend = np.zeros((t_len, n))
        trend[0] = rng.normal(0, 1, n)
        phi_t = 0.985
        s_t = np.sqrt(1 - phi_t**2)
        for t in range(1, t_len):
            trend[t] = phi_t * trend[t - 1] + s_t * rng.normal(0, 1, n)

        # Value + quality states: much slower (fundamental characteristics)
        value = np.zeros((t_len, n))
        quality = np.zeros((t_len, n))
        value[0] = rng.normal(0, 1, n)
        quality[0] = rng.normal(0, 1, n)
        phi_s = 0.9985
        s_s = np.sqrt(1 - phi_s**2)
        for t in range(1, t_len):
            value[t] = phi_s * value[t - 1] + s_s * rng.normal(0, 1, n)
            quality[t] = phi_s * quality[t - 1] + s_s * rng.normal(0, 1, n)

        alpha = (
            TRUE_PREMIA_ANN["trend"] / 252.0 * trend
            + TRUE_PREMIA_ANN["value"] / 252.0 * value
            + TRUE_PREMIA_ANN["quality"] / 252.0 * quality
        )

        idio = rng.normal(0.0, (idio_vol_ann / np.sqrt(252.0))[None, :], (t_len, n))

        rets = (
            alpha
            + betas[None, :] * market[:, None]
            + sector_load[None, :] * sector_ret[:, sec_idx]
            + idio
        )
        rets[0] = 0.0

        # -- prices -----------------------------------------------------
        p0 = rng.uniform(20.0, 220.0, n)
        prices = pd.DataFrame(
            p0[None, :] * np.cumprod(1.0 + rets, axis=0),
            index=dates,
            columns=tickers,
        )

        base_vol = rng.uniform(3e5, 9e6, n)
        vol_mult = np.exp(rng.normal(0, 0.35, (t_len, n))) * (1.0 + 2.5 * np.abs(rets))
        volumes = pd.DataFrame(base_vol[None, :] * vol_mult, index=dates, columns=tickers).round()

        # -- benchmark: float-cap-weighted index of the universe --------
        caps = prices.to_numpy() * shares[None, :]
        w = caps / caps.sum(axis=1, keepdims=True)
        bench_ret = (w[:-1] * rets[1:]).sum(axis=1)
        bench = pd.Series(
            100.0 * np.cumprod(np.r_[1.0, 1.0 + bench_ret]), index=dates, name=benchmark
        )

        # -- risk-free: slow stochastic path, floored at zero -----------
        rf = 0.02 + np.cumsum(rng.normal(0, 0.00012, t_len))
        risk_free = pd.Series(np.clip(rf, 0.0, 0.09), index=dates, name="risk_free")

        # -- fundamentals ----------------------------------------------
        fundamentals = self._build_fundamentals(dates, prices, value, quality, tickers, shares, rng)

        meta = {
            tk: AssetMeta(ticker=tk, name=f"Synthetic Corp {tk}", sector=sc)
            for tk, sc in zip(tickers, sectors, strict=True)
        }

        # Ground truth kept for validation tests only; never used by the
        # factor engine or the backtester.
        truth = {
            "regime_path": pd.Series([REGIME_NAMES[s] for s in states], index=dates, name="regime"),
            "trend_state": pd.DataFrame(trend, index=dates, columns=tickers),
            "value_state": pd.DataFrame(value, index=dates, columns=tickers),
            "quality_state": pd.DataFrame(quality, index=dates, columns=tickers),
            "true_premia_ann": dict(TRUE_PREMIA_ANN),
            "betas": pd.Series(betas, index=tickers),
        }

        return MarketDataBundle(
            prices=prices,
            volumes=volumes,
            benchmark=bench,
            risk_free=risk_free,
            fundamentals=fundamentals,
            meta=meta,
            provider="synthetic",
            notes=[
                f"Synthetic DGP, seed={self.seed}, n_assets={n}.",
                "Results are DGP validation, NOT historical performance.",
                f"True annualised premia: {TRUE_PREMIA_ANN}",
            ],
            ground_truth=truth,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_fundamentals(
        dates: pd.DatetimeIndex,
        prices: pd.DataFrame,
        value: np.ndarray,
        quality: np.ndarray,
        tickers: list[str],
        shares: np.ndarray,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Quarterly fundamentals stamped with a realistic publication lag.

        The latent `value` state is mapped into book value per share so that
        book/price is a monotone function of the true value state; likewise
        `quality` maps into ROE. A reporting lag of 45-75 calendar days is
        applied so that point-in-time discipline is testable.
        """
        # Quarter ends present in the calendar
        q_ends = (
            pd.Series(dates, index=dates).groupby([dates.year, dates.quarter]).last().to_numpy()
        )
        rows: list[dict] = []
        pos = {d: i for i, d in enumerate(dates)}
        for qe in q_ends:
            qe = pd.Timestamp(qe)
            i = pos[qe]
            lag_days = int(rng.integers(45, 76))
            avail = qe + pd.Timedelta(days=lag_days)
            px = prices.iloc[i].to_numpy()
            # book/price ~ exp(0.35 * value_state) * 0.5  => P/B = 1/that
            bp = 0.50 * np.exp(0.35 * value[i])
            bvps = px * bp
            roe = np.clip(0.14 + 0.05 * quality[i], -0.05, 0.55)
            eps = bvps * roe
            for j, tk in enumerate(tickers):
                equity = bvps[j]
                debt = max(equity * float(np.clip(0.6 + 0.25 * rng.normal(), 0.0, 2.5)), 0.0)
                assets = equity + debt
                rows.append(
                    {
                        "ticker": tk,
                        "period_end": qe.date(),
                        "available_from": avail.date(),
                        "book_value_per_share": float(bvps[j]),
                        "eps": float(eps[j]),
                        "ebitda_per_share": float(eps[j] * 1.9),
                        "free_cash_flow_per_share": float(eps[j] * 0.85),
                        "revenue_per_share": float(bvps[j] * 1.4),
                        "total_debt_per_share": float(debt),
                        "total_equity_per_share": float(equity),
                        "net_income_per_share": float(eps[j]),
                        "total_assets_per_share": float(assets),
                    }
                )
        return pd.DataFrame(rows, columns=FUNDAMENTAL_COLUMNS)
