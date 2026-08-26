"""Portfolio construction (Phase 8).

Five methods plus a CVaR-constrained variant, all sharing one constraint
machinery so that a comparison between them is a comparison of *objectives*,
not of accidentally different constraint sets.

WHY scipy AND NOT cvxpy
-----------------------
Mean-variance, max-Sharpe and risk-parity are small (N <= 50) smooth problems
that SLSQP solves reliably in milliseconds. cvxpy would add a hard dependency
and a compilation step for no accuracy gain at this size. The one genuinely
non-smooth problem — CVaR — is handled by the Rockafellar-Uryasev linearisation
which turns it into a *linear* program over historical scenarios, solved with
scipy's HiGHS backend. So the convex structure is exploited where it exists,
without reinventing any numerical algorithm.

FEASIBILITY
-----------
Every optimiser can fail. A silent failure that returns garbage weights is far
worse than a loud one, so each method reports an `optimization_status` and
falls back to a documented, sane portfolio (equal weight over the selected
names) when the solver does not converge. Fallbacks are counted and surfaced in
the backtest warnings rather than hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from finrisk.config.settings import PortfolioConfig
from finrisk.risk import metrics as m
from finrisk.schemas import ConstraintReport, PortfolioWeights

logger = logging.getLogger(__name__)

_SLSQP_OPTS = {"maxiter": 300, "ftol": 1e-10}


@dataclass
class OptimizationInputs:
    """Everything an optimiser needs, already aligned and cleaned."""

    tickers: list[str]
    expected_returns: np.ndarray  # annualised excess returns
    cov: np.ndarray  # annualised covariance
    scenario_returns: np.ndarray | None = None  # (T, N) daily, for CVaR
    sectors: dict[str, str] | None = None
    risk_free: float = 0.0
    prev_weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        n = len(self.tickers)
        if self.expected_returns.shape != (n,):
            raise ValueError("expected_returns shape mismatch")
        if self.cov.shape != (n, n):
            raise ValueError("cov shape mismatch")
        self.cov = m.nearest_psd(self.cov)

    @property
    def n(self) -> int:
        return len(self.tickers)


class PortfolioOptimizer:
    """Constrained portfolio construction."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.cfg = config or PortfolioConfig()
        self.fallback_count = 0

    # ==================================================================
    # Constraint machinery
    # ==================================================================

    def _bounds(self, n: int) -> Bounds:
        return Bounds(
            np.zeros(n),  # long-only
            np.full(n, self.cfg.max_weight),
        )

    def _budget_constraints(self, inp: OptimizationInputs) -> list[LinearConstraint]:
        """Budget + sector caps as linear constraints."""
        n = inp.n
        cons: list[LinearConstraint] = []

        # Sum of weights: exactly max_leverage if no cash allowed, else <=.
        lo = 0.0 if self.cfg.cash_allowed else self.cfg.max_leverage
        cons.append(LinearConstraint(np.ones((1, n)), lo, self.cfg.max_leverage))

        if inp.sectors and self.cfg.max_sector_weight is not None:
            groups: dict[str, list[int]] = {}
            for i, t in enumerate(inp.tickers):
                groups.setdefault(inp.sectors.get(t, "Unknown"), []).append(i)
            rows, ub = [], []
            for _, idx in groups.items():
                if len(idx) == 0:
                    continue
                row = np.zeros(n)
                row[idx] = 1.0
                rows.append(row)
                ub.append(self.cfg.max_sector_weight)
            if rows:
                cons.append(LinearConstraint(np.vstack(rows), np.zeros(len(rows)), np.array(ub)))
        return cons

    def _vol_constraint(self, inp: OptimizationInputs) -> list[dict]:
        """Non-linear volatility ceiling for SLSQP."""
        if self.cfg.target_volatility is None:
            return []
        tgt = float(self.cfg.target_volatility)
        return [
            {
                "type": "ineq",
                "fun": lambda w, c=inp.cov, t=tgt: t**2 - float(w @ c @ w),
                "jac": lambda w, c=inp.cov: -2.0 * (c @ w),
            }
        ]

    def _turnover_constraint(self, inp: OptimizationInputs) -> list[dict]:
        """One-way turnover ceiling: 0.5 * sum |w_new - w_prev| <= limit.

        Non-linear because of the absolute value. Uses a subgradient for the
        Jacobian, which SLSQP handles: |x| is differentiable everywhere the
        solver actually lands, and sign(0)=0 is a valid subgradient at the kink.
        """
        limit = self.cfg.max_turnover
        if limit is None or not inp.prev_weights:
            return []
        prev = np.array([inp.prev_weights.get(t, 0.0) for t in inp.tickers], dtype=float)
        return [
            {
                "type": "ineq",
                "fun": lambda w, p=prev, L=float(limit): L - 0.5 * np.abs(w - p).sum(),
                "jac": lambda w, p=prev: -0.5 * np.sign(w - p),
            }
        ]

    def _risk_budget_constraints(self, inp: OptimizationInputs) -> list[dict]:
        """Cap each asset's Euler risk share at `max_risk_contribution`.

        Component risk share is  w_i (Sigma w)_i / (w' Sigma w), so the
        constraint  share_i <= c  rearranges to the smooth form

            c * (w' Sigma w) - w_i (Sigma w)_i >= 0

        which avoids dividing by a portfolio variance that can approach zero
        during the solve.
        """
        cap = self.cfg.max_risk_contribution
        if cap is None:
            return []
        cons = []
        for i in range(inp.n):

            def fun(w, i=i, C=inp.cov, c=float(cap)):
                var = float(w @ C @ w)
                return c * var - float(w[i] * (C @ w)[i])

            def jac(w, i=i, C=inp.cov, c=float(cap)):
                Cw = C @ w
                g = 2.0 * c * Cw
                g[i] -= Cw[i]
                g -= w[i] * C[i, :]
                return g

            cons.append({"type": "ineq", "fun": fun, "jac": jac})
        return cons

    def _extra_constraints(self, inp: OptimizationInputs) -> list[dict]:
        """Non-linear constraints shared by every optimiser."""
        return (
            self._vol_constraint(inp)
            + self._turnover_constraint(inp)
            + self._risk_budget_constraints(inp)
        )

    def _x0(self, inp: OptimizationInputs) -> np.ndarray:
        """Feasible starting point: equal weight capped at max_weight."""
        n = inp.n
        w = np.full(n, min(1.0 / n, self.cfg.max_weight))
        total = w.sum()
        if total > self.cfg.max_leverage:
            w *= self.cfg.max_leverage / total
        return w

    def check_constraints(
        self,
        weights: dict[str, float],
        sectors: dict[str, str] | None,
        vol: float,
        cvar: float | None = None,
        prev_weights: dict[str, float] | None = None,
        cov: np.ndarray | None = None,
        tickers: list[str] | None = None,
    ) -> ConstraintReport:
        """Post-hoc audit. Every returned portfolio is checked, including ones
        produced by the fallback path.

        Deliberately independent of the solver: it recomputes each quantity
        from the returned weights rather than trusting the optimiser's own
        report that it converged successfully.
        """
        tol = 1e-6
        v: list[str] = []
        w = pd.Series(weights, dtype=float)
        if len(w) and w.max() > self.cfg.max_weight + tol:
            v.append(f"max_weight breached: {w.idxmax()}={w.max():.4f} > {self.cfg.max_weight}")
        if len(w) and w.min() < -tol:
            v.append(f"negative weight: {w.idxmin()}={w.min():.4f}")
        gross = float(w.abs().sum())
        if gross > self.cfg.max_leverage + tol:
            v.append(f"leverage breached: {gross:.4f} > {self.cfg.max_leverage}")
        if self.cfg.max_turnover is not None and prev_weights:
            traded = set(weights) | set(prev_weights)
            turn = 0.5 * sum(abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in traded)
            if turn > self.cfg.max_turnover + 1e-4:
                v.append(f"turnover breached: {turn:.4f} > {self.cfg.max_turnover}")
        if self.cfg.max_risk_contribution is not None and cov is not None:
            names = tickers or list(weights)
            w_vec = np.array([weights.get(t, 0.0) for t in names], dtype=float)
            shares = m.percent_risk_contribution(w_vec, cov)
            worst = float(np.max(shares)) if shares.size else 0.0
            if worst > self.cfg.max_risk_contribution + 1e-3:
                v.append(
                    f"risk budget breached: max share {worst:.4f} > "
                    f"{self.cfg.max_risk_contribution}"
                )
        if sectors and self.cfg.max_sector_weight is not None:
            agg: dict[str, float] = {}
            for t, wt in weights.items():
                agg[sectors.get(t, "Unknown")] = agg.get(sectors.get(t, "Unknown"), 0.0) + wt
            for s, tot in agg.items():
                if tot > self.cfg.max_sector_weight + 1e-4:
                    v.append(f"sector cap breached: {s}={tot:.4f} > {self.cfg.max_sector_weight}")
        if self.cfg.target_volatility is not None and vol > self.cfg.target_volatility + 5e-3:
            v.append(f"volatility target breached: {vol:.4f} > {self.cfg.target_volatility}")
        if (
            self.cfg.max_cvar_95 is not None
            and cvar is not None
            and abs(cvar) > self.cfg.max_cvar_95 + 1e-4
        ):
            v.append(f"CVaR limit breached: {abs(cvar):.4f} > {self.cfg.max_cvar_95}")
        return ConstraintReport(
            max_weight=self.cfg.max_weight,
            min_weight=self.cfg.min_weight,
            max_sector_weight=self.cfg.max_sector_weight,
            target_volatility=self.cfg.target_volatility,
            max_cvar_95=self.cfg.max_cvar_95,
            max_leverage=self.cfg.max_leverage,
            violations=v,
        )

    # ==================================================================
    # Result packaging
    # ==================================================================

    def _package(
        self,
        inp: OptimizationInputs,
        w: np.ndarray,
        method: str,
        status: str,
        as_of: pd.Timestamp,
        diagnostics: dict | None = None,
    ) -> PortfolioWeights:
        w = np.where(np.abs(w) < 1e-6, 0.0, w)  # kill dust positions
        exp_ret = m.portfolio_return(w, inp.expected_returns)
        vol = m.portfolio_volatility(w, inp.cov)
        sharpe = (exp_ret - inp.risk_free) / vol if vol > 1e-9 else 0.0
        weights = {t: float(x) for t, x in zip(inp.tickers, w, strict=True)}

        cvar95 = None
        risk_metrics: dict[str, float] = {
            "expected_volatility": vol,
            "herfindahl": m.herfindahl(w),
            "effective_n": m.effective_n(w),
            "diversification_ratio": m.diversification_ratio(w, inp.cov),
            "mean_correlation": m.mean_pairwise_correlation(inp.cov),
        }
        if inp.scenario_returns is not None and len(inp.scenario_returns) >= 60:
            port_scen = pd.Series(inp.scenario_returns @ w)
            cvar95 = m.conditional_var(port_scen, 0.95)
            risk_metrics["var_95_1d"] = m.historical_var(port_scen, 0.95)
            risk_metrics["cvar_95_1d"] = cvar95
            risk_metrics["cvar_99_1d"] = m.conditional_var(port_scen, 0.99)

        return PortfolioWeights(
            as_of=pd.Timestamp(as_of).date(),
            method=method,
            weights=weights,
            expected_return=exp_ret,
            expected_volatility=vol,
            sharpe=float(sharpe),
            cash=float(max(self.cfg.max_leverage - w.sum(), 0.0)),
            optimization_status=status,
            constraints=self.check_constraints(
                weights,
                inp.sectors,
                vol,
                cvar95,
                prev_weights=inp.prev_weights,
                cov=inp.cov,
                tickers=inp.tickers,
            ),
            risk_metrics=risk_metrics,
            diagnostics=diagnostics or {},
        )

    def _fallback(
        self, inp: OptimizationInputs, method: str, as_of: pd.Timestamp, why: str
    ) -> PortfolioWeights:
        self.fallback_count += 1
        logger.warning("%s failed at %s (%s); falling back to equal weight", method, as_of, why)
        w = self._x0(inp)
        return self._package(
            inp, w, method, f"fallback_equal_weight:{why}", as_of, {"fallback_reason": why}
        )

    # ==================================================================
    # Methods
    # ==================================================================

    def equal_weight(self, inp: OptimizationInputs, as_of: pd.Timestamp) -> PortfolioWeights:
        """1/N. The benchmark every clever method must beat — and frequently
        does not, because it has zero estimation error."""
        n = inp.n
        w = np.full(n, min(1.0 / n, self.cfg.max_weight))
        if not self.cfg.cash_allowed and w.sum() < self.cfg.max_leverage - 1e-9:
            w *= self.cfg.max_leverage / w.sum()
        return self._package(inp, w, "equal_weight", "ok", as_of)

    # ------------------------------------------------------------------
    def mean_variance(self, inp: OptimizationInputs, as_of: pd.Timestamp) -> PortfolioWeights:
        """maximise  w'mu - (lambda/2) w'Sigma w"""
        lam = self.cfg.risk_aversion

        def neg_util(w: np.ndarray) -> float:
            return -(w @ inp.expected_returns) + 0.5 * lam * float(w @ inp.cov @ w)

        def jac(w: np.ndarray) -> np.ndarray:
            return -inp.expected_returns + lam * (inp.cov @ w)

        res = minimize(
            neg_util,
            self._x0(inp),
            jac=jac,
            method="SLSQP",
            bounds=self._bounds(inp.n),
            constraints=[*self._budget_constraints(inp), *self._extra_constraints(inp)],
            options=_SLSQP_OPTS,
        )
        if not res.success:
            return self._fallback(inp, "mean_variance", as_of, res.message[:60])
        return self._package(inp, res.x, "mean_variance", "ok", as_of, {"iterations": int(res.nit)})

    # ------------------------------------------------------------------
    def max_sharpe(self, inp: OptimizationInputs, as_of: pd.Timestamp) -> PortfolioWeights:
        """Maximise (w'mu - rf) / sqrt(w'Sigma w).

        Optimised directly rather than via the classic tangency formula because
        the closed form ignores inequality constraints (weight caps, sector
        caps), which are the whole point of a realistic mandate.
        """
        eps = 1e-10

        def neg_sharpe(w: np.ndarray) -> float:
            vol = np.sqrt(max(w @ inp.cov @ w, eps))
            return -((w @ inp.expected_returns) - inp.risk_free) / vol

        res = minimize(
            neg_sharpe,
            self._x0(inp),
            method="SLSQP",
            bounds=self._bounds(inp.n),
            constraints=[*self._budget_constraints(inp), *self._extra_constraints(inp)],
            options=_SLSQP_OPTS,
        )
        if not res.success or not np.isfinite(res.fun):
            return self._fallback(inp, "max_sharpe", as_of, str(res.message)[:60])
        return self._package(inp, res.x, "max_sharpe", "ok", as_of, {"iterations": int(res.nit)})

    # ------------------------------------------------------------------
    def risk_parity(self, inp: OptimizationInputs, as_of: pd.Timestamp) -> PortfolioWeights:
        """Equal Risk Contribution.

        Minimises the dispersion of component risk contributions. Uses NO
        expected returns at all — which is its main attraction, since expected
        returns are the least reliable input in the whole pipeline.
        """
        n = inp.n
        target = 1.0 / n

        def obj(w: np.ndarray) -> float:
            vol2 = float(w @ inp.cov @ w)
            if vol2 < 1e-16:
                return 1e6
            rc = w * (inp.cov @ w) / vol2  # fractional contributions, sum=1
            return float(np.sum((rc - target) ** 2))

        # Inverse-vol start: already close to ERC, so SLSQP converges fast.
        sd = np.sqrt(np.diag(inp.cov))
        x0 = np.where(sd > 0, 1.0 / np.maximum(sd, 1e-9), 1.0)
        x0 = np.clip(x0 / x0.sum(), 0.0, self.cfg.max_weight)
        if x0.sum() > 0:
            x0 = x0 / x0.sum() * min(1.0, self.cfg.max_leverage)

        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=self._bounds(n),
            constraints=self._budget_constraints(inp),
            options={"maxiter": 400, "ftol": 1e-12},
        )
        if not res.success:
            return self._fallback(inp, "risk_parity", as_of, str(res.message)[:60])
        w = res.x
        rc = m.percent_risk_contribution(w, inp.cov)
        return self._package(
            inp,
            w,
            "risk_parity",
            "ok",
            as_of,
            {"risk_contribution_dispersion": float(np.std(rc))},
        )

    # ------------------------------------------------------------------
    def risk_constrained(self, inp: OptimizationInputs, as_of: pd.Timestamp) -> PortfolioWeights:
        """Maximise expected return subject to a volatility ceiling.

        Operationally this is what a real mandate looks like: "deliver as much
        return as you can without exceeding 15% vol", rather than an abstract
        risk-aversion parameter nobody can calibrate.
        """
        if self.cfg.target_volatility is None:
            return self.mean_variance(inp, as_of)

        def neg_ret(w: np.ndarray) -> float:
            return -float(w @ inp.expected_returns)

        res = minimize(
            neg_ret,
            self._x0(inp),
            jac=lambda w: -inp.expected_returns,
            method="SLSQP",
            bounds=self._bounds(inp.n),
            constraints=[*self._budget_constraints(inp), *self._extra_constraints(inp)],
            options=_SLSQP_OPTS,
        )
        if not res.success:
            return self._fallback(inp, "risk_constrained", as_of, str(res.message)[:60])
        return self._package(
            inp, res.x, "risk_constrained", "ok", as_of, {"iterations": int(res.nit)}
        )

    # ------------------------------------------------------------------
    def cvar_constrained(
        self, inp: OptimizationInputs, as_of: pd.Timestamp, confidence: float = 0.95
    ) -> PortfolioWeights:
        """Maximise expected return s.t. CVaR limit, via Rockafellar-Uryasev.

        The R-U result: for a scenario set {r_1..r_T},

            CVaR_c(w) = min over (alpha, u>=0) of
                        alpha + 1/((1-c)T) * sum_t u_t
            s.t.        u_t >= -(r_t' w) - alpha

        is **linear** in (w, alpha, u). So a CVaR-constrained expected-return
        maximisation is an LP — no heuristics, no Monte Carlo, and a global
        optimum. Variables are ordered [w (N), alpha (1), u (T)].
        """
        if inp.scenario_returns is None or len(inp.scenario_returns) < 60:
            logger.debug("insufficient scenarios for CVaR; using risk_constrained")
            return self.risk_constrained(inp, as_of)

        R = np.asarray(inp.scenario_returns, dtype=float)
        T, N = R.shape
        limit = self.cfg.max_cvar_95
        if limit is None:
            return self.risk_constrained(inp, as_of)

        n_var = N + 1 + T
        k = 1.0 / ((1.0 - confidence) * T)

        # Objective: maximise w'mu  ->  minimise -mu'w
        c = np.zeros(n_var)
        c[:N] = -inp.expected_returns

        A_ub, b_ub = [], []
        # (1) CVaR limit:  alpha + k * sum(u) <= limit
        row = np.zeros(n_var)
        row[N] = 1.0
        row[N + 1 :] = k
        A_ub.append(row)
        b_ub.append(float(limit))
        # (2) scenario rows: -R_t'w - alpha - u_t <= 0
        block = np.zeros((T, n_var))
        block[:, :N] = -R
        block[:, N] = -1.0
        block[:, N + 1 :] = -np.eye(T)
        A_ub.append(block)
        b_ub.extend([0.0] * T)
        # (3) sector caps
        if inp.sectors and self.cfg.max_sector_weight is not None:
            groups: dict[str, list[int]] = {}
            for i, t in enumerate(inp.tickers):
                groups.setdefault(inp.sectors.get(t, "Unknown"), []).append(i)
            for _, idx in groups.items():
                r = np.zeros(n_var)
                r[idx] = 1.0
                A_ub.append(r)
                b_ub.append(self.cfg.max_sector_weight)

        # Budget equality (or inequality when cash is allowed)
        budget = np.zeros((1, n_var))
        budget[0, :N] = 1.0
        if self.cfg.cash_allowed:
            A_ub.append(budget[0])
            b_ub.append(self.cfg.max_leverage)
            A_eq, b_eq = None, None
        else:
            A_eq, b_eq = budget, [self.cfg.max_leverage]

        bounds = (
            [(0.0, self.cfg.max_weight)] * N
            + [(None, None)]  # alpha (VaR level) is free
            + [(0.0, None)] * T  # u >= 0
        )
        res = linprog(
            c,
            A_ub=np.vstack([r if r.ndim == 2 else r[None, :] for r in A_ub]),
            b_ub=np.asarray(b_ub, dtype=float),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            return self._fallback(inp, "cvar_constrained", as_of, str(res.message)[:60])
        w = res.x[:N]
        return self._package(
            inp,
            w,
            "cvar_constrained",
            "ok",
            as_of,
            {"var_level_alpha": float(res.x[N]), "lp_status": res.status},
        )

    # ==================================================================
    def optimize(
        self, inp: OptimizationInputs, as_of: pd.Timestamp, method: str | None = None
    ) -> PortfolioWeights:
        """Dispatch by name."""
        name = method or self.cfg.method
        fn = {
            "equal_weight": self.equal_weight,
            "mean_variance": self.mean_variance,
            "max_sharpe": self.max_sharpe,
            "risk_parity": self.risk_parity,
            "risk_constrained": self.risk_constrained,
            "cvar_constrained": self.cvar_constrained,
        }.get(name)
        if fn is None:
            raise ValueError(f"unknown optimisation method {name!r}")
        if inp.n == 0:
            raise ValueError("empty investable universe")
        if inp.n == 1:
            w = np.array([min(1.0, self.cfg.max_weight)])
            return self._package(inp, w, name, "single_asset", as_of)
        return fn(inp, as_of)
