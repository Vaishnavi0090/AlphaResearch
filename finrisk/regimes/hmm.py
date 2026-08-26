"""Gaussian Hidden Markov Model, fitted by Expectation-Maximisation.

WHY AN HMM (and not clustering, and not a vol threshold)
-------------------------------------------------------
Phase 6 asks for ONE defensible method. An HMM is chosen because market
regimes have three properties that an HMM models natively and k-means does not:

1. **Persistence.** Regimes last months, not days. The transition matrix
   encodes this directly and yields an expected duration `1/(1-a_ii)`.
   k-means treats each day as i.i.d. and flips state on noise.
2. **Probabilistic output.** "72% bear" is actionable for scaling risk;
   a hard cluster label is not.
3. **Filtering.** The forward algorithm gives P(state_t | data up to t), which
   is exactly the point-in-time object a backtest is allowed to use.

That last point is the reason this is implemented rather than imported. Most
HMM libraries expose `predict()`, which runs Viterbi/smoothing over the WHOLE
sample — every historical label is informed by future data. Using that in a
backtest is look-ahead bias that is nearly invisible because the code looks
innocent. Here, `filter_states()` is forward-only and is the only accessor the
backtester is allowed to call.

Diagonal covariances are used: with 2-3 features and 3 states the full
covariance adds parameters without changing the state assignments materially,
and it can go singular on quiet sub-samples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

logger = logging.getLogger(__name__)

_LOG_2PI = float(np.log(2.0 * np.pi))


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp.

    Hand-rolled rather than imported from scipy: the forward-backward
    recursions are inherently sequential, so this is called O(T) times per EM
    iteration on tiny (K,) and (K,K) arrays where scipy's per-call dispatch
    overhead dominates the arithmetic by an order of magnitude.
    """
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return out if keepdims else np.squeeze(out, axis=axis) if axis is not None else out.reshape(())


@dataclass
class HMMParams:
    start_log_prob: np.ndarray  # (K,)
    log_transition: np.ndarray  # (K, K)
    means: np.ndarray  # (K, D)
    variances: np.ndarray  # (K, D) diagonal
    n_iter: int = 0
    log_likelihood: float = float("-nan")
    converged: bool = False
    # Log-likelihood after each EM iteration. EM guarantees this sequence is
    # monotonically non-decreasing; retaining it makes that guarantee testable
    # rather than assumed (see tests/test_regimes.py).
    ll_history: list[float] = dataclass_field(default_factory=list)

    @property
    def n_states(self) -> int:
        return self.means.shape[0]

    @property
    def transition(self) -> np.ndarray:
        return np.exp(self.log_transition)

    def expected_durations(self) -> np.ndarray:
        """1 / (1 - a_ii) in periods."""
        diag = np.clip(np.diag(self.transition), 0.0, 1.0 - 1e-9)
        return 1.0 / (1.0 - diag)


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM trained with Baum-Welch (EM)."""

    def __init__(
        self,
        n_states: int = 3,
        max_iter: int = 120,
        tol: float = 1e-5,
        seed: int = 0,
        min_variance: float = 1e-8,
    ) -> None:
        self.n_states = int(n_states)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.min_variance = float(min_variance)
        self.params_: HMMParams | None = None

    # ------------------------------------------------------------------
    def _log_emission(self, X: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
        """(T, K) log N(x_t; mu_k, diag(var_k))."""
        var = np.maximum(var, self.min_variance)
        # (T,1,D) - (1,K,D)
        diff = X[:, None, :] - means[None, :, :]
        quad = np.sum(diff**2 / var[None, :, :], axis=2)
        norm = np.sum(np.log(var), axis=1) + X.shape[1] * _LOG_2PI
        return -0.5 * (quad + norm[None, :])

    # ------------------------------------------------------------------
    def _init_params(self, X: np.ndarray) -> HMMParams:
        """Initialise means by quantiles of the first feature.

        Quantile seeding (rather than random) makes the fit deterministic and
        makes state 0 reliably the low end of feature 0, which keeps the
        semantic labelling stable across refits.
        """
        K, D = self.n_states, X.shape[1]
        qs = np.linspace(0.15, 0.85, K)
        means = np.empty((K, D))
        for d in range(D):
            means[:, d] = np.quantile(X[:, d], qs)
        var = np.tile(np.maximum(X.var(axis=0, ddof=1), self.min_variance), (K, 1))
        # Persistent prior: 0.95 on the diagonal.
        A = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(A, 0.95)
        return HMMParams(
            start_log_prob=np.log(np.full(K, 1.0 / K)),
            log_transition=np.log(A),
            means=means,
            variances=var,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _forward(
        log_e: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """alpha_t(k) = log P(x_1..x_t, state_t=k)."""
        T = log_e.shape[0]
        la = np.empty_like(log_e)
        la[0] = log_pi + log_e[0]
        At = log_A.T  # (j, i) so the reduction is over the last axis
        for t in range(1, T):
            m = la[t - 1].max()
            la[t] = np.log(np.exp(At + (la[t - 1] - m)).sum(axis=1)) + m + log_e[t]
        m = la[-1].max()
        return la, float(np.log(np.exp(la[-1] - m).sum()) + m)

    @staticmethod
    def _backward(log_e: np.ndarray, log_A: np.ndarray) -> np.ndarray:
        """beta_t(k) = log P(x_{t+1}..x_T | state_t=k)."""
        T = log_e.shape[0]
        lb = np.zeros_like(log_e)
        for t in range(T - 2, -1, -1):
            v = log_e[t + 1] + lb[t + 1]
            m = v.max()
            lb[t] = np.log(np.exp(log_A + (v - m)[None, :]).sum(axis=1)) + m
        return lb

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> GaussianHMM:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (T, D)")
        T = X.shape[0]
        if 20 * self.n_states > T:
            raise ValueError(f"need >= {20 * self.n_states} observations, got {T}")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")

        p = self._init_params(X)
        prev_ll = -np.inf
        history: list[float] = []

        for it in range(1, self.max_iter + 1):
            log_e = self._log_emission(X, p.means, p.variances)
            la, ll = self._forward(log_e, p.start_log_prob, p.log_transition)
            lb = self._backward(log_e, p.log_transition)

            # gamma_t(k) = P(state_t = k | X)
            log_gamma = la + lb
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # xi_t(i,j) = P(state_t=i, state_{t+1}=j | X), summed over t.
            # Vectorised over the time axis: the naive Python loop over T
            # dominates fit time (it is O(T) logaddexp calls per EM iteration,
            # i.e. ~500k numpy dispatches for a 15-year daily sample).
            fwd = la[:-1, :, None]  # (T-1, K, 1)
            bwd = (log_e[1:] + lb[1:])[:, None, :]  # (T-1, 1, K)
            acc = logsumexp(fwd + bwd, axis=0) + p.log_transition - ll

            # M-step
            log_A = acc - logsumexp(acc, axis=1, keepdims=True)
            start = np.log(np.maximum(gamma[0], 1e-300))
            start -= logsumexp(start)

            nk = gamma.sum(axis=0)  # (K,)
            nk_safe = np.maximum(nk, 1e-12)
            means = (gamma.T @ X) / nk_safe[:, None]
            diff2 = (X[:, None, :] - means[None, :, :]) ** 2
            variances = np.einsum("tk,tkd->kd", gamma, diff2) / nk_safe[:, None]
            variances = np.maximum(variances, self.min_variance)

            history.append(ll)
            p = HMMParams(
                start_log_prob=start,
                log_transition=log_A,
                means=means,
                variances=variances,
                n_iter=it,
                log_likelihood=ll,
                ll_history=list(history),
            )

            if it > 1 and abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                p.converged = True
                break
            prev_ll = ll

        if not p.converged:
            logger.debug("HMM hit max_iter=%d without converging", self.max_iter)
        self.params_ = p
        return self

    # ------------------------------------------------------------------
    def filter_states(self, X: np.ndarray) -> np.ndarray:
        """FILTERED posteriors P(state_t | x_1..x_t) — forward pass only.

        This is the ONLY method a backtest may use. It never conditions on
        observations after t.
        """
        if self.params_ is None:
            raise RuntimeError("model not fitted")
        p = self.params_
        X = np.atleast_2d(np.asarray(X, dtype=float))
        log_e = self._log_emission(X, p.means, p.variances)
        la, _ = self._forward(log_e, p.start_log_prob, p.log_transition)
        la -= logsumexp(la, axis=1, keepdims=True)
        return np.exp(la)

    def smoothed_states(self, X: np.ndarray) -> np.ndarray:
        """SMOOTHED posteriors P(state_t | all data). Uses future information —
        legitimate for *ex-post* regime description in a report, and forbidden
        in a backtest. Named explicitly so misuse is a conscious act."""
        if self.params_ is None:
            raise RuntimeError("model not fitted")
        p = self.params_
        X = np.atleast_2d(np.asarray(X, dtype=float))
        log_e = self._log_emission(X, p.means, p.variances)
        la, _ = self._forward(log_e, p.start_log_prob, p.log_transition)
        lb = self._backward(log_e, p.log_transition)
        lg = la + lb
        lg -= logsumexp(lg, axis=1, keepdims=True)
        return np.exp(lg)

    def score(self, X: np.ndarray) -> float:
        if self.params_ is None:
            raise RuntimeError("model not fitted")
        p = self.params_
        log_e = self._log_emission(np.asarray(X, float), p.means, p.variances)
        return self._forward(log_e, p.start_log_prob, p.log_transition)[1]

    def bic(self, X: np.ndarray) -> float:
        """Model selection for the number of states. K free params:
        transitions K(K-1) + starts (K-1) + means K*D + vars K*D."""
        X = np.asarray(X, float)
        T, D = X.shape
        K = self.n_states
        n_params = K * (K - 1) + (K - 1) + 2 * K * D
        return float(-2.0 * self.score(X) + n_params * np.log(T))
