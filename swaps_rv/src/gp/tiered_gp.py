"""
tiered_gp.py
============

Multi-layer Gaussian-Process engine (“frequency-layer” framework) for
instantaneous-forward-rate term-structures.

A *tier* represents a liquidity layer: the most liquid instruments (e.g. 30 y
anchor, on-the-run swaps, …) are fitted first, their basis functions are frozen,
and residual illiquid constraints are calibrated on successive layers.

The posterior mean is the additive sum of per-tier contributions; the posterior
covariance (optionally) follows the usual conditional-GP update.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as npl
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

# --------------------------------------------------------------------- optional JIT
try:
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover
    # graceful degradation if Numba is not available
    def njit(*_args, **_kw):
        def _decorator(func):
            return func

        return _decorator


@njit(cache=True, fastmath=True)
def _brownian_cov(t: np.ndarray) -> np.ndarray:
    n = t.size
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = min(t[i], t[j])
    return out


@njit(cache=True, fastmath=True)
def _ou_cov(t: np.ndarray, hl: float) -> np.ndarray:
    lam = np.log(2.0) / hl
    n = t.size
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = 0.5 / lam * np.exp(-lam * abs(t[i] - t[j]))
    return out


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class TierConfig:
    """Describe one liquidity layer."""

    knots: Sequence[float]            # knot locations for the interpolator
    constraints: Sequence[int]        # indices in the input vector
    interp: str = "cubic"             # registered interpolator name
    sigma: float | Sequence[float] = 0.0  # tension parameter(s) if relevant


USD_TIERS: list[TierConfig] = [
    TierConfig(knots=[30.0], constraints=[-1]),
    TierConfig(knots=[2.0, 5.0, 10.0], constraints=[0, 1, 2]),
    TierConfig(knots=[1.0, 7.0, 20.0], constraints=[3, 4, 5]),
    TierConfig(knots=[3.0, 15.0, 25.0], constraints=[6, 7, 8]),
]


# ---------------------------------------------------------------- GP core
class TieredGP:
    """
    Tiered Gaussian-Process interpolator for IFR curves.

    Parameters
    ----------
    times
        Strictly increasing training grid (years).
    tiers
        List of :class:`TierConfig`; defaults to *USD_TIERS*.
    prior
        `'BM'` Brownian motion or `'OU'` Ornstein–Uhlenbeck.
    hl
        Half-life for the OU prior (ignored if *prior* is `'BM'`).
    store_posterior
        If *True* retain Σ\_post for error bands / sampling.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        times: Sequence[float],
        tiers: list[TierConfig] | None = None,
        *,
        prior: Literal["BM", "OU"] = "BM",
        hl: float = 5.0,
        store_posterior: bool = False,
    ):
        self.t = np.ascontiguousarray(times, float)
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("`times` must be strictly increasing.")

        self.tiers: list[TierConfig] = tiers or USD_TIERS
        self.prior, self.hl = prior, hl
        self._keep_cov = store_posterior

        self._Σ_post: np.ndarray | None = None
        self._alpha: list[np.ndarray] = []
        self._basis_funcs: list[Callable[[np.ndarray], np.ndarray]] = []

    # ------------------------------------------------------------------ helpers
    def _prior_cov(self) -> np.ndarray:
        return _brownian_cov(self.t) if self.prior == "BM" else _ou_cov(self.t, self.hl)

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        y_mkt: np.ndarray,
        *,
        design: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "TieredGP":
        """
        Sequentially calibrate the GP to *y_mkt* tier-by-tier.

        *design* maps implied-IFR contributions (Λ α) into the observable
        metric (par-swap residuals, yields, etc.).  If *None*, the identity
        map is used.
        """
        y_mkt = np.ascontiguousarray(y_mkt, float)
        Σ = self._prior_cov()
        resid = y_mkt.copy()

        design = (lambda x: x) if design is None else design

        for tier in self.tiers:
            if not tier.constraints:  # purely interpolatory layer
                interp_cls = get_interp(tier.interp)
                knots = np.asarray(tier.knots or self.t, float)
                zeros = np.zeros_like(knots)
                spl = interp_cls(knots, zeros, sigma=tier.sigma)
                self._basis_funcs.append(lambda x, s=spl: s(x))
                continue

            # ----------------------------------------------------------------
            idx = np.asarray(tier.constraints, int)
            Λ = Σ[:, idx]                 # n × m
            Σy = Σ[np.ix_(idx, idx)]      # m × m

            α = npl.solve(Σy, resid[idx])
            self._alpha.append(α.copy())

            contrib_train = Λ @ α
            resid -= design(contrib_train)

            # capture the vector once to avoid late binding inside the lambda
            contrib_train = contrib_train.copy()

            def _bf(x, train_t=self.t, train_contrib=contrib_train):
                return np.interp(np.asarray(x, float), train_t, train_contrib)

            self._basis_funcs.append(_bf)

            if self._keep_cov:
                Σ -= Λ @ npl.solve(Σy, Λ.T)

        if self._keep_cov:
            self._Σ_post = Σ
        return self

    # ------------------------------------------------------------------ API
    def predict(self, grid: Sequence[float]) -> np.ndarray:
        g = np.asarray(grid, float)
        out = np.zeros_like(g)
        for bf in self._basis_funcs:
            out += bf(g)
        return out

    def posterior_cov(self, grid: Sequence[float]) -> np.ndarray:
        if self._Σ_post is None:
            raise RuntimeError("Call fit(..., store_posterior=True) first.")
        g = np.asarray(grid, float)
        if g.size == self.t.size and np.allclose(g, self.t):
            return self._Σ_post

        # full kernel recomputation for arbitrary grid
        K = (
            _brownian_cov(g)
            if self.prior == "BM"
            else _ou_cov(g, self.hl)
        )
        # project through the learnt conditioning terms
        Σ = K.copy()
        start = 0
        for tier, α in zip(self.tiers, self._alpha):
            m = len(tier.constraints)
            idx = start + np.arange(m)
            Λg = Σ[:, idx]
            Σy = Σ[np.ix_(idx, idx)]
            Σ -= Λg @ npl.solve(Σy, Λg.T)
            start += m
        return Σ

    def sample(self, n_path: int = 1, *, random_state=None) -> np.ndarray:
        """
        Draw IFR paths from the posterior (grid = training grid).
        """
        if self._Σ_post is None:
            raise RuntimeError("Call fit(..., store_posterior=True) first.")
        rng = np.random.default_rng(random_state)
        L = npl.cholesky(self._Σ_post + 1e-16 * np.eye(len(self.t)))
        z = rng.standard_normal((n_path, len(self.t)))
        return (L @ z.T).T + self.predict(self.t)

    # ------------------------------------------------------------------ plot helper
    def plot(self, ax=None, **kw):
        import utils.plots as _p

        fx = self.predict(self.t)
        ax = _p.plot_curve(self.t, fx, ax=ax, label="Tiered-GP", **kw)
        if self._Σ_post is not None:
            _p.plot_posterior(self.t, fx, self._Σ_post, ax=ax)
        return ax
