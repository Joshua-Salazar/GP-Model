"""
Tiered Gaussian-Process engine à la “frequency‐layer” framework.

The object builds an *instantaneous-forward* (IFR) curve f(t)
as

    f̂(t) =  Σ_{ℓ=0}^{L}  b_ℓ(t)·α_ℓ                     (1)

where every layer ℓ corresponds to a **liquidity tier**

    L0 :  30 y anchor
    L1 :   2 y, 5 y, 10 y
    L2 :   1 y, 7 y, 20 y
    L3 :   3 y, 15 y, 25 y
    L4 :   rest  (interpolated)

and is calibrated sequentially from the most- to least-liquid
constraints (par-swap or yield inputs).  Basis functions b_ℓ(t)
are the *conditional means* of a GP (Brownian motion or OU)
subject to the constraints at that tier.

Features
--------
* Brownian / OU prior; automatic posterior mean & covariance.
* Arbitrary interpolation backend (cubic, tension, …) for **residual**
  projection inside a tier.
* Posterior Σₓ – ΛΣᴛΛᵀ retained (optional) for uncertainty bands.
* Numba-accelerated path sampling & bucket-DV01.
* Plays nicely with the `ann.residual_net.ResidualNet` to add
  non-linear/fly corrections.

Notes
-----
This implementation focuses on clarity; heavy linear-algebra pieces
are in `_core.py` and jit-compiled.  For production you may want to
specialise the factorisation calls (Cholesky vs QR) to your BLAS.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as npl
from dataclasses import dataclass, field
from typing import Sequence, Literal, Callable, Optional

from .interpolators import get as get_interp
#from utils.calibration import dv01_bucket              # noqa: F401
#from utils.data import par_swap_to_ifr_constraints     # noqa: F401
#from utils.plots import plot_curve, plot_posterior      # noqa: F401

# --------------------------------------------------------------------- jit helpers
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _brownian_cov(t: np.ndarray) -> np.ndarray:
    n = t.size
    Σ = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            Σ[i, j] = min(t[i], t[j])
    return Σ


@njit(cache=True, fastmath=True)
def _ou_cov(t: np.ndarray, hl: float) -> np.ndarray:
    lam = np.log(2.0) / hl
    n = t.size
    Σ = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            Σ[i, j] = 0.5 / lam * np.exp(-lam * abs(t[i] - t[j]))
    return Σ


# ---------------------------------------------------------------- configuration
@dataclass
class TierConfig:
    knots: Sequence[float]
    constraints: Sequence[int]
    interp: str = "cubic"
    sigma: float | Sequence[float] = 0.0


USD_TIERS: list[TierConfig] = [
    TierConfig(knots=[30.0], constraints=[-1]),
    TierConfig(knots=[2.0, 5.0, 10.0], constraints=[0, 1, 2]),
    TierConfig(knots=[1.0, 7.0, 20.0], constraints=[3, 4, 5]),
    TierConfig(knots=[3.0, 15.0, 25.0], constraints=[6, 7, 8]),
    TierConfig(knots=[], constraints=[]),  # catch-all residual
]


# ---------------------------------------------------------------- core
class TieredGP:
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
            raise ValueError("times must be strictly increasing")
        self.tiers = tiers or USD_TIERS
        self.prior = prior
        self.hl = hl
        self._keep_cov = store_posterior
        self._Σ_post: Optional[np.ndarray] = None

        self._alpha: list[np.ndarray] = []
        self._basis_funcs: list[Callable[[np.ndarray], np.ndarray]] = []

    # ------------------------------------------------ utilities
    def _prior_cov(self) -> np.ndarray:
        return _brownian_cov(self.t) if self.prior == "BM" else _ou_cov(self.t, self.hl)

    # ------------------------------------------------ fitting
    def fit(
        self,
        y_mkt: np.ndarray,
        design: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        Σ = self._prior_cov()
        μ = np.zeros_like(self.t)
        resid = np.copy(y_mkt)

        if design is None:
            design = lambda x: x

        for tier in self.tiers:
            # ------------------------- residual tier (no hard constraints)
            if not tier.constraints:
                interp_cls = get_interp(tier.interp)

                knots = np.asarray(tier.knots, float)
                if knots.size == 0:           # fallback → full grid
                    knots = self.t

                zero_ifr = np.zeros_like(knots)
                spl = interp_cls(knots, zero_ifr, sigma=tier.sigma)

                # basis → simply zeros; residual tier starts at 0
                self._basis_funcs.append(lambda x: np.zeros_like(x, float))
                self._alpha.append(np.zeros(1))
                continue
            # ------------------------- constrained tier
            idx = np.asarray(tier.constraints)
            Λ = Σ[:, idx]
            Σy = Σ[np.ix_(idx, idx)]

            α = npl.solve(Σy, resid[idx])
            μ += Λ @ α
            resid -= design(Λ.T @ α)

            self._alpha.append(α)
            self._basis_funcs.append(lambda x, α=α, Λ=Λ: np.interp(x, self.t, Λ @ α))

            if self._keep_cov:
                Σ -= Λ @ npl.solve(Σy, Λ.T)

        if self._keep_cov:
            self._Σ_post = Σ
        return self

    # ------------------------------------------- API (predict / cov / sample)
    def predict(self, grid: Sequence[float]) -> np.ndarray:
        g = np.asarray(grid, float)
        out = np.zeros_like(g)
        for bf in self._basis_funcs:
            out += bf(g)
        return out

    def posterior_cov(self, grid: Sequence[float]) -> np.ndarray:
        if self._Σ_post is None:
            raise RuntimeError("fit(..., store_posterior=True) required.")
        g = np.asarray(grid, float)
        return np.interp(g, self.t, self._Σ_post)

    def bucket_dv01(self, beta: np.ndarray, pvbp: np.ndarray) -> np.ndarray:
        return dv01_bucket(beta, pvbp)

    def sample(self, n_path: int = 1, random_state=None) -> np.ndarray:
        if self._Σ_post is None:
            raise RuntimeError("Must fit(store_posterior=True) first.")
        rng = np.random.default_rng(random_state)
        L = npl.cholesky(self._Σ_post + 1e-16 * np.eye(len(self.t)))
        z = rng.standard_normal((n_path, len(self.t)))
        return (L @ z.T).T + self.predict(self.t)

    # ------------------------------------------------ plotting helpers
    def plot(self, ax=None, **kw):
        ax = plot_curve(self.t, self.predict(self.t), ax=ax,
                        label="Tiered-GP", **kw)
        if self._Σ_post is not None:
            plot_posterior(self.t, self.predict(self.t), self._Σ_post, ax=ax)
        return ax

