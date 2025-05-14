"""
tiered_gp.py
============

Multi-layer Gaussian‑Process engine ("frequency‑layer" framework) for
instantaneous‑forward‑rate term‑structures.

A *tier* represents a liquidity layer: the most‑liquid instruments (e.g. 30 y
anchor, on‑the‑run swaps, …) are fitted first, their basis functions are frozen,
and residual illiquid constraints are calibrated on successive layers.

The posterior mean is the additive sum of per‑tier contributions; the posterior
covariance (optionally) follows the usual conditional‑GP update.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import numpy.linalg as npl

# --------------------------------------------------------------------- optional JIT
try:
    from numba import njit  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – run without numba

    def njit(*_args, **_kw):  # type: ignore
        def _decorator(func):
            return func

        return _decorator

# --------------------------------------------------------------------- local imports
# (imported lazily earlier – bring to top for clarity)
from gp.interpolators import get as get_interp  # registry helper

__all__ = [
    "TieredGP",
    "TierConfig",
    "USD_TIERS",
]


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

    knots: Sequence[float]  # knot locations for the interpolator
    constraints: Sequence[int]  # indices in the input vector
    interp: str = "cubic"  # registered interpolator name
    sigma: float | Sequence[float] = 0.0  # tension parameter(s) if relevant


USD_TIERS: list[TierConfig] = [
    TierConfig(knots=[30.0], constraints=[-1]),
    TierConfig(knots=[2.0, 5.0, 10.0], constraints=[0, 1, 2]),
    TierConfig(knots=[1.0, 7.0, 20.0], constraints=[3, 4, 5]),
    TierConfig(knots=[3.0, 15.0, 25.0], constraints=[6, 7, 8]),
]


# ---------------------------------------------------------------- GP core
class TieredGP:
    """Tiered Gaussian‑Process interpolator for IFR curves."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        times: Sequence[float],
        tiers: list[TierConfig] | None = None,
        *,
        prior: Literal["BM", "OU"] = "BM",
        hl: float = 5.0,
        store_posterior: bool = False,
        value_date: _dt.date | None = None,
    ):
        self.t = np.ascontiguousarray(times, float)
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("`times` must be strictly increasing.")

        self.tiers: list[TierConfig] = tiers or USD_TIERS
        self.prior, self.hl = prior, hl
        self._keep_cov = store_posterior
        self.value_date: _dt.date | None = value_date  # external assignment allowed

        self._Σ_post: np.ndarray | None = None  # posterior cov (if kept)
        self._alpha: list[np.ndarray] = []  # per‑tier weights
        self._basis_funcs: list[Callable[[np.ndarray], np.ndarray]] = []

        # placeholders filled by `.fit()`
        self._y_mkt: np.ndarray | None = None
        self._residual: np.ndarray | None = None

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
        """Sequentially calibrate the GP to *y_mkt* tier‑by‑tier."""
        y_mkt = np.ascontiguousarray(y_mkt, float)
        Σ = self._prior_cov()
        resid = y_mkt.copy()

        design = (lambda x: x) if design is None else design

        self._basis_funcs.clear()
        self._alpha.clear()

        for tier in self.tiers:
            # purely interpolatory layer -----------------------------------
            if not tier.constraints:
                interp_cls = get_interp(tier.interp)
                knots = np.asarray(tier.knots or self.t, float)
                zeros = np.zeros_like(knots)
                spl = interp_cls(knots, zeros, sigma=tier.sigma)
                self._basis_funcs.append(lambda x, s=spl: s(x))
                continue

            # --------------------------------------------------------------
            idx = np.asarray(tier.constraints, int)
            Λ = Σ[:, idx]  # n × m
            Σy = Σ[np.ix_(idx, idx)]  # m × m

            α = npl.solve(Σy, resid[idx])
            self._alpha.append(α.copy())

            contrib_train = Λ @ α
            resid -= design(contrib_train)

            # capture current values for closure
            contrib_train = contrib_train.copy()
            train_t = self.t.copy()

            def _bf(x, _t=train_t, _c=contrib_train):
                return np.interp(np.asarray(x, float), _t, _c)

            self._basis_funcs.append(_bf)

            if self._keep_cov:
                Σ -= Λ @ npl.solve(Σy, Λ.T)

        # store bookkeeping
        self._y_mkt = y_mkt
        self._residual = resid
        if self._keep_cov:
            self._Σ_post = Σ
        return self

    # ------------------------------------------------------------------ API accessors (added for utils)
    # These thin wrappers provide the attributes expected by utils.plots & calibration.

    # geometry
    @property
    def knots(self) -> np.ndarray:
        """Return training‑grid times (alias for *t*)."""
        return self.t

    # core prediction helpers
    def ifr(self, x: Sequence[float] | float) -> np.ndarray:
        """Instantaneous forward rate at *x* (vectorised)."""
        return self.predict(x)

    # liquidity masks (based on whether a tier has constraints)
    def liquid_knots(self) -> np.ndarray:
        idx = []
        for tier in self.tiers:
            if tier.constraints:
                idx.extend(tier.constraints)
        return self.t[np.asarray(sorted(set(idx)), int)] if idx else np.asarray([], float)

    def illiquid_knots(self) -> np.ndarray:
        liq = set(self.liquid_knots())
        return self.t[[i for i, x in enumerate(self.t) if x not in liq]]

    # simple par‑swap placeholder (should be replaced with proper pricing)
    def par_swap_rates(self) -> np.ndarray:
        """Mock par‑swap rates = IFR at knot locations (placeholder)."""
        return self.ifr(self.knots)

    def residual(self) -> np.ndarray:
        if self._residual is None:
            raise RuntimeError("Call fit(...) first to populate residuals.")
        return self._residual

    def posterior_var_scalar(self) -> float:
        if self._Σ_post is None:
            raise RuntimeError("Run fit(..., store_posterior=True) to keep Σ_post.")
        return float(np.mean(np.diag(self._Σ_post)))

    # ------------------------------------------------------------------ prediction / sampling
    def predict(self, grid: Sequence[float] | float) -> np.ndarray:
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

        # full kernel recomputation for arbitrary grid --------------------
        K = _brownian_cov(g) if self.prior == "BM" else _ou_cov(g, self.hl)
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
        """Draw IFR paths from the posterior (grid = training grid)."""
        if self._Σ_post is None:
            raise RuntimeError("Call fit(..., store_posterior=True) first.")
        rng = np.random.default_rng(random_state)
        L = npl.cholesky(self._Σ_post + 1e-16 * np.eye(len(self.t)))
        z = rng.standard_normal((n_path, len(self.t)))
        return (L @ z.T).T + self.predict(self.t)

    # ------------------------------------------------------------------ plot helper (compat with new utils.plots API)
    def plot(self, ax=None, **kw):
        """Quick visual – wraps ``utils.plots.curve``."""
        import utils.plots as _p  # local import to avoid cycles

        fig = _p.curve(self, ax=ax, **kw)
        return fig

    # ------------------------------------------------------------------ convenience persistence helpers
    def to_pickle(self) -> bytes:  # for backward‑compat with old CLI code
        import pickle

        return pickle.dumps(self)

