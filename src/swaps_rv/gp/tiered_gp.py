"""
tiered_gp.py
============

Multi‑layer Gaussian‑Process engine ("frequency‑layer" framework) for
instantaneous‑forward‑rate term‑structures.

A *tier* represents a liquidity layer: the most‑liquid instruments (e.g. 30 y
anchor, on‑the‑run swaps, …) are fitted first, their basis functions are frozen,
and residual illiquid constraints are calibrated on successive layers.

This version supports **optional linear constraints** at each tier.  If a tier
specifies a matrix *W* and vector *c*, calibration solves the Karush–Kuhn–Tucker
(KKT) block system

```
[ Σ_y   Wᵀ ] [α] = [r]
[  W    0 ] [λ]   [c]
```

where Σ_y = ΛᵀΣΛ is the centred prior covariance of the tier’s observed
instruments and *r* is the current residual.  Setting *W = None* (default)
reduces to plain GP conditioning.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import numpy.linalg as npl

try:
    from numba import njit  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – run without numba

    def njit(*_a, **_k):  # type: ignore
        def _decor(fn):
            return fn

        return _decor

# --------------------------------------------------------------------------
# NOTE: fixed intra-package import – use relative path
# --------------------------------------------------------------------------
from .interpolators import get as get_interp  # ← was “from gp.interpolators …”

__all__ = ["TieredGP", "TierConfig", "USD_TIERS"]


@njit(cache=True, fastmath=True)
def _brownian_cov(t: np.ndarray) -> np.ndarray:  # noqa: D401 – fast helper
    n = t.size
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = min(t[i], t[j])
    return out


@njit(cache=True, fastmath=True)
def _ou_cov(t: np.ndarray, hl: float) -> np.ndarray:  # noqa: D401
    lam = np.log(2.0) / hl
    n = t.size
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = 0.5 / lam * np.exp(-lam * abs(t[i] - t[j]))
    return out


# ---------------------------------------------------------------------------
# Tier description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierConfig:
    """Definition of one liquidity tier.

    Parameters
    ----------
    knots       : Knot locations passed to the interpolator.
    constraints : Indices of *y_mkt* that belong to this tier.
    interp      : Registered interpolator name (default ``"cubic"``).
    sigma       : Spline tension parameter (if the interpolator supports it).
    W, c        : Optional linear constraint *W α = c* (arrays).  ``W`` must have
                  shape (r, m) where *m = len(constraints)* and ``c`` shape (r,).
    """

    knots: Sequence[float]
    constraints: Sequence[int]
    interp: str = "cubic"
    sigma: float | Sequence[float] = 0.0
    W: np.ndarray | None = None
    c: np.ndarray | None = None


USD_TIERS: list[TierConfig] = [
    TierConfig(knots=[30.0], constraints=[-1]),
    TierConfig(knots=[2.0, 5.0, 10.0], constraints=[0, 1, 2]),
    TierConfig(knots=[1.0, 7.0, 20.0], constraints=[3, 4, 5]),
    TierConfig(knots=[3.0, 15.0, 25.0], constraints=[6, 7, 8]),
]


# ---------------------------------------------------------------------------
# Core GP class
# ---------------------------------------------------------------------------


class TieredGP:  # noqa: D101 – high-level class
    def __init__(
        self,
        times: Sequence[float],
        tiers: list[TierConfig] | None = None,
        *,
        prior: Literal["BM", "OU"] = "BM",
        hl: float = 5.0,
        store_posterior: bool = False,
        value_date: _dt.date | None = None,
    ) -> None:
        self.t = np.ascontiguousarray(times, float)
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("`times` must be strictly increasing.")

        self.tiers = tiers or USD_TIERS
        self.prior, self.hl = prior, hl
        self._keep_cov = store_posterior
        self.value_date = value_date

        self._Σ_post: np.ndarray | None = None
        self._alpha: list[np.ndarray] = []
        self._basis_funcs: list[Callable[[np.ndarray], np.ndarray]] = []

        self._y_mkt: np.ndarray | None = None
        self._residual: np.ndarray | None = None

    # ------------------------------------------------------------- helpers
    def _prior_cov(self) -> np.ndarray:  # noqa: D401 – quick selector
        return _brownian_cov(self.t) if self.prior == "BM" else _ou_cov(self.t, self.hl)

    # ------------------------------------------------------------- fitting
    def fit(
        self,
        y_mkt: np.ndarray,
        *,
        design: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "TieredGP":
        """Calibrate all tiers sequentially (optionally with linear constraints)."""
        y_mkt = np.ascontiguousarray(y_mkt, float)
        Σ = self._prior_cov()
        resid = y_mkt.copy()

        design = (lambda x: x) if design is None else design

        self._basis_funcs.clear()
        self._alpha.clear()

        for tier in self.tiers:
            # -------------------------------------------------- pure interpolator
            if not tier.constraints:
                interp_cls = get_interp(tier.interp)
                knots = np.asarray(tier.knots or self.t, float)
                spl = interp_cls(knots, np.zeros_like(knots), sigma=tier.sigma)
                self._basis_funcs.append(lambda x, s=spl: s(x))
                continue

            idx = np.asarray(tier.constraints, int)
            Λ = Σ[:, idx]  # n × m
            Σy = Σ[np.ix_(idx, idx)]  # m × m
            rhs = resid[idx]

            # -------------------------- optional KKT linear constraints W α = c
            if tier.W is not None and tier.c is not None and tier.W.size > 0:
                W = np.asarray(tier.W, float)
                c_vec = np.asarray(tier.c, float)
                m = Σy.shape[0]
                r = W.shape[0]
                kkt = np.block([[Σy, W.T], [W, np.zeros((r, r))]])
                sol = npl.solve(kkt, np.concatenate([rhs, c_vec]))
                α = sol[:m]
            else:
                α = npl.solve(Σy, rhs)

            self._alpha.append(α.copy())

            contrib_train = Λ @ α
            resid -= design(contrib_train)

            # closure capture for basis func
            contrib_train = contrib_train.copy()
            train_t = self.t.copy()

            def _bf(x, _t=train_t, _c=contrib_train):  # noqa: D401
                return np.interp(np.asarray(x, float), _t, _c)

            self._basis_funcs.append(_bf)

            if self._keep_cov:
                Σ -= Λ @ npl.solve(Σy, Λ.T)

        self._y_mkt = y_mkt
        self._residual = resid
        if self._keep_cov:
            self._Σ_post = Σ
        return self

    # ------------------------------------------------------------- public accessors
    @property
    def knots(self) -> np.ndarray:
        return self.t

    def ifr(self, x: Sequence[float] | float) -> np.ndarray:
        return self.predict(x)

    def liquid_knots(self) -> np.ndarray:
        idx: list[int] = []
        for tier in self.tiers:
            if tier.constraints:
                idx.extend(tier.constraints)
        return self.t[np.asarray(sorted(set(idx)), int)] if idx else np.asarray([], float)

    def illiquid_knots(self) -> np.ndarray:
        return np.setdiff1d(self.knots, self.liquid_knots(), assume_unique=True)

    def par_swap_rates(self) -> np.ndarray:  # placeholder
        return self.ifr(self.knots)

    def residual(self) -> np.ndarray:
        if self._residual is None:
            raise RuntimeError("Call fit() first to populate residuals.")
        return self._residual

    def posterior_var_scalar(self) -> float:
        if self._Σ_post is None:
            raise RuntimeError("Run fit(..., store_posterior=True) first.")
        return float(np.mean(np.diag(self._Σ_post)))

    # ------------------------------------------------------------- GP prediction
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
        if self._Σ_post is None:
            raise RuntimeError("Call fit(..., store_posterior=True) first.")
        rng = np.random.default_rng(random_state)
        L = npl.cholesky(self._Σ_post + 1e-16 * np.eye(len(self.t)))
        z = rng.standard_normal((n_path, len(self.t)))
        return (L @ z.T).T + self.predict(self.t)

    # ------------------------------------------------------------- misc helpers
    def plot(self, ax=None, **kw):  # noqa: D401 – thin wrapper
        import utils.plots as _p

        return _p.curve(self, ax=ax, **kw)

    def to_pickle(self) -> bytes:
        import pickle

        return pickle.dumps(self)
