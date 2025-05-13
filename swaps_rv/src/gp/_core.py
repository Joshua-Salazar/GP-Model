"""
Low-level numerics – JIT-accelerated kernels & linear-algebra snippets
used by the Tiered GP engine.

The point of this helper is to keep heavy Numba compilation out of the
import path for users who *only* want analytical utilities; the module is
lazy-imported by `tiered_gp.py` *iff* sampling or other JIT-intensive
functions are called.

All routines are purely functional (no globals) so that they can be
heap-allocated by Numba without relying on Python objects.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as npl

from numba import njit, prange

# ------------------------------------------------------------------ covariances


@njit(cache=True, fastmath=True)
def brownian_cov(t: np.ndarray) -> np.ndarray:
    """
    Σ_ij = min(t_i, t_j)   (Brownian motion).

    Parameters
    ----------
    t : 1-D float array  (strictly increasing)

    Returns
    -------
    Σ : 2-D float array  (len(t) × len(t))
    """
    n = t.size
    Σ = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            Σ[i, j] = t[i] if t[i] < t[j] else t[j]
    return Σ


@njit(cache=True, fastmath=True)
def ou_cov(t: np.ndarray, half_life: float) -> np.ndarray:
    """
    Ornstein–Uhlenbeck stationary covariance with half-life `half_life`.

    Σ_ij = (σ² / 2λ) · exp[-λ |t_i−t_j|]   with λ = ln2 / h_l, σ ≡ 1

    The marginal variance is scaled to unity so that the GP matches the
    Brownian limiting variance at small lags; rescaling can be applied on
    top if needed.
    """
    lam = np.log(2.0) / half_life
    n = t.size
    Σ = np.empty((n, n))
    for i in range(n):
        Σ[i, i] = 0.5 / lam                      # |Δ| = 0
        for j in range(i + 1, n):
            v = 0.5 / lam * np.exp(-lam * (t[j] - t[i]))
            Σ[i, j] = v
            Σ[j, i] = v
    return Σ


# ------------------------------------------------------------------ conditionals


@njit(cache=True, fastmath=True)
def gp_conditional_mean(
    Σ_x: np.ndarray,            # (n, m)  Σ_x,y
    Σ_y: np.ndarray,            # (m, m)  Σ_y
    y:  np.ndarray,             # (m,)    observed y
) -> np.ndarray:
    """
    Return Λ Σ_y⁻¹ y   (eq. 10 in the white-paper)  where Λ = Σ_x,y.

    Works for both Brownian & OU; user supplies the appropriate blocks.
    """
    α = npl.solve(Σ_y, y)
    return Σ_x @ α


@njit(cache=True, fastmath=True)
def gp_conditional_cov(
    Σ: np.ndarray,              # (n, n)  prior
    Σ_x: np.ndarray,
    Σ_y: np.ndarray,
) -> np.ndarray:
    """
    Posterior Σ_x − Λ Σ_y⁻¹ Λᵀ    (eq. 9).
    """
    L = npl.solve(Σ_y, Σ_x.T)      # Σ_y⁻¹ Λᵀ
    return Σ - Σ_x @ L


# ------------------------------------------------------------------ fast DV01


@njit(cache=True, parallel=True)
def dv01_bucket(beta: np.ndarray, pvbp: np.ndarray) -> np.ndarray:
    """
    Very simple bucket-by-bucket dollar value of a basis point for
    pre-scaled beta coefficients (Jacobian comes from autodiff if JAX is
    enabled).

    Parameters
    ----------
    beta :  (n_buckets,)  sensitivity of PV to β_i      (∂PV/∂β_i)
    pvbp :  (n_buckets,)  ∂β_i / ∂(1 bp)  of the underlying knot

    Returns
    -------
    dv01 : (n_buckets,)   contribution to ∂PV/∂(1 bp)
    """
    out = np.empty_like(beta)
    for i in prange(beta.size):
        out[i] = beta[i] * pvbp[i]
    return out
