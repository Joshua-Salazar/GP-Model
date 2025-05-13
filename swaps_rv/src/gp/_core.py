"""
Low‑level numerics – JIT‑accelerated kernels & linear‑algebra snippets
used by the Tiered GP engine.

The helper keeps heavy Numba compilation off the import path for users
who *only* need analytical utilities; the module is lazy‑imported by
`tiered_gp.py` **iff** sampling or other JIT‑intensive functions are
called.

All routines are purely functional (no globals) so that they can be
heap‑allocated by Numba without relying on Python objects.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as npl  # kept for non‑JIT code paths; *inside* @njit we use np.linalg

# ---------------------------------------------------------------------------
# Optional Numba import – graceful degradation on lightweight installs
# ---------------------------------------------------------------------------

try:
    from numba import njit, prange  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional acceleration

    def njit(*args, **kwargs):  # noqa: D401 – dummy decorator
        """Fallback decorator when Numba is unavailable (no‑op)."""

        def _decorator(fn):  # pylint: disable=missing-docstring
            return fn

        return _decorator

    def prange(n: int):  # noqa: D401
        """Serial replacement for numba.prange."""

        return range(n)

# ---------------------------------------------------------------------------
# Covariance matrices
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def brownian_cov(t: np.ndarray) -> np.ndarray:  # noqa: D401 – clear naming
    """Return Σ_ij = min(t_i, t_j) for Brownian motion.

    Parameters
    ----------
    t : 1‑D float array (strictly increasing)

    Returns
    -------
    Σ : 2‑D float array  (len(t) × len(t))
    """
    n = t.size
    Σ = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            Σ[i, j] = t[i] if t[i] < t[j] else t[j]
    return Σ


@njit(cache=True, fastmath=True)
def ou_cov(t: np.ndarray, half_life: float) -> np.ndarray:  # noqa: D401
    """Ornstein–Uhlenbeck stationary covariance with half‑life ``half_life``.

    Σ_ij = (σ² / 2λ) · exp[‑λ |t_i−t_j|]  with λ = ln2 / h_l,  σ ≡ 1.

    The marginal variance is scaled to unity, matching Brownian limiting
    variance at small lags; rescale afterwards if different σ² is needed.
    """
    lam = np.log(2.0) / half_life
    n = t.size
    Σ = np.empty((n, n))
    for i in range(n):
        Σ[i, i] = 0.5 / lam  # |Δ| = 0
        for j in range(i + 1, n):
            v = 0.5 / lam * np.exp(-lam * (t[j] - t[i]))
            Σ[i, j] = v
            Σ[j, i] = v
    return Σ


# ---------------------------------------------------------------------------
# Conditional GP blocks
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def gp_conditional_mean(
    Σ_x: np.ndarray,  # (n, m)  Σ_x,y
    Σ_y: np.ndarray,  # (m, m)  Σ_y
    y: np.ndarray,  # (m,)     observed y
) -> np.ndarray:  # noqa: D401 – equation ref
    """Compute Λ Σ_y⁻¹ y  (eq. 10).

    Works for Brownian *and* OU – caller supplies matching covariance
    blocks.
    """
    α = np.linalg.solve(Σ_y, y)  # np.linalg works inside Numba
    return Σ_x @ α


@njit(cache=True, fastmath=True)
def gp_conditional_cov(
    Σ: np.ndarray,  # (n, n) prior full block
    Σ_x: np.ndarray,  # (n, m)
    Σ_y: np.ndarray,  # (m, m)
) -> np.ndarray:
    """Posterior covariance Σ_x − Λ Σ_y⁻¹ Λᵀ  (eq. 9)."""
    L = np.linalg.solve(Σ_y, Σ_x.T)  # Σ_y⁻¹ Λᵀ
    return Σ - Σ_x @ L



