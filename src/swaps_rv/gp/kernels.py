"""
gp.kernels
==========

Stationary and non-stationary covariance kernels used by the Gaussian-process
layer.  Implementations are **JAX-compatible** (pure functions + `jax.numpy`)
but safely fall back to NumPy when JAX is unavailable.

All kernels expose the canonical signature

    k(x: ndarray[..., 1], y: ndarray[..., 1], θ: dict) -> ndarray[...].

The last two array dimensions are broadcast as usual so that batched evaluation
over points or hyper-parameter grids is trivial.

Examples
--------
>>> import numpy as np
>>> from fixed_income_rv.gp.kernels import SE, Brownian
>>> x = np.linspace(0, 30, 200)[:, None]
>>> K  = SE(x, x, {"σ": 0.03, "ℓ": 4.0})   # squared-exp (RBF) kernel
>>> K0 = Brownian(x, x, {"σ": 0.03})

---------------------------------------------------------------------------
NOTE:
  A very thin object-oriented wrapper (`Kernel` subclasses) lives in
  ``gp/_core.py`` for type safety and auto-diff; here we keep only stateless
  functional helpers so that users can call them directly.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict

try:  # use JAX if available
    import jax.numpy as np  # type: ignore
except ModuleNotFoundError:  # fall back to NumPy
    import numpy as np  # type: ignore

###############################################################################
# helpers
###############################################################################


def _as_col(x):
    """Ensure x is an (N,1) column array."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _pairwise_diffs(x, y):
    """
    Compute the (N,M) matrix |x_i - y_j| with broadcasting; works for batched
    arrays of shape (..., N, 1) and (..., M, 1).
    """
    return np.abs(x - np.swapaxes(y, -2, -1))


###############################################################################
# kernels
###############################################################################


def SE(x, y, θ: Dict[str, float]):
    """
    Squared-exponential (RBF) kernel

        k(r) = σ² * exp(-0.5 * r² / ℓ²).

    Parameters
    ----------
    θ : dict
        Must contain keys ``"σ"`` (amplitude) and ``"ℓ"`` (length-scale).
    """
    σ = θ["σ"]
    ell = θ["ℓ"]
    x, y = _as_col(x), _as_col(y)
    r = _pairwise_diffs(x / ell, y / ell)
    return σ * σ * np.exp(-0.5 * r * r)


def Matern52(x, y, θ: Dict[str, float]):
    """
    Matérn ν = 5/2 kernel

        k(r) = σ² * (1 + √5 r/ℓ + 5 r²/(3ℓ²)) * exp(-√5 r/ℓ)
    """
    σ = θ["σ"]
    ell = θ["ℓ"]
    x, y = _as_col(x), _as_col(y)
    r = _pairwise_diffs(x, y) / ell
    f = 1.0 + np.sqrt(5.0) * r + 5.0 * r * r / 3.0
    return σ * σ * f * np.exp(-np.sqrt(5.0) * r)


def Brownian(x, y, θ: Dict[str, float]):
    """
    Brownian-motion (Wiener) covariance

        k(t,s) = σ² * min(t, s)

    Works for any `x`, `y` >= 0 (interpreted as times).
    """
    σ = θ["σ"]
    x, y = _as_col(x), _as_col(y)
    return σ * σ * np.minimum(x, np.swapaxes(y, -2, -1))


def OU(x, y, θ: Dict[str, float]):
    """
    Ornstein-Uhlenbeck kernel (stationary, exponential)

        k(r) = σ² * exp(-r / ℓ)

    Equivalent to Matérn-½.
    """
    σ = θ["σ"]
    ell = θ["ℓ"]
    x, y = _as_col(x), _as_col(y)
    r = _pairwise_diffs(x, y)
    return σ * σ * np.exp(-r / ell)


###############################################################################
# registry – exposed via gp.__init__
###############################################################################

_REGISTRY = {
    "SE": SE,
    "RBF": SE,
    "Matern52": Matern52,
    "Brownian": Brownian,
    "OU": OU,
}


def get(name: str):
    """
    Retrieve a kernel by name (case-insensitive).

    >>> kfun = get("SE")
    """
    key = name.strip().lower()
    for k, v in _REGISTRY.items():
        if k.lower() == key:
            return v
    raise KeyError(f"kernel '{name}' not found. available: {list(_REGISTRY)}")


# tidy namespace
__all__ = ["SE", "Matern52", "Brownian", "OU", "get"]
