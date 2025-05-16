"""
gp.interpolators.flat
=====================

Piece-wise **flat-forward** interpolation.

The curve is specified by a vector of *instantaneous forward rates* defined on
a knot grid t₀ < t₁ < … < tₙ. For any *t* in \[tᵢ, tᵢ₊₁) the IFR is taken to be
constant and equal to fᵢ.  Discount factors are therefore

    DF(t) = DF(tᵢ) · exp(-fᵢ · (t - tᵢ)).

Because the segment rate is constant the implementation is especially simple
and embarrassingly fast – no need for splines or linear solves.

The API matches the other back-ends so that users can hot-swap:

>>> from fixed_income_rv.gp.interpolators import get
>>> interp = get("flat")(knots, ifr)        # build once
>>> df = interp.discount(7.25)              # DF at 7.25 y
>>> grid = interp.discount_grid()           # DF on full grid

Notes
-----
* The class is *stateless* after construction: all heavy work is done in `__init__`.
* Vectorised over `numpy` **or** `jax.numpy` thanks to the local `_np` shim.
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    import jax.numpy as _np  # type: ignore
except ModuleNotFoundError:  # graceful fall-back
    import numpy as _np  # type: ignore


class FlatForward:
    """Piece-wise constant instantaneous-forward interpolator."""

    # --------------------------------------------------------------------- #
    def __init__(self, knots: Sequence[float], fwd: Sequence[float]):
        """
        Parameters
        ----------
        knots : 1-D array-like
            Strictly increasing knot locations (in **years**).
        fwd   : 1-D array-like
            Instantaneous forward rate attached to each segment.  Must have
            ``len(fwd) == len(knots)``.  The last entry is used beyond the
            final knot for simple log-linear extrapolation.
        """
        self.t = _np.asarray(knots, dtype=float)
        self.f = _np.asarray(fwd, dtype=float)

        if self.t.ndim != 1:
            raise ValueError("`knots` must be 1-D.")
        if self.f.shape != self.t.shape:
            raise ValueError("`fwd` must have same length as `knots`.")
        if _np.any(self.t[1:] <= self.t[:-1]):
            raise ValueError("`knots` must be strictly increasing.")

        # Pre-compute discount factors at knots: DF₀ = 1
        tau = _np.diff(_np.concatenate([_np.zeros(1), self.t]))
        self.df = _np.exp(-_np.cumsum(self.f * tau))

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def discount(self, t: float | _np.ndarray) -> _np.ndarray:
        """
        Discount factor **DF(t)** with flat-fwd convention.

        Supports scalar or vector input; broadcasting follows `numpy` rules.
        """
        t = _np.asarray(t, dtype=float)

        idx = _np.minimum(
            _np.searchsorted(self.t, t, side="right") - 1,
            len(self.t) - 1,
        )
        # piece-wise constant rate for the segment
        f_seg = self.f[idx]
        t_anchor = self.t[idx]
        df_anchor = self.df[idx]

        return df_anchor * _np.exp(-f_seg * (t - t_anchor))

    # ------------------------------------------------------------------ #

    def discount_grid(self) -> _np.ndarray:
        """Vector of DF evaluated at the original knot grid (copy)."""
        return _np.array(self.df, copy=True)

    # ------------------------------------------------------------------ #

    def ifr_grid(self) -> _np.ndarray:
        """Instantaneous forward rates _f_ at knots (copy)."""
        return _np.array(self.f, copy=True)

    # ------------------------------------------------------------------ #

    def __call__(self, t):
        """Alias for `discount` so the object behaves like a function."""
        return self.discount(t)


# -------------------------------------------------------------------------
# Registry hook used by gp.interpolators.__init__.py
# -------------------------------------------------------------------------

def factory(*args, **kwargs):
    """Entry-point required by the registry (`get("flat")`)."""
    return FlatForward(*args, **kwargs)

# ---------------------------------------------------------------------------
# public alias expected by the registry
Interpolator = FlatForward

