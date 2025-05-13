"""
gp.interpolators.linear
=======================

Piece-wise **linear** instantaneous-forward interpolator.

* On each interval *(tᵢ, tᵢ₊₁)* the forward rate is

        f(t) = fᵢ + sᵢ (t − tᵢ),
        sᵢ = (fᵢ₊₁ − fᵢ) / (tᵢ₊₁ − tᵢ).

* Discount factors follow from analytical integration ⇒ a quadratic
  primitive per segment.

This is the “classical” linear-IFR boot-strap used in many trading
systems, but wrapped into the same interface as the other spline
interpolators.
"""

from __future__ import annotations

from typing import Sequence

try:  # prefer JAX if present
    import jax.numpy as _np  # type: ignore
except ModuleNotFoundError:
    import numpy as _np  # type: ignore


class LinearFwd:
    """Piece-wise linear forward-rate curve."""

    # ------------------------------------------------------------------ #
    def __init__(self, knots: Sequence[float], fwd: Sequence[float]):
        self.t = _np.asarray(knots, dtype=float)
        self.f = _np.asarray(fwd, dtype=float)

        if self.t.ndim != 1:
            raise ValueError("`knots` must be 1-D.")
        if self.f.shape != self.t.shape:
            raise ValueError("`fwd` length mismatch.")
        if _np.any(self.t[1:] <= self.t[:-1]):
            raise ValueError("`knots` must be strictly increasing.")

        # slopes per segment
        h = _np.diff(self.t)
        self._s = _np.diff(self.f) / h          # slope
        self._h = h

        # cumulative primitive F(t) = ∫₀ᵗ f(u) du on knots
        F = _np.zeros_like(self.t)
        F_vals = F  # alias for JAX compat
        for i in range(len(h)):
            a = self.f[i]
            s = self._s[i]
            dt = h[i]
            F_vals = F_vals.at[i + 1].set(
                F_vals[i] + a * dt + 0.5 * s * dt**2
            ) if hasattr(F_vals, "at") else _np.concatenate(
                (F_vals[: i + 1], [F_vals[i] + a * dt + 0.5 * s * dt**2])
            )
        self._Fgrid = F_vals

    # ------------------------------------------------------------------ #
    def _primitive(self, t):
        """Compute ∫₀ᵗ f(u) du."""
        t = _np.asarray(t, dtype=float)
        idx = _np.minimum(
            _np.searchsorted(self.t, t, side="right") - 1,
            len(self.t) - 2,
        )

        dt = t - self.t[idx]
        return self._Fgrid[idx] + self.f[idx] * dt + 0.5 * self._s[idx] * dt**2

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def discount(self, t):
        """Return DF(t)."""
        return _np.exp(-self._primitive(t))

    def discount_grid(self):
        """Discount factors on the original knots."""
        return _np.exp(-self._Fgrid)

    def ifr_grid(self):
        """Forward rates on knots."""
        return _np.array(self.f, copy=True)

    __call__ = discount


# ------------------------------------------------------------------ #
# Registry entry point expected by gp.interpolators.__init__
# ------------------------------------------------------------------ #

def factory(*args, **kwargs):
    return LinearFwd(*args, **kwargs)
