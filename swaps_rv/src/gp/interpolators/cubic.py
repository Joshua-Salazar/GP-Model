"""
gp.interpolators.cubic
======================

Natural **C³ cubic‐spline** forward‐rate interpolation.

Constructs a *natural* cubic spline *f(t)* that passes through the instantaneous
forward‐rate knots  *(tᵢ, fᵢ)*.  Discount factors are obtained by analytical
integration of the piece‐wise cubic polynomial on each segment.

Implementation notes
--------------------
*  Uses **SciPy**’s `CubicSpline` when available (fast & well-tested).
*  Falls back to an internal *O(n)* tri-diagonal solver — good enough for
   typical knot counts (< 200) and fully NumPy/JAX compatible.
*  After building the spline we pre-compute the cumulative integral

        F(t) = ∫₀ᵗ f(u) du

   on every segment so that `discount(t)` is just `exp(-F(t))`.

Example
-------
>>> from fixed_income_rv.gp.interpolators import get
>>> knots = [0.5, 1, 2, 3, 5, 7, 10, 30]
>>> fwd   = [0.03, 0.032, 0.034, 0.035, 0.037, 0.039, 0.041, 0.042]
>>> cubic = get("cubic")(knots, fwd)
>>> cubic.discount(4.2)          # DF at 4.2y
"""

from __future__ import annotations

from typing import Sequence

try:  # JAX first
    import jax.numpy as _np  # type: ignore
except ModuleNotFoundError:  # noqa: E501 – fall back
    import numpy as _np  # type: ignore

# SciPy is optional – pure-NumPy backup is provided
try:
    from scipy.interpolate import CubicSpline as _SciCubic  # type: ignore
    _HAVE_SCIPY = True
except ModuleNotFoundError:  # pragma: no cover
    _HAVE_SCIPY = False


class _TriDiag:
    """
    Minimal *pure NumPy/JAX* C² natural cubic‐spline builder.

    Solves the classical tri-diagonal system to obtain second derivatives y″ᵢ.
    """

    @staticmethod
    def build(x: _np.ndarray, y: _np.ndarray):
        n = len(x)
        h = _np.diff(x)

        # right-hand side
        rhs = 6.0 * _np.diff((y[1:] - y[:-1]) / h)
        # tri-diagonal coefficients
        lower = h[:-1]
        diag = 2.0 * (h[:-1] + h[1:])
        upper = h[1:]

        # natural boundary: prepend/append zeros
        M = _np.zeros(n)
        if n > 2:
            # solve via Thomas algorithm (compatible with JAX)
            c = _np.empty_like(diag)
            d = _np.empty_like(rhs)

            c[0] = upper[0] / diag[0]
            d[0] = rhs[0] / diag[0]

            for i in range(1, n - 2):
                denom = diag[i] - lower[i - 1] * c[i - 1]
                c[i] = upper[i] / denom
                d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom

            # back-substitute
            M_inner = _np.empty(n - 2, dtype=y.dtype)
            M_inner[-1] = d[-1]
            for i in range(n - 4, -1, -1):
                M_inner[i] = d[i] - c[i] * M_inner[i + 1]

            M = M.at[1:-1].set(M_inner) if hasattr(M, "at") else _np.insert(
                M_inner, [0, len(M_inner)], 0.0
            )
        return M, h

    # ------------------------------------------------------------------ #

    @staticmethod
    def coeffs(x: _np.ndarray, y: _np.ndarray, M: _np.ndarray, h: _np.ndarray):
        """
        Returns piece-wise cubic polynomial coefficients such that on
        [xᵢ, xᵢ₊₁]:

            f(t) = aᵢ + bᵢ (t−xᵢ) + cᵢ (t−xᵢ)² + dᵢ (t−xᵢ)³
        """
        a = y[:-1]
        b = (y[1:] - y[:-1]) / h - h * (2 * M[:-1] + M[1:]) / 6.0
        c = M[:-1] / 2.0
        d = (M[1:] - M[:-1]) / (6.0 * h)
        return a, b, c, d


class CubicFwd:
    """Natural cubic‐spline forward interpolator."""

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

        # -----------------------------------------------------------------
        # Build spline representation
        # -----------------------------------------------------------------
        if _HAVE_SCIPY:
            cs = _SciCubic(self.t, self.f, bc_type="natural", axis=0)
            # SciPy already provides integral
            self._cs = cs
            self._F = cs.antiderivative()       # ∫ f dt piece-wise
        else:  # “poor-man’s” builder
            M, h = _TriDiag.build(self.t, self.f)
            a, b, c, d = _TriDiag.coeffs(self.t, self.f, M, h)
            self._poly = (a, b, c, d)

            # cumulative integral ∫₀^{tᵢ} f(u) du
            F = _np.zeros_like(self.t)
            for i in range(len(h)):
                a_i, b_i, c_i, d_i = a[i], b[i], c[i], d[i]
                dt = h[i]
                F = F.at[i + 1].set(
                    F[i] + a_i * dt + 0.5 * b_i * dt**2 + (1 / 3) * c_i * dt**3 + 0.25 * d_i * dt**4
                ) if hasattr(F, "at") else _np.concatenate(
                    (F[: i + 1], [F[i] + a_i * dt + 0.5 * b_i * dt**2 +
                                  (1 / 3) * c_i * dt**3 + 0.25 * d_i * dt**4])
                )
            self._Fgrid = F

    # ------------------------------------------------------------------ #
    def _primitive(self, t):
        """Compute ∫₀ᵗ f(u) du (supports vector input)."""
        t = _np.asarray(t, dtype=float)
        idx = _np.minimum(
            _np.searchsorted(self.t, t, side="right") - 1,
            len(self.t) - 2,
        )

        if _HAVE_SCIPY:
            return self._F(t)
        else:
            # manual polynomial integral
            a, b, c, d = self._poly
            dt = t - self.t[idx]
            return (
                self._Fgrid[idx]
                + a[idx] * dt
                + 0.5 * b[idx] * dt**2
                + (1 / 3) * c[idx] * dt**3
                + 0.25 * d[idx] * dt**4
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def discount(self, t):
        """DF(t) via log-integral of the cubic spline."""
        return _np.exp(-self._primitive(t))

    def discount_grid(self):
        """DF on original knots (copy)."""
        return _np.exp(-self._primitive(self.t))

    def ifr_grid(self):
        """Forward rates at knots (copy)."""
        return _np.array(self.f, copy=True)

    # alias
    __call__ = discount


# -------------------------------------------------------------------------
# Registry hook
# -------------------------------------------------------------------------

def factory(*args, **kwargs):
    return CubicFwd(*args, **kwargs)

# ---------------------------------------------------------------------------
# public alias expected by the registry
Interpolator=CubicFwd

