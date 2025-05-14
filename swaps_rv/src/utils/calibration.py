"""
utils.calibration
=================

Lightweight utilities that connect the Gaussian‑process curve (**TieredGP**)
with the residual ANN (**ResidualNet**).

This module is **risk‑metric free** – no DV01 or carry/roll helpers live here.
Everything is written in pure NumPy; `_JIT` turns into a no‑op when Numba
is not installed.

Key functions
-------------
residual_dataset(curves)  → (X, y)
    Build supervised pairs (GP features → residuals) for ANN training. Accepts
    either a single *TieredGP* or an *iterable* of them.

ann_features(gp) → 1‑D ndarray
    Feature engineering for a single calibrated GP snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable as _Iterable
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from gp.tiered_gp import TieredGP

# ---------------------------------------------------------------------------
# Lazy import of TieredGP at runtime (avoids circular dep in static checkers)
# ---------------------------------------------------------------------------
from gp.tiered_gp import TieredGP as _TieredGP  # noqa: E402

# ---------------------------------------------------------------------------
# Optional Numba JIT decorator
# ---------------------------------------------------------------------------
try:
    import numba as _nb  # type: ignore

    _JIT = _nb.njit(cache=True, fastmath=True)  # noqa: N818 – simple alias
except ModuleNotFoundError:  # pragma: no cover

    def _JIT(fn):  # type: ignore
        return fn

# public API ---------------------------------------------------------
__all__ = [
    "residual_dataset",
    "ann_features",
]

Array = np.ndarray

# ---------------------------------------------------------------------------
# Feature builders for the residual ANN
# ---------------------------------------------------------------------------

def residual_dataset(curves: _TieredGP | _Iterable[_TieredGP]) -> Tuple[Array, Array]:
    """Return (X, y) matrices derived from one or many *TieredGP* objects."""

    if isinstance(curves, _TieredGP):
        curves_iter: _Iterable[_TieredGP] = [curves]
    elif isinstance(curves, _Iterable):
        curves_iter = curves
    else:  # pragma: no cover
        raise TypeError("curves must be a TieredGP or an iterable of TieredGP")

    feats, resids = [], []
    for gp in curves_iter:
        feats.append(ann_features(gp))
        resids.append(gp.residual())  # illiquid‑tier residuals

    return np.stack(feats), np.stack(resids)


def ann_features(gp: _TieredGP) -> Array:  # noqa: D401 – simple helper
    """Compute Kondratyev‑style shape metrics plus posterior variance."""

    f_liq = gp.ifr(gp.liquid_knots())
    level = f_liq.mean()
    sf, sb = f_liq[1] - f_liq[0], f_liq[-1] - f_liq[-2]
    curvature = f_liq.std(ddof=0)
    sigma_lv = math.sqrt(gp.posterior_var_scalar())

    return np.concatenate([f_liq, np.array([level, sf, sb, curvature, sigma_lv])])

# ---------------------------------------------------------------------------
# Additional calibration helpers can be placed below
# ---------------------------------------------------------------------------
