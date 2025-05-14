"""
utils.calibration
=================

Numerical glue between the Gaussian-process layer (**TieredGP**) and the
supervised-learning residual layer (**ResidualNet**).

*Pure Python + NumPy*; hot loops can optionally be JIT-compiled with Numba.
If Numba is not installed the ``_JIT`` decorator becomes a no-op.

Key symbols
-----------
residual_dataset(curves)  -> (X, y)
    Builds supervised pairs (GP features → residuals) for ANN training.
    Accepts either **a single TieredGP** or **an iterable of TieredGP** objects.

ann_features(gp)          -> np.ndarray[n_feats]
    Shape-aware feature engineering for a single *TieredGP* snapshot.

---------------------------------------------------------------------------
Place new calibration utilities (e.g. λ-search for tension splines, Bayesian
hyper-parameter tuning, …) *below* the existing definitions, keeping them
stateless and free of I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable as _Iterable
from typing import Tuple, TYPE_CHECKING

import numpy as np

# --------------------------------------------------------------------------- #
# Forward refs for type checkers only
# --------------------------------------------------------------------------- #
if TYPE_CHECKING:  # pragma: no cover
    from gp.tiered_gp import TieredGP  # noqa: F401 (mypy / pyright hint)

# --------------------------------------------------------------------------- #
# Runtime import (safe: TieredGP does NOT import this module)
# --------------------------------------------------------------------------- #
from gp.tiered_gp import TieredGP as _TieredGP  # noqa: E402

# --------------------------------------------------------------------------- #
# optional Numba JIT
# --------------------------------------------------------------------------- #
try:
    import numba as _nb  # noqa: WPS433

    _JIT = _nb.njit(cache=True, fastmath=True)  # noqa: N818 (keep simple name)
except ModuleNotFoundError:  # pragma: no cover
    def _JIT(fn):  # type: ignore[override]
        return fn


# --------------------------------------------------------------------------- #
# ANN helpers
# --------------------------------------------------------------------------- #
Array = np.ndarray  # small alias for annotations


def residual_dataset(
    curves: _TieredGP | _Iterable[_TieredGP],
) -> Tuple[Array, Array]:
    """
    Convert one **TieredGP** *or* an iterable of TieredGP objects into the
    feature/target matrices (X, y) required by the residual ANN.

    Parameters
    ----------
    curves : TieredGP | Iterable[TieredGP]
        Either a single calibrated curve or a collection (e.g. history).

    Returns
    -------
    X : (n_obs, n_feats) ndarray[float64]
        Engineered feature matrix (see :pyfunc:`ann_features`).
    y : (n_obs, n_targets) ndarray[float64]
        Mis-pricing residuals at the illiquid tiers.
    """
    # ---- normalise input to an iterable ---------------------------------
    if isinstance(curves, _TieredGP):
        curves_iter: _Iterable[_TieredGP] = [curves]
    elif isinstance(curves, _Iterable):
        curves_iter = curves
    else:  # pragma: no cover  (should never happen with correct typing)
        raise TypeError(
            "curves must be a TieredGP instance or an iterable of TieredGP objects"
        )

    # ---- build lists ----------------------------------------------------
    feat_list, res_list = [], []
    for gp in curves_iter:
        feat_list.append(ann_features(gp))
        res_list.append(gp.residual())  # shape: (n_illiquid,)

    return np.stack(feat_list), np.stack(res_list)


def ann_features(gp: _TieredGP) -> Array:
    """
    Feature vector based on Kondratyev-style regime descriptors.

    Components
    ----------
    f_liq : IFR at each *liquid* knot (dimension n_liq)
    L     : level          = mean(f_liq)
    SF    : front slope    = f_liq[1] − f_liq[0]
    SB    : back  slope    = f_liq[-1] − f_liq[-2]
    C     : curvature      = std(f_liq)
    σ_lv  : posterior-level variance √(αᵀ Σ α)
    """
    f_liq = gp.ifr(gp.liquid_knots())  # 1-D NumPy array
    level = f_liq.mean()
    sf = f_liq[1] - f_liq[0]
    sb = f_liq[-1] - f_liq[-2]
    curvature = f_liq.std(ddof=0)
    sigma_lv = math.sqrt(gp.posterior_var_scalar())

    return np.concatenate(
        [f_liq, np.array([level, sf, sb, curvature, sigma_lv])]
    )


# --------------------------------------------------------------------------- #
# (additional calibration helpers can be appended below)
# --------------------------------------------------------------------------- #