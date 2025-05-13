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

ann_features(gp)         -> np.ndarray[n_feats]
    Shape-aware feature engineering for a single *TieredGP* snapshot.

---------------------------------------------------------------------------
Place new calibration utilities (e.g. λ-search for tension splines, Bayesian
hyper-parameter tuning, …) *below* the existing definitions, keeping them
stateless and free of I/O.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gp.tiered_gp import TieredGP  # forward ref only for type checkers

# --------------------------------------------------------------------------- #
# optional Numba JIT
# --------------------------------------------------------------------------- #
try:
    import numba as _nb

    _JIT = _nb.njit(cache=True, fastmath=True)  # noqa: N818  (keep simple)
except ModuleNotFoundError:  # pragma: no cover
    def _JIT(fn):  # type: ignore[override]
        return fn


# --------------------------------------------------------------------------- #
# ANN helpers
# --------------------------------------------------------------------------- #
Array: type = np.ndarray  # small alias for signatures


def residual_dataset(curves: Iterable["TieredGP"]) -> Tuple[Array, Array]:
    """
    Convert a sequence of *TieredGP* objects into (X, y) matrices for the
    residual ANN.

    Returns
    -------
    X : (n_obs, n_feats)  Engineered features (see ``ann_features``).
    y : (n_obs, n_targets) Mis-pricing residuals at illiquid tiers.
    """
    feat_list, res_list = [], []

    for gp in curves:
        feat_list.append(ann_features(gp))
        res_list.append(gp.residual())           # shape (n_illiquid,)

    return np.stack(feat_list), np.stack(res_list)


def ann_features(gp: "TieredGP") -> Array:
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

    return np.concatenate([f_liq,
                           np.array([level, sf, sb, curvature, sigma_lv])])


# --------------------------------------------------------------------------- #
# (additional calibration helpers can be appended below)
# --------------------------------------------------------------------------- #
