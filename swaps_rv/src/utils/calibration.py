"""
utils.calibration
=================

Numerical helpers that glue **Gaussian-process curve objects** to
risk / PnL analytics used by the RV back-tester and live engine.

The module is *pure-Python* + NumPy/Pandas; hot-loops can be JIT’ed under
Numba – toggled by the caller with ``jit=True``.  (Import‐side choices keep
Numba optional.)

Key symbols
-----------

residual_dataset(curves)  -> (X, y)
    Builds supervised pairs (GP features → residual) for ANN fit.

ann_features(gp)         -> np.ndarray[n_feats]
    Feature engineering given a single *TieredGP* curve instance.

bucket_dv01(gp)          -> pd.Series
    Per-knot DV01 in *instrument currency*, using 1 bp bump–revalue.

notional_used(gp, alpha) -> float
    Gross notional implied by signal weights ``alpha``.

realised_pnl(gp, alpha)  -> float
    Mark-to-market Δ plus carry/roll for one trading day.

performance_tearsheet(pnl: pd.Series) -> pd.Series
    Classic perf metrics (CAGR, vol, Sharpe, max-drawdown, hit-ratio…)
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import numba as nb

    _JIT = nb.njit(cache=True, fastmath=True)
except ModuleNotFoundError:  # pragma: no cover – numba optional
    # graceful fallback – acts as identity decorator
    def _JIT(fn):  # type: ignore[override]
        return fn


# --------------------------------------------------------------------------- #
# ANN helpers
# --------------------------------------------------------------------------- #


def residual_dataset(
    curves: Iterable["TieredGP"],  # forward ref
) -> Tuple[np.ndarray, np.ndarray]:
    """Return design matrix **X** and target residual **y** (stack over history)."""
    X_, y_ = [], []

    for gp in curves:
        X_.append(ann_features(gp))
        y_.append(gp.residual())  # shape (n_illiquid,)

    X = np.stack(X_, axis=0)
    y = np.stack(y_, axis=0)
    return X, y


def ann_features(gp: "TieredGP") -> np.ndarray:
    """
    Feature vector from one *TieredGP* snapshot.

    * level risk = IFR at liquid knots
    * shape stats = (level, front-slope, back-slope, curvature, sigmaᵀΣpostσ)
    """
    level = gp.ifr(gp.liquid_knots())  # shape (n_liq,)
    fwd = level

    L = fwd.mean()
    sf = fwd[1] - fwd[0]
    sb = fwd[-1] - fwd[-2]
    C = fwd.std()

    sigma_lv = math.sqrt(gp.posterior_var_scalar())  # αᵀ Σ α – single number

    return np.concatenate([fwd, np.array([L, sf, sb, C, sigma_lv])])


# --------------------------------------------------------------------------- #
# Risk (DV01, notional, carry/roll)
# --------------------------------------------------------------------------- #


def bucket_dv01(gp: "TieredGP") -> pd.Series:
    """
    1 bp PV01 per interpolation knot *under the GP prior*.

    For fixed-for-float IRS that implies: DV01 = -∂PV/∂r * 1e-4
    """
    bump = 1e-4
    knots = gp.all_knots()

    base_df = gp.df_curve()

    dv01 = []
    for t in knots:
        bumped = gp.bump_ifr({t: bump})
        dv01.append((bumped.df_curve() - base_df).sum())  # ΔPV

    return pd.Series(dv01, index=knots, name="dv01")


def notional_used(gp: "TieredGP", alpha: np.ndarray) -> float:
    """Gross notional required to express positions α * DV01."""
    dv01 = bucket_dv01(gp).values
    notionals = np.abs(alpha * 1e4) / np.maximum(np.abs(dv01), 1e-12)
    return notionals.sum()


# --------------------------------------------------------------------------- #
# PnL
# --------------------------------------------------------------------------- #


@_JIT
def _pnl_vec(base: np.ndarray, nxt: np.ndarray, alpha: np.ndarray) -> float:
    """Low-level numba’d helper."""
    return np.dot(alpha, nxt - base)


def realised_pnl(gp_today: "TieredGP", alpha: np.ndarray) -> float:
    """
    Single-day realised PnL:

    ΔPV (mark-to-market) + carry/roll (automatic in GP evolution).
    """
    try:
        gp_prev = gp_today.prev_curve
    except AttributeError:
        warnings.warn("First day – PnL = 0")
        return 0.0

    pv_today = gp_today.par_swap_rates()  # vector
    pv_prev = gp_prev.par_swap_rates()

    return _pnl_vec(pv_prev, pv_today, alpha)


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #


def performance_tearsheet(pnl: pd.Series) -> pd.Series:
    """Return standard tear-sheet KPIs."""
    daily = pnl.values
    dt = 252.0  # trading days

    tot = pnl.iloc[-1]
    mu = daily.mean()
    vol = daily.std(ddof=0)
    sharpe = mu / vol * math.sqrt(dt) if vol > 0 else np.nan
    cagr = (1 + tot) ** (dt / len(daily)) - 1

    roll = pnl.cumsum()
    high = np.maximum.accumulate(roll)
    dd = roll - high
    max_dd = dd.min()

    hit = (daily > 0).mean()

    return pd.Series(
        {
            "total_pnl": tot,
            "CAGR": cagr,
            "daily_vol": vol,
            "Sharpe": sharpe,
            "max_drawdown": max_dd,
            "hit_ratio": hit,
        }
    )
