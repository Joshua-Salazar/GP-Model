"""
utils.plots
===========

Matplotlib helpers – **not** imported by the core engine, so `matplotlib`
remains an *optional* dependency (only required for exploratory notebooks or
the CLI “plot” flag).

All functions return the figure handle, allowing the caller to further tweak /
save / show.  The colour-map is kept minimal and avoids hard-coding styles so
that downstream code (or mplrc) stays in control.

Functions
---------

curve(ax, gp, *, show_liq=True)            -> Figure
    IFR + par-swap dots; liquid knots highlighted when *show_liq*.

residuals(ax, gp)                          -> Figure
    Stem plot of *Tier-j* residual (illiquid bucket) in bp.

dv01_bar(ax, gp)                           -> Figure
    Horizontal bar-chart of bucket DV01.

ann_surface(ax, net, X, y, dim=0)          -> Figure
    Quick mesh-plot of ANN output vs. two selected inputs.

"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def _ax(ax):
    """Return provided axis or new fig/ax."""
    if ax is None:
        fig, ax_ = plt.subplots(figsize=(8, 4))
        return fig, ax_
    return ax.figure, ax


# --------------------------------------------------------------------------- #
# GP visualisations
# --------------------------------------------------------------------------- #


def curve(gp: "TieredGP", *, ax=None, show_liq: bool = True):
    """Instantaneous-forward curve + par-swap markers."""
    fig, ax = _ax(ax)

    t_grid = np.linspace(0.0, gp.knots[-1], 600)
    ax.plot(t_grid, gp.ifr(t_grid), lw=1.5, label="IFR")

    par = gp.par_swap_rates()
    ax.scatter(gp.knots, par, marker="o", s=30, zorder=3, label="Par swap")

    if show_liq:
        liq = gp.liquid_knots()
        ax.scatter(liq, gp.ifr(liq), marker="D", c="tab:red", s=40, label="Liquid")

    ax.set_xlabel("Maturity (y)")
    ax.set_ylabel("Instantaneous fwd / Par-swap")
    ax.set_title(f"Curve snapshot – {gp.value_date:%Y-%m-%d}")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    return fig


def residuals(gp: "TieredGP", *, ax=None):
    """Stem plot in basis-points for illiquid residual."""
    fig, ax = _ax(ax)
    res = gp.residual() * 1e4  # bp

    illiq = gp.illiquid_knots()
    markerline, stemlines, _ = ax.stem(illiq, res, use_line_collection=True)
    plt.setp(markerline, marker='x', markersize=6, mec='k', mfc='none')
    plt.setp(stemlines, lw=1.2)

    ax.axhline(0, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Maturity (y)")
    ax.set_ylabel("Residual (bp)")
    ax.set_title("Level-hedged residual")
    ax.grid(True, ls="--", alpha=0.3)
    return fig


def dv01_bar(gp: "TieredGP", *, ax=None):
    """Horizontal bar chart of bucket DV01."""
    fig, ax = _ax(ax)
    ser = gp.bucket_dv01()

    ax.barh(ser.index, ser.values * 1e6)  # convert to $ per bp on 1mm
    ax.set_xlabel("$ DV01 (per $1mm notional)")
    ax.set_ylabel("Knot maturity (y)")
    ax.set_title("Bucket PV01 profile")
    ax.grid(axis="x", ls="--", alpha=0.3)
    return fig


# --------------------------------------------------------------------------- #
# ANN diagnostics
# --------------------------------------------------------------------------- #


def ann_surface(
    net: "ResidualNet",
    X: np.ndarray,
    y: np.ndarray,
    *,
    dim: int = 0,
    grid: Sequence[float] | None = None,
    ax=None,
):
    """
    2-D response surface along *dim* vs. target residual.

    *All other* inputs are clamped at median.
    """
    fig, ax = _ax(ax)
    if grid is None:
        x_min, x_max = X[:, dim].min(), X[:, dim].max()
        grid = np.linspace(x_min, x_max, 200)

    X0 = np.median(X, axis=0, keepdims=True).repeat(len(grid), 0)
    X0[:, dim] = grid

    y_hat = net.predict(X0).ravel()

    ax.plot(grid, y_hat, lw=1.8, label="ANN")
    # scatter true obs along dim for sanity-check
    ax.scatter(X[:, dim], y, s=8, alpha=0.4, label="train pts")

    ax.set_xlabel(f"Feature[{dim}]")
    ax.set_ylabel("Residual")
    ax.set_title("ANN cross-section")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    return fig
