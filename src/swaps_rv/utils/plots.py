"""
Utility plotting helpers for the GP + ANN framework (risk‑metrics removed).

Functions
---------
curve(ax, gp, *, show_liq=True)            -> Figure
    IFR + par‑swap dots; liquid knots highlighted when *show_liq*.

residuals(ax, gp)                          -> Figure
    Stem plot of Tier‑J residuals (bp).

ann_surface(ax, net, X, y, dim=0)          -> Figure
    1‑D response slice of the residual ANN.

plot_ifr(gp, *, save_to=None, **kwargs)     -> Figure
    Thin wrapper around ``curve`` that also handles ``save_to``.

alpha_panel(gp, net, *, save_to=None)      -> Figure
    Two‑row summary panel: IFR curve + residual stem plot.

Note
----
No DV01 / bucket‑risk code is kept in this trimmed version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "curve",
    "residuals",
    "ann_surface",
    "plot_ifr",
    "alpha_panel",
]

if TYPE_CHECKING:  # pragma: no cover – import only for static typing
    from swaps_rv.ann.residual_net import ResidualNet
    from swaps_rv.gp.tiered_gp import TieredGP

# --------------------------------------------------------------------------- #
# internal helper
# --------------------------------------------------------------------------- #


def _ax(ax):
    if ax is None:
        fig, ax_ = plt.subplots(figsize=(8, 4))
        return fig, ax_
    return ax.figure, ax


# --------------------------------------------------------------------------- #
# GP visualisations
# --------------------------------------------------------------------------- #


def curve(gp: "TieredGP", *, ax=None, show_liq: bool = True):
    """Instantaneous‑forward curve plus par‑swap markers."""
    fig, ax = _ax(ax)

    t_grid = np.linspace(0.0, gp.knots[-1], 600)
    ax.plot(t_grid, gp.ifr(t_grid), lw=1.5, label="IFR")

    par = gp.par_swap_rates()
    ax.scatter(gp.knots, par, marker="o", s=30, zorder=3, label="Par swap")

    if show_liq:
        liq = gp.liquid_knots()
        ax.scatter(liq, gp.ifr(liq), marker="D", c="tab:red", s=40, label="Liquid")

    ax.set(
        xlabel="Maturity (y)",
        ylabel="Rate",
        title=f"Curve snapshot – {gp.value_date:%Y-%m-%d}",
    )
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    return fig


def residuals(gp: "TieredGP", *, ax=None):
    """Stem plot of hedged residuals in basis points."""
    fig, ax = _ax(ax)
    res = gp.residual() * 1e4  # bp
    illiq = gp.illiquid_knots()

    markerline, stemlines, _ = ax.stem(illiq, res, use_line_collection=True)
    markerline.set(marker="x", markersize=6, mec="k", mfc="none")
    stemlines.set(lw=1.2)

    ax.axhline(0, ls="--", c="k", lw=0.8)
    ax.set(
        xlabel="Maturity (y)",
        ylabel="Residual (bp)",
        title="Level‑hedged residuals",
    )
    ax.grid(True, ls="--", alpha=0.3)
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
    """2‑D response surface along one feature axis vs. the target residual."""
    fig, ax = _ax(ax)

    if grid is None:
        x_min, x_max = X[:, dim].min(), X[:, dim].max()
        grid = np.linspace(x_min, x_max, 200)

    X0 = np.median(X, axis=0, keepdims=True).repeat(len(grid), 0)
    X0[:, dim] = grid

    y_hat = net.predict(X0).ravel()

    ax.plot(grid, y_hat, lw=1.8, label="ANN")
    ax.scatter(X[:, dim], y, s=8, alpha=0.4, label="train pts")

    ax.set(
        xlabel=f"Feature[{dim}]",
        ylabel="Residual",
        title="ANN cross‑section",
    )
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    return fig


# --------------------------------------------------------------------------- #
# Thin wrappers to keep backward‑compat with the original CLI scripts
# --------------------------------------------------------------------------- #


def plot_ifr(gp: "TieredGP", *, save_to: str | None = None, **kwargs):
    """Wrapper around :pyfunc:`curve` that also handles file output."""
    fig = curve(gp, **kwargs)
    if save_to is not None:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def alpha_panel(gp: "TieredGP", net: "ResidualNet", *, save_to: str | None = None):
    """Two‑row summary panel consisting of IFR curve and residual stem plot."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    # Top: IFR
    curve(gp, ax=axes[0])

    # Bottom: residuals (ANN currently unused but kept for signature parity)
    residuals(gp, ax=axes[1])

    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
