"""
Gaussian‑process backbone for the relative‑value framework.

Public interface
----------------
* **TieredGPModel** – canonical façade that builds a curve, conditions on
  liquidity tiers, and produces IFR / DF arrays.
* **TieredGP**      – *deprecated* alias kept for backward compatibility.
* **kernels.get(name)**
* **interpolators.get(name)**
"""

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Public re‑exports
# ---------------------------------------------------------------------------

# kernels -------------------------------------------------------------------
from . import kernels as kernels  # noqa: F401 – re‑export

# interpolators -------------------------------------------------------------
from .interpolators import get as _get_interp  # noqa: F401 – re‑export

# ---------------------------------------------------------------------------
# Tiered GP façade
# ---------------------------------------------------------------------------
# ``tiered_gp.py`` defines ``TieredGP``.  We expose it under two names so that
# old code continues to work while new code follows the updated terminology.

from .tiered_gp import TieredGP as _TieredGP

# preferred handle ----------------------------------------------------------
TieredGPModel = _TieredGP  # type: ignore[misc]

# legacy alias --------------------------------------------------------------
TieredGP = _TieredGP  # noqa: N816  – preserve original camel‑caps symbol

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def interpolators(name: str) -> Callable[..., Any]:
    """Shortcut so callers can do ``gp.interpolators("cubic")``."""

    return _get_interp(name)


# ---------------------------------------------------------------------------
# Public namespace
# ---------------------------------------------------------------------------

__all__ = [
    "TieredGPModel",
    "TieredGP",
    "kernels",
    "interpolators",
]
