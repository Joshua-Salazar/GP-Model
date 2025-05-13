"""
Gaussian-process backbone for the relative-value framework.

Public interface
----------------
* **TieredGPModel** – canonical façade that builds a curve, conditions on
  liquidity tiers, and produces IFR / DF / DV01 arrays.
* **TieredGP**      – *deprecated* alias kept for backward compatibility.
* **kernels.get(name)**
* **interpolators.get(name)**
"""

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

# kernels -------------------------------------------------------------------
from . import kernels as kernels  # noqa: F401 re-export

# interpolators -------------------------------------------------------------
from .interpolators import get as _get_interp  # noqa: F401

# Tiered GP façade ----------------------------------------------------------
from .tiered_gp import TieredGPModel  # <-- new canonical name

# Backward-compat shim ------------------------------------------------------
class _DeprecatedAlias(TieredGPModel):        # type: ignore[misc]
    """Alias for code written against the old ``TieredGP`` symbol.

    Will be removed in a future major release.
    """
    pass


TieredGP = _DeprecatedAlias  # noqa: N816  (keep original cap-style)


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
    "TieredGPModel",   # new preferred handle
    "TieredGP",        # deprecated alias
    "kernels",
    "interpolators",
]