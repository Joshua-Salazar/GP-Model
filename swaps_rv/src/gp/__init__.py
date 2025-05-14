"""
Gaussian-process backbone for the relative-value framework.

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
# Public re-exports
# ---------------------------------------------------------------------------

# kernels -------------------------------------------------------------------
from . import kernels as kernels  # noqa: F401 – re-export

# interpolators -------------------------------------------------------------
from .interpolators import get as _get_interp  # noqa: F401 – re-export

# ---------------------------------------------------------------------------
# Tiered GP façade
# ---------------------------------------------------------------------------
# `tiered_gp.py` defines `TieredGP`; we expose both a modern handle
# (`TieredGPModel`) and the legacy alias (`TieredGP`).

from .tiered_gp import TieredGP as _TieredGP


class TieredGPModel(_TieredGP):  # type: ignore[misc]
    """Canonical façade for the tiered Gaussian-process curve."""
    # No extra logic; subclass purely for naming consistency.
    pass


# Backward-compat shim ------------------------------------------------------
TieredGP = _TieredGP  # noqa: N816 – preserve original camel-caps symbol

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
    "TieredGPModel",  # new preferred handle
    "TieredGP",       # deprecated alias
    "kernels",
    "interpolators",
]