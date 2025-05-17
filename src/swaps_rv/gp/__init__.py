"""
Gaussian-process backbone for the relative-value framework.

Public interface
----------------
* **TieredGPModel** – canonical façade that builds a curve, conditions on
  liquidity tiers, and produces IFR / DF arrays.
* **TieredGP**      – *deprecated* alias kept for backward compatibility.
* **TierConfig**, **USD_TIERS** – tier-description helpers.
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
from .tiered_gp import (
    USD_TIERS as _USD_TIERS,
)
from .tiered_gp import (
    TierConfig as _TierConfig,
)

# ---------------------------------------------------------------------------
# Tiered GP façade
# ---------------------------------------------------------------------------
# ``tiered_gp.py`` defines ``TieredGP`` (curve engine) and the tier helpers.
# We expose them at the package level so that users can simply write
#     from swaps_rv.gp import TieredGP, TierConfig, USD_TIERS
from .tiered_gp import (  # noqa: E402  – keep grouped import
    TieredGP as _TieredGP,
)

# preferred handle ----------------------------------------------------------
TieredGPModel = _TieredGP  # type: ignore[misc]

# legacy alias --------------------------------------------------------------
TieredGP = _TieredGP  # noqa: N816 – preserve original camel-caps symbol

# tier descriptors ----------------------------------------------------------
TierConfig = _TierConfig
USD_TIERS = _USD_TIERS

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
    "TierConfig",
    "USD_TIERS",
    "kernels",
    "interpolators",
]
