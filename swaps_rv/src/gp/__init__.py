"""
fixed_income_rv.gp
==================

Gaussian-process backbone for the relative-value framework.

Public interface
----------------
The sub-package exposes exactly three public objects:

* **TieredGP**   – high-level façade that builds a curve, conditions on
  liquidity tiers, and produces IFR / DF / DV01 arrays.

* **kernels.get(name)** – factory for common covariance kernels
  ("SE", "Brownian", "OU", …).

* **interpolators.get(name)** – registry returning a deterministic
  back-end (flat, linear, cubic, tension_spline).

Everything else is considered *private* implementation detail.

Example
-------
>>> from fixed_income_rv.gp import TieredGP, kernels, interpolators
>>> gp = TieredGP(
...         kernel=kernels.get("Brownian"),
...         interpolator=interpolators.get("tension_spline"),
...         tiers=[(0, ["30y"]),
...                (1, ["2y","5y","10y"]),
...                (2, ["1y","7y","20y"]),
...                (3, ["3y","15y","25y"])]
...     )
>>> gp.fit(swap_quotes)
>>> df = gp.discount_factor_grid()
>>> dv01 = gp.bucket_dv01()

Design notes
------------
* The heavy linear-algebra and JIT code lives in ``gp._core``.
* Kernels are stateless functional helpers in ``gp.kernels``.
* Interpolators reside in ``gp.interpolators`` and are swappable at runtime.

"""

from importlib import import_module
from types import ModuleType
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

# Kernels  ------------------------------------------------------------------
from . import kernels as kernels  # noqa: F401  (re-export)

# Interpolators -------------------------------------------------------------
from .interpolators import get as _get_interp   # noqa: F401

# Tiered GP façade ----------------------------------------------------------
from .tiered_gp import TieredGP  # noqa: F401


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def interpolators(name: str) -> Callable[..., Any]:
    """Shortcut so callers can do `gp.interpolators("cubic")`."""
    return _get_interp(name)


# tidy up namespace
del import_module, ModuleType, Callable, Any
__all__ = ["TieredGP", "kernels", "interpolators"]
