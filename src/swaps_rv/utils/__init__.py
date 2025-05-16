"""
utils
=====

Stateless helper sub‑package used across **fixed‑income‑rv**.

* :pymod:`utils.data`        – lightweight I/O, calendars, holiday helpers
* :pymod:`utils.calibration` – GP/ANN feature builders (no DV01 code)
* :pymod:`utils.plots`       – Matplotlib/JAX plotting shortcuts

The top‑level package intentionally avoids heavy imports (Numba, JAX, …) so you can do::

    >>> import utils as U
    >>> df = U.data.load_quotes("usd_swaps.csv")

without dragging large dependencies into memory until the first function call.
"""

from __future__ import annotations

import importlib
import types
from typing import Any, Callable

__all__ = [
    "data",
    "calibration",
    "plots",
]

# ---------------------------------------------------------------------------
# Lazy sub‑module loader (keeps import footprint tiny)
# ---------------------------------------------------------------------------

def _lazy(name: str) -> Callable[[], types.ModuleType]:
    """Return a proxy loader so ``import utils.<name>`` is truly lazy."""

    def _load() -> types.ModuleType:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod  # cache for subsequent look‑ups
        return mod

    return _load


# Create placeholder modules that forward attribute access on demand ---------
for _sub in __all__:
    _proxy = types.ModuleType(f"utils.{_sub}")

    def __getattr__(self, attr, _n=_sub):  # type: ignore[override]
        return getattr(_lazy(_n)(), attr)

    _proxy.__getattr__ = __getattr__.__get__(  # type: ignore[attr-defined]
        _proxy, types.ModuleType
    )
    globals()[_sub] = _proxy
