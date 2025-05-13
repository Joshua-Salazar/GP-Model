"""
utils
=====

Small, *stateless* helper sub-package shared by **fixed-income-rv**:

* :pymod:`utils.data`          – lightweight I/O and calendars
* :pymod:`utils.calibration`   – DV01 math, carry-roll, posterior helper fns
* :pymod:`utils.plots`         – Matplotlib / JAX plotting shortcuts

The module intentionally holds **no heavy imports** at top-level so that
interactive workflows can do::

    import utils as U
    df = U.data.load_quotes("usd_swaps.csv")

without triggering `jax`, `numba`, *etc.* unless required inside the
specific function.
"""

from __future__ import annotations

import importlib
import types
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Lazy sub-module loader
# --------------------------------------------------------------------------- #

__all__: list[str] = ["data", "calibration", "plots"]  # public API


def _lazy(name: str) -> Callable[[], types.ModuleType]:
    """Return a proxy loader so `import utils.data` is loaded on first access."""

    def _load() -> types.ModuleType:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod  # memoise – subsequent access is direct
        return mod

    return _load


# create `types.ModuleType` proxies that defer real import
data = types.ModuleType("utils.data")
calibration = types.ModuleType("utils.calibration")
plots = types.ModuleType("utils.plots")

for _m in (("data", data), ("calibration", calibration), ("plots", plots)):
    _name, _proxy = _m
    _proxy.__getattr__ = lambda _a, n, _loader=_name: getattr(_lazy(_loader)(), n)  # type: ignore[attr-defined]
    globals()[_name] = _proxy  # expose at package level
