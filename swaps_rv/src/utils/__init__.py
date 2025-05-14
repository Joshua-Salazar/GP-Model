"""
utils
=====

Stateless helper sub-package used across **fixed-income-rv**.

* :pymod:`utils.data`        – lightweight I/O, calendars, holiday helpers
* :pymod:`utils.calibration` – GP/ANN feature builders, posterior utilities
* :pymod:`utils.plots`       – Matplotlib / JAX plotting shortcuts

The top-level package intentionally avoids heavy imports (Numba, JAX, …)
so interactive sessions can do:

    >>> import utils as U
    >>> df = U.data.load_quotes("usd_swaps.csv")

without pulling large dependencies until the specific function is called.
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
    """Return a proxy loader so `import utils.<name>` happens on first access."""

    def _load() -> types.ModuleType:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod  # memoise – next access is direct
        return mod

    return _load


# Create proxy modules that defer the real import until first attribute access.
data = types.ModuleType("utils.data")
calibration = types.ModuleType("utils.calibration")
plots = types.ModuleType("utils.plots")

for _mod_name, _proxy in (("data", data), ("calibration", calibration), ("plots", plots)):
    _proxy.__getattr__ = lambda _self, attr, _n=_mod_name: getattr(_lazy(_n)(), attr)  # type: ignore[attr-defined]
    globals()[_mod_name] = _proxy  # expose at package level