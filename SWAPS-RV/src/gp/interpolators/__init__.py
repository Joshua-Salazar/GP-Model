"""
Back-end registry for instantaneous-forward-rate interpolators.

Each concrete interpolator exposes a NumPy-ish call-signature
    f = interp(time_grid, beta)
where:

    * `time_grid` – 1-D array (float) of evaluation times,
    * `beta`      – knot / coefficient vector **excluding** the two natural
                    boundary weights; those are implied inside each class.

Implemented back-ends
---------------------
flat     – piece-wise constant fwd-rate (“flat forward”)
linear   – C⁰ linear IFR
cubic    – C² cubic spline (natural or clamped)
tension  – Andersen (2005) hyperbolic / GB tension spline  ← NEW

Adding new schemes merely requires dropping a module implementing
`class Interpolator` **or** calling `register(tag, cls)` manually.
"""

from importlib import import_module
from types import MappingProxyType
from typing import Type, Dict

# ---------------------------------------------------------------- registry

_REGISTRY: Dict[str, Type] = {}


def register(tag: str, cls):
    """
    Add a concrete interpolator to the registry.

    Raises
    ------
    ValueError  if `tag` already taken.
    """
    if tag in _REGISTRY:
        raise ValueError(f"Interpolator '{tag}' already registered")
    _REGISTRY[tag] = cls


def get(tag: str):
    """
    Retrieve interpolator class by short name (e.g. ``'cubic'``).
    """
    try:
        return _REGISTRY[tag]
    except KeyError as exc:
        raise KeyError(f"Unknown interpolator '{tag}'. "
                       f"Available: {list(_REGISTRY)}") from exc


# ---------------------------------------------------------------- built-ins  – import triggers class definitions
# NB: each module must define a public `Interpolator` class.

for _name in ("flat", "linear", "cubic"):
    mod = import_module(f".{_name}", __name__)
    register(_name, mod.Interpolator)

# tension spline lives in its own file; import and register

from .tension_spline import TensionSpline  # noqa: E402  (after registry helpers)

register("tension", TensionSpline)

# immutable public view -------------------------------------------------------

available = MappingProxyType(_REGISTRY)          # read-only dict proxy

__all__ = ["register", "get", "available", "TensionSpline"]
