"""
Back-end registry for instantaneous-forward-rate interpolators.

Each concrete interpolator exposes a NumPy-ish call-signature
    f = interp(time_grid, beta)

Implemented back-ends
---------------------
flat     – piece-wise constant fwd-rate (“flat forward”ssssssssss
linear   – C⁰ linear IFR
cubic    – C² cubic spline (natural or clamped)
tension  – Andersen (2005) hyperbolic / GB tension spline
"""

from importlib import import_module
from types import MappingProxyType
from typing import Dict, Type

# ---------------------------------------------------------------- registry

_REGISTRY: Dict[str, Type] = {}


def register(tag: str, cls: Type):
    """Add a concrete interpolator to the registry."""
    if tag in _REGISTRY:
        raise ValueError(f"Interpolator '{tag}' already registered")
    _REGISTRY[tag] = cls


def get(tag: str) -> Type:
    """Retrieve interpolator class by short name (e.g. ``'cubic'``)."""
    try:
        return _REGISTRY[tag]
    except KeyError as exc:
        raise KeyError(
            f"Unknown interpolator '{tag}'. Available: {list(_REGISTRY)}"
        ) from exc


# ---------------------------------------------------------------- built-ins
# For each plug-in module we try, in order:
#   1) public symbol ``Interpolator``
#   2) ``<Name>Interpolator`` where <Name> is Flat / Linear / …

def _import_and_register(name: str):
    mod = import_module(f".{name}", __name__)  # gp.interpolators.<name>

    # primary key: "Interpolator"
    cls = getattr(mod, "Interpolator", None)

    # fallback:  FlatInterpolator, LinearInterpolator, …
    if cls is None:
        fallback = f"{name.capitalize()}Interpolator"
        cls = getattr(mod, fallback, None)

    if cls is None:
        raise AttributeError(
            f"Module 'gp.interpolators.{name}' exports neither "
            "'Interpolator' nor '{fallback}'."
        )

    register(name, cls)


for _name in ("flat", "linear", "cubic"):
    _import_and_register(_name)

# tension spline lives in its own file
from .tension_spline import TensionSpline  # noqa: E402

register("tension", TensionSpline)

# immutable public view -------------------------------------------------------

available = MappingProxyType(_REGISTRY)  # read-only dict proxy

__all__ = ["register", "get", "available", "TensionSpline"]
