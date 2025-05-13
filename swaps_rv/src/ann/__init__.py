"""
ann package
===========

High-level helpers that wrap the residual neural network used to capture
non-linear (fly / curvature) effects on top of the Gaussian-process baseline.

Public API
----------
get_resnet(cfg: dict) -> haiku.Transformed
    Factory that returns a Haiku‐transformed forward / init pair according to
    a simple JSON-style config.

load_weights(path: str) -> dict
    Thin convenience wrapper around `jax.numpy.load`.

Notes
-----
* The heavy lifting lives in `ann.residual_net.ResidualNet`.
* This file purposefully re-exports only the user-friendly objects so that
  `import fixed_income_rv.ann as ann` yields an uncluttered namespace.
"""

from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import haiku as hk

from .residual_net import ResidualNet


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_resnet(cfg: Dict) -> hk.Transformed:
    """
    Build a ResidualNet from a lightweight configuration dictionary.

    Parameters
    ----------
    cfg
        Example::

            {
                "hidden_layers": [64, 64],
                "activation": "gelu",
                "dropout": 0.0,
                "crest_mode": true
            }

    Returns
    -------
    haiku.Transformed
        Haiku pair with .init(rng, x) and .apply(params, rng, x) call-signatures.
    """
    def _forward(x, is_training: bool = False):
        net = ResidualNet(**cfg)
        return net(x, is_training=is_training)

    return hk.without_apply_rng(hk.transform(_forward))


def load_weights(path: str) -> Dict:
    """
    Utility to load serialised Haiku/JAX parameters saved with `jnp.savez`.

    Parameters
    ----------
    path : str
        Location of the ``.npz`` file.

    Returns
    -------
    dict
        PyTree of parameters compatible with `get_resnet(cfg).apply`.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"weights file not found: {path}")

    flat = jnp.load(str(path), allow_pickle=True)
    # Haiku parameters are stored as a flat dict; rely on haiku-data-struct conv.
    return {k: flat[k] for k in flat}


__all__ = ["get_resnet", "load_weights"]
