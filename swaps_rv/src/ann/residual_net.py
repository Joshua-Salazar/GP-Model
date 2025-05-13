"""Hybrid GP + ANN residual model with optional JAX backend.

This module wraps a two–hidden–layer MLP that learns the *non‑Gaussian*
residual

    Δf(t) := f_market(t) − f_GP(t)

conditioned on a design vector **X** (knot values + shape factors).

The reference implementation relies on **Haiku/JAX/Optax** but those
libraries are *optional*.  The rest of the code‑base (GP calibration,
analytics) stays fully importable even when the deep‑learning stack is
absent.  Attempting to *instantiate* the network without JAX installed
will raise a helpful `ImportError` that suggests the extra requirement:

    pip install swaps-rv[jax]
"""

from __future__ import annotations

import importlib
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

__all__ = [
    "ResidualNetConfig",
    "ResidualNet",
    "build_residual_net",
]

# ---------------------------------------------------------------------------
# Optional‑dependency helper
# ---------------------------------------------------------------------------


def _require_jax():  # noqa: D401 – internal helper
    """Return imported (hk, jnp, optax) or raise a clear error message."""

    missing = [lib for lib in ("jax", "haiku", "optax") if importlib.util.find_spec(lib) is None]
    if missing:
        libs = ", ".join(missing)
        raise ImportError(
            "Residual‑net requires the JAX stack, but the following modules are missing: "
            f"{libs}.  Install extras with `pip install swaps-rv[jax]`."
        )

    import jax.numpy as jnp  # type: ignore  # noqa: WPS433 – late import is the point
    import haiku as hk  # type: ignore  # noqa: WPS433
    import optax  # type: ignore  # noqa: WPS433

    return hk, jnp, optax


# ---------------------------------------------------------------------------
# Minimal functional builder (for inference‑only use‑cases)
# ---------------------------------------------------------------------------


def build_residual_net(n_input: int, n_output: int) -> Callable[[object], object]:
    """Return (init_fn, apply_fn) of a 2‑layer tanh MLP.

    A convenience wrapper when you only need a forward‑pass function and
    plan to manage parameters externally (e.g. in JAX jit/scan loops).
    """

    hk, jnp, _optax = _require_jax()

    def _forward(x: jnp.ndarray) -> jnp.ndarray:  # noqa: D401
        x = hk.Linear(64)(x)
        x = jnp.tanh(x)
        x = hk.Linear(32)(x)
        x = jnp.tanh(x)
        return hk.Linear(n_output)(x)

    return hk.without_apply_rng(hk.transform(_forward))


# ---------------------------------------------------------------------------
# OO wrapper that includes training logic
# ---------------------------------------------------------------------------


@dataclass
class ResidualNetConfig:
    in_dim: int
    out_dim: int
    hidden: tuple[int, ...] = (64, 32)
    lr: float = 3e-3
    l2: float = 1e-4
    seed: int = 0


class ResidualNet:  # noqa: D101 – extended docstring below
    """Trainable residual network.

    Notes
    -----
    *Instantiation* triggers the lazy JAX import; merely importing the
    module does **not**.  This keeps unit‑testing of non‑NN parts light.

    Examples
    --------
    >>> cfg = ResidualNetConfig(in_dim=18, out_dim=8)
    >>> net = ResidualNet(cfg)  # requires JAX stack installed
    >>> net.fit(X_train, y_train, n_iter=2000, verbose=500)
    >>> delta_curve = net(X_new)
    """

    # -------------------------------------------------------------------
    # Construction / helpers
    # -------------------------------------------------------------------

    def __init__(self, cfg: ResidualNetConfig):
        self.cfg = cfg
        self._lazy_init()

    def _lazy_init(self):  # noqa: D401 – helper
        """Import JAX stack and build the Haiku transform lazily."""

        hk, jnp, optax = _require_jax()
        self._jnp = jnp  # stash for later use inside methods

        # Build network --------------------------------------------------
        def _mlp(x):  # noqa: D401 – local scope to capture cfg
            for h in self.cfg.hidden:
                x = hk.Linear(h)(x)
                x = jnp.tanh(x)
            return hk.Linear(self.cfg.out_dim)(x)

        self._forward = hk.without_apply_rng(hk.transform(_mlp))

        key = jnp.array([self.cfg.seed, self.cfg.seed ^ 0xDEADBEEF], dtype=jnp.uint32)
        dummy = jnp.zeros((1, self.cfg.in_dim))
        self.params = self._forward.init(key, dummy)

        # Optimiser ------------------------------------------------------
        opt = optax.adamw(self.cfg.lr, weight_decay=self.cfg.l2)
        self.opt_state = opt.init(self.params)
        self._opt_update = opt.update

        @jax.jit  # type: ignore  # noqa: WPS430 – jit compiles closed‑over fns, fine
        def _step(params, opt_state, xb, yb):  # noqa: D401 – JIT kernel
            preds = self._forward.apply(params, xb)
            loss = jnp.mean((preds - yb) ** 2)
            grads = jax.grad(lambda p: jnp.mean((self._forward.apply(p, xb) - yb) ** 2))(params)
            updates, opt_state2 = self._opt_update(grads, opt_state)
            params2 = optax.apply_updates(params, updates)
            return params2, opt_state2, loss

        self._step = _step

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def __call__(self, x):  # noqa: D401 – mimic function call
        """Return network prediction Δf."""

        return self._forward.apply(self.params, x)

    # ------------------------------ training ----------------------------

    def fit(
        self,
        x_train,
        y_train,
        n_iter: int = 5000,
        batch_size: int | None = None,
        verbose: int = 0,
    ) -> "ResidualNet":  # noqa: D401 – fluent API
        jnp = self._jnp
        key = jnp.array([self.cfg.seed ^ 123, self.cfg.seed], dtype=jnp.uint32)

        if batch_size is None or batch_size >= x_train.shape[0]:
            # full‑batch SGD
            for k in range(n_iter):
                self.params, self.opt_state, loss = self._step(
                    self.params, self.opt_state, x_train, y_train
                )
                if verbose and k % verbose == 0:
                    print(f"[{k}] MSE = {loss:.4e}")
        else:
            # mini‑batch
            for k in range(n_iter):
                key = jax.random.split(key)[0]
                idx = jax.random.choice(key, x_train.shape[0], (batch_size,), replace=False)
                xb, yb = x_train[idx], y_train[idx]
                self.params, self.opt_state, loss = self._step(self.params, self.opt_state, xb, yb)
                if verbose and k % verbose == 0:
                    print(f"[{k}] mini MSE = {loss:.4e}")

        return self

    # ------------------------------ persistence ------------------------

    def save(self, path: str | Path):  # noqa: D401 – utility I/O
        payload = {"cfg": asdict(self.cfg), "params": self.params}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualNet":  # noqa: D401 – alt constructor
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        net = cls(ResidualNetConfig(**blob["cfg"]))
        net.params = blob["params"]
        return net

