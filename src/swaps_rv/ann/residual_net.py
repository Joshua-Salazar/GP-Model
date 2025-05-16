"""Hybrid GP + ANN residual model with optional JAX backend.

This module wraps a two–hidden–layer MLP that learns the *non‑Gaussian*
residual

    Δf(t) := f_market(t) − f_GP(t)

conditioned on a design vector **X** (knot values + shape factors).

The reference implementation relies on **Haiku/JAX/Optax**. Those
libraries are *optional* – importing this file works even if the DL stack
is missing; the network is instantiated lazily and raises a clear
``ImportError`` if JAX/Haiku/Optax are absent.
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
    "ResidualANN",  # legacy alias
    "build_residual_net",
]

# ---------------------------------------------------------------------------
# Optional‑dependency guard
# ---------------------------------------------------------------------------

def _require_jax():  # noqa: D401 – helper
    """Import (haiku, jax.numpy, optax) or raise an informative error."""
    missing = [lib for lib in ("jax", "haiku", "optax") if importlib.util.find_spec(lib) is None]
    if missing:
        raise ImportError(
            "Residual‑net requires the JAX stack; missing: "
            + ", ".join(missing)
            + ".  Run  `pip install swaps-rv[jax]`  to add the extras."
        )

    import jax  # type: ignore  # noqa: WPS433 – late import intentional
    import jax.numpy as jnp  # type: ignore  # noqa: WPS433
    import haiku as hk  # type: ignore  # noqa: WPS433
    import optax  # type: ignore  # noqa: WPS433

    return hk, jnp, optax


# ---------------------------------------------------------------------------
# Minimal functional builder (inference‑only)
# ---------------------------------------------------------------------------

def build_residual_net(n_input: int, n_output: int):  # -> (init_fn, apply_fn)
    """Return a (init_fn, apply_fn) pair for a 2‑layer tanh MLP."""
    hk, jnp, _ = _require_jax()

    def _forward(x: jnp.ndarray):  # noqa: D401
        x = hk.Linear(64)(x)
        x = jnp.tanh(x)
        x = hk.Linear(32)(x)
        x = jnp.tanh(x)
        return hk.Linear(n_output)(x)

    return hk.without_apply_rng(hk.transform(_forward))


# ---------------------------------------------------------------------------
# OO wrapper with training logic
# ---------------------------------------------------------------------------


@dataclass
class ResidualNetConfig:
    in_dim: int
    out_dim: int
    hidden: Sequence[int] = (64, 32)
    lr: float = 3e‑3
    l2: float = 1e‑4
    seed: int = 0


class ResidualNet:
    """Small MLP to learn residuals Δf given design matrix X."""

    # ----------------------------- construction -----------------------------

    def __init__(self, cfg: ResidualNetConfig):
        self.cfg = cfg
        self._lazy_init()

    def _lazy_init(self):  # noqa: D401
        hk, jnp, optax = _require_jax()
        import jax  # type: ignore  # noqa: WPS433

        self._jnp = jnp
        self._jax = jax

        # network -----------------------------------------------------------
        def _mlp(x):  # capture cfg via closure
            for h in self.cfg.hidden:
                x = hk.Linear(h)(x)
                x = jnp.tanh(x)
            return hk.Linear(self.cfg.out_dim)(x)

        self._forward = hk.without_apply_rng(hk.transform(_mlp))

        key = jnp.array([self.cfg.seed, self.cfg.seed ^ 0xDEADBEEF], dtype=jnp.uint32)
        dummy = jnp.zeros((1, self.cfg.in_dim))
        self.params = self._forward.init(key, dummy)

        # optimiser --------------------------------------------------------
        opt = optax.adamw(self.cfg.lr, weight_decay=self.cfg.l2)
        self.opt_state = opt.init(self.params)
        self._opt_update = opt.update

        @jax.jit  # type: ignore[misc]
        def _step(params, opt_state, xb, yb):
            preds = self._forward.apply(params, xb)
            loss = jnp.mean((preds - yb) ** 2)
            grads = jax.grad(lambda p: jnp.mean((self._forward.apply(p, xb) - yb) ** 2))(params)
            updates, opt_state2 = self._opt_update(grads, opt_state)
            params2 = optax.apply_updates(params, updates)
            return params2, opt_state2, loss

        self._step = _step

    # ----------------------------- public API ------------------------------

    def __call__(self, x):  # noqa: D401
        """Forward pass Δf(X)."""
        return self._forward.apply(self.params, x)

    # training --------------------------------------------------------------

    def fit(self, x_train, y_train, *, epochs: int = 2000, batch_size: int | None = None, verbose: int = 0):
        jnp, jax = self._jnp, self._jax
        key = jax.random.PRNGKey(self.cfg.seed ^ 0x123456)

        if batch_size is None or batch_size >= x_train.shape[0]:
            # full‑batch
            for k in range(epochs):
                self.params, self.opt_state, loss = self._step(self.params, self.opt_state, x_train, y_train)
                if verbose and k % verbose == 0:
                    print(f"[{k}] MSE = {loss:.4e}")
        else:
            n = x_train.shape[0]
            for k in range(epochs):
                key, subk = jax.random.split(key)
                idx = jax.random.choice(subk, n, (batch_size,), replace=False)
                xb, yb = x_train[idx], y_train[idx]
                self.params, self.opt_state, loss = self._step(self.params, self.opt_state, xb, yb)
                if verbose and k % verbose == 0:
                    print(f"[{k}] mini MSE = {loss:.4e}")
        return self

    # persistence -----------------------------------------------------------

    def save(self, path: str | Path):
        payload = {"cfg": asdict(self.cfg), "params": self.params}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        net = cls(ResidualNetConfig(**blob["cfg"]))
        net.params = blob["params"]
        return net

    # compatibility helper --------------------------------------------------

    def to_pickle(self) -> bytes:
        """Return a ``pickle.dumps`` payload (CLI backward‑compat)."""
        return pickle.dumps(self)


# ---------------------------------------------------------------------------
# Backward‑compat alias
# ---------------------------------------------------------------------------

ResidualANN = ResidualNet
