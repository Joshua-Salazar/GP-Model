"""
Hybrid “GP + ANN” residual model.

The network learns the *non–Gaussian* residual
    Δf(t) := f_markt(t) − f_GP(t)
conditional on

    X  =  [ f_GP(t₁), …, f_GP(tₙ),  R ]          (1)

where R = (level, slope_front, slope_back, curvature, σ_lv) are the
**Crest** macro-descriptors pre-computed upstream.

The class is a *thin* wrapper around JAX/Haiku/Optax so that the rest of
the code-base never touches deep-learning libraries directly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import haiku as hk
import optax

# ------------------------------------------------------- architecture helpers


def _mlp(hidden: Sequence[int], out_dim: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return an MLP builder for Haiku transform."""
    def net(x):
        for h in hidden:
            x = hk.Linear(h)(x)
            x = jax.nn.tanh(x)
        return hk.Linear(out_dim)(x)

    return net


# ------------------------------------------------------- main dataclass


@dataclass
class ResidualNetConfig:
    in_dim: int
    out_dim: int
    hidden: tuple[int, ...] = (32, 32)
    lr: float = 3e-3
    l2: float = 1e-4
    seed: int = 0


class ResidualNet:
    """
    Examples
    --------
    >>> cfg = ResidualNetConfig(in_dim=9, out_dim=60)
    >>> net = ResidualNet(cfg)
    >>> net.fit(X_train, y_train, 2000)
    >>> delta = net(X_new)
    """

    def __init__(self, cfg: ResidualNetConfig):
        self.cfg = cfg
        self._build()

    # --------------------------------------------------- private

    def _build(self):
        mlp = _mlp(self.cfg.hidden, self.cfg.out_dim)
        self._forward = hk.without_apply_rng(hk.transform(mlp))

        key = jax.random.PRNGKey(self.cfg.seed)
        dummy = jnp.zeros((1, self.cfg.in_dim))
        self.params = self._forward.init(key, dummy)

        opt = optax.adamw(self.cfg.lr, weight_decay=self.cfg.l2)
        self.opt_state = opt.init(self.params)
        self._opt_update = opt.update

        # jit-ed step --------------------------------------------------------

        @jax.jit
        def _step(params, opt_state, x, y):
            def loss_fn(p, xb, yb):
                preds = self._forward.apply(p, xb)
                return jnp.mean((preds - yb) ** 2)

            grads = jax.grad(loss_fn)(params, x, y)
            updates, opt_state2 = self._opt_update(grads, opt_state)
            params2 = optax.apply_updates(params, updates)
            loss_val = jnp.mean((self._forward.apply(params2, x) - y) ** 2)
            return params2, opt_state2, loss_val

        self._step = _step

    # --------------------------------------------------- API

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict Δf(t) given design vector `x`."""
        return self._forward.apply(self.params, x)

    # ---------------------------------------------- training routine

    def fit(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        n_iter: int = 5000,
        batch_size: int | None = None,
        verbose: int = 0,
    ):
        key = jax.random.PRNGKey(self.cfg.seed + 1)

        if batch_size is None or batch_size >= x_train.shape[0]:
            # full-batch
            for k in range(n_iter):
                self.params, self.opt_state, loss = self._step(
                    self.params, self.opt_state, x_train, y_train
                )
                if verbose and k % verbose == 0:
                    print(f"[{k}] MSE = {loss:.4e}")
        else:
            # mini-batch
            for k in range(n_iter):
                key, sub = jax.random.split(key)
                idx = jax.random.choice(sub, x_train.shape[0], (batch_size,), replace=False)
                xb, yb = x_train[idx], y_train[idx]
                self.params, self.opt_state, loss = self._step(self.params, self.opt_state, xb, yb)
                if verbose and k % verbose == 0:
                    print(f"[{k}] mini-batch MSE = {loss:.4e}")

        return self

    # ------------------------------------------------ persistence

    def save(self, path: str | Path):
        data = {
            "cfg": asdict(self.cfg),
            "params": self.params,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualNet":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        net = cls(ResidualNetConfig(**blob["cfg"]))
        net.params = blob["params"]
        return net
