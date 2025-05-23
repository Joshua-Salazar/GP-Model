#!/usr/bin/env python3
"""
cli.build_curve
===============

Command-line helper that builds a **tiered GP + ANN** relative-value curve
from a simple CSV with market quotes and dumps:

* calibrated discount / IFR grids
* residual-ANN alpha series
* prettified PNG / PDF plots

Risk analytics (DV01, carry/roll, tear sheets) have been **removed**.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
from datetime import datetime

import numpy as np  # noqa: F401  (kept for future extensions)
import pandas as pd

from swaps_rv.ann.residual_net import ResidualNet
from swaps_rv.gp.tiered_gp import TieredGP
from swaps_rv.utils import calibration as ucal
from swaps_rv.utils import data as udata
from swaps_rv.utils import plots as uplt


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_curve",
        description="Calibrate tiered GP + ANN RV framework from EOD quotes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--csv", required=True, help="Market quotes CSV.")
    p.add_argument(
        "--currency",
        required=True,
        choices=["USD", "EUR", "GBP", "JPY"],
        help="Quote currency – drives calendar/holidays.",
    )
    p.add_argument("--ois", help="Optional OIS discount-curve bootstrap CSV.")
    p.add_argument(
        "--tiers",
        help="YAML/JSON defining liquidity tiers; default = canonical USD tiers.",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("build") / datetime.now().strftime("%Y-%m-%d"),
        help="Output folder.",
    )
    p.add_argument(
        "--plot/--no-plot",
        dest="plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle generation of PNG/PDF diagnostics.",
    )
    p.add_argument(
        "--jit",
        default=False,
        action="store_true",
        help="Numba @njit calibration for large grids.",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load data
    quotes = pd.read_csv(args.csv)

    # canonical tier map if none given
    tier_cfg = (
        udata.load_yaml(args.tiers)
        if args.tiers
        else {
            0: [30],
            1: [2, 5, 10],
            2: [1, 7, 20],
            3: [3, 15, 25],
            4: "rest",
        }
    )

    # ------------------------------------------------------------------ #
    # 2. GP calibration
    times = quotes["tenor"].astype(float).values
    ifr = quotes["ifr"].astype(float).values

    gp = TieredGP(times, tiers=tier_cfg, store_posterior=False)
    gp.fit(ifr)

    # ------------------------------------------------------------------ #
    # 3. Residual ANN
    ann = ResidualNet(
        hidden_dims=(64, 64),
        reg=1e-4,
        device="cpu",
    )
    # residual_dataset expects an iterable of curves → wrap gp in a list
    X_train, y_train = ucal.residual_dataset([gp])
    ann.fit(X_train, y_train, epochs=200, batch_size=128)

    # ------------------------------------------------------------------ #
    # 4. Persist artefacts
    (args.out / "curve.pkl").write_bytes(pickle.dumps(gp))
    (args.out / "alpha.pkl").write_bytes(pickle.dumps(ann))

    # ------------------------------------------------------------------ #
    # 5. Diagnostics
    if args.plot:
        uplt.curve(gp, save_to=args.out / "ifr.png")  # IFR curve
        uplt.ann_surface(
            ann, X_train, y_train, save_to=args.out / "ann.png"
        )  # ANN residual map

    print("✅  Build finished:", args.out)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
