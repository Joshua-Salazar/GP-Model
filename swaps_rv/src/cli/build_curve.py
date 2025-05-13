#!/usr/bin/env python3
"""
cli.build_curve
===============

Command–line helper that builds a **tiered GP + ANN** relative-value curve
from a simple CSV with market quotes and dumps:

* calibrated discount / IFR grids
* residual ANN alpha series
* bucket-DV01, carry/roll tables
* prettified PNG / PDF plots

Typical usage
-------------

.. code-block:: console

    $ build_curve \
        --csv usd_swaps_eod_2025-05-09.csv \
        --currency USD \
        --ois usd_ois_curve.csv \
        --out build/2025-05-09

The output folder will contain

* `curve.pkl`      –   pickled :class:`gp.tiered_gp.TieredGP`
* `alpha.pkl`      –   pickled :class:`ann.residual_net.ResidualANN`
* `dv01.csv`, `carry_roll.csv`
* `ifr.png`, `alpha_panel.pdf`, ...

All heavy lifting is delegated to **core library** modules; this file is a
thin argparse + orchestration wrapper.

"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from gp.tiered_gp import TieredGP
from ann.residual_net import ResidualANN
from utils import data as udata
from utils import calibration as ucal
from utils import plots as uplt


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
    p.add_argument("--ois", help="Optional OIS discount curve bootstrap CSV.")
    p.add_argument(
        "--tiers",
        help="YAML/JSON defining liquidity tiers; default = canonical USD tiers.",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("build") / datetime.now().strftime("%Y-%m-d"),
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
    if args.ois:
        ois_quotes = pd.read_csv(args.ois)
    else:
        ois_quotes = None

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
    gp = TieredGP(
        quotes,
        tiers=tier_cfg,
        ois_curve=ois_quotes,
        kernel="Brownian",  # default
        optimize_prior=True,
        jit=args.jit,
    )
    gp.calibrate()

    # ------------------------------------------------------------------ #
    # 3. Residual ANN
    ann = ResidualANN(
        hidden_dims=(64, 64),
        reg=1e-4,
        device="cpu",
    )
    X_train, y_train = ucal.residual_dataset(gp, hist_days=750)
    ann.fit(X_train, y_train, epochs=200, batch_size=128)

    # ------------------------------------------------------------------ #
    # 4. Risk tables
    dv01 = ucal.bucket_dv01(gp)
    carry_roll = ucal.carry_roll(gp)

    # ------------------------------------------------------------------ #
    # 5. Persist artefacts
    (args.out / "curve.pkl").write_bytes(gp.to_pickle())
    (args.out / "alpha.pkl").write_bytes(ann.to_pickle())
    dv01.to_csv(args.out / "dv01.csv", index=False)
    carry_roll.to_csv(args.out / "carry_roll.csv", index=False)

    # ------------------------------------------------------------------ #
    # 6. Diagnostics
    if args.plot:
        uplt.plot_ifr(gp, save_to=args.out / "ifr.png")
        uplt.alpha_panel(
            gp,
            ann,
            save_to=args.out / "alpha_panel.pdf",
        )

    print("✅  Build finished:", args.out)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
