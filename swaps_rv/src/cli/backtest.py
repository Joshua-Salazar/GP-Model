#!/usr/bin/env python3
"""
cli.backtest (risk-stripped)
============================

Light-weight event-loop that **replays a historical data folder** through the
GP + ANN stack and records the daily residual-alpha signal – nothing else.

Removed functionality
---------------------
* bucket-DV01, carry/roll, notional usage
* realised PnL & performance tear-sheet
* all DV01 / PnL plots
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import pickle                         # ➜ needed for on-disk caching
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from gp.tiered_gp import TieredGP
from ann.residual_net import ResidualNet       # ➜ class is called ResidualNet
from utils import calibration as ucal
from utils import data as udata   # still handy if you later add live feeds


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Historical replay of GP + ANN calibration (risk blocks removed).",
    )

    p.add_argument(
        "--quotes",
        required=True,
        nargs="+",
        help="Glob to EOD quote CSV files (one date per file).",
    )
    p.add_argument(
        "--currency",
        required=True,
        choices=["USD", "EUR", "GBP", "JPY"],
        help="Market currency (holidays, calendars).",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=750,
        help="Days of history to feed residual-ANN (≈3y EOD).",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("bt") / datetime.now().strftime("%Y-%m-%d_%H%M"),
        help="Output folder.",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help="Cache daily curve artefacts for speed (unsafe for prod).",
    )
    p.add_argument("--jit", action="store_true", help="Enable Numba JIT.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main back-tester (signals-only)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None):  # pragma: no cover
    args = _parse(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(pathlib.Path(f).expanduser() for f in args.quotes)
    if not files:
        raise FileNotFoundError("No CSV files matched --quotes")

    # Containers
    dates: list[str] = []
    signal_hist: list[np.ndarray] = []

    # Rolling ANN fit object (refit monthly for realism)
    ann = ResidualNet(hidden_dims=(64, 64), reg=1e-4)

    # Rolling window buffer of past curves
    hist_curves: list[TieredGP] = []

    for f in tqdm(files, desc="↻ back-testing", unit="day"):
        date = pathlib.Path(f).stem  # expects YYYY-MM-DD in filename stem
        quotes = pd.read_csv(f)

        gp = TieredGP(
            quotes,
            kernel="Brownian",
            tiers=None,          # default canonical tiers inside the class
            optimize_prior=False,
            jit=args.jit,
        )
        gp.calibrate()

        # ------------------------------ #
        # ANN fit / update
        # ------------------------------ #
        hist_curves.append(gp)
        if len(hist_curves) > args.lookback:
            hist_curves.pop(0)

        # Re-train ANN once per 20 business days
        if len(hist_curves) % 20 == 0:
            X, y = ucal.residual_dataset(hist_curves)
            ann.fit(X, y, epochs=100, batch_size=128)

        # Current-day residual alpha
        alpha = ann.predict(ucal.ann_features(gp)[None, :])[0]  # shape (n_knots,)
        signal_hist.append(alpha)
        dates.append(date)

        # Optional cache for debugging
        if args.cache:
            curve_dir = args.out / "curves"
            curve_dir.mkdir(exist_ok=True)
            (curve_dir / f"{date}.pkl").write_bytes(pickle.dumps(gp))  # ➜ use stdlib pickle

    # ------------------------------------------------------------------ #
    # Aggregate & dump signals
    signals = pd.DataFrame(signal_hist, index=pd.to_datetime(dates))
    signals.to_csv(args.out / "alpha.csv")

    print("✅  Back-test finished (signals only):", args.out)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())