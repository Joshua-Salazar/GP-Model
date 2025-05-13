#!/usr/bin/env python
#!/usr/bin/env python3
"""
cli.backtest
============

Light-weight event–loop that **replays a historical data folder** through the
GP + ANN stack and writes a performance report:

* daily RV signal for every illiquid knot
* realised PnL (mark-to-market + carry/roll)
* notional & DV01 usage
* tear-sheet summary (CAGR, vol, Sharpe, max-DD)

Only **marginal glue** lives here – all finance/math is in
`utils.calibration`, `gp.tiered_gp`, `ann.residual_net`, *etc.*.

Typical usage
-------------

.. code-block:: console

    $ backtest \
        --quotes ./data/eod/*.csv \
        --currency USD \
        --lookback 750 \
        --out ./bt-usd-2020_2025

The engine assumes each `*.csv` contains end-of-day swap quotes for a single
date (filename → `YYYY-MM-DD.csv`).  Curve artefacts are **not** cached – every
date is rebuilt to avoid look-ahead bias; turn on `--cache` for speedier dev
runs.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from gp.tiered_gp import TieredGP
from ann.residual_net import ResidualANN
from utils import calibration as ucal
from utils import data as udata
from utils import plots as uplt

# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Historical replay of GP + ANN RV framework.",
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
# Main back-tester
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None):  # pragma: no cover
    args = _parse(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(pathlib.Path(f).expanduser() for f in args.quotes)
    if not files:
        raise FileNotFoundError("No CSV files matched --quotes")

    # Pre-allocate result containers
    pnl, dv01_hist, notional_hist = [], [], []
    dates, signal_hist = [], []

    # Rolling ANN fit object (refit monthly for realism)
    ann = ResidualANN(hidden_dims=(64, 64), reg=1e-4)

    # Rolling window buffer of past curves
    hist_curves: list[TieredGP] = []

    for f in tqdm(files, desc="↻ back-testing", unit="day"):
        date = pathlib.Path(f).stem  # expects YYYY-MM-DD
        quotes = pd.read_csv(f)

        gp = TieredGP(
            quotes,
            kernel="Brownian",
            tiers=None,  # default canonical
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

        # Current-day alpha
        alpha = ann.predict(ucal.ann_features(gp)[None, :])[0]  # shape (n_knots,)
        signal_hist.append(alpha)
        dates.append(date)

        # ------------------------------ #
        # Risk / PnL bookkeeping
        # ------------------------------ #
        dv01 = ucal.bucket_dv01(gp)
        dv01_hist.append(dv01)
        notional_hist.append(ucal.notional_used(gp, alpha))

        pnl.append(ucal.realised_pnl(gp, alpha))

        # Optional cache for debugging
        if args.cache:
            (args.out / "curves").mkdir(exist_ok=True)
            (args.out / "curves" / f"{date}.pkl").write_bytes(gp.to_pickle())

    # ------------------------------------------------------------------ #
    # Teardown → aggregate & dump
    pnl = pd.Series(pnl, index=pd.to_datetime(dates), name="pnl")
    dv01_hist = pd.DataFrame(dv01_hist, index=pnl.index)
    signals = pd.DataFrame(signal_hist, index=pnl.index)

    pnl.to_csv(args.out / "pnl.csv")
    dv01_hist.to_csv(args.out / "dv01_daily.csv")
    signals.to_csv(args.out / "alpha.csv")

    stats = ucal.performance_tearsheet(pnl)
    stats.to_csv(args.out / "stats.csv")

    # Fancy charts
    uplt.pnl_curve(pnl, save_to=args.out / "pnl.png")
    uplt.signal_heatmap(signals, save_to=args.out / "alpha_heat.png")

    print("✅  Back-test finished:", args.out)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
