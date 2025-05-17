#!/usr/bin/env python3
"""
cli.backtest (risk‑stripped)
===========================

Replay a folder of end‑of‑day swap quote CSVs through **TieredGP → ANN** and
store the daily *alpha* signal (the ANN‑predicted residual at the knot grid).

Removed functionality (compared with the original repo)
------------------------------------------------------
* No DV01 / carry‑roll, notional, or PnL calculations
* No tear‑sheet; only the alpha time‑series gets written
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from ann.residual_net import ResidualNet, ResidualNetConfig
from gp.tiered_gp import TieredGP
from tqdm import tqdm
from utils import calibration as ucal
from utils import data as udata  # noqa: F401 – reserved for future live-feed work

# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #


def _parse(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Historical replay of GP + ANN calibration (risk blocks removed).",
    )

    p.add_argument("quotes", nargs="+", help="Glob of EOD CSV files – one per date.")
    p.add_argument(
        "--currency",
        required=True,
        choices=["USD", "EUR", "GBP", "JPY"],
        help="Market currency (holiday calendar).",
    )
    p.add_argument(
        "--lookback", type=int, default=750, help="Days to feed the ANN (≈3y)."
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("bt") / datetime.now().strftime("%Y-%m-%d_%H%M"),
        help="Output folder.",
    )
    p.add_argument(
        "--cache", action="store_true", help="Cache per‑day GP pickles for debug."
    )
    p.add_argument("--jit", action="store_true", help="Enable Numba JIT in TieredGP.")

    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Back‑test driver (signals only)
# --------------------------------------------------------------------------- #


def main(argv: List[str] | None = None):  # pragma: no cover
    args = _parse(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(pathlib.Path(f).expanduser() for f in args.quotes)
    if not files:
        raise FileNotFoundError("No CSV files matched the --quotes pattern.")

    dates: list[str] = []
    signal_hist: list[np.ndarray] = []
    hist_curves: list[TieredGP] = []

    ann: ResidualNet | None = None  # will be initialised after first curve

    for f in tqdm(files, desc="↻ back‑testing", unit="day"):
        date_str = f.stem  # expects YYYY‑MM‑DD in filename
        df_quotes = pd.read_csv(f)

        # ------------------------------------------------------------------
        # 1. Build GP curve for the day (placeholder fit – user should adapt
        #    to their quote format and design matrix).
        # ------------------------------------------------------------------
        times = df_quotes["tenor"].values.astype(float)
        market_ifr = df_quotes["ifr"].values.astype(float)

        gp = TieredGP(times, store_posterior=False, prior="BM", hl=5.0)
        gp.fit(market_ifr)  # identity design‑matrix assumption

        # ------------------------------------------------------------------
        # 2. Maintain rolling history & (re)fit ANN
        # ------------------------------------------------------------------
        hist_curves.append(gp)
        if len(hist_curves) > args.lookback:
            hist_curves.pop(0)

        # Initialise ANN once we know input / output dims
        if ann is None and len(hist_curves) >= 5:  # wait for a few curves
            X0, y0 = ucal.residual_dataset(hist_curves)
            cfg = ResidualNetConfig(in_dim=X0.shape[1], out_dim=y0.shape[1])
            ann = ResidualNet(cfg)
            ann.fit(X0, y0, epochs=300, batch_size=128)

        # Monthly refit (every 20 business days)
        if ann is not None and len(hist_curves) % 20 == 0:
            X, y = ucal.residual_dataset(hist_curves)
            ann.fit(X, y, epochs=200, batch_size=128)

        # ------------------------------------------------------------------
        # 3. Generate alpha signal for today
        # ------------------------------------------------------------------
        if ann is not None:
            x_today = ucal.ann_features(gp)[None, :]
            alpha = ann(x_today)[0]  # type: ignore[index]
            signal_hist.append(alpha)
        else:
            signal_hist.append(np.full(gp.knots.shape, np.nan))

        dates.append(date_str)

        # Optional caching of the curve object
        if args.cache:
            curve_dir = args.out / "curves"
            curve_dir.mkdir(exist_ok=True)
            (curve_dir / f"{date_str}.pkl").write_bytes(pickle.dumps(gp))

    # ------------------------------------------------------------------
    # Dump alpha panel
    # ------------------------------------------------------------------
    signals = pd.DataFrame(signal_hist, index=pd.to_datetime(dates))
    signals.to_csv(args.out / "alpha.csv")
    print("✅  Back‑test finished – alpha saved to:", args.out / "alpha.csv")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
