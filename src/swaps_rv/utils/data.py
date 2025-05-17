"""
utils.data
==========

Thin I/O layer – keeps *all* external data access in one place so the core
GP / ANN logic stays completely file-system agnostic.

Only a handful of lightweight helpers are exposed; anything more complex
(full-blown data-base adapters, async feeds, …) should live in the caller’s
code-base and feed *DataFrame*s into these functions.

Functions
---------

load_quotes(path, *, tz="UTC")      -> pd.DataFrame
    CSV ↦ tidy frame with **timestamp index** (tz-aware) and _mid_ column.

load_yaml(path)                     -> Any
    Load a YAML/JSON configuration file and return the parsed object.

load_curve_snapshots(dir_)         -> list[TieredGP]
    Read a directory of pickled GP snapshots (chronological sort).

dump_curve(snapshot, dir_)         -> Path
    Persist a single *TieredGP* object as `YYYY-MM-DD.pkl` (gzip -9).

The module has **no** dependencies beyond *pandas >=1.5* and *pyarrow* for
fast parquet, both already listed in *requirements.txt*.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from gp.tiered_gp import TieredGP

# --------------------------------------------------------------------------- #
# Quotes loader
# --------------------------------------------------------------------------- #


def load_quotes(path: str | Path, *, tz: str = "UTC") -> pd.DataFrame:
    """
    Load mid-market quotes CSV.

    Expected columns: ``timestamp,ticker,mid`` – anything else is ignored.
    Result is **multi-index**: (timestamp, ticker) → mid (float).
    """
    df = (
        pd.read_csv(path, parse_dates=["timestamp"])
        .loc[:, ["timestamp", "ticker", "mid"]]
        .dropna()
    )
    df["timestamp"] = df["timestamp"].dt.tz_localize(tz)
    return df.set_index(["timestamp", "ticker"]).sort_index().astype({"mid": "float64"})


# --------------------------------------------------------------------------- #
# Lightweight YAML/JSON loader
# --------------------------------------------------------------------------- #


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML or JSON file and return the deserialised object."""
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        if p.suffix.lower() == ".json":
            return json.load(fh)

        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            msg = "PyYAML is required to parse YAML files"
            raise ImportError(msg) from exc

        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# GP snapshots – simple pickle-store
# --------------------------------------------------------------------------- #


def _ts_from_filename(fname: str) -> _dt.date:
    return _dt.datetime.strptime(fname[:10], "%Y-%m-%d").date()


def load_curve_snapshots(dir_: str | Path) -> List["TieredGP"]:
    """
    Walk *dir_* and unpickle every ``*.pkl(.gz)`` file.

    Files must be named ``YYYY-MM-DD.pkl`` (optionally ``.gz`` compressed).
    Returned list **is sorted** by value-date ascending.
    """
    dir_path = Path(dir_).expanduser().resolve()
    snaps: list[tuple[_dt.date, "TieredGP"]] = []

    for p in dir_path.glob("*.pkl*"):
        ts = _ts_from_filename(p.name)
        with gzip.open(p, "rb") if p.suffix == ".gz" else p.open("rb") as fh:
            snaps.append((ts, pickle.load(fh)))

    snaps.sort(key=lambda t: t[0])
    return [s for _, s in snaps]


def dump_curve(gp: "TieredGP", dir_: str | Path) -> Path:
    """
    Pickle one *TieredGP* snapshot as ``YYYY-MM-DD.pkl.gz`` under *dir_*.

    Overwrites if file already exists.
    """
    dir_path = Path(dir_).expanduser().resolve()
    dir_path.mkdir(parents=True, exist_ok=True)

    fname = f"{gp.value_date:%Y-%m-%d}.pkl.gz"
    out = dir_path / fname
    with gzip.open(out, "wb", compresslevel=9) as fh:
        pickle.dump(gp, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out
