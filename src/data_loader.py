"""Load and lightly normalize ACIS insurance tabular data (pipe-delimited)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Default path relative to project root (parent of src/)
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "insurance_data.csv"


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Return path to the dataset, from arg, INSURANCE_DATA_PATH env, or default."""
    if path is not None:
        return Path(path).expanduser().resolve()
    env = os.environ.get("INSURANCE_DATA_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_DATA


def load_insurance_data(
    path: str | Path | None = None,
    *,
    nrows: int | None = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Read pipe-delimited insurance data (same format as MachineLearningRating_v3).

    Parameters
    ----------
    path
        File path; defaults to data/insurance_data.csv or INSURANCE_DATA_PATH.
    nrows
        Optional row limit for faster exploration.
    low_memory
        Passed to pandas.read_csv (False coerces mixed types more eagerly).
    """
    p = resolve_data_path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Data file not found: {p}. "
            "Copy the challenge dataset to data/insurance_data.csv or set INSURANCE_DATA_PATH."
        )
    df = pd.read_csv(
        p,
        sep="|",
        nrows=nrows,
        low_memory=low_memory,
        parse_dates=["TransactionMonth"],
        dayfirst=False,
    )
    # Harmonise column names to PascalCase-style used in documentation where helpful
    rename = {
        "mmcode": "Mmcode",
        "make": "Make",
        "cubiccapacity": "Cubiccapacity",
        "kilowatts": "Kilowatts",
        "bodytype": "Bodytype",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df
