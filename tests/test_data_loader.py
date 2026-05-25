"""Tests for data_loader."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_insurance_data, resolve_data_path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_insurance.txt"


def test_resolve_default_ends_with_insurance_data():
    p = resolve_data_path()
    assert p.name == "insurance_data.csv"


def test_load_fixture_smoke():
    df = load_insurance_data(FIXTURE)
    assert len(df) == 500
    assert "TotalPremium" in df.columns
    assert "Make" in df.columns  # renamed from make
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionMonth"])


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_insurance_data("/nonexistent/path/file.csv")
