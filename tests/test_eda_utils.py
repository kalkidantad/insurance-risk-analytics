"""Tests for eda_utils."""

from pathlib import Path

import pandas as pd

from src.data_loader import load_insurance_data
from src.eda_utils import (
    add_loss_ratio_and_margin,
    loss_ratio_by_group,
    missing_report,
    portfolio_loss_ratio,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_insurance.txt"


def test_add_loss_ratio_and_margin():
    df = pd.DataFrame({"TotalPremium": [100, 0], "TotalClaims": [30, 10]})
    out = add_loss_ratio_and_margin(df)
    assert out["LossRatio"].iloc[0] == 0.3
    assert pd.isna(out["LossRatio"].iloc[1])
    assert out["Margin"].iloc[0] == 70


def test_portfolio_loss_ratio_on_fixture():
    df = load_insurance_data(FIXTURE)
    lr = portfolio_loss_ratio(df)
    assert lr == lr and lr >= 0  # finite non-negative


def test_loss_ratio_by_group():
    df = load_insurance_data(FIXTURE)
    by = loss_ratio_by_group(df, "Province")
    assert "LossRatio" in by.columns
    assert len(by) >= 1


def test_missing_report_runs():
    df = load_insurance_data(FIXTURE)
    r = missing_report(df)
    assert "missing" in r.columns
