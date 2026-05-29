"""Tests for modeling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    ModelTrainer,
    engineer_features,
    naive_premium_baseline,
    risk_adjusted_premium,
    save_shap_summary,
)


def _synthetic_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "PolicyID": np.repeat(np.arange(n // 4), 4),
            "TransactionMonth": pd.date_range("2015-01-01", periods=n, freq="ME"),
            "RegistrationYear": rng.integers(2000, 2012, size=n),
            "TotalClaims": np.zeros(n),
            "TotalPremium": rng.uniform(20, 200, size=n),
            "CalculatedPremiumPerTerm": rng.uniform(20, 200, size=n),
            "SumInsured": rng.uniform(1e4, 1e5, size=n),
            "Province": rng.choice(["A", "B"], size=n),
            "VehicleType": "Passenger Vehicle",
            "CoverType": "X",
            "Gender": rng.choice(["Male", "Female"], size=n),
            "PostalCode": rng.choice([1, 2], size=n),
            "Cubiccapacity": rng.integers(1200, 3000, size=n),
            "Kilowatts": rng.integers(50, 200, size=n),
            "NumberOfDoors": rng.integers(3, 5, size=n),
        }
    )
    df.loc[: n // 3, "TotalClaims"] = rng.uniform(50, 800, size=n // 3 + 1)
    return df


def test_engineer_features_adds_columns():
    df = _synthetic_frame(120)
    out = engineer_features(df)
    assert "vehicle_age_years" in out.columns
    assert "has_claim" in out.columns
    assert out["has_claim"].isin([0, 1]).all()


def test_model_trainer_severity_and_classifier(tmp_path):
    df = _synthetic_frame(500)
    trainer = ModelTrainer(random_state=0)
    reg_tbl, reg_models = trainer.fit_eval_severity(
        df, test_size=0.25, n_estimators=40, max_depth_rf=6
    )
    assert set(reg_tbl["model"]) >= {"linear_regression", "random_forest"}
    assert reg_tbl["rmse"].notna().all()

    cls_tbl, _ = trainer.fit_eval_claim_classifier(
        df, test_size=0.25, n_estimators=40, n_estimators_xgb=60
    )
    assert {"accuracy", "f1"}.issubset(cls_tbl.columns)

    pipe = reg_models["random_forest"]
    d = engineer_features(df)
    d = d[pd.to_numeric(d["TotalClaims"], errors="coerce").fillna(0) > 0]
    num = [
        c
        for c in [
            "RegistrationYear",
            "CalculatedPremiumPerTerm",
            "SumInsured",
            "Cubiccapacity",
            "Kilowatts",
            "NumberOfDoors",
            "vehicle_age_years",
            "policy_row_index",
        ]
        if c in d.columns
    ]
    cat = [
        c
        for c in ["Province", "VehicleType", "CoverType", "Gender", "PostalCode"]
        if c in d.columns
    ]
    img = save_shap_summary(pipe, d[num + cat].head(120), tmp_path / "shap.png")
    assert img.is_file() and img.stat().st_size > 1000


def test_naive_premium_and_risk_formula():
    df = pd.DataFrame({"CalculatedPremiumPerTerm": [10.0, np.nan, 5.0]})
    base = naive_premium_baseline(df)
    assert base.shape == (3,)
    prem = risk_adjusted_premium(np.array([0.2, 0.1, 0.0]), np.array([1000.0, 500.0, 200.0]))
    assert prem.shape == (3,)
    assert prem[2] == pytest.approx(50.0, rel=1e-9)  # expense only when pure premium zero
