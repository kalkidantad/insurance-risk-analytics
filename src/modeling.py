"""Predictive modeling for claim severity, claim incidence, and pricing (Task 4).

This module builds a reusable tabular pipeline: feature engineering, preprocessing
(imputation + one-hot encoding), train/test split, and model comparison for:

* **Severity regression** — rows with ``TotalClaims > 0``, target ``TotalClaims``.
* **Claim classifier** — all rows (or a policy-level frame), target ``has_claim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional until pip install
    XGBRegressor = None  # type: ignore[misc, assignment]
    XGBClassifier = None  # type: ignore[misc, assignment]


def build_preprocess_transformer(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Default sklearn preprocessing for mixed tabular insurance features."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple business features used for pricing / risk.

    * ``vehicle_age_years`` — approximate years since registration vs transaction month.
    * ``policy_row_index`` — ordinal row index within each ``PolicyID`` (proxy for exposure length).
    """
    out = df.copy()
    if "TransactionMonth" in out.columns:
        tm = pd.to_datetime(out["TransactionMonth"], errors="coerce")
        yr = tm.dt.year
        if "RegistrationYear" in out.columns:
            ry = pd.to_numeric(out["RegistrationYear"], errors="coerce")
            out["vehicle_age_years"] = (yr - ry).clip(lower=0, upper=80)
        else:
            out["vehicle_age_years"] = np.nan
    else:
        out["vehicle_age_years"] = np.nan

    if "PolicyID" in out.columns:
        out["policy_row_index"] = out.groupby("PolicyID").cumcount()
    else:
        out["policy_row_index"] = 0

    out["has_claim"] = (pd.to_numeric(out.get("TotalClaims"), errors="coerce").fillna(0) > 0).astype(
        int
    )
    return out


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric / categorical columns present in ``df`` for modeling."""
    numeric_candidates = [
        "RegistrationYear",
        "CalculatedPremiumPerTerm",
        "SumInsured",
        "Cubiccapacity",
        "Kilowatts",
        "NumberOfDoors",
        "vehicle_age_years",
        "policy_row_index",
    ]
    categorical_candidates = [
        "Province",
        "VehicleType",
        "CoverType",
        "Gender",
        "PostalCode",
    ]
    num = [c for c in numeric_candidates if c in df.columns]
    cat = [c for c in categorical_candidates if c in df.columns]
    return num, cat


class ModelTrainer:
    """Train and evaluate sklearn / XGBoost models on engineered insurance frames."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit_eval_severity(
        self,
        df: pd.DataFrame,
        *,
        test_size: float = 0.2,
        max_depth_rf: int = 12,
        n_estimators: int = 200,
    ) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
        """Train severity models on rows with ``TotalClaims > 0``."""
        d = engineer_features(df)
        d = d[pd.to_numeric(d["TotalClaims"], errors="coerce").fillna(0) > 0].copy()
        y = pd.to_numeric(d["TotalClaims"], errors="coerce")
        num, cat = select_feature_columns(d)
        if not num and not cat:
            raise ValueError("No usable feature columns after engineering.")

        X = d[num + cat]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        pre = build_preprocess_transformer(num, cat)

        models: dict[str, Any] = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth_rf,
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }
        if XGBRegressor is not None:
            models["xgboost"] = XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
            )

        fitted: dict[str, Pipeline] = {}
        rows: list[dict[str, float | str]] = []
        for name, est in models.items():
            pipe = Pipeline([("prep", clone(pre)), ("model", est)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            r2 = float(r2_score(y_test, pred))
            rows.append({"model": name, "rmse": rmse, "r2": r2})
            fitted[name] = pipe

        return pd.DataFrame(rows), fitted

    def fit_eval_claim_classifier(
        self,
        df: pd.DataFrame,
        *,
        test_size: float = 0.2,
        n_estimators: int = 300,
        n_estimators_xgb: int = 200,
    ) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
        """Binary claim classifier at row level (``has_claim``)."""
        d = engineer_features(df)
        y = d["has_claim"]
        num, cat = select_feature_columns(d)
        X = d[num + cat]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        pre = build_preprocess_transformer(num, cat)
        models: dict[str, Any] = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=14,
                random_state=self.random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        }
        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(
                n_estimators=n_estimators_xgb,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="logloss",
            )

        fitted: dict[str, Pipeline] = {}
        rows: list[dict[str, float | str]] = []
        for name, est in models.items():
            pipe = Pipeline([("prep", clone(pre)), ("model", est)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            rows.append(
                {
                    "model": name,
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "precision": float(precision_score(y_test, pred, zero_division=0)),
                    "recall": float(recall_score(y_test, pred, zero_division=0)),
                    "f1": float(f1_score(y_test, pred, zero_division=0)),
                }
            )
            fitted[name] = pipe

        return pd.DataFrame(rows), fitted


def naive_premium_baseline(df: pd.DataFrame) -> np.ndarray:
    """Return ``CalculatedPremiumPerTerm`` as a vector (naive benchmark)."""
    return pd.to_numeric(df["CalculatedPremiumPerTerm"], errors="coerce").fillna(0).to_numpy()


def risk_adjusted_premium(
    p_claim: np.ndarray,
    severity_hat: np.ndarray,
    *,
    expense_loading: float = 50.0,
    profit_margin: float = 0.12,
) -> np.ndarray:
    """
    Simple pricing skeleton:

    ``Premium ≈ P(claim) * E[severity | claim] * (1 + profit_margin) + expense_loading``

    ``p_claim`` and ``severity_hat`` must be aligned to the same row index.
    """
    p_claim = np.clip(np.asarray(p_claim, dtype=float), 0, 1)
    severity_hat = np.asarray(severity_hat, dtype=float)
    pure = p_claim * severity_hat
    return pure * (1.0 + profit_margin) + expense_loading


def save_shap_summary(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    out_path: str | Path,
    *,
    max_display: int = 10,
) -> Path:
    """
    Write a SHAP summary bar plot for a tree-based ``Pipeline`` (prep + tree model).

    Raises ``ImportError`` if ``shap`` is not installed.
    """
    import matplotlib.pyplot as plt

    import shap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prep: ColumnTransformer = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_t = prep.transform(X_sample)
    feature_names = prep.get_feature_names_out()

    if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_t)
        if isinstance(shap_vals, list):  # binary classifier
            shap_vals = shap_vals[1]
        vec = shap.Explanation(shap_vals, data=X_t, feature_names=feature_names)
    elif XGBRegressor is not None and isinstance(
        model,
        (XGBRegressor, XGBClassifier),  # type: ignore[name-defined]
    ):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_t)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        vec = shap.Explanation(shap_vals, data=X_t, feature_names=feature_names)
    else:
        raise TypeError("SHAP summary is implemented for tree ensembles in this project.")

    shap.plots.bar(vec, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path
