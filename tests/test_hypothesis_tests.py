"""Smoke tests for hypothesis_tests."""

import numpy as np
import pandas as pd

from src.hypothesis_tests import (
    aggregate_policy_kpis,
    chi2_claim_frequency_two_samples,
    chi2_independence,
    run_hypothesis_suite,
    welch_ttest,
)


def test_chi2_runs():
    table = pd.DataFrame([[10, 20], [30, 40]])
    chi2, p, dof, _ = chi2_independence(table)
    assert chi2 >= 0
    assert 0 <= p <= 1
    assert dof == 1


def test_welch_ttest_runs():
    a = np.random.default_rng(0).normal(0, 1, 50)
    b = np.random.default_rng(1).normal(0.5, 1.2, 50)
    stat, p = welch_ttest(a, b)
    assert np.isfinite(stat)
    assert 0 <= p <= 1


def test_aggregate_policy_kpis_and_chi2():
    rows = [
        [1, 100.0, 0.0, "P1", 10, "Passenger Vehicle", "Male"],
        [1, 50.0, 20.0, "P1", 10, "Passenger Vehicle", "Male"],
        [2, 80.0, 0.0, "P1", 10, "Passenger Vehicle", "Female"],
        [2, 80.0, 0.0, "P1", 10, "Passenger Vehicle", "Female"],
        [3, 200.0, 50.0, "P2", 20, "Passenger Vehicle", "Male"],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "PolicyID",
            "TotalPremium",
            "TotalClaims",
            "Province",
            "PostalCode",
            "VehicleType",
            "Gender",
        ],
    )
    pol = aggregate_policy_kpis(df)
    assert len(pol) == 3
    assert pol.set_index("PolicyID").loc[1, "has_claim"] == 1
    chi2, p = chi2_claim_frequency_two_samples([1, 1], [0, 0])
    assert np.isfinite(chi2) and 0 <= p <= 1


def test_run_hypothesis_suite_synthetic():
    """Enough policies in each cell for default thresholds in ``run_hypothesis_suite``."""
    rng = np.random.default_rng(42)
    n = 120
    base = pd.DataFrame(
        {
            "PolicyID": np.arange(1, n + 1),
            "TotalPremium": rng.uniform(50, 200, size=n),
            "TotalClaims": np.zeros(n),
            "Province": ["Western Cape"] * (n // 2) + ["Gauteng"] * (n // 2),
            "PostalCode": [2000] * (n // 2) + [122] * (n // 2),
            "VehicleType": "Passenger Vehicle",
            "Gender": ["Male"] * (n - 40) + ["Female"] * 40,
        }
    )
    base.loc[base["Province"] == "Gauteng", "PostalCode"] = np.where(
        base.loc[base["Province"] == "Gauteng"].index % 2 == 0, 2000, 122
    )
    # inject claims
    claim_idx = rng.choice(base.index, size=25, replace=False)
    base.loc[claim_idx, "TotalClaims"] = rng.uniform(10, 80, size=len(claim_idx))
    res = run_hypothesis_suite(base)
    assert len(res) == 4
    assert all(0 <= r.p_value <= 1 for r in res)
