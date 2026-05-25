"""Smoke tests for hypothesis_tests."""

import numpy as np
import pandas as pd

from src.hypothesis_tests import chi2_independence, welch_ttest


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
