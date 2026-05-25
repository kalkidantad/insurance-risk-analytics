"""Statistical hypothesis tests for A/B and segmentation analysis (Week 3 Task 2+)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def chi2_independence(table: pd.DataFrame) -> tuple[float, float, int, int]:
    """
    Chi-squared test of independence on a contingency table.

    Returns
    -------
    statistic, pvalue, dof, expected_shape (rows, cols) as last tuple ints for dof check.
    """
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p), int(dof), int(expected.size)


def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test (unequal variance). Returns statistic, pvalue."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    return stats.ttest_ind(a, b, equal_var=False)
