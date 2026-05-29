"""Statistical hypothesis tests for A/B and segmentation analysis (ACIS Week 3 Task 3).

KPI definitions (aligned with the challenge brief)
--------------------------------------------------
* **Claim frequency** — proportion of *policies* with at least one claim on any row.
* **Claim severity** — mean ``TotalClaims`` among policies (or rows) where a claim occurred.
* **Margin** — ``TotalPremium − TotalClaims``, aggregated per policy as sums then difference.

Tests
-----
* Claim frequency vs a binary split (e.g. two provinces): **chi-squared** on a 2×2 contingency
  (claim vs no-claim × group A vs group B).
* Severity or margin (continuous): **Welch t-test** on independent samples (unequal variance).

``p < 0.05`` ⇒ reject H₀.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

KPI = Literal["claim_frequency", "claim_severity", "margin"]


@dataclass(frozen=True)
class HypothesisTestResult:
    """Structured output for reporting tables and notebooks."""

    hypothesis_id: str
    null_hypothesis: str
    kpi: KPI
    group_a_label: str
    group_b_label: str
    test_name: str
    p_value: float
    reject_h0: bool
    alpha: float
    n_a: int
    n_b: int
    detail: str


def chi2_independence(table: pd.DataFrame | np.ndarray) -> tuple[float, float, int, int]:
    """
    Chi-squared test of independence on a contingency table.

    Returns
    -------
    statistic, pvalue, dof, expected cell count (flattened size for sanity checks).
    """
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p), int(dof), int(expected.size)


def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test (unequal variance). Returns statistic, pvalue."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def aggregate_policy_kpis(
    df: pd.DataFrame,
    *,
    policy_col: str = "PolicyID",
    premium_col: str = "TotalPremium",
    claims_col: str = "TotalClaims",
    carry_first: tuple[str, ...] = (
        "Province",
        "PostalCode",
        "VehicleType",
        "Gender",
    ),
) -> pd.DataFrame:
    """
    Roll transaction-level rows up to one row per policy with KPI-ready fields.

    ``has_claim`` is 1 if any row for the policy has ``TotalClaims > 0``.
    ``severity_if_claim`` is mean ``TotalClaims`` over rows with ``TotalClaims > 0`` (0 if none).

    Only ``carry_first`` dimension columns are retained (``first`` per policy) to keep the
    aggregation fast on large extracts.
    """
    need = {policy_col, premium_col, claims_col, *carry_first}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"aggregate_policy_kpis: missing columns {sorted(missing)}")

    cols = [policy_col, premium_col, claims_col, *carry_first]
    d = df.loc[:, cols].copy()
    d[claims_col] = pd.to_numeric(d[claims_col], errors="coerce").fillna(0.0)
    d[premium_col] = pd.to_numeric(d[premium_col], errors="coerce").fillna(0.0)
    d["_row_claim"] = (d[claims_col] > 0).astype(int)

    agg_spec: dict[str, str] = {
        premium_col: "sum",
        claims_col: "sum",
        "_row_claim": "max",
    }
    for c in carry_first:
        agg_spec[c] = "first"

    out = d.groupby(policy_col, as_index=False).agg(agg_spec)
    out = out.rename(columns={"_row_claim": "has_claim"})
    out["margin"] = out[premium_col] - out[claims_col]
    sev = d[d[claims_col] > 0].groupby(policy_col)[claims_col].mean().rename("severity_if_claim")
    out = out.merge(sev, on=policy_col, how="left")
    out["severity_if_claim"] = out["severity_if_claim"].fillna(0.0)
    return out


def chi2_claim_frequency_two_samples(
    has_claim_a: np.ndarray | pd.Series,
    has_claim_b: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """
    Chi-squared test for difference in claim incidence between two independent policy groups.

    Builds a 2×2 table: rows = group A / group B, columns = no claim / claim.
    """
    a = np.asarray(has_claim_a).astype(int).ravel()
    b = np.asarray(has_claim_b).astype(int).ravel()
    a_claim, a_noclaim = int(a.sum()), int(len(a) - a.sum())
    b_claim, b_noclaim = int(b.sum()), int(len(b) - b.sum())
    table = [[a_noclaim, a_claim], [b_noclaim, b_claim]]
    chi2, p, _, _ = chi2_independence(np.array(table))
    return chi2, p


def cohort_mask_two_categories(
    policies: pd.DataFrame,
    column: str,
    value_a: object,
    value_b: object,
    *,
    extra_masks: tuple[pd.Series, ...] = (),
) -> tuple[pd.Series, pd.Series]:
    """Boolean masks for policies in group A or B, optionally intersected with ``extra_masks``."""
    base = policies[column].notna()
    for m in extra_masks:
        base = base & m
    m_a = base & (policies[column] == value_a)
    m_b = base & (policies[column] == value_b)
    return m_a, m_b


def run_hypothesis_suite(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    province_a: str = "Western Cape",
    province_b: str = "Gauteng",
    postal_a: int | str = 2000,
    postal_b: int | str = 122,
    cohort_province: str = "Gauteng",
    cohort_vehicle_type: str = "Passenger Vehicle",
) -> list[HypothesisTestResult]:
    """
    Run the four standard ACIS hypotheses on the raw transaction dataframe.

    Zip-code hypotheses use a **matched cohort**: same ``Province`` and ``VehicleType`` so
    client/vehicle mix is aligned before comparing two high-volume postal codes.
    """
    pol = aggregate_policy_kpis(df)
    results: list[HypothesisTestResult] = []

    # --- H1: provinces (claim frequency) ---
    m_a, m_b = cohort_mask_two_categories(pol, "Province", province_a, province_b)
    if m_a.sum() < 30 or m_b.sum() < 30:
        raise ValueError("Insufficient policies for province test after filtering.")
    chi2, p = chi2_claim_frequency_two_samples(pol.loc[m_a, "has_claim"], pol.loc[m_b, "has_claim"])
    freq_a = float(pol.loc[m_a, "has_claim"].mean())
    freq_b = float(pol.loc[m_b, "has_claim"].mean())
    results.append(
        HypothesisTestResult(
            hypothesis_id="H_province",
            null_hypothesis="No risk difference across provinces (claim incidence).",
            kpi="claim_frequency",
            group_a_label=f"Province={province_a}",
            group_b_label=f"Province={province_b}",
            test_name="Chi-squared (2×2 claim incidence)",
            p_value=float(p),
            reject_h0=bool(p < alpha),
            alpha=alpha,
            n_a=int(m_a.sum()),
            n_b=int(m_b.sum()),
            detail=f"Claim frequency {province_a}={freq_a:.3%}, {province_b}={freq_b:.3%}; chi2={chi2:.4f}",
        )
    )

    # --- H2: zip codes within matched cohort (claim frequency) ---
    cohort = (pol["Province"] == cohort_province) & (pol["VehicleType"] == cohort_vehicle_type)
    m_za, m_zb = cohort_mask_two_categories(
        pol, "PostalCode", postal_a, postal_b, extra_masks=(cohort,)
    )
    if m_za.sum() < 25 or m_zb.sum() < 25:
        raise ValueError("Insufficient policies for postal code test in matched cohort.")
    chi2z, pz = chi2_claim_frequency_two_samples(pol.loc[m_za, "has_claim"], pol.loc[m_zb, "has_claim"])
    freq_za = float(pol.loc[m_za, "has_claim"].mean())
    freq_zb = float(pol.loc[m_zb, "has_claim"].mean())
    results.append(
        HypothesisTestResult(
            hypothesis_id="H_postal_risk",
            null_hypothesis="No risk difference between zip codes (claim incidence, matched cohort).",
            kpi="claim_frequency",
            group_a_label=f"PostalCode={postal_a} ({cohort_province}, {cohort_vehicle_type})",
            group_b_label=f"PostalCode={postal_b} ({cohort_province}, {cohort_vehicle_type})",
            test_name="Chi-squared (2×2 claim incidence)",
            p_value=float(pz),
            reject_h0=bool(pz < alpha),
            alpha=alpha,
            n_a=int(m_za.sum()),
            n_b=int(m_zb.sum()),
            detail=f"Claim frequency zip {postal_a}={freq_za:.3%}, zip {postal_b}={freq_zb:.3%}; chi2={chi2z:.4f}",
        )
    )

    # --- H3: margin difference between same two zips (Welch t on policy margin) ---
    mar_a = pol.loc[m_za, "margin"].astype(float)
    mar_b = pol.loc[m_zb, "margin"].astype(float)
    _, pm = welch_ttest(mar_a.values, mar_b.values)
    mean_a, mean_b = float(mar_a.mean()), float(mar_b.mean())
    results.append(
        HypothesisTestResult(
            hypothesis_id="H_postal_margin",
            null_hypothesis="No margin difference between zip codes (matched cohort).",
            kpi="margin",
            group_a_label=f"PostalCode={postal_a}",
            group_b_label=f"PostalCode={postal_b}",
            test_name="Welch t-test (policy margin)",
            p_value=float(pm),
            reject_h0=bool(pm < alpha),
            alpha=alpha,
            n_a=int(m_za.sum()),
            n_b=int(m_zb.sum()),
            detail=f"Mean margin zip {postal_a}={mean_a:.2f}, zip {postal_b}={mean_b:.2f}",
        )
    )

    # --- H4: gender (claim frequency, Male vs Female) ---
    gmask = pol["Gender"].isin(["Male", "Female"])
    m_m = gmask & (pol["Gender"] == "Male")
    m_f = gmask & (pol["Gender"] == "Female")
    if m_m.sum() < 30 or m_f.sum() < 30:
        raise ValueError("Insufficient Male/Female policies for gender test.")
    chi2g, pg = chi2_claim_frequency_two_samples(pol.loc[m_m, "has_claim"], pol.loc[m_f, "has_claim"])
    freq_m = float(pol.loc[m_m, "has_claim"].mean())
    freq_f = float(pol.loc[m_f, "has_claim"].mean())
    results.append(
        HypothesisTestResult(
            hypothesis_id="H_gender",
            null_hypothesis="No risk difference between women and men (claim incidence).",
            kpi="claim_frequency",
            group_a_label="Male",
            group_b_label="Female",
            test_name="Chi-squared (2×2 claim incidence)",
            p_value=float(pg),
            reject_h0=bool(pg < alpha),
            alpha=alpha,
            n_a=int(m_m.sum()),
            n_b=int(m_f.sum()),
            detail=f"Claim frequency Male={freq_m:.3%}, Female={freq_f:.3%}; chi2={chi2g:.4f}",
        )
    )

    return results


def results_to_dataframe(results: list[HypothesisTestResult]) -> pd.DataFrame:
    """Flatten dataclass results for notebook / export tables."""
    rows = [r.__dict__ for r in results]
    return pd.DataFrame(rows)


def business_interpretation(r: HypothesisTestResult) -> str:
    """Short business copy when H₀ is rejected (caller may still use for fail-to-reject)."""
    decision = "reject H₀" if r.reject_h0 else "do not reject H₀"
    return f"{r.hypothesis_id}: we {decision} at α={r.alpha} (p={r.p_value:.4g}). {r.detail}"
