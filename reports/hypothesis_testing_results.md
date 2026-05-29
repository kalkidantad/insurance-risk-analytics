# Task 3 — Hypothesis testing results (ACIS)

**Dataset:** `data/insurance_data.csv` (pipe-delimited), policy-level KPIs after aggregating transactions by `PolicyID`.

**Decision rule:** reject H₀ when *p* < 0.05.

## Results table

| Hypothesis | KPI | Statistical test | *p*-value | Decision |
|------------|-----|------------------|-----------|----------|
| No risk difference across provinces | Claim frequency | Chi-squared (2×2) | ≈ 0.019 | **Reject H₀** |
| No risk difference between zip codes (matched: Gauteng, Passenger Vehicle, 2000 vs 122) | Claim frequency | Chi-squared (2×2) | ≈ 0.82 | Fail to reject H₀ |
| No margin difference between the same zip codes | Margin (policy sums) | Welch *t*-test | ≈ 0.53 | Fail to reject H₀ |
| No risk difference between women and men | Claim frequency | Chi-squared (2×2) | ≈ 0.32 | Fail to reject H₀ |

*Exact numeric outputs may differ slightly by environment; re-run `notebooks/02_hypothesis_testing.ipynb` or `run_hypothesis_suite(load_insurance_data())`.*

## Business recommendations

### Rejected: provincial claim incidence

We **reject** H₀ for provinces (*p* < 0.05). In the default comparison (**Western Cape** vs **Gauteng**), Gauteng shows a **higher** share of policies with at least one claim than the Western Cape. **Recommendation:** treat province as a rating factor (e.g. regional loadings or separate loss-cost curves) and monitor loss ratio by province after repricing.

### Fail to reject: zip codes (matched cohort)

After holding **province** and **vehicle type** constant (Gauteng, Passenger Vehicle), claim rates for postal codes **2000** and **122** are **not** statistically distinguishable at α = 0.05. **Recommendation:** do not split these two codes on risk alone; if commercial strategy still differs by suburb, validate with a larger panel or geo-level models.

### Fail to reject: margin by zip (same cohort)

Mean **policy margin** does not differ materially between the same two postal codes at conventional significance. **Recommendation:** align marketing or channel tests with this finding—avoid margin-based price cuts targeted only at one of these codes without new evidence.

### Fail to reject: gender (Male vs Female)

With available **Male** vs **Female** labels (female policies are fewer in this book), we **do not** reject equal claim incidence. **Recommendation:** avoid gender-based premium differentiation on this evidence alone; focus on validated risk drivers and comply with fairness regulation. If more granular gender-linked confounders exist, revisit with richer data and fairness review.

---

*Implementation:* `src/hypothesis_tests.py`, notebook `notebooks/02_hypothesis_testing.ipynb`.
