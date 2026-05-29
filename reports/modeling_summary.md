# Task 4 — Modeling summary (ACIS)

Run `notebooks/03_modeling.ipynb` on your machine to refresh **model comparison tables** and the **SHAP** figure (`reports/figures/shap_severity_best.png`). Metrics depend on the random seed and subsample (`NROWS`).

## Severity (TotalClaims | claim > 0)

| Algorithm | Metrics |
|-----------|---------|
| Linear regression | RMSE, R² (test) |
| Random forest | RMSE, R² |
| XGBoost | RMSE, R² |

**Takeaway:** tree ensembles typically capture non-linear vehicle and cover interactions better than a linear baseline; use the notebook table to pick the lowest RMSE model for production experiments.

## Claim incidence (row-level `has_claim`)

| Algorithm | Metrics |
|-----------|---------|
| Logistic regression | accuracy, precision, recall, F1 |
| Random forest | same |
| XGBoost | same |

**Takeaway:** class imbalance is extreme; interpret **precision/recall** jointly rather than accuracy alone when tuning marketing or underwriting thresholds.

## Risk-adjusted premium (illustrative)

Technical premium uses:

\\[
\\text{Premium} \\approx P(\\text{claim}) \\times \\hat{S} \\times (1 + \\text{profit margin}) + \\text{expense loading}
\\]

with `risk_adjusted_premium` in `src/modeling.py`. Compare against `CalculatedPremiumPerTerm` as a naive benchmark.

## SHAP — interpreting top drivers

After generating `reports/figures/shap_severity_best.png`, read **larger bars** as features that move predicted severity most on average:

- **`vehicle_age_years`** — older vehicles tend to increase predicted claim cost, supporting age-based rating adjustments.
- **`SumInsured` / `CalculatedPremiumPerTerm`** — scale of cover and current pricing are strong anchors; large SHAP here suggests the model is pricing exposure consistently with the tariff already in the book.
- **Geo / vehicle / cover one-hot segments** — SHAP mass on `Province_*` or `CoverType_*` buckets highlights segments deserving targeted loss-cost reviews.

These statements should be validated against the latest SHAP export from your notebook run.
