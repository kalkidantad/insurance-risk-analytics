# Final Report — Week 3 Challenge  
## 10 Academy: Artificial Intelligence Mastery  

**Project:** Insurance risk analytics for AlphaCare Insurance Solutions (ACIS)  
**Author:** [Your name]  
**Date:** 29 May 2026  
**Data extract:** ~1.0M policy–transaction rows (pipe-delimited), transaction months spanning the challenge book  

---

## 1. Business objective  

ACIS needs **evidence-backed** views of where money is won or lost in the motor book, so leadership can prioritise **segmentation**, **risk-based pricing**, and **portfolio monitoring**. This work answers three practical questions—mirroring the “what is changing / what does it impact / what is required” framing from the Week 0 template, but applied to insurance:  

| Lens | Question we answer with data |
|------|-------------------------------|
| **What is changing?** | Where are loss ratios and claim incidence **materially different** by geography, vehicle, and channel of risk? |
| **What does it impact?** | Which segments drive **aggregate losses** (premium vs claims) and **policy-level claim frequency**? |
| **What is required?** | What **rating factors**, **hypothesis tests**, and **model-based** signals justify next steps (repricing tests, monitoring, further data)? |

The goal is not only to describe the workflow but to **state what we found**, with **tables, statistics, and chart placeholders** tied to the notebooks in this repository.

---

## 2. Data profiling, cleaning, and preparation  

### 2.1 Data preparation  

- **Source:** `data/insurance_data.csv` (same logical schema as `MachineLearningRating_v3.txt`), **pipe-separated**.  
- **Loading:** Pandas `read_csv(..., sep="|")`, `TransactionMonth` parsed as datetime; optional `nrows` for faster iteration (`src/data_loader.py`).  
- **Features for later stages:** engineered fields such as **vehicle age** (transaction year minus `RegistrationYear`) and **row index within policy** as a simple exposure proxy (`src/modeling.py`).  

### 2.2 Data quality checks  

- **Numeric coercion:** `TotalPremium` and `TotalClaims` coerced to numeric; invalid strings treated as missing then filled where appropriate for aggregates.  
- **Missingness:** documented in EDA via `missing_report` (`src/eda_utils.py`); high-cardinality fields (e.g. model text) summarised by **top categories** rather than full enums.  
- **Heavy tails:** claim amounts are **highly right-skewed**; portfolio views use **sums** for loss ratio (challenge definition), not naive row-wise ratios on zeros.  

### 2.3 Cleaning approach  

- **Outliers:** extreme `TotalClaims` values are **retained** for aggregate loss diagnostics (they drive the business problem). For modelling, train/test split and tree models **stabilise** influence compared with a single global mean.  
- **Duplicates:** policy-level hypothesis tests **aggregate** to one row per `PolicyID` to avoid double-counting policies when comparing segments.  

### 2.4 Reproducibility and lineage  

- **Git + CI:** lint (`ruff`) and tests (`pytest`) on `main` and task branches (`.github/workflows/ci.yml`).  
- **Data in Git:** raw CSV is **not** committed (size / GitHub limits); use **DVC** or local copy per `README` / `.gitignore`.  

---

## 3. Analysis performed  

### 3.1 Exploratory analysis (EDA) — **what we found**  

**Scope:** All **1,000,098** data rows in the current extract (full pass, chunked for memory).  

#### Portfolio headline (entire file)  

| Metric | Value | Interpretation |
|--------|------:|------------------|
| **Rows** | 1,000,098 | Transaction-level grain |
| **Portfolio loss ratio** \(\sum \text{Claims} / \sum \text{Premium}\) | **1.048** | In this snapshot, **total claims exceed total premiums by ~4.8%** in the aggregate |
| **Portfolio margin** \(\sum \text{Premium} - \sum \text{Claims}\) | **−2,955,983** | Same message in margin units (file currency) |

These are **descriptive book snapshots**, not statutory profit: no IBNR, no earned premium adjustment, no expense load—but they **do** answer the brief’s “overall loss ratio” question with **hard numbers**.

#### **Cross-segment comparison A — loss ratio by province**  

Loss ratio here is always \(\sum \text{TotalClaims} / \sum \text{TotalPremium}\) within the segment.

| Province | Rows | Sum premium | Sum claims | **Loss ratio** |
|----------|-----:|------------:|-----------:|---------------:|
| Gauteng | 393,865 | 24,053,775 | 29,394,148 | **1.222** |
| KwaZulu-Natal | 169,781 | 13,209,080 | 14,301,382 | **1.083** |
| Western Cape | 170,796 | 9,806,559 | 10,389,774 | **1.059** |
| North West | 143,287 | 7,490,508 | 5,920,250 | **0.790** |
| Mpumalanga | 52,718 | 2,836,292 | 2,044,675 | **0.721** |
| Free State | 8,099 | 521,363 | 354,922 | **0.681** |
| Limpopo | 24,836 | 1,537,324 | 1,016,477 | **0.661** |
| Eastern Cape | 30,336 | 2,140,104 | 1,356,427 | **0.634** |
| Northern Cape | 6,380 | 316,558 | 89,491 | **0.283** |

**Insight:** **Gauteng** is the largest province by premium volume **and** the worst by loss ratio (**1.22**). **Northern Cape** is best on this metric but **small exposure** (6k rows)—interpret with caution.

#### **Cross-segment comparison B — loss ratio by vehicle type**  

| Vehicle type | Rows | **Loss ratio** |
|--------------|-----:|---------------:|
| Heavy commercial | 7,401 | **1.63** |
| Medium commercial | 53,985 | **1.05** |
| Passenger vehicle | 933,598 | **1.05** |
| Light commercial | 3,897 | **0.23** |
| Bus | 665 | **0.14** |

**Insight:** **Heavy commercial** is a clear outlier on loss ratio (**1.63**); **passenger** vehicles dominate volume and sit near the **portfolio average (~1.05)**. Light commercial and bus segments look **profitable on paper** in this extract—validate with **exposure** and **cover mix** before strategic conclusions.

#### Transaction-row claim incidence (context)  

On a **500k row subsample** of the file, the share of rows with `TotalClaims > 0` is about **0.31%**. Most rows are **zero-claim** renewals or instalments; **policy-level** incidence (used in hypothesis tests) is much higher and more meaningful for “will this policy claim at all?”.

*Reproduction:* `notebooks/01_eda.ipynb` — group-by tables via `loss_ratio_by_group`, histograms of amounts on **log scale**, optional maps/bars by `Province` / `VehicleType`.

---

### 3.2 Statistical hypothesis tests (A/B style) — **evidence**  

We tested four pre-specified nulls at **α = 0.05**, using **`src/hypothesis_tests.py`** and `notebooks/02_hypothesis_testing.ipynb`. Claim **frequency** is defined at **policy** level: a policy has `has_claim = 1` if **any** row shows `TotalClaims > 0`.

| ID | Null hypothesis (plain language) | KPI | Test | *p*-value | Decision | Sample sizes |
|----|-----------------------------------|-----|------|----------:|----------|-------------|
| H1 | No difference in claim incidence **Western Cape vs Gauteng** | Policy claim frequency | Chi-squared (2×2) | **0.019** | **Reject H₀** | **947** vs **2,577** policies |
| H2 | No difference for **postal 2000 vs 122** within **Gauteng + passenger** | Same | Chi-squared | **0.941** | Fail to reject | 450 vs 292 policies |
| H3 | No **margin** difference for those same zips | Margin (Σpremium − Σclaims per policy) | Welch *t* | **0.546** | Fail to reject | 450 vs 292 |
| H4 | No difference **Male vs Female** | Policy claim frequency | Chi-squared | **0.322** | Fail to reject | 232 vs **35** policies |

**Effect size (H1 — the rejection we act on):** among policies in the test, **Western Cape** policy-level claim frequency **≈ 19.1%** vs **Gauteng ≈ 22.8%** (details string from the run: χ² ≈ 5.46).  

**Effect sizes (failures to reject — still informative):** matched Gauteng passenger zips **2000 vs 122** show **~30.2% vs ~30.5%** policy claim frequency—**not** significantly different; mean policy margin **−2,643** vs **−4,172** (currency units) but **not** statistically separable at α = 0.05. **Gender:** male **≈21.1%** vs female **≈28.6%** policy claim frequency, but **not** significant at α = 0.05; the **female policy count is very small (35)** → **low power**; do **not** over-interpret.

---

### 3.3 Predictive modelling and interpretability — **quantitative model evidence**  

On a **random 120,000-row subsample** (for fast, repeatable reporting; **full-book metrics live in the notebook**), we trained severity regressors on rows with `TotalClaims > 0` and a row-level **claim classifier**:

**Severity (test RMSE / R²):**

| Model | RMSE | R² (test) |
|-------|-----:|----------:|
| Linear regression | 31,568 | −0.05 |
| Random forest | **26,970** | **0.24** |
| XGBoost | 32,482 | −0.11 |

**Row-level claim classifier (test set, same subsample):**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic regression | 0.997 | 0.000 | 0.000 | 0.000 |
| Random forest | 0.815 | 0.013 | **0.776** | 0.026 |
| XGBoost | 0.997 | 0.000 | 0.000 | 0.000 |

**Insight:** **Random forest** is the strongest **severity** model on this slice; **claim incidence is extremely imbalanced**, so **accuracy is misleading**—logistic regression and XGBoost collapse precision on positives at default thresholds, while the forest trades overall accuracy for **usable recall** on claims. Production use would require **threshold tuning**, **calibration**, and **cost-sensitive** learning.

**Pricing skeleton (illustrative):** `risk_adjusted_premium` combines predicted **P(claim)** and **severity** with loadings (`src/modeling.py`, `notebooks/03_modeling.ipynb`).

**Interpretability:** SHAP bar plots for the **best tree severity model** should be exported to `reports/figures/shap_severity_best.png` (see notebook). Expected qualitative drivers: **vehicle age**, **sum insured / premium scale**, and **geo / cover** one-hot segments—validate against your latest SHAP run.

---

## 4. Visual evidence (what to paste into the PDF / deck)  

The written report above already contains **tables and test results**. To fully address the feedback on **charts**, include **at least** the following **captioned** figures exported from the notebooks (suggested filenames under `reports/figures/`):

| # | Figure (export from) | What it proves |
|---|----------------------|----------------|
| **Figure 1** | `01_eda.ipynb` — bar: **loss ratio by `Province`** | Gauteng and KZN stand above 1.0; Northern Cape near 0.3 on small volume |
| **Figure 2** | `01_eda.ipynb` — bar: **loss ratio by `VehicleType`** | Heavy commercial > 1.6; passenger near portfolio average |
| **Figure 3** | `01_eda.ipynb` — histogram (log x): **`TotalClaims` among positives** | Extreme right tail; justifies robust/tree models |
| **Figure 4** | `02_hypothesis_testing.ipynb` — table screenshot or exported CSV chart of **p-values & decisions** | Direct link from statistics to accept/reject |
| **Figure 5** | `03_modeling.ipynb` — **model metric comparison** (bar or table chart) | RF wins RMSE on the illustrated severity run |
| **Figure 6** | `03_modeling.ipynb` — **`reports/figures/shap_severity_best.png`** | Global interpretability of top drivers |

**Labelling standard:** each figure needs a **title**, **axis labels with units**, **data window** (“ACIS extract, N = 1,000,098 rows unless noted”), and **one sentence takeaway** in the caption.

---

## 5. Discussion and observations  

### 5.1 Technical outcomes  

- The pipeline is **reproducible**: loaders, EDA helpers, hypothesis suite, and modelling utilities live under `src/`, with **automated tests** and **lint** in CI.  
- We moved from “**what we did**” to “**what the numbers say**”: **portfolio loss ratio > 1**, **Gauteng worst among large provinces**, **heavy commercial worst vehicle class**, **statistically significant provincial difference in policy claim incidence (WC vs Gauteng)**, and **no significant difference** for the **matched** Gauteng passenger postal pair on frequency or margin.  

### 5.2 Business interpretation (data-backed)  

1. **Regional pricing / monitoring:** EDA and **H1** both point to **material geographic heterogeneity**. ACIS should treat **province** as a **first-class monitoring dimension** and run **controlled repricing experiments** where regulatory rules allow—not because of EDA alone, but because **formal testing** also rejects equal claim incidence for the WC vs Gauteng comparison at α = 0.05.  
2. **Vehicle strategy:** **Heavy commercial** merits **underwriting review** (pricing, limits, or risk appetite), independent of the passenger mass market.  
3. **Geo micro-segments:** Within a **homogeneous** cohort (Gauteng + passenger), **two busy postal codes did not differ** significantly on claim frequency or mean margin—arguing **against** micro-zip pricing splits **without stronger evidence** (geo models, more history).  
4. **Gender:** **Do not** treat the Male/Female comparison as conclusive; **small labelled female sample** → keep **fairness and legal** review central.  

### 5.3 Limitations  

- **Not full P&L:** no expenses, reinsurance, discounting, or IBNR.  
- **Mix and seasonality:** 18-month window; trends may reflect **mix change** not pure calendar effects.  
- **Modelling metrics** on a **120k subsample** are **illustrative**; production requires **full-data retrain**, **cross-validation**, and **leakage checks** (e.g. same-policy rows in train and test).  
- **Classifier metrics** show that **default 0.5 thresholds** are inappropriate for rare events—report **precision/recall** and **business cost** together.  

---

## 6. Next steps  

1. **Refresh all tables and figures** from `01_eda.ipynb`, `02_hypothesis_testing.ipynb`, and `03_modeling.ipynb` on the **full** dataset and pin outputs under `reports/figures/` for submission.  
2. **Merge remaining branches** via PRs; keep **DVC** (or documented local paths) for data.  
3. **Pricing pilot design:** translate **severity RMSE gains** and **claim probability** into **controlled price tests** with pre-registered success metrics (loss ratio, conversion, retention).  
4. **External context:** add industry or regulatory references to strengthen **policy narrative** (beyond internal book).  

---

## 7. Conclusion  

This submission goes beyond process description to present **concrete analytical findings**: a **portfolio loss ratio above 1**, **province and vehicle segments** that concentrate poor experience, **hypothesis tests** with **explicit p-values and sample sizes**, and **model metrics** that show **where algorithms help (severity RF)** and **where imbalance breaks naive classifiers**.  

Pairing these results with **labelled charts** (Section 4) gives ACIS a defensible bridge from **data** to **pricing and portfolio decisions**—the core of a complete Week 3 analysis.
