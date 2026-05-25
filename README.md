# AlphaCare Insurance Solutions — Risk Analytics

End-to-end insurance risk analytics for the **10 Academy Week 3** challenge: EDA, hypothesis testing, DVC-tracked data, and predictive modeling for South African auto insurance (ACIS).

## Repository layout

```
insurance-risk-analytics/
├── .github/workflows/ci.yml   # Ruff + pytest on push/PR
├── data/                      # Place dataset here (not committed; use DVC in Task 2)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── final_report.md
├── tests/
└── requirements.txt
```

## Quick start

```bash
cd insurance-risk-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

Download the challenge file (`MachineLearningRating_v3.txt`, pipe-separated). Either:

```bash
cp /path/to/MachineLearningRating_v3.txt data/insurance_data.csv
```

or keep the file as `MachineLearningRating_v3.txt` in the **parent** folder of this repo (same level as `insurance-risk-analytics/`), which matches the default layout when both live under `week3/`.

Optional: limit rows in `notebooks/01_eda.ipynb` via `NROWS = 200_000` if memory is tight.

## Running tests

From the repository root (`insurance-risk-analytics/`), with dependencies installed:

```bash
cd insurance-risk-analytics
source .venv/bin/activate          # if you use a venv; Windows: .venv\Scripts\activate
pytest tests/ -v                   # verbose, all tests
pytest tests/ -q                   # quiet summary only
pytest tests/test_eda_utils.py -v  # single file
ruff check src tests               # lint (same as CI)
```

CI runs the same checks: `ruff check src tests` then `pytest tests/ -v`. You do not need the full dataset for tests; they use `tests/fixtures/sample_insurance.txt`.

## Running the application

This project is a **notebook-driven analytics workspace** (there is no separate API or web server). The main “app” is **Jupyter Lab** (or Jupyter Notebook) to run the EDA and modeling notebooks.

```bash
cd insurance-risk-analytics
source .venv/bin/activate
jupyter lab                        # opens the Lab UI; open files under notebooks/
# or open one notebook directly:
jupyter lab notebooks/01_eda.ipynb
```

Alternative (classic Notebook UI):

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Headless check that imports work:

```bash
python -c "from src.data_loader import load_insurance_data; print('OK')"
```

## Task 1 deliverables

- GitHub Actions CI: lint (`ruff`) and `pytest` on pushes to `main`, `task-1`, and `task-2`.
- `task-1` branch: EDA notebook + reusable `src/` modules.
- At least three insight-focused plots in `01_eda.ipynb` (loss-ratio ladder, premium–claims hexbin, vehicle-type exposure vs loss ratio).

## Task 2 — DVC (after merging Task 1)

1. Merge `task-1` → `main` via PR, then branch `task-2`.
2. `pip install dvc && dvc init`
3. Create storage outside the repo, e.g. `mkdir -p ~/dvc-local-storage`, then  
   `dvc remote add -d localstorage ~/dvc-local-storage`
4. `dvc add data/insurance_data.csv` (raw) and optionally a cleaned copy; commit `.dvc` / `dvc.yaml` / `.gitignore` updates; `dvc push`.

**Reproduce data locally:** clone the repo, `pip install -r requirements.txt dvc`, `dvc pull` (with remote configured), then open notebooks or run pipelines defined in `dvc.yaml` once you add stages.

## License / attribution

Educational use for 10 Academy. Dataset use per challenge terms.
