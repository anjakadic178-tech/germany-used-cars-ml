# Germany Used Cars — ML Capstone Project

End-to-end machine learning project predicting used-car price segments and prices
on the Germany Used Cars 2023 dataset (AutoScout24 / Kaggle, 251,079 raw rows).

## Live App

🔗 **https://germany-used-cars-ml.streamlit.app/**

## Results

| Task | Target | Best model | Key metric |
|---|---|---|---|
| Classification | `price_segment` (HIGH / LOW) | XGBoost | F1_macro 0.934 · AUC 0.982 |
| Regression | `price_in_euro` | XGBoost | R² 0.905 · RMSE €4,430 |

Price segment threshold: **€19,486** (median of training set only — leakage-safe)

## Repo structure

```
Project-MADA/
├── data/
│   ├── raw/                        # original CSV — NOT committed (download from Kaggle)
│   └── processed/                  # cleaned splits — NOT committed (run src/ scripts)
├── models/
│   ├── classifier_pipeline.pkl     # full sklearn Pipeline → XGBoost classifier
│   ├── regressor_pipeline.pkl      # full sklearn Pipeline → XGBoost regressor
│   └── price_segment_threshold.json
├── src/
│   ├── clean.py                    # Phase 2A: load, audit, clean, EDA figures
│   ├── features.py                 # Phase 2B/C: feature engineering, train/test split
│   └── train.py                    # Phase 2D/E: model ladder, evaluation, save pipelines
├── app/
│   ├── app.py                      # Streamlit prediction app
│   └── brand_model_options.csv     # dropdown lookup (brand + model names only)
├── reports/
│   ├── figures/                    # EDA and evaluation plots
│   ├── tables/                     # leaderboard CSVs
│   └── project_walkthrough_for_anja.md
├── requirements.txt
└── README.md
```

## How to run locally

```bash
# 1. Clone the repo
git clone <repo-url>
cd Project-MADA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the raw dataset from Kaggle and place it at:
#    data/raw/germany_cars.csv
#    https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023

# 4. Reproduce the cleaned data and trained models (optional — models already committed)
python src/clean.py
python src/features.py
python src/train.py

# 5. Run the Streamlit app
streamlit run app/app.py
```

## Data source

Germany Used Cars Dataset 2023 — Kaggle / AutoScout24
251,079 raw rows → 240,017 after general-market filtering (95.6% retained)

## Key design decisions

- **General-market filter:** cars before year 2000, price outside €500–€80,000,
  zero-mileage (new/demo) cars, and implausible power/mileage values removed.
  Follows teacher guidance to build a general used-car model, not a collector-car model.
- **Leakage-safe classification:** `price_segment` threshold (€19,486) derived from
  training-set median only. `price_in_euro` excluded from classification features.
- **Pipeline:** all encoding, scaling, and feature selection fitted on training data
  only via `sklearn.Pipeline` + `ColumnTransformer`. Test set used only for final evaluation.
- **Feature engineering:** `car_age = 2023 − year` replaces raw year;
  `mileage_per_year = mileage_in_km / max(car_age, 1)` captures usage intensity.
- **`random_state=42`** throughout for full reproducibility.
