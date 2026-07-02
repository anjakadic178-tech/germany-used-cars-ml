# Germany Used Cars — ML Capstone Project

End-to-end machine learning project predicting used-car prices on the
**Germany Used Cars 2023** dataset (AutoScout24 / Kaggle, 251,079 raw listings).
Two models are trained: a **classifier** that predicts whether a car is in the HIGH
or LOW price segment, and a **regressor** that predicts the exact listing price in euros.
Both models are deployed in a live Streamlit web application.

---

## Live App

**[https://germany-used-cars-ml.streamlit.app/](https://germany-used-cars-ml.streamlit.app/)**

Enter any car's brand, model, fuel type, year range, mileage, and horsepower to receive:
- An estimated market price in euros
- A price segment classification (HIGH / LOW) with confidence percentage
- A rule-based explanation of why the model predicted that price
- A table of the 8 most similar real cars from the dataset
- A side-by-side "What if?" comparison between two cars

---

## Results

| Task | Target | Best model | Key metric |
|---|---|---|---|
| Classification | `price_segment` (HIGH / LOW) | XGBoost (tuned) | F1_macro **0.934** · AUC **0.982** |
| Regression | `price_in_euro` | XGBoost (tuned) | R² **0.905** · RMSE **€4,430** · MAE **€2,602** |

**What the results mean:**
The regression model predicts the listing price of an unseen used car with a mean
error of **€2,602** using only 7 input fields. The classification model correctly
identifies the price segment in **93.4%** of cases on 48,004 held-out cars it never
saw during training. Both XGBoost models outperform a Logistic Regression and
Decision Tree baseline, but the close performance of the linear models reveals
that used-car pricing is largely an **additive structure** (brand + age + mileage
+ power) with XGBoost adding a small but consistent improvement from non-linear
interaction effects.

**Price segment threshold:** €19,486 — derived from the training-set median only (leakage-safe).

---

## Report and Slides

| Document | File | How to open |
|---|---|---|
| Full analysis report | `reports/report.html` | Open in any browser — no installation needed |
| Presentation slides | `slides/slides.html` | Open in any browser — use arrow keys to navigate |
| Report source | `reports/report.qmd` | Requires Quarto + Python (see Reproducing below) |
| Slides source | `slides/slides.qmd` | Requires Quarto |

---

## Repo Structure

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
│   ├── clean.py                    # load, audit, clean, save EDA figures
│   ├── features.py                 # feature engineering, train/test split
│   └── train.py                    # model ladder, evaluation, save pipelines
├── app/
│   ├── app.py                      # Streamlit prediction app
│   ├── brand_model_options.csv     # dropdown lookup (brand + model names)
│   └── cars_sample.csv             # 13,887-row sample for similar-cars feature
├── reports/
│   ├── report.html                 # rendered analysis report (open directly)
│   ├── report.qmd                  # report source
│   ├── figures/                    # EDA and evaluation plots
│   └── tables/                     # leaderboard CSVs
├── slides/
│   ├── slides.html                 # rendered presentation (open directly)
│   └── slides.qmd                  # slides source
└── requirements.txt
```

---

## How to Run the App Locally

```bash
# 1. Clone the repo
git clone https://github.com/anjakadic178-tech/germany-used-cars-ml.git
cd germany-used-cars-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (models are already committed — no training needed)
streamlit run app/app.py
```

---

## Reproducing the Full Analysis from Scratch

The trained `.pkl` model files are committed and the app works without re-running
any scripts. To reproduce the full pipeline from raw data:

```bash
# 1. Download the dataset from Kaggle and place it at:
#    data/raw/germany_cars.csv
#    https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023

# 2. Run scripts in order
python src/clean.py       # cleans data, saves EDA figures
python src/features.py    # engineers features, creates train/test split
python src/train.py       # trains 4-model ladder, saves pipelines

# 3. Re-render the report (requires Quarto + quarto CLI)
quarto render reports/report.qmd
```

Note: `data/raw/` and `data/processed/` are excluded from git (large files).
The pre-rendered `reports/report.html` can be opened directly without any of the above.

---

## Data Source

Germany Used Cars Dataset 2023 — Kaggle / AutoScout24
251,079 raw rows → 240,017 after general-market filtering (95.6% retained)

---

## Key Design Decisions

- **General-market filter:** pre-2000 cars, prices outside €500–€80,000, zero-mileage
  (new/demo) cars, and implausible power/mileage values removed. Follows teacher
  guidance to build a general used-car model, not a collector-car model.
- **Leakage-safe classification:** `price_segment` threshold (€19,486) derived from
  training-set median only. `price_in_euro` excluded from classification features.
- **Pipeline:** all encoding, scaling, and feature selection fitted on training data
  only via `sklearn.Pipeline` + `ColumnTransformer`. Test set used only for final evaluation.
- **Feature engineering:** `car_age = 2023 − year` replaces raw year;
  `mileage_per_year = mileage_in_km / max(car_age, 1)` captures usage intensity.
- **`random_state=42`** throughout for full reproducibility.
