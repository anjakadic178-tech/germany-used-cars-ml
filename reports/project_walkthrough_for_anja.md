# Project Walkthrough — Study Guide for Anja
## Germany Used Cars 2023 · End-to-End ML Capstone

> **Read this before your presentation or oral exam.**
> Every number comes from the actual project files, not memory.

---

## 1. Project Goal and Dataset

**Dataset:** Germany Used Cars 2023, scraped from AutoScout24 and published on Kaggle.
- Raw size: **251,079 rows × 15 columns**
- Each row is one used-car listing

**Two ML tasks:**

| Task | Question | Target column |
|---|---|---|
| Classification | Is this car in the HIGH or LOW price segment? | `price_segment` (derived from `price_in_euro`) |
| Regression | What is this car's listed price in euros? | `price_in_euro` |

**Why this is a real problem:**
Used-car pricing is opaque. A buyer or seller cannot easily know if a car is fairly priced. A model trained on 240,000 real listings can give an objective, data-driven estimate in seconds. This is exactly the kind of problem ML is useful for — many variables interact in complex ways (brand, age, mileage, engine) that a simple formula cannot capture.

**Why it satisfies the assignment:**
- Public dataset (Kaggle, AutoScout24 — shareable)
- 240,000+ rows (well above the ~500 minimum)
- Supports both tasks naturally (price is regression; segment is classification)
- Not a lecture/toy dataset

**"General-market used-car model" explained:**
The teacher warned not to build a model that tries to price everything from a €800 Dacia Sandero to a €300,000 Ferrari. Those cars follow completely different pricing logic. Instead, we built a model for the realistic everyday used-car buyer — cars from 2000 onwards, priced €500–€80,000, with ordinary mileage and power. This is what the filters in cleaning.py enforce.

---

## 2. Raw Data Audit — What We Actually Found

This is important because the real dataset had differences from the Kaggle description.

**Type problems:**
- `price_in_euro` was stored as a **string**, not a number → had to convert with `pd.to_numeric(errors="coerce")`
- `power_ps` was stored as a **string** → same conversion
- `year` was stored as a **string** → same conversion
- `mileage_in_km` was already a float (no conversion needed)

**Missing column:**
- The Kaggle description mentioned an `offer_type` column. It did not exist in the actual file. Instead there was `offer_description` (free text, 191,824 unique values — unusable for modelling, dropped).

**Corruption in `fuel_type`:**
- Some rows had dates, mileage values, or transmission strings in the `fuel_type` field
- These rows were structurally corrupted — the entire row was unrecoverable, so they were dropped (88 rows)

**Columns present in raw data (15 total):**
`brand, model, color, registration_date, year, price_in_euro, power_kw, power_ps, transmission_type, fuel_type, mileage_in_km, fuel_consumption_l_100km, fuel_consumption_g_km, offer_description, Unnamed: 0`

**Why this audit mattered:**
If you skip the audit, you might try to train on string columns and get silent errors, or include corrupted rows that damage model quality. The audit reveals what you're actually working with, not what you think you're working with.

---

## 3. Cleaning Decisions — All 12 Steps (Sequential)

All 12 rules were applied **in order**, each one operating on the already-filtered dataset from the previous step.

| Step | Rule | Rows removed | Reason |
|---|---|---|---|
| 1 | Drop NaN `price_in_euro` | 199 | Can't train or evaluate without a target |
| 2 | Drop NaN `year` | 0 | Needed for car_age engineering |
| 3 | Drop NaN `power_ps` | 128 | Primary price driver; imputing would be misleading |
| 4 | Drop NaN `mileage_in_km` | 60 | Primary price driver; imputing would be misleading |
| 5 | Drop corrupted `fuel_type` | 88 | Rows had dates/mileage in wrong field — unrecoverable |
| 6 | Drop `transmission_type == "Unknown"` | 1,134 | Unknown transmission can't be encoded meaningfully |
| 7 | Drop `year < 2000` | 1,924 | Oldtimers/classics — different pricing logic |
| 8 | Drop `price_in_euro < €500` | 77 | Scrap/parts listings, not general-market cars |
| 9 | Drop `price_in_euro > €80,000` | 7,078 | Ultra-luxury cars (Porsche, Ferrari, Bentley) — different pricing dynamics |
| 10 | Drop `power_ps < 40 or > 800` | 64 | Data entry errors or exotic supercars outside scope |
| 11 | Drop `mileage_in_km > 500,000` | 116 | Odometer errors |
| 12 | Drop `mileage_in_km == 0` | 194 | New/demo dealer cars follow MSRP pricing, not depreciation |

**Final result: 240,017 rows retained (95.59% of the raw data)**

**For each major filter — why it matters:**

**`year >= 2000` (removes oldtimers):**
Pre-2000 cars are collector items. A 1968 Porsche 911 is worth more than a 2010 Porsche 911. Including them would create a non-monotonic age-price relationship that confuses the model. The teacher explicitly warned about this.

**`price ≤ €80,000` (removes ultra-luxury):**
A Ferrari or Bentley costs €150,000+ for brand prestige, not depreciation curves. Training on these would teach the model wrong patterns for the majority of cars. The €80,000 cutoff was chosen because it covers the entire mainstream and near-premium market (BMW 5 Series, Mercedes E-Class, Audi A6) while excluding collector/exotic cars.

**`mileage > 0` (removes new/demo cars):**
Zero-mileage listings on AutoScout24 are dealer-new cars. New-car pricing is based on MSRP and dealer margins, not used-car depreciation. Including them would teach the model a completely different pricing logic that breaks for real used cars.

**Dropped columns:**
- `color` — low predictive signal, adds ~13 sparse dummy columns, and would make the model harder to interpret
- `power_kw` — redundant with `power_ps` (same information, different unit)
- `registration_date` — year is sufficient; exact date adds noise
- `fuel_consumption_l_100km`, `fuel_consumption_g_km` — many missing values, correlated with fuel_type and power_ps
- `offer_description` — free text, 191k unique values, not useful for structured ML
- `Unnamed: 0` — auto-generated index column from original CSV, not a feature

---

## 4. Row-Count Correction — The Audit Bug

**What happened:**
Early in the project, the cleaning audit showed inconsistent row counts. When I added up all the rows dropped by each filter, the total didn't match the final row count.

**Why it happened:**
The original code computed all 12 filter masks **simultaneously on the original dataframe** and then counted them. But filters overlap — a car that violates both the year rule and the price rule gets counted twice.

**The fix:**
Apply each filter **sequentially** — each step runs on the output of the previous step, not on the original data. This is how `src/clean.py` works.

**Why this matters for the grade:**
The cleaning audit is a core deliverable. If the numbers don't add up, the professor will notice. After the fix: 251,079 raw → 240,017 clean = 11,062 rows dropped, which exactly matches the sum of per-step removals.

---

## 5. EDA and Visualizations

**Key findings (all saved in `reports/figures/`):**

**Price distribution:**
- Raw data: extremely right-skewed, prices from €0 to millions
- After filtering: much cleaner bell curve centred around €15,000–€25,000, skewness ≈ 1.20 (manageable for tree models)

**Year distribution:**
- Heavy concentration 2015–2022, with a long left tail back to 2000
- Post-filter: no pre-2000 cars

**Mileage distribution:**
- Most cars: 20,000–200,000 km
- After filter: clean range 1–500,000 km, no outlier spikes

**Correlation heatmap (`correlation_heatmap.png`):**
- Only true numeric columns used: `price_in_euro`, `power_ps`, `mileage_in_km`, `car_age`
- Strongest correlations: `power_ps` ↔ `price` (positive), `car_age` ↔ `price` (negative), `mileage_in_km` ↔ `car_age` (positive — older cars have more km)

**Why categorical columns were NOT label-encoded for the heatmap:**
Label-encoding `brand` as 1,2,3… implies alphabetical order = price order, which is meaningless. Correlation between an arbitrary number and price tells you nothing. Only genuine numeric variables belong in a correlation analysis.

**Price vs mileage (`price_vs_mileage.png`):**
Clear negative relationship — higher mileage = lower price. This confirms mileage is a strong predictor for both tasks.

---

## 6. Feature Engineering

**What we started with:** 8 columns in `cars_clean.csv`: brand, model, year, price_in_euro, power_ps, transmission_type, fuel_type, mileage_in_km

**What went into the model (8 feature columns):**
`brand, model, fuel_type, transmission_type, car_age, power_ps, mileage_in_km, mileage_per_year`

Note: `year` is **not** in the model features. `price_in_euro` is **not** in the classification features.

**`car_age = 2023 − year`**
- Why: Car age is the meaningful signal, not the calendar year. A 10-year-old car costs less than a 2-year-old car — this relationship holds regardless of what year it is now.
- Why drop raw `year`: If you keep both `year` and `car_age`, they are perfectly collinear (`car_age = 2023 − year` exactly). This breaks Ridge regression (singular matrix) and hurts interpretability.
- 2023 is used because that is when AutoScout24 scraped the data.

**`mileage_per_year = mileage_in_km / max(car_age, 1)`**
- Why: A car with 200,000 km is priced very differently depending on whether it's 20 years old (normal) or 5 years old (heavily used). Raw mileage alone doesn't capture this.
- Division-by-zero safety: `max(car_age, 1)` ensures that if `car_age = 0` (a 2023 car), we divide by 1 instead of 0. In code: `df["mileage_per_year"] = df["mileage_in_km"] / df["car_age"].clip(lower=1)`

---

## 7. Brand/Model Cardinality — Why We Kept Both

**The problem:**
The `model` column has 1,191 unique values (e.g., "Volkswagen Golf", "BMW 3 Series", "Audi A4"…). One-hot encoding 1,191 values is expensive and creates many sparse columns for rare models.

**Solution — `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")`:**
- Models with **≥ 50 listings** get their own dummy column
- Models with **< 50 listings** are all collapsed into one shared "infrequent" column
- At inference time, if a model the app has never seen is entered, it goes into the infrequent bucket (no crash)

**Why we kept `brand` even though `model` already implies it:**
Consider this: after OHE with min_frequency=50, a rare Porsche model and a rare Opel model both collapse into the same "infrequent_model" bucket. Without `brand`, the model cannot tell these apart — they look identical. With `brand`, the model knows one is Porsche and one is Opel, and can price them accordingly. Brand carries real pricing information beyond what model alone provides.

**Leakage safety:**
The `min_frequency=50` threshold is counted from training data only (the encoder is fitted on `X_train` only inside the Pipeline). At test time, the encoder uses the vocabulary it learned from training — it never re-counts frequencies from test data.

---

## 8. Leakage-Safe Design — The Most Important Section

**What is data leakage?**
When information from the future (test set) or from the target variable is used during training. It makes models look better in development but fail in production.

**Specific leakage risks in this project and how we avoided them:**

**1. Price → classification target:**
`price_in_euro` must NOT be a feature when predicting `price_segment`. If it were, the model would know the exact price and the classification becomes trivial (not real ML). We excluded `price_in_euro` entirely from `FEATURE_COLS`.

**2. Price segment threshold:**
The HIGH/LOW boundary (€19,486) was computed as **the median of `y_reg_train` only** — the training portion of prices. If we computed it from the full dataset, the test set would have "seen" its own prices during threshold computation. The threshold is saved in `models/price_segment_threshold.json` and the derivation is documented: *"median of y_reg_train (training set only — leakage-safe)"*.

**3. Train/test split:**
- 80% training (192,013 rows), 20% test (48,004 rows)
- `random_state=42` for reproducibility
- **Stratified** by `price_segment` for classification — ensures both splits have the same HIGH/LOW ratio (approximately 50/50)
- After the split: the test set is **never used again until final evaluation**

**4. Preprocessing inside Pipeline:**
All encoders, scalers, and feature selectors are fitted inside `sklearn.Pipeline` using `.fit(X_train)`. This guarantees they never see `X_test`. If preprocessing happened before the split, the scaler would know the test set's mean and standard deviation — that's leakage.

**5. Proof the scaler was fitted on training data only:**
The StandardScaler inside the pipeline stores the mean it was fitted on. We verified that this mean matches `X_train.mean()` exactly (to 4 decimal places) and differs from the full-dataset mean. This is documented in `modeling_metadata.json` under `"leakage_proof"`.

---

## 9. Pipeline Structure

Every prediction uses this exact pipeline structure:

```
ColumnTransformer
├── OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")
│   └── applied to: brand, model, fuel_type, transmission_type
└── StandardScaler()
    └── applied to: car_age, power_ps, mileage_in_km, mileage_per_year
↓
VarianceThreshold(threshold=0.0)
↓
XGBoostClassifier  (or XGBoostRegressor)
```

**After OHE, there are 559 features:**
- 46 brand dummies
- 496 model dummies (+ 1 infrequent bucket)
- 13 fuel type and transmission dummies
- 4 scaled numeric features

**VarianceThreshold(threshold=0.0) — the feature selection step:**
This removes any feature with zero variance — i.e., a column that has the exact same value in every training row. Such columns carry no information. In this project, it removed **0 features**, which means the OHE with `min_frequency=50` already did its job — there are no dead-weight constant columns. The VarianceThreshold still counts as a feature-selection step because:
- It is inside the Pipeline (leakage-safe)
- It is documented and auditable
- It would remove features automatically if they appeared (e.g., if a category had zero variance after the split)

**Why preprocessing is inside the Pipeline:**
`Pipeline.fit(X_train)` calls each step's `.fit_transform()` in sequence. `Pipeline.predict(X_test)` calls each step's `.transform()` only — the scaler and encoder are never re-fitted on test data.

---

## 10. Model Ladder — Both Tasks

The assignment requires a progression from dumbest to smartest.

### Classification Ladder

| Model | Why included | Test F1_macro | Test AUC |
|---|---|---|---|
| DummyClassifier | Honest baseline — "what if we always guess HIGH?" | 0.333 | 0.500 |
| Logistic Regression | Linear model — tests if price segment is linearly separable | 0.929 | 0.978 |
| Decision Tree (depth=15) | Non-linear, interpretable, tests tree-based approach | 0.919 | 0.945 |
| XGBoost (tuned) | Best model — ensemble of trees with regularization | **0.934** | **0.982** |

**DummyClassifier:** Always predicts HIGH (the most frequent class). This is our baseline — any real model must beat this.

**Logistic Regression:** Tests whether a straight line in 559-dimensional feature space can separate HIGH from LOW. It can — F1_macro = 0.929. This tells us the problem has strong linear structure.

**Decision Tree (depth=15):** Tests a single non-linear tree. It overfits slightly (train F1 = 0.949, test F1 = 0.919 — gap of 0.030), but it's still good and interpretable.

**XGBoost:** Hundreds of small trees combined, with regularization and randomness. Best test performance (F1 0.934, AUC 0.982) with moderate overfitting (train-test gap 0.012). Selected for deployment.

### Regression Ladder

| Model | Why included | Test R² | Test RMSE |
|---|---|---|---|
| DummyRegressor | Baseline — "what if we always predict the mean price?" | −0.000 | €14,369 |
| Ridge Regression | Linear model — tests additive price structure | 0.841 | €5,726 |
| Decision Tree (depth=15) | Non-linear tree — tests local price patterns | 0.856 | €5,452 |
| XGBoost (tuned) | Best model | **0.905** | **€4,430** |

**DummyRegressor:** Predicts the training mean (≈€19,000) for every car. R² = 0 means "explains nothing." RMSE = €14,369 means on average it's €14,369 wrong.

**Ridge Regression (R² = 0.841, gap nearly zero):** The surprise result. A linear model with L2 regularization explains 84% of price variance — better than expected for 559 sparse features. The near-zero train-test gap (train R² = 0.841, test R² = 0.841) proves no overfitting. This shows that car pricing has a strong additive structure: price ≈ brand effect + age effect + mileage effect + power effect.

**Decision Tree (depth=15, R² = 0.856):** Slightly better than Ridge on test, but overfits more (train R² = 0.911 vs test 0.856 — gap of 0.055).

**XGBoost (R² = 0.905, RMSE €4,430):** Explains 90.5% of price variance. Average error of €2,602 (MAE). The final €4,430 RMSE means for most cars the estimate is within ~€4,400 of the actual listing price. Selected for deployment.

---

## 11. Metrics — Plain Language

### Classification

**Accuracy (0.934 for XGBoost):** Out of 48,004 test cars, 93.4% were classified correctly as HIGH or LOW.

**Precision (0.937):** Of all the cars the model predicted HIGH, 93.7% actually were HIGH. Low precision = many false alarms.

**Recall (0.931):** Of all the actual HIGH cars, the model found 93.1% of them. Low recall = many misses.

**F1_binary:** Harmonic mean of precision and recall. Used when you care about both. For balanced classes, it ≈ accuracy.

**F1_macro:** Average of F1 scores computed separately for each class, then averaged. This is the **honest metric** for this project.

**The F1 correction — very important:**
The DummyClassifier always predicts HIGH.
- F1_binary = 0.667 — this looks decent! But it's meaningless — the model never predicts LOW, so its F1 on class LOW is 0.
- F1_macro = 0.333 — this is honest. Average F1 across both classes: (F1_HIGH + F1_LOW) / 2 = (0.667 + 0.0) / 2 = 0.333.

This distinction matters because if you report F1_binary for the dummy, you make a terrible model look 67% effective. F1_macro = 0.333 correctly shows it as barely above chance.

**ROC-AUC (0.982 for XGBoost):** Probability that the model ranks a random HIGH car above a random LOW car. 0.982 is excellent; 0.5 is random guessing.

**Confusion Matrix:** Shows exact counts of true positives, true negatives, false positives, false negatives. Located in `reports/figures/confusion_matrix.png`.

### Regression

**R² (0.905):** The model explains 90.5% of the variance in used-car prices. R² = 1 would be perfect. R² = 0 is the dummy baseline.

**MAE — Mean Absolute Error (€2,602):** On average, the model's prediction is €2,602 away from the actual listed price. This is the most interpretable metric.

**RMSE — Root Mean Squared Error (€4,430):** Penalises large errors more than MAE. Higher than MAE because a few expensive cars are harder to predict.

**Predicted vs Actual plot (`regression_predicted_vs_actual.png`):** If the model were perfect, all points would lie on a diagonal line. Scatter around the diagonal shows error. Points far from the line at high prices reveal where the model struggles — very expensive cars in the training range have less data.

---

## 12. Results Interpretation

**Classification:**
XGBoost wins with F1_macro = 0.934, AUC = 0.982. But the interesting finding is that Logistic Regression (F1_macro = 0.929, AUC = 0.978) comes very close while being a much simpler model. This tells us: **car price segment is almost linearly separable.** The features — brand, age, mileage, power — add up nearly additively to determine whether a car is above or below €19,486. XGBoost captures a small additional non-linear structure worth ~0.5% F1.

**Regression:**
XGBoost wins with R² = 0.905, RMSE = €4,430. Ridge R² = 0.841 is remarkably close for a linear model. This confirms the additive price structure.

**Overfitting analysis:**

| Model | Train-test gap (F1_macro / R²) | Assessment |
|---|---|---|
| Logistic Regression | 0.9328 → 0.9291 (gap 0.004) | Minimal overfitting — excellent |
| Decision Tree (cls) | 0.9489 → 0.9189 (gap 0.030) | Moderate overfitting |
| XGBoost (cls) | 0.9461 → 0.9341 (gap 0.012) | Small overfitting — well-regularized |
| Ridge | 0.8412 → 0.8412 (gap 0.000) | No overfitting |
| Decision Tree (reg) | 0.9112 → 0.8560 (gap 0.055) | Notable overfitting |
| XGBoost (reg) | 0.9223 → 0.9049 (gap 0.017) | Small overfitting — well-regularized |

**Business meaning:**
For a buyer or seller in the general German used-car market, this model predicts the listed price within about €4,400 on average, and correctly identifies HIGH/LOW segment 93.4% of the time. This is good enough to detect significantly mispriced cars (great deal or overpriced) in the €500–€80,000 range.

---

## 13. The Training Script Crash — Honest Correction

**What happened:**
The first complete training run (`src/train.py`) crashed at the very last step (step 8 out of 9) with:
```
TypeError: 'int' object is not iterable
```
This was in the metadata-saving code, not in any model training. A malformed Python expression tried to call `.__len__()` on an integer.

**Why it was a problem:**
The script produced all model files (`.pkl`) and leaderboards correctly before crashing, but the exit code was 1 (error), not 0. The assignment requires reproducible scripts that run cleanly.

**How it was fixed:**
The expression was replaced with `int(len(best_cls_pipeline.named_steps["selector"].get_support()))`. Since all artifacts (PKL files, leaderboards) were already saved before the crash, only the metadata JSON needed regenerating.

**The slowness problem and how it was fixed:**
The initial full re-run attempted to tune Logistic Regression's `C` via `GridSearchCV` on 192,013 rows × 559 features. This took 40+ minutes and was impractical. Fix: use `C=10` directly (from the validated prior run) and switch XGBoost to `tree_method="hist"` (optimized histogram-based training). Runtime dropped from 45+ minutes to ~10 minutes with identical results.

**Final confirmation:**
Full clean rerun of `src/train.py` → exit code 0 → all artifacts regenerated → smoke tests pass.

---

## 14. Streamlit App

**What files the app loads at startup (cached):**
- `models/classifier_pipeline.pkl` — complete sklearn Pipeline with XGBoost
- `models/regressor_pipeline.pkl` — complete sklearn Pipeline with XGBoost
- `models/price_segment_threshold.json` — €19,486 threshold for labelling
- `data/processed/cars_clean.csv` — for brand/model dropdown lists
- `reports/tables/classification_leaderboard.csv` + `regression_leaderboard.csv`

**What the user enters (7 inputs):**
brand, model, fuel_type, transmission_type, year, power_ps, mileage_in_km

**What the app computes internally (2 derived features):**
```python
car_age          = 2023 - year
mileage_per_year = mileage_in_km / max(car_age, 1)
```

**What goes into the pipeline (exactly 8 columns, in this order):**
```
brand, model, fuel_type, transmission_type,
car_age, power_ps, mileage_in_km, mileage_per_year
```

**Why raw `year` is NOT passed directly:**
The pipeline was trained on `car_age`, not `year`. Passing `year` directly would cause the ColumnTransformer to fail (unknown column name) or silently use wrong features. The app replicates exactly the same feature engineering done in `src/features.py`.

**What the app returns:**
1. Estimated market price (from the regression pipeline)
2. HIGH or LOW price segment (from the classification pipeline)
3. Segment confidence (probability from `predict_proba`)

**Scope warning:**
The app displays a notice that it is designed for general-market cars (year 2000–2023, €500–€80,000, 40–800 PS). This matches the training data filter and is honest — the model was not trained on cars outside this range.

---

## 15. Design Changes

**What changed:**
- `app/app.py` — full visual redesign: light ivory background (#faf9f6), dark charcoal text, gold accent (#b8960c), premium white cards, clean typography
- `.streamlit/config.toml` — switched from dark to light theme

**What was never touched:**
- `src/train.py`, `src/features.py`, `src/clean.py`
- `models/classifier_pipeline.pkl`, `models/regressor_pipeline.pkl`
- `models/price_segment_threshold.json`
- All leaderboard tables and figures

**How we verified the model was not changed:**
MD5 fingerprints of the `.pkl` files were checked before and after design changes — identical.

**Why smoke tests after design changes:**
Even though only CSS changed, Streamlit re-executes Python on every render. A coding mistake in design HTML could theoretically break the prediction call. The smoke tests confirm predictions are still correct.

---

## 16. Limitations (Be Honest About These)

**What the model should NOT be used for:**
- Cars built before 2000 (oldtimers, classics)
- New cars (0 km, dealer listings)
- Ultra-luxury or collector cars (Porsche 911 GT3, Ferrari, Bentley, etc. above €80,000)
- Rare or exotic cars with very few training examples
- Cars outside Germany (different market, different depreciation curves)
- Non-AutoScout24 listings (private sales, auction prices, dealer purchase prices)

**Listed price ≠ final sale price:**
The model predicts the AutoScout24 **asking price**, not the negotiated sale price. Real transactions are typically 5–15% below asking price.

**Temporal drift:**
The data is from 2023. Car prices change with fuel prices, economic conditions, and new model releases. By 2025 or later, the model may be meaningfully out of date.

**What future improvements could include:**
- Add color (currently dropped — adds 13 features but some signal exists)
- Log-transform the price target to reduce skewness impact on RMSE
- Cross-validation instead of a single 80/20 split for more robust estimates
- Include geographic data (city vs rural listing price differences)
- Retrain periodically on fresh AutoScout24 data
- Calibrated probabilities (Platt scaling) for the classifier

---

## 17. Assignment Compliance Checklist

| Requirement | How satisfied | File/Output |
|---|---|---|
| Approved public dataset | Germany Used Cars 2023, Kaggle/AutoScout24 | Raw CSV in data/raw/ |
| EDA with visualizations | 15 figures: distributions, heatmap, brand analysis, price-mileage, etc. | reports/figures/*.png |
| Data cleaning documented | 12 sequential steps, metadata with row counts per step | data/processed/cleaning_metadata.json |
| Feature engineering | car_age, mileage_per_year (replaces raw year) | src/features.py |
| Feature selection | VarianceThreshold(0.0) inside Pipeline | src/train.py, modeling_metadata.json |
| Leakage-safe pipeline | All preprocessing fitted on X_train only inside Pipeline | src/features.py, src/train.py |
| Stratified split | train_test_split(stratify=y_cls) | src/features.py |
| Classification ladder | Dummy → Logistic Regression → Decision Tree → XGBoost | src/train.py, reports/tables/ |
| Regression ladder | Dummy → Ridge → Decision Tree → XGBoost | src/train.py, reports/tables/ |
| XGBoost with tuning | RandomizedSearchCV (n_iter=6, cv=3) | src/train.py |
| Leaderboard | Both tasks compared | reports/tables/classification_leaderboard.csv, regression_leaderboard.csv |
| Dummy baseline | Both tasks include baseline | Both leaderboards |
| Overfitting discussion | Train vs test gap for all models | Leaderboards (Train_F1_macro, Train_R2 columns) |
| Classification metrics | Accuracy, Precision, Recall, F1_binary, F1_macro, AUC, confusion matrix | reports/tables/, reports/figures/ |
| Regression metrics | R², MAE, RMSE, predicted-vs-actual plot | reports/tables/, reports/figures/ |
| Deployed web app | Streamlit app, premium light design | app/app.py (live on port 8510) |
| Quarto report (D1) | **TODO — not yet built** | reports/report.qmd (pending) |
| Slides (D4) | **TODO — not yet built** | pending |
| Executive summary (D5) | **TODO — not yet built** | pending |
| AI reflection (D3) | **TODO — not yet written** | pending |
| README / GitHub | **TODO — not yet created** | pending |

---

## 18. Questions Your Professor Might Ask

**Q: Why did you filter out cars above €80,000?**
A: The teacher explicitly warned to build a general-market model. Cars above €80,000 — Porsche 911, Ferrari, Bentley, Lamborghini — are priced on brand prestige and collector demand, not normal depreciation curves. Including them would corrupt the model's understanding of how ordinary cars age. The €80,000 cutoff still includes the full premium mainstream market: BMW 5 Series, Mercedes E-Class, Audi A6, Volvo XC90.

**Q: Why did you use a training-set median for the price segment threshold?**
A: Because computing the threshold from the full dataset would be leakage. The test set would influence the threshold used to label it. By computing the median from `y_reg_train` only (€19,486), the threshold is derived from information that was legitimately available before the test set was "seen." This is the correct procedure for any target-derived feature in ML.

**Q: How did you prove there was no data leakage?**
A: Three ways. First, `price_in_euro` is entirely excluded from classification features. Second, the segment threshold is documented as training-median-only. Third, the StandardScaler mean embedded in the saved pipeline matches `X_train.mean()` exactly and differs from the full-dataset mean — proving the scaler was fitted only on training data.

**Q: Why did XGBoost win over Logistic Regression by such a small margin?**
A: Because car pricing has a strong additive structure. Brand + age + mileage + power add up nearly linearly to determine price. XGBoost's advantage comes from capturing non-linear interaction effects (e.g., mileage matters more for cheap cars than expensive ones), but these interactions are not large enough to give a major advantage. The dataset is also large (192k training rows), which benefits linear models.

**Q: Why did Ridge regression perform so well?**
A: Ridge with L2 regularization handles sparse OHE features well — it shrinks small brand/model coefficients toward zero without eliminating them. Car pricing is fundamentally additive (a car's price ≈ brand baseline + age penalty + mileage penalty + power premium), and Ridge captures this structure perfectly. The near-zero train-test gap (R² identical on both) confirms it is not overfitting.

**Q: Why F1_macro and not F1_binary?**
A: F1_binary for the DummyClassifier was 0.667 because it always predicts HIGH and gets perfect recall for that class. But it never predicts LOW, so F1 for the LOW class is zero. F1_macro = (0.667 + 0.0) / 2 = 0.333, which honestly shows the dummy is barely above chance. Using F1_binary would make the dummy look competitive with real models, which is dishonest.

**Q: Why didn't you log-transform the price target?**
A: After filtering to €500–€80,000, the price skewness is 1.20 — moderate, not extreme. Tree-based models (XGBoost, Decision Tree) are not affected by target skewness at all because they split on ranks. Ridge would benefit from log-transformation but still achieved R² = 0.841 without it. We chose to keep raw EUR because it makes predictions directly interpretable: €21,221, not e^9.96. A future improvement could log-transform for better Ridge performance and residual normality.

**Q: Why did you keep both `brand` and `model`?**
A: With `min_frequency=50`, rare models collapse into a single infrequent bucket. A rare Porsche and a rare Opel would be indistinguishable if only `model` were used. Keeping `brand` separately lets the model know which manufacturer the rare car belongs to, and Porsche pricing is very different from Opel pricing even for rare models.

**Q: How does the app make predictions without retraining?**
A: The complete sklearn Pipeline (preprocessor + VarianceThreshold + XGBoost) is serialized to a `.pkl` file with `joblib.dump()`. The app loads this file once at startup (cached). When a user clicks Estimate Value, the app constructs a one-row DataFrame with the 8 feature columns and calls `pipeline.predict()`. The pipeline applies OHE, scaling, and XGBoost inference in sequence. No training happens in the app.

**Q: What would you improve if you had more time?**
A: Log-transform the regression target for better linear model residuals; add calibrated probabilities to the classifier; retrain on more recent data (2024/2025 listings to account for market changes since 2023); add a confidence interval around the price estimate (e.g., "likely between €18,000 and €24,000") using quantile regression; and include geographic features if available.

**Q: Is a €4,430 RMSE good for a used-car pricing model?**
A: Yes, it is competitive. The baseline (always predicting mean) has RMSE = €14,369. We reduced error by 69%. For the typical car in the €5,000–€35,000 range (the bulk of the dataset), a €4,430 average error is roughly 12–88% of the price — so it works better for mid-range cars than cheap ones. A professional AutoScout24 model would likely use far more features (exact trim, accident history, photos). Given only 7 user inputs, the results are strong.

---

## 19. Two-Minute Oral Explanation

*Practice saying this until it feels natural:*

---

"I worked with a dataset of 251,000 real used-car listings from AutoScout24 Germany, scraped in 2023. I had two ML tasks: first, classifying whether a car belongs to the HIGH or LOW price segment — above or below about €19,500 — and second, predicting the actual listed price in euros.

Before any modelling, I did a full data cleaning pass. The dataset had type problems — prices and horsepower were stored as strings — and there were some corrupted rows where the fuel type field contained mileage values instead. I also applied scope filters based on my teacher's feedback: I removed cars before 2000 because oldtimers follow collector pricing logic, not depreciation curves. I removed cars above €80,000 because Ferraris and Bentleys are priced on prestige, not the same factors that price a VW Golf. I also removed zero-mileage listings because those are dealer-new cars, not used cars. After all this, I kept 240,017 rows — about 95.6% of the data.

For features, I engineered two new variables. Instead of using raw registration year, I computed car age by subtracting from 2023, because age is what directly drives depreciation — not what year it was built. I also computed mileage per year, which tells you how intensively a car was used, which is a stronger signal than raw mileage alone.

To avoid data leakage, all preprocessing — one-hot encoding, scaling, and feature selection — happens inside a sklearn Pipeline that is fitted only on the training data. The classification threshold was computed from training prices only. I verified this by checking the scaler's internal mean values match the training set exactly.

I trained four models for each task — a dummy baseline, a linear model, a decision tree, and XGBoost. For classification, XGBoost achieved 93.4% accuracy and a ROC-AUC of 0.982. For regression, XGBoost achieved an R² of 0.905 with an average error of about €2,600. The interesting finding was that logistic regression came very close to XGBoost, which tells me that car pricing has a strong additive structure that even a linear model can mostly capture.

I deployed the best models in a Streamlit web app where users enter seven car details and get an instant price estimate and segment classification. The app was smoke-tested on three representative cars to confirm predictions are correct and the pipeline runs without errors."

---

## 20. File Reference Map

| File | What it does |
|---|---|
| `src/clean.py` | Loads raw CSV, converts types, applies 12 sequential filters, saves EDA figures |
| `src/features.py` | Computes car_age and mileage_per_year, splits data 80/20 (stratified), builds ColumnTransformer |
| `src/train.py` | Trains 4-model ladder for both tasks, saves best pipelines as PKL, saves leaderboards |
| `app/app.py` | Streamlit UI — loads PKL files, takes 7 inputs, returns predictions |
| `data/raw/germany_cars.csv` | Original dataset (251,079 rows, not committed to git if large) |
| `data/processed/cars_clean.csv` | Cleaned dataset (240,017 rows) |
| `data/processed/cleaning_metadata.json` | Step-by-step cleaning audit with row counts |
| `data/processed/split_metadata.json` | Train/test split details, feature columns |
| `data/processed/X_train.csv / X_test.csv` | Feature matrices (192,013 / 48,004 rows, 8 columns) |
| `models/classifier_pipeline.pkl` | Complete sklearn Pipeline (preprocessor → VarianceThreshold → XGBoost) |
| `models/regressor_pipeline.pkl` | Complete sklearn Pipeline (preprocessor → VarianceThreshold → XGBoost) |
| `models/price_segment_threshold.json` | €19,486 — HIGH/LOW boundary, training-median-only |
| `reports/tables/classification_leaderboard.csv` | All 4 classifiers compared on test set |
| `reports/tables/regression_leaderboard.csv` | All 4 regressors compared on test set |
| `reports/figures/*.png` | 15 EDA and evaluation figures |
| `reports/modeling_metadata.json` | Full training log, leakage proof, pipeline details |
| `.streamlit/config.toml` | Light ivory premium theme |

---

*This document was generated from the actual project files. All numbers are real.*
