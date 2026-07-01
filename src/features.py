"""
Phase 2B/2C — Target Creation, Train/Test Split, Feature Engineering,
and Preprocessing Pipeline Design
======================================================================
Key design decisions (all leakage-safe):
  • car_age = 2023 - year   (replaces raw year entirely)
  • mileage_per_year = mileage_in_km / max(car_age, 1)
  • price_segment threshold derived from y_train median only
  • ColumnTransformer fitted on training data only
  • Two separate complete pipelines saved: classifier + regressor

Run from the project root:
    python src/features.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "processed" / "cars_clean.csv"
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR   = ROOT / "reports" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
MIN_FREQ_MODEL = 50      # models with fewer rows → infrequent OHE bucket
REFERENCE_YEAR = 2023    # dataset scraped in 2023; used for car_age


def banner(text: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD CLEANED DATA
# ══════════════════════════════════════════════════════════════════════════
banner("1. Loading cleaned dataset")

df = pd.read_csv(CLEAN_CSV)
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
banner("2. Feature engineering")

# car_age replaces year entirely.
# Using age instead of raw year makes the linear component meaningful
# (older cars cost less, independent of which calendar year it is).
# Raw year is then dropped — keeping both would introduce near-perfect
# collinearity that harms Ridge/Logistic convergence and interpretability.
df["car_age"] = REFERENCE_YEAR - df["year"]

# mileage_per_year: intensity of use, a strong price signal.
# max(car_age, 1) prevents division by zero for brand-new cars (car_age=0).
# Cars with car_age=0 are year-2023 listings with very low mileage;
# clamping to 1 still produces a defensible estimate.
df["mileage_per_year"] = df["mileage_in_km"] / df["car_age"].clip(lower=1)

print(f"  car_age:           range {df['car_age'].min():.0f}–{df['car_age'].max():.0f} yrs  "
      f"mean {df['car_age'].mean():.1f} yrs")
print(f"  mileage_per_year:  range {df['mileage_per_year'].min():.0f}–{df['mileage_per_year'].max():,.0f}  "
      f"median {df['mileage_per_year'].median():,.0f} km/yr")
print(f"  NOTE: raw 'year' column is NOT included in model features — "
      f"car_age carries the same information without collinearity risk.")


# ══════════════════════════════════════════════════════════════════════════
# 3. DEFINE FEATURE SETS AND TARGETS
# ══════════════════════════════════════════════════════════════════════════
banner("3. Feature sets and targets")

Y_REG = "price_in_euro"

# ── Categorical columns ──────────────────────────────────────────────────
CAT_BRAND     = ["brand"]               # 46 unique values
CAT_MODEL     = ["model"]               # 1,191 → min_frequency groups rare ones
CAT_OTHER     = ["fuel_type",           # 10 values
                 "transmission_type"]   #  3 values

# ── Numeric columns (after engineering) ──────────────────────────────────
# year is intentionally excluded — car_age captures all its price-relevant
# information without creating a collinear pair.
NUM_COLS = [
    "car_age",           # 2023 - year
    "power_ps",          # horsepower
    "mileage_in_km",     # total odometer reading
    "mileage_per_year",  # mileage_in_km / max(car_age, 1)
]

FEATURE_COLS = CAT_BRAND + CAT_MODEL + CAT_OTHER + NUM_COLS

print(f"\n  Categorical features ({len(CAT_BRAND+CAT_MODEL+CAT_OTHER)}):")
print(f"    {CAT_BRAND + CAT_MODEL + CAT_OTHER}")
print(f"\n  Numeric features ({len(NUM_COLS)}):")
print(f"    {NUM_COLS}")
print(f"\n  Total feature columns: {len(FEATURE_COLS)}")
print(f"\n  Excluded from features:")
print(f"    year           → collinear with car_age; raw year dropped")
print(f"    price_in_euro  → regression target only")
print(f"    price_segment  → classification target (created after split)")

# Explicit leakage confirmation
print(f"\n  === LEAKAGE CHECK ===")
print(f"  price_in_euro NOT in FEATURE_COLS: {Y_REG not in FEATURE_COLS}")
print(f"  'year' NOT in FEATURE_COLS:        {'year' not in FEATURE_COLS}")
print(f"  'price_segment' NOT in FEATURE_COLS: {'price_segment' not in FEATURE_COLS}")


# ══════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════
banner("4. Train/test split (80/20, random_state=42)")

X     = df[FEATURE_COLS].copy()
y_reg = df[Y_REG].copy()

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

print(f"  X_train: {X_train.shape[0]:,} rows  X_test: {X_test.shape[0]:,} rows")
print(f"  Ratio:   {X_train.shape[0]/len(X)*100:.1f}% / {X_test.shape[0]/len(X)*100:.1f}%")

# Sanity: year column must not be in train/test
assert "year" not in X_train.columns, "BUG: raw year is in training features"
print(f"  'year' absent from X_train: confirmed ✓")


# ══════════════════════════════════════════════════════════════════════════
# 5. LEAKAGE-SAFE CLASSIFICATION TARGET
# ══════════════════════════════════════════════════════════════════════════
banner("5. Leakage-safe price_segment (threshold from y_train only)")

# CRITICAL: threshold computed from training set ONLY.
# Using the full-dataset median would embed test-set price information
# into the target labels, inflating test accuracy.
threshold = float(y_reg_train.median())
print(f"  Training median (→ threshold):  €{threshold:,.0f}")
print(f"  Full-dataset median (NOT used): €{y_reg.median():,.0f}")

y_cls_train = (y_reg_train >= threshold).astype(int).rename("price_segment")
y_cls_test  = (y_reg_test  >= threshold).astype(int).rename("price_segment")

train_bal = y_cls_train.value_counts(normalize=True)
test_bal  = y_cls_test.value_counts(normalize=True)
print(f"\n  Class balance — train:  LOW={train_bal.get(0,0)*100:.1f}%  HIGH={train_bal.get(1,0)*100:.1f}%")
print(f"  Class balance — test:   LOW={test_bal.get(0,0)*100:.1f}%  HIGH={test_bal.get(1,0)*100:.1f}%")
print(f"  (Balanced by construction: median splits distribution at 50/50)")


# ══════════════════════════════════════════════════════════════════════════
# 6. PRICE SEGMENT VISUALISATION
# ══════════════════════════════════════════════════════════════════════════
banner("6. Saving price_segment_split.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(y_reg_train, bins=80, color="#4C72B0", edgecolor="none", alpha=0.8,
             label="Training prices")
axes[0].axvline(threshold, color="#C44E52", linewidth=2,
                label=f"Median threshold = €{threshold:,.0f}")
axes[0].set_title("Training price distribution + segment threshold")
axes[0].set_xlabel("Price (EUR)")
axes[0].set_ylabel("Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"€{x/1_000:.0f}k"))
axes[0].legend()

labels = ["LOW (0)", "HIGH (1)"]
train_counts = [y_cls_train.value_counts().get(i, 0) for i in [0, 1]]
test_counts  = [y_cls_test.value_counts().get(i, 0) for i in [0, 1]]
xp = np.arange(len(labels))
w  = 0.35
axes[1].bar(xp - w/2, train_counts, w, label="Train", color="#4C72B0")
axes[1].bar(xp + w/2, test_counts,  w, label="Test",  color="#DD8452")
axes[1].set_title("Class balance — price_segment")
axes[1].set_xticks(xp)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Count")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda y, _: f"{y/1_000:.0f}k"))
axes[1].legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "price_segment_split.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: price_segment_split.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. PREPROCESSING — ColumnTransformer (fitted on training data ONLY)
# ══════════════════════════════════════════════════════════════════════════
banner("7. Building ColumnTransformer (fitted on X_train only)")

# Design rationale:
# ─────────────────────────────────────────────────────────────────────────
# brand (46 categories):
#   OHE, handle_unknown="ignore" → new brand at inference → all-zero row.
#
# model (1,191 categories, min_frequency=50):
#   Models with < 50 training rows → single "_infrequent_" dummy.
#   handle_unknown="infrequent_if_exist" → unseen inference model routes
#   to the infrequent bucket rather than all-zero or error.
#   Fitted on X_train only — thresholds are computed from training counts.
#   Brand is retained separately (not dropped) because rare models collapse
#   into one bucket; without brand, a rare Porsche and a rare Opel are
#   indistinguishable. Brand preserves that signal.
#
# fuel_type, transmission_type:
#   Standard OHE, handle_unknown="ignore".
#
# Numeric (car_age, power_ps, mileage_in_km, mileage_per_year):
#   StandardScaler. Needed for Ridge and Logistic Regression to converge
#   on fair footing; scale-invariant for tree models (no harm).
#
# Caveat for the report (linear model structural disadvantage):
#   Total output features ≈ 559. OHE produces a high-dimensional sparse
#   binary matrix. Ridge/Logistic are at a structural disadvantage vs
#   XGBoost because:
#     1. Sparsity: most rows have only ~3–4 non-zero categorical columns.
#     2. Collinearity: brand ↔ model dummies are partially redundant.
#     3. Residual price skew (1.21): linear residuals will be non-normal.
#   If linear models underperform in the ladder, the explanation in the
#   report should mention these structural factors, not just "model is weak."
#   Hyperparameter tuning (C for LogReg, alpha for Ridge) will still be
#   applied to give them a fair chance. See src/train.py.
# ─────────────────────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("brand_ohe",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CAT_BRAND),

        ("model_ohe",
         OneHotEncoder(
             min_frequency=MIN_FREQ_MODEL,
             handle_unknown="infrequent_if_exist",
             sparse_output=False,
         ),
         CAT_MODEL),

        ("cat_ohe",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CAT_OTHER),

        ("num_scaler",
         StandardScaler(),
         NUM_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# Fit ONLY on training data
preprocessor.fit(X_train)

X_train_t = preprocessor.transform(X_train)
X_test_t  = preprocessor.transform(X_test)

n_out = X_train_t.shape[1]
brand_feats = preprocessor.named_transformers_["brand_ohe"].get_feature_names_out(CAT_BRAND)
model_feats = preprocessor.named_transformers_["model_ohe"].get_feature_names_out(CAT_MODEL)
cat_feats   = preprocessor.named_transformers_["cat_ohe"].get_feature_names_out(CAT_OTHER)
model_cats  = preprocessor.named_transformers_["model_ohe"].categories_[0]
model_infreq= preprocessor.named_transformers_["model_ohe"].infrequent_categories_
n_infreq    = len(model_infreq[0]) if model_infreq[0] is not None else 0

print(f"  Input features:              {len(FEATURE_COLS)}")
print(f"  Output features after OHE:   {n_out}")
print(f"    brand dummies:             {len(brand_feats)}")
print(f"    model dummies:             {len(model_feats)}  ({len(model_cats)-n_infreq} frequent + 1 infrequent)")
print(f"    fuel+transmission dummies: {len(cat_feats)}")
print(f"    numeric (scaled):          {len(NUM_COLS)}")
print(f"  Model OHE: {n_infreq} infrequent models grouped into '_infrequent_' bucket")
print(f"  Fitted on X_train only ✓")


# ══════════════════════════════════════════════════════════════════════════
# 8. SAVE SPLITS AND PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════
banner("8. Saving splits and preprocessor artifact")

X_train.to_csv(PROC_DIR / "X_train.csv", index=False)
X_test.to_csv( PROC_DIR / "X_test.csv",  index=False)
y_reg_train.to_csv(PROC_DIR / "y_reg_train.csv", index=False, header=True)
y_reg_test.to_csv( PROC_DIR / "y_reg_test.csv",  index=False, header=True)
y_cls_train.to_csv(PROC_DIR / "y_cls_train.csv", index=False, header=True)
y_cls_test.to_csv( PROC_DIR / "y_cls_test.csv",  index=False, header=True)

joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")

print(f"  X_train.csv / X_test.csv")
print(f"  y_reg_train.csv / y_reg_test.csv  (price_in_euro)")
print(f"  y_cls_train.csv / y_cls_test.csv  (price_segment: 0/1)")
print(f"  preprocessor.pkl  (ColumnTransformer fitted on X_train)")


# ══════════════════════════════════════════════════════════════════════════
# 9. SAVE SPLIT METADATA
# ══════════════════════════════════════════════════════════════════════════
banner("9. Saving split_metadata.json")

metadata = {
    "random_state":    RANDOM_STATE,
    "test_size":       TEST_SIZE,
    "reference_year":  REFERENCE_YEAR,
    "n_train":         int(X_train.shape[0]),
    "n_test":          int(X_test.shape[0]),
    "feature_cols":    FEATURE_COLS,
    "engineered_features": ["car_age", "mileage_per_year"],
    "dropped_before_modelling": {
        "year": "collinear with car_age — dropped to avoid ridge/logistic instability",
    },
    "regression_target":     Y_REG,
    "classification_target": "price_segment",
    "price_segment_threshold": threshold,
    "threshold_derivation":    "median of y_reg_train only — leakage-safe",
    "class_balance_train": {"LOW_0": float(train_bal.get(0, 0)),
                            "HIGH_1": float(train_bal.get(1, 0))},
    "class_balance_test":  {"LOW_0": float(test_bal.get(0, 0)),
                            "HIGH_1": float(test_bal.get(1, 0))},
    "preprocessing": {
        "brand":           "OneHotEncoder(handle_unknown='ignore')",
        "model":           f"OneHotEncoder(min_frequency={MIN_FREQ_MODEL}, "
                           "handle_unknown='infrequent_if_exist')",
        "fuel_type":       "OneHotEncoder(handle_unknown='ignore')",
        "transmission_type": "OneHotEncoder(handle_unknown='ignore')",
        "numeric":         "StandardScaler",
        "n_output_features":    int(n_out),
        "n_frequent_models":    int(len(model_cats) - n_infreq),
        "n_infrequent_models":  int(n_infreq),
    },
    "leakage_checks": {
        "price_in_euro_excluded_from_features": Y_REG not in FEATURE_COLS,
        "year_excluded_from_features":          "year" not in FEATURE_COLS,
        "threshold_from_training_only":         True,
        "preprocessor_fitted_on_train_only":    True,
        "test_data_only_transformed_not_fitted": True,
    },
    "report_caveats": [
        f"price_segment threshold = €{threshold:,.0f} from training median only",
        "brand retained alongside model because rare models collapse to infrequent "
        "bucket; without brand, rare Porsche (med €63k) ≡ rare Opel in the feature space",
        f"~{n_out} output features; linear models (Ridge, Logistic) structurally "
        "disadvantaged by sparse OHE + collinear brand/model dummies + mild price skew",
        "If linear models underperform, report should mention structural factors, "
        "not only model weakness; C and alpha will be tuned to give fair comparison",
        "mileage_per_year = mileage_in_km / max(car_age, 1); clamp prevents div-by-zero "
        "for car_age=0 (year-2023 new listings with near-zero mileage)",
    ],
}

with open(PROC_DIR / "split_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved: data/processed/split_metadata.json")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY — PIPELINE DESIGN FOR CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════
banner("PHASE 2B/2C COMPLETE — PIPELINE DESIGN FOR CONFIRMATION")

print(f"""
  CORRECTED FEATURE LIST (after engineering):
  ─────────────────────────────────────────────────────────────────────
  Categorical:
    brand              (46 values)   → OHE, handle_unknown="ignore"
    model              (1,191 values)→ OHE, min_frequency=50,
                                        handle_unknown="infrequent_if_exist"
    fuel_type          (10 values)   → OHE, handle_unknown="ignore"
    transmission_type  ( 3 values)   → OHE, handle_unknown="ignore"

  Numeric (StandardScaler):
    car_age            (2023 − year; raw year DROPPED ✓)
    power_ps
    mileage_in_km
    mileage_per_year   (mileage_in_km / max(car_age, 1) ✓)

  NOT in features:
    year               → dropped (collinear with car_age) ✓
    price_in_euro      → regression target only ✓
    price_segment      → classification target only ✓
  ─────────────────────────────────────────────────────────────────────
  Total input features:   {len(FEATURE_COLS)}
  Total after OHE/scale:  {n_out}
  ─────────────────────────────────────────────────────────────────────

  PLANNED PIPELINE STRUCTURE (two complete, independent pipelines):
  ─────────────────────────────────────────────────────────────────────
  classifier_pipeline.pkl:
    Pipeline([
      ("preprocessor", ColumnTransformer(...)),  ← fitted on X_train
      ("model", <best classifier>)               ← fitted on X_train
    ])
    Ladder: DummyClassifier → LogisticRegression(C tuned via CV)
            → DecisionTreeClassifier → XGBClassifier
    Metrics: Accuracy, F1, ROC-AUC (all on test set)
    CV:      5-fold StratifiedKFold for LogReg C tuning

  regressor_pipeline.pkl:
    Pipeline([
      ("preprocessor", ColumnTransformer(...)),  ← fitted on X_train
      ("model", <best regressor>)                ← fitted on X_train
    ])
    Ladder: DummyRegressor → Ridge(alpha tuned via CV)
            → DecisionTreeRegressor → XGBRegressor
    Metrics: MAE, RMSE, R² (all on test set)
    CV:      5-fold KFold for Ridge alpha tuning

  LEAKAGE GUARANTEES:
    ✓ price_in_euro excluded from classification features
    ✓ price_segment threshold = training median only
    ✓ ColumnTransformer fitted on X_train only
    ✓ X_test only transformed (never fitted)
    ✓ No preprocessor step touches the full dataset after splitting
    ✓ Each complete Pipeline is refitted end-to-end on X_train
    ✓ Final evaluation: pipeline.predict(X_test) only

  Next step: confirm this design, then → src/train.py
""")
