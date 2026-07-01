"""
Phase 2A — Data Loading, EDA, Cleaning, and Figure Export
==========================================================
Loads the raw Germany Used Cars 2023 dataset, runs a full sequential
cleaning audit, saves the cleaned CSV, produces EDA figures for the
Quarto report, and writes a metadata JSON for traceability.

Run from the project root:
    python src/clean.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all machines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
RAW_CSV   = ROOT / "data" / "raw"   / "germany_cars.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "cars_clean.csv"
FIG_DIR   = ROOT / "reports" / "figures"
META_JSON = ROOT / "data" / "processed" / "cleaning_metadata.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── Valid category sets ────────────────────────────────────────────────────────
VALID_FUELS = {
    "Petrol", "Diesel", "Hybrid", "Electric",
    "LPG", "CNG", "Diesel Hybrid", "Ethanol", "Hydrogen", "Other",
}

# ── Helper ─────────────────────────────────────────────────────────────────────
def banner(text: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
banner("1. Loading raw dataset")

if not RAW_CSV.exists():
    sys.exit(f"ERROR: raw CSV not found at {RAW_CSV}\n"
             "Place germany_cars.csv in data/raw/ and re-run.")

df_raw = pd.read_csv(RAW_CSV)
n_raw  = len(df_raw)
print(f"  Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"  Columns: {df_raw.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TYPE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
banner("2. Converting column types")

df = df_raw.copy()
df["price_in_euro"] = pd.to_numeric(df["price_in_euro"], errors="coerce")
df["power_ps"]      = pd.to_numeric(df["power_ps"],      errors="coerce")
df["year"]          = pd.to_numeric(df["year"],           errors="coerce")
# mileage_in_km is already float

print("  price_in_euro → numeric  ✓")
print("  power_ps      → numeric  ✓")
print("  year          → numeric  ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INITIAL AUDIT (before cleaning)
# ══════════════════════════════════════════════════════════════════════════════
banner("3. Initial data audit")

print(f"\n  Missing values per column:")
missing = df.isnull().sum()
for col, n in missing.items():
    if n > 0:
        print(f"    {col:<35} {n:>6,}  ({n/len(df)*100:.2f}%)")

print(f"\n  Duplicate rows: {df.duplicated().sum():,}")

print(f"\n  Price distribution (raw):")
print(df["price_in_euro"].describe().apply(lambda x: f"{x:,.1f}").to_string())

print(f"\n  Year range: {df['year'].min():.0f} – {df['year'].max():.0f}")
print(f"  Power_ps range: {df['power_ps'].min():.0f} – {df['power_ps'].max():.0f}")
print(f"  Mileage range: {df['mileage_in_km'].min():,.0f} – {df['mileage_in_km'].max():,.0f}")

print(f"\n  Fuel type unique values ({df['fuel_type'].nunique()}):")
print(df["fuel_type"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4. EDA FIGURES — BEFORE CLEANING
# ══════════════════════════════════════════════════════════════════════════════
banner("4. EDA figures — raw data")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# 4a. Raw price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
price_raw = df["price_in_euro"].dropna()
axes[0].hist(price_raw, bins=100, color="#4C72B0", edgecolor="none")
axes[0].set_title("Price distribution — raw (all rows)")
axes[0].set_xlabel("Price (EUR)")
axes[0].set_ylabel("Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"€{x/1_000:.0f}k"))

axes[1].hist(np.log1p(price_raw), bins=100, color="#DD8452", edgecolor="none")
axes[1].set_title("Log1p(price) — raw")
axes[1].set_xlabel("log1p(Price in EUR)")
axes[1].set_ylabel("Count")
plt.tight_layout()
fig.savefig(FIG_DIR / "price_raw.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: price_raw.png")

# 4b. Year distribution (raw)
fig, ax = plt.subplots(figsize=(10, 4))
year_vc = df["year"].dropna().astype(int).value_counts().sort_index()
ax.bar(year_vc.index, year_vc.values, color="#4C72B0", width=0.8)
ax.set_title("Year of registration — raw dataset")
ax.set_xlabel("Year")
ax.set_ylabel("Count")
ax.axvline(2000, color="red", linestyle="--", linewidth=1.5, label="year = 2000 cutoff")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "year_distribution_raw.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: year_distribution_raw.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SEQUENTIAL CLEANING WITH AUDIT TABLE
# ══════════════════════════════════════════════════════════════════════════════
banner("5. Sequential cleaning audit")

audit_rows = []

def clean_step(label, df, mask, reason):
    """Apply mask, record audit entry, return cleaned dataframe."""
    n_before = len(df)
    df_out   = df[~mask].reset_index(drop=True)
    n_removed = n_before - len(df_out)
    audit_rows.append({
        "step":     label,
        "reason":   reason,
        "removed":  n_removed,
        "remaining": len(df_out),
    })
    print(f"  {label:<52} removed {n_removed:>6,}  remaining {len(df_out):>8,}")
    return df_out

print(f"  {'Step':<52} {'Removed':>13}  {'Remaining':>12}")
print(f"  {'─'*52} {'─'*13}  {'─'*12}")
print(f"  {'Raw dataset':<52} {'—':>13}  {n_raw:>12,}")

df = df.copy()

df = clean_step(
    "Drop NaN price_in_euro", df,
    df["price_in_euro"].isna(),
    "Cannot train or evaluate without a price target.",
)
df = clean_step(
    "Drop NaN year", df,
    df["year"].isna(),
    "Year is required for car_age feature engineering.",
)
df = clean_step(
    "Drop NaN power_ps", df,
    df["power_ps"].isna(),
    "Horsepower is a primary price driver; imputing it would be misleading.",
)
df = clean_step(
    "Drop NaN mileage_in_km", df,
    df["mileage_in_km"].isna(),
    "Mileage is a primary price driver; imputing it would be misleading.",
)
df = clean_step(
    "Remove invalid/corrupted fuel_type", df,
    ~df["fuel_type"].isin(VALID_FUELS),
    "Structural data corruption: rows contain dates, mileage values, or "
    "transmission types in the fuel_type field — entire rows unrecoverable.",
)
df = clean_step(
    "Remove transmission_type == 'Unknown'", df,
    df["transmission_type"] == "Unknown",
    "Unknown transmission cannot be encoded meaningfully.",
)
df = clean_step(
    "Remove year < 2000 (oldtimers/classics)", df,
    df["year"] < 2000,
    "Pre-2000 cars are oldtimers or collector cases. "
    "The teacher explicitly warned to build a general used-car model.",
)
df = clean_step(
    "Remove price_in_euro < 500 (unrealistic)", df,
    df["price_in_euro"] < 500,
    "Prices below €500 are scrap-value listings, not general-market cars.",
)
df = clean_step(
    "Remove price_in_euro > 80000 (luxury/collector)", df,
    df["price_in_euro"] > 80_000,
    "Ultra-luxury and collector cars (Porsche 911, Ferrari, Bentley, etc.) "
    "follow different pricing dynamics and are outside the general-market scope.",
)
df = clean_step(
    "Remove power_ps < 40 or > 800 (implausible)", df,
    (df["power_ps"] < 40) | (df["power_ps"] > 800),
    "Values outside 40–800 PS are data entry errors or exotic supercars.",
)
df = clean_step(
    "Remove mileage_in_km > 500000 (implausible)", df,
    df["mileage_in_km"] > 500_000,
    "Odometer readings above 500,000 km are data entry errors.",
)
df = clean_step(
    "Remove mileage_in_km == 0 (new/demo cars)", df,
    df["mileage_in_km"] == 0,
    "Zero-mileage cars (median year 2023) are dealer-new listings on AutoScout24. "
    "New-car pricing follows MSRP/margin logic, not used-car depreciation. "
    "Including them would bias the general used-car model.",
)

n_clean = len(df)
print(f"  {'─'*52} {'─'*13}  {'─'*12}")
print(f"  {'FINAL clean rows':<52} {'':>13}  {n_clean:>12,}")
print(f"  {'% data retained':<52} {'':>13}  {n_clean/n_raw*100:>11.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DROP LOW-VALUE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
banner("6. Dropping low-value columns")

DROP_COLS = [
    "Unnamed: 0",               # row index artifact from Kaggle CSV
    "registration_date",        # redundant with year
    "power_kw",                 # redundant with power_ps
    "color",                    # low signal, adds ~13 dummy columns for negligible gain
    "fuel_consumption_l_100km", # 10.7% missing in raw data; not a useful predictor
    "fuel_consumption_g_km",    # same as above
    "offer_description",        # free text, ~192k unique values, cannot be encoded
]

cols_before = df.columns.tolist()
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
cols_after  = df.columns.tolist()

print(f"  Columns before drop: {len(cols_before)}  →  after drop: {len(cols_after)}")
print(f"  Kept columns: {cols_after}")
print(f"  Dropped: {[c for c in DROP_COLS if c in cols_before]}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL DATA QUALITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
banner("7. Final data quality check on clean dataset")

print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n  Missing values:")
miss = df.isnull().sum()
if miss.sum() == 0:
    print("    None — clean dataset has no missing values ✓")
else:
    print(miss[miss > 0])

print(f"\n  Column dtypes:")
print(df.dtypes.to_string())

print(f"\n  Numeric summaries:")
print(df[["price_in_euro", "power_ps", "year", "mileage_in_km"]]
      .describe()
      .map(lambda x: f"{x:,.1f}")
      .to_string())

print(f"\n  Categorical value counts:")
for col in ["brand", "fuel_type", "transmission_type"]:
    print(f"\n  {col} ({df[col].nunique()} unique):")
    print(df[col].value_counts().head(10).to_string())

print(f"\n  Model column: {df['model'].nunique():,} unique values "
      f"(handled via OneHotEncoder min_frequency=50)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. EDA FIGURES — AFTER CLEANING
# ══════════════════════════════════════════════════════════════════════════════
banner("8. EDA figures — cleaned data")

# 8a. Price distribution before vs after
price_clean = df["price_in_euro"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(price_clean, bins=80, color="#4C72B0", edgecolor="none")
axes[0].set_title("Price distribution — after cleaning\n(general-market cars, €500–€80,000)")
axes[0].set_xlabel("Price (EUR)")
axes[0].set_ylabel("Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"€{x/1_000:.0f}k"))

axes[1].hist(np.log1p(price_clean), bins=80, color="#DD8452", edgecolor="none")
axes[1].set_title(f"Log1p(price) — after cleaning\nskewness = {np.log1p(price_clean).skew():.2f}")
axes[1].set_xlabel("log1p(Price in EUR)")
axes[1].set_ylabel("Count")

plt.tight_layout()
fig.savefig(FIG_DIR / "price_clean.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: price_clean.png")

price_raw_num = pd.to_numeric(df_raw["price_in_euro"], errors="coerce").dropna()
print(f"\n  Price skewness — raw:     {price_raw_num.skew():.2f}")
print(f"  Price skewness — cleaned: {price_clean.skew():.2f}")
print(f"  Log1p skewness — cleaned: {np.log1p(price_clean).skew():.2f}")

# 8b. Mileage distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["mileage_in_km"] / 1_000, bins=80, color="#55A868", edgecolor="none")
ax.set_title("Mileage distribution — cleaned dataset")
ax.set_xlabel("Mileage (thousands of km)")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(FIG_DIR / "mileage_clean.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: mileage_clean.png")

# 8c. Power distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["power_ps"], bins=60, color="#C44E52", edgecolor="none")
ax.set_title("Horsepower distribution — cleaned dataset")
ax.set_xlabel("Power (PS)")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(FIG_DIR / "power_clean.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: power_clean.png")

# 8d. Year distribution after cleaning
fig, ax = plt.subplots(figsize=(10, 4))
year_vc_clean = df["year"].astype(int).value_counts().sort_index()
ax.bar(year_vc_clean.index, year_vc_clean.values, color="#4C72B0", width=0.8)
ax.set_title("Year of registration — cleaned dataset (year ≥ 2000)")
ax.set_xlabel("Year")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(FIG_DIR / "year_clean.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: year_clean.png")

# 8e. Fuel type bar chart
fig, ax = plt.subplots(figsize=(9, 4))
fuel_vc = df["fuel_type"].value_counts()
fuel_vc.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="none")
ax.set_title("Fuel type distribution — cleaned dataset")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "fuel_type_clean.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fuel_type_clean.png")

# 8f. Top 20 brands by count
fig, ax = plt.subplots(figsize=(10, 5))
brand_vc = df["brand"].value_counts().head(20)
brand_vc.sort_values().plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="none")
ax.set_title("Top 20 brands by listing count — cleaned dataset")
ax.set_xlabel("Count")
plt.tight_layout()
fig.savefig(FIG_DIR / "top_brands.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: top_brands.png")

# 8g. Price vs mileage scatter (10k random sample for speed)
sample = df.sample(10_000, random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(
    sample["mileage_in_km"] / 1_000,
    sample["price_in_euro"],
    alpha=0.15, s=8, c=sample["year"],
    cmap="viridis"
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Year")
ax.set_title("Price vs Mileage (sample of 10,000 rows, coloured by year)")
ax.set_xlabel("Mileage (thousands of km)")
ax.set_ylabel("Price (EUR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1_000:.0f}k"))
plt.tight_layout()
fig.savefig(FIG_DIR / "price_vs_mileage.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: price_vs_mileage.png")

# 8h. Median price by fuel type
fig, ax = plt.subplots(figsize=(9, 4))
med_price = (df.groupby("fuel_type")["price_in_euro"]
               .median()
               .sort_values(ascending=False))
med_price.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="none")
ax.set_title("Median price by fuel type — cleaned dataset")
ax.set_xlabel("")
ax.set_ylabel("Median price (EUR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1_000:.0f}k"))
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "median_price_by_fuel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: median_price_by_fuel.png")

# 8i. Correlation heatmap (numeric columns only)
num_cols = ["price_in_euro", "year", "power_ps", "mileage_in_km"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title("Pearson correlation — numeric features vs price")
plt.tight_layout()
fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE CLEANED CSV
# ══════════════════════════════════════════════════════════════════════════════
banner("9. Saving cleaned CSV")

df.to_csv(CLEAN_CSV, index=False)
saved_rows = pd.read_csv(CLEAN_CSV).shape[0]
print(f"  Saved: {CLEAN_CSV}")
print(f"  Written rows:    {saved_rows:,}")
print(f"  Expected rows:   {n_clean:,}")

if saved_rows != n_clean:
    sys.exit(f"ERROR: row count mismatch! Written={saved_rows}, expected={n_clean}")
else:
    print(f"  Row count verified ✓  ({saved_rows:,} rows match audit total)")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE METADATA JSON
# ══════════════════════════════════════════════════════════════════════════════
banner("10. Saving cleaning metadata")

metadata = {
    "raw_rows":        n_raw,
    "clean_rows":      n_clean,
    "pct_retained":    round(n_clean / n_raw * 100, 2),
    "final_columns":   cols_after,
    "dropped_columns": DROP_COLS,
    "cleaning_steps":  audit_rows,
    "design_decisions": {
        "year_cutoff": "year >= 2000 — removes oldtimers and collector-style cases",
        "price_range": "500 <= price_in_euro <= 80000 — general used-car market scope",
        "power_range": "40 <= power_ps <= 800 — removes implausible values",
        "mileage_max": "mileage_in_km <= 500000 — removes odometer errors",
        "zero_mileage": "dropped — new/demo dealer cars follow new-car pricing, not used-car depreciation",
        "fuel_type_corruption": "corrupted rows (dates/mileage/transmission in fuel_type field) are unrecoverable and dropped",
        "price_target": "raw price_in_euro used for regression (skewness 1.20 post-filter, manageable for tree models)",
        "color_dropped": "low signal, adds ~13 dummy columns; excluded to keep model clean and explainable",
        "model_encoding": "OneHotEncoder(min_frequency=50, handle_unknown='infrequent_if_exist') — leakage-safe, handles rare and unseen models",
        "log_price_caveat": "log1p(price) not used as target; if linear models show skewed residuals, this is partly due to moderate price skewness (1.20), not only model weakness",
    },
}

with open(META_JSON, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Saved: {META_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
banner("PHASE 2A COMPLETE")
print(f"  Raw rows:      {n_raw:>10,}")
print(f"  Clean rows:    {n_clean:>10,}  ({n_clean/n_raw*100:.1f}% retained)")
print(f"  Columns kept:  {len(cols_after)}")
print(f"  Figures saved: {len(list(FIG_DIR.glob('*.png')))} PNG files in reports/figures/")
print(f"  Cleaned CSV:   data/processed/cars_clean.csv")
print(f"  Metadata:      data/processed/cleaning_metadata.json")
print()
