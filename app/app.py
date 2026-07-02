"""
Germany Used Cars — Price Prediction App (Premium Light)
=========================================================
Prediction logic (unchanged):
  • user enters: brand, model, fuel_type, transmission_type, year, power_ps, mileage_in_km
  • app computes: car_age = 2023 − year
  • app computes: mileage_per_year = mileage_in_km / max(car_age, 1)
  • pipeline receives exactly these 8 columns:
        brand, model, fuel_type, transmission_type,
        car_age, power_ps, mileage_in_km, mileage_per_year

Run:
    streamlit run app/app.py --server.port 8510
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "reports" / "tables"
FIG_DIR   = ROOT / "reports" / "figures"

REFERENCE_YEAR = 2023
FEATURE_COLS = [
    "brand", "model", "fuel_type", "transmission_type",
    "car_age", "power_ps", "mileage_in_km", "mileage_per_year",
]

# ══════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Used Car Value Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS — premium light automotive theme (no images, no external assets)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>

/* ── Base & layout ─────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: #faf9f6 !important;
}
.block-container {
    padding-top: 1.8rem !important;
    max-width: 1180px !important;
}
[data-testid="stSidebar"] {
    background-color: #f2efe9 !important;
    border-right: 1px solid #e0dbd0 !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem !important;
}

/* ── Hero ──────────────────────────────────────────────────────────── */
.hero {
    background: #ffffff;
    border: 1px solid #e8e4d8;
    border-radius: 14px;
    padding: 2.2rem 2.5rem 1.8rem 2.5rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}
/* Subtle gold top accent line */
.hero::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, #b8960c 35%, #d4aa3c 65%, transparent 100%);
}
.hero-badge {
    display: inline-block;
    background: rgba(184,150,12,0.10);
    border: 1px solid rgba(184,150,12,0.28);
    color: #7a6005;
    font-size: 0.70rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 0.22rem 0.65rem;
    border-radius: 20px;
    margin-bottom: 0.85rem;
    font-weight: 600;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1c1c1c;
    letter-spacing: -0.02em;
    margin: 0 0 0.45rem 0;
    line-height: 1.15;
}
.hero-accent { color: #b8960c; }
.hero-sub {
    font-size: 0.92rem;
    color: #6a6560;
    margin: 0;
    max-width: 580px;
    line-height: 1.62;
}

/* ── Scope notice ──────────────────────────────────────────────────── */
.scope-notice {
    background: #fffdf5;
    border: 1px solid #e8dfb8;
    border-left: 3px solid #b8960c;
    border-radius: 6px;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #7a7060;
    margin-bottom: 1.5rem;
    line-height: 1.55;
}
.scope-notice strong { color: #7a6005; }

/* ── Result metric cards ───────────────────────────────────────────── */
.cards-row {
    display: flex;
    gap: 1rem;
    margin: 1.4rem 0 0.9rem 0;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 160px;
    background: #ffffff;
    border: 1px solid #e0dbd0;
    border-radius: 10px;
    padding: 1.2rem 1.3rem 1.1rem 1.3rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d4aa3c, transparent);
    opacity: 0.45;
}
.metric-card.primary {
    border-color: rgba(184,150,12,0.38);
    background: linear-gradient(150deg, #fffdf5 0%, #ffffff 60%);
    box-shadow: 0 2px 16px rgba(184,150,12,0.09);
}
.metric-card.primary::after { opacity: 1.0; }
.card-label {
    font-size: 0.68rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #a09880;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.card-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #1c1c1c;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.card-value.gold { color: #8a6e05; }
.card-sub { font-size: 0.74rem; color: #a09880; }

/* ── Probability bars ──────────────────────────────────────────────── */
.conf-wrap {
    background: #ffffff;
    border: 1px solid #e8e4d8;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0 1.3rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
}
.conf-row:last-child { margin-bottom: 0; }
.conf-lbl {
    font-size: 0.73rem;
    color: #7a7060;
    width: 48px;
    flex-shrink: 0;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.conf-track {
    flex: 1;
    background: #f0ede7;
    border-radius: 4px;
    height: 7px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #c8a030, #b8960c);
}
.conf-pct {
    font-size: 0.73rem;
    color: #5a5248;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
    font-weight: 600;
}

/* ── Out-of-range warning ──────────────────────────────────────────── */
.warn-card {
    background: #fff8f6;
    border: 1px solid #f0d0c8;
    border-left: 3px solid #c84030;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.81rem;
    color: #8a5048;
    margin-top: 0.6rem;
}

/* ── Section divider ───────────────────────────────────────────────── */
.section-header {
    font-size: 0.70rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a09880;
    border-bottom: 1px solid #e8e4d8;
    padding-bottom: 0.45rem;
    margin: 1.8rem 0 1rem 0;
    font-weight: 600;
}

/* ── Sidebar section labels ────────────────────────────────────────── */
.sidebar-group {
    font-size: 0.67rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: #a09880;
    margin: 1.1rem 0 0.35rem 0;
    padding-top: 0.75rem;
    border-top: 1px solid #ddd8cc;
    font-weight: 600;
}
.sidebar-group.first {
    border-top: none;
    margin-top: 0.1rem;
    padding-top: 0;
}

/* ── Predict button ────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #9a7a08, #b8960c, #c8a030);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.84rem;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    width: 100%;
    margin-top: 0.5rem;
    box-shadow: 0 2px 10px rgba(184,150,12,0.22);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #b8960c, #c8a030, #d4b040);
    box-shadow: 0 4px 16px rgba(184,150,12,0.32);
}

/* ── Idle placeholder ──────────────────────────────────────────────── */
.idle-box {
    text-align: center;
    padding: 3.2rem 1rem 3.8rem 1rem;
    background: #ffffff;
    border: 1px dashed #ddd8cc;
    border-radius: 10px;
    margin: 1.4rem 0;
}
.idle-icon { font-size: 2.8rem; margin-bottom: 0.75rem; opacity: 0.45; }
.idle-text { font-size: 0.92rem; color: #a09880; }
.idle-cta  { color: #b8960c; font-weight: 600; }

/* ── Charts & tables ───────────────────────────────────────────────── */
img { border-radius: 8px; }
[data-testid="stDataFrame"] {
    border: 1px solid #e8e4d8 !important;
    border-radius: 8px !important;
}
.stExpander {
    border: 1px solid #e8e4d8 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading valuation models…")
def load_models():
    cls_pipe = joblib.load(MODEL_DIR / "classifier_pipeline.pkl")
    reg_pipe = joblib.load(MODEL_DIR / "regressor_pipeline.pkl")
    with open(MODEL_DIR / "price_segment_threshold.json") as f:
        threshold_meta = json.load(f)
    return cls_pipe, reg_pipe, threshold_meta


@st.cache_data(show_spinner=False)
def load_leaderboards():
    cls_lb = pd.read_csv(TABLE_DIR / "classification_leaderboard.csv")
    reg_lb = pd.read_csv(TABLE_DIR / "regression_leaderboard.csv")
    return cls_lb, reg_lb


@st.cache_data(show_spinner=False)
def load_car_options():
    try:
        df = pd.read_csv(
            Path(__file__).resolve().parent / "brand_model_options.csv"
        )
        brands = sorted(df["brand"].unique().tolist())
        brand_models: dict[str, list[str]] = {}
        for b in brands:
            brand_models[b] = df[df["brand"] == b]["model"].tolist()
        return brands, brand_models
    except FileNotFoundError:
        return [], {}


@st.cache_data(show_spinner=False)
def load_dataset():
    try:
        return pd.read_csv(Path(__file__).resolve().parent / "cars_sample.csv")
    except FileNotFoundError:
        return None


cls_pipe, reg_pipe, threshold_meta = load_models()
threshold = threshold_meta["price_segment_threshold"]
cls_lb, reg_lb = load_leaderboards()
brands, brand_models = load_car_options()
df_sample = load_dataset()

brand_options = brands if brands else [
    "volkswagen", "bmw", "audi", "mercedes-benz", "opel", "ford",
    "skoda", "seat", "toyota", "hyundai",
]


# ══════════════════════════════════════════════════════════════════════════
# Feature engineering — IDENTICAL to src/features.py (do not modify)
# ══════════════════════════════════════════════════════════════════════════
def build_input_row(brand, model, fuel_type, transmission_type,
                    year, power_ps, mileage_in_km):
    car_age          = REFERENCE_YEAR - year
    mileage_per_year = mileage_in_km / max(car_age, 1)
    return pd.DataFrame([{
        "brand":             brand,
        "model":             model,
        "fuel_type":         fuel_type,
        "transmission_type": transmission_type,
        "car_age":           float(car_age),
        "power_ps":          float(power_ps),
        "mileage_in_km":     float(mileage_in_km),
        "mileage_per_year":  float(mileage_per_year),
    }])[FEATURE_COLS]


# ══════════════════════════════════════════════════════════════════════════
# Feature explanation & similarity helpers
# ══════════════════════════════════════════════════════════════════════════
def explain_prediction(brand, model_name, fuel_type, year, power_ps,
                       mileage_in_km, price, df):
    if df is None:
        return ["Explanation unavailable — sample dataset not loaded."]
    car_age       = float(REFERENCE_YEAR - year)
    med_mileage   = df["mileage_in_km"].median()
    med_age       = (REFERENCE_YEAR - df["year"]).median()
    med_power     = df["power_ps"].median()
    overall_med   = df["price_in_euro"].median()
    bullets = []

    if mileage_in_km < med_mileage * 0.75:
        bullets.append(
            f"**Low mileage** ({mileage_in_km:,.0f} km vs dataset avg {med_mileage:,.0f} km)"
            f" — in this dataset, lower mileage is likely associated with higher prices"
        )
    elif mileage_in_km > med_mileage * 1.25:
        bullets.append(
            f"**High mileage** ({mileage_in_km:,.0f} km vs dataset avg {med_mileage:,.0f} km)"
            f" — in this dataset, higher mileage is likely associated with lower prices"
        )
    else:
        bullets.append(
            f"**Average mileage** ({mileage_in_km:,.0f} km)"
            f" — close to the dataset average of {med_mileage:,.0f} km"
        )

    if car_age < med_age * 0.75:
        bullets.append(
            f"**Newer registration** ({year}, {car_age:.0f} yr old)"
            f" — in this dataset, newer cars are likely priced higher"
        )
    elif car_age > med_age * 1.25:
        bullets.append(
            f"**Older car** ({year}, {car_age:.0f} yr old)"
            f" — in this dataset, older registration is likely associated with lower prices"
        )
    else:
        bullets.append(
            f"**Average age** ({car_age:.0f} years)"
            f" — similar to the dataset average of {med_age:.1f} years"
        )

    if power_ps > med_power * 1.35:
        bullets.append(
            f"**High horsepower** ({power_ps:.0f} PS vs avg {med_power:.0f} PS)"
            f" — in this dataset, higher PS is likely associated with higher prices"
        )
    elif power_ps < med_power * 0.70:
        bullets.append(
            f"**Lower horsepower** ({power_ps:.0f} PS vs avg {med_power:.0f} PS)"
            f" — typically seen in more affordable segments in this dataset"
        )

    brand_df = df[df["brand"] == brand]
    if len(brand_df) >= 10:
        brand_med   = brand_df["price_in_euro"].median()
        brand_label = brand.replace("-", " ").title()
        if brand_med > overall_med * 1.2:
            bullets.append(
                f"**{brand_label}** median price €{brand_med:,.0f} in this dataset"
                f" — above the overall median (€{overall_med:,.0f})"
            )
        elif brand_med < overall_med * 0.85:
            bullets.append(
                f"**{brand_label}** median price €{brand_med:,.0f} in this dataset"
                f" — below the overall median (€{overall_med:,.0f})"
            )

    if "electric" in fuel_type.lower():
        bullets.append(
            "**Electric** fuel type — EVs often carry higher prices for newer model years"
            " in this dataset"
        )
    elif "hybrid" in fuel_type.lower():
        bullets.append(
            "**Hybrid** fuel type — hybrids tend to be in the mid-to-high price range"
            " in this dataset"
        )

    return bullets[:5]


def find_similar_cars(brand, model_name, year, power_ps, mileage_in_km, df, n=8):
    if df is None:
        return None
    car_age = float(REFERENCE_YEAR - year)

    candidate = df[df["model"] == model_name].copy()
    if len(candidate) < 5:
        candidate = df[df["brand"] == brand].copy()
    if len(candidate) < 5:
        candidate = df.copy()

    std_mileage = max(df["mileage_in_km"].std(), 1.0)
    std_age     = max((REFERENCE_YEAR - df["year"]).std(), 1.0)
    std_power   = max(df["power_ps"].std(), 1.0)

    candidate = candidate.copy()
    candidate["_age"]   = REFERENCE_YEAR - candidate["year"]
    candidate["_score"] = (
        (candidate["mileage_in_km"] - mileage_in_km).abs() / std_mileage
        + (candidate["_age"] - car_age).abs() / std_age
        + (candidate["power_ps"] - power_ps).abs() / std_power
    )

    result = candidate.nsmallest(n, "_score").copy()
    result = result.drop(columns=["_age", "_score"])
    result = result.rename(columns={
        "brand":             "Brand",
        "model":             "Model",
        "year":              "Year",
        "mileage_in_km":     "Mileage (km)",
        "fuel_type":         "Fuel",
        "transmission_type": "Transmission",
        "power_ps":          "HP (PS)",
        "price_in_euro":     "Price (€)",
    })
    result["Year"]         = result["Year"].astype(int)
    result["Mileage (km)"] = result["Mileage (km)"].astype(int)
    result["HP (PS)"]      = result["HP (PS)"].astype(int)
    result["Price (€)"]    = result["Price (€)"].apply(lambda v: f"€{v:,.0f}")
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — inputs
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Valuation Input")
    st.caption("General-market used cars · Germany 2023")

    st.markdown('<div class="sidebar-group first">Car Identity</div>',
                unsafe_allow_html=True)

    brand = st.selectbox(
        "Brand",
        options=brand_options,
        index=brand_options.index("volkswagen") if "volkswagen" in brand_options else 0,
        format_func=lambda x: x.replace("-", " ").title(),
    )

    available_models = brand_models.get(brand, [])
    if available_models:
        model = st.selectbox("Model", options=available_models)
    else:
        model = st.text_input("Model", value=f"{brand.title()} Model")

    fuel_type = st.selectbox(
        "Fuel type",
        options=["Petrol", "Diesel", "Hybrid", "Electric",
                 "LPG", "CNG", "Diesel Hybrid", "Ethanol", "Hydrogen", "Other"],
    )
    transmission_type = st.selectbox(
        "Transmission",
        options=["Manual", "Automatic", "Semi-automatic"],
    )

    st.markdown('<div class="sidebar-group">Year & Performance</div>',
                unsafe_allow_html=True)

    year     = st.slider("Registration year", min_value=2000, max_value=2023,
                         value=2015, step=1)
    power_ps = st.number_input("Horsepower (PS)", min_value=41, max_value=799,
                                value=150, step=5)

    st.markdown('<div class="sidebar-group">Usage</div>', unsafe_allow_html=True)

    mileage_in_km = st.number_input(
        "Mileage (km)", min_value=1, max_value=500_000, value=80_000, step=1_000,
    )

    _age = REFERENCE_YEAR - year
    _mpy = mileage_in_km // max(_age, 1)
    st.caption(f"car age: {_age} yr  ·  ≈ {_mpy:,} km/yr")

    st.write("")
    predict_btn = st.button("Estimate Value", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════
# MAIN — hero header
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI Valuation &nbsp;·&nbsp; Germany Used Cars 2023</div>
  <div class="hero-title">
    Used Car <span class="hero-accent">Value Intelligence</span>
  </div>
  <p class="hero-sub">
    Enter your car's details in the sidebar to receive an instant price
    estimate and market segment classification — powered by XGBoost models
    trained on 240,000 real AutoScout24 listings.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="scope-notice">
  <strong>Scope:</strong> Designed for general-market used cars (year 2000–2023,
  €500–€80,000, 40–800 PS, mileage ≤ 500,000 km). Estimates may be less reliable
  for pre-2000 vehicles, collector cars, or ultra-luxury models.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════
if predict_btn:
    input_df = build_input_row(
        brand, model, fuel_type, transmission_type,
        year, power_ps, mileage_in_km,
    )

    car_age          = REFERENCE_YEAR - year
    mileage_per_year = mileage_in_km / max(car_age, 1)

    seg_pred  = int(cls_pipe.predict(input_df)[0])
    seg_probs = cls_pipe.predict_proba(input_df)[0]
    price     = float(reg_pipe.predict(input_df)[0])

    seg_label  = "HIGH" if seg_pred == 1 else "LOW"
    seg_arrow  = "▲" if seg_pred == 1 else "▼"
    seg_desc   = (f"above €{threshold:,.0f} threshold"
                  if seg_pred == 1 else f"below €{threshold:,.0f} threshold")
    confidence = seg_probs[seg_pred] * 100
    prob_high  = seg_probs[1] * 100
    prob_low   = seg_probs[0] * 100

    # ── Three metric cards ────────────────────────────────────────────
    st.markdown(f"""
<div class="cards-row">
  <div class="metric-card primary">
    <div class="card-label">Estimated Market Price</div>
    <div class="card-value gold">€{price:,.0f}</div>
    <div class="card-sub">XGBoost regression · test R² 0.905</div>
  </div>
  <div class="metric-card">
    <div class="card-label">Price Segment</div>
    <div class="card-value">{seg_arrow} {seg_label}</div>
    <div class="card-sub">{seg_desc}</div>
  </div>
  <div class="metric-card">
    <div class="card-label">Segment Confidence</div>
    <div class="card-value">{confidence:.1f}<span style="font-size:1rem;color:#a09880">%</span></div>
    <div class="card-sub">XGBoost classifier · test F1 0.934</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Probability bars ──────────────────────────────────────────────
    st.markdown(f"""
<div class="conf-wrap">
  <div class="conf-row">
    <span class="conf-lbl">LOW</span>
    <div class="conf-track">
      <div class="conf-fill" style="width:{prob_low:.1f}%"></div>
    </div>
    <span class="conf-pct">{prob_low:.1f}%</span>
  </div>
  <div class="conf-row">
    <span class="conf-lbl">HIGH</span>
    <div class="conf-track">
      <div class="conf-fill" style="width:{prob_high:.1f}%"></div>
    </div>
    <span class="conf-pct">{prob_high:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Out-of-range warning ──────────────────────────────────────────
    if price < 500 or price > 80_000:
        st.markdown(
            f'<div class="warn-card">⚠ Predicted price €{price:,.0f} is outside the '
            f'general-market training range (€500–€80,000). This car may be outside '
            f"the model's intended scope.</div>",
            unsafe_allow_html=True,
        )

    # ── Feature detail expander ───────────────────────────────────────
    with st.expander("Feature values passed to the pipeline", expanded=False):
        st.caption(
            f"Raw year is never passed to the pipeline.  "
            f"car_age = {REFERENCE_YEAR} − {year} = **{car_age}**  ·  "
            f"mileage_per_year = {mileage_in_km:,} ÷ {max(car_age, 1)}"
            f" = **{mileage_per_year:,.0f} km/yr**"
        )
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    # ── Why this price? ───────────────────────────────────────────────
    with st.expander("Why this price?", expanded=False):
        st.caption(
            "Rule-based comparison of this car against the dataset sample. "
            "All statements say *in this dataset* or *likely* and are not guarantees."
        )
        bullets = explain_prediction(
            brand, model, fuel_type, year, power_ps, mileage_in_km, price, df_sample
        )
        for b in bullets:
            st.markdown(f"- {b}")

    # ── Similar cars ──────────────────────────────────────────────────
    with st.expander("Similar cars in the dataset", expanded=False):
        st.caption(
            "Matched first by model, then by brand, then sorted by numeric closeness "
            "(mileage · age · horsepower). Prices are actual listing prices from the dataset."
        )
        similar = find_similar_cars(
            brand, model, year, power_ps, mileage_in_km, df_sample, n=8
        )
        if similar is not None and len(similar) > 0:
            st.dataframe(similar, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Sample dataset not available. "
                "Place cars_sample.csv in the app/ folder to enable this feature."
            )

else:
    # ── Idle state ────────────────────────────────────────────────────
    st.markdown("""
<div class="idle-box">
  <div class="idle-icon">🚗</div>
  <div class="idle-text">
    Configure the car in the sidebar and click
    <span class="idle-cta">Estimate Value</span> to see results.
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# WHAT IF COMPARISON
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">What If Comparison</div>',
    unsafe_allow_html=True,
)

with st.expander("What if I compare two cars?", expanded=False):
    st.caption(
        "Configure two cars and click **Compare** to see their predicted prices side by side."
    )
    col_a, col_mid, col_b = st.columns([5, 0.4, 5])

    with col_a:
        st.markdown("**Car A**")
        wa_brand = st.selectbox(
            "Brand", brand_options, key="wa_brand",
            index=brand_options.index("volkswagen") if "volkswagen" in brand_options else 0,
            format_func=lambda x: x.replace("-", " ").title(),
        )
        wa_avail = brand_models.get(wa_brand, [wa_brand.title() + " model"])
        wa_model = st.selectbox("Model", wa_avail, key="wa_model")
        wa_fuel  = st.selectbox(
            "Fuel type",
            ["Petrol","Diesel","Hybrid","Electric","LPG","CNG",
             "Diesel Hybrid","Ethanol","Hydrogen","Other"],
            key="wa_fuel",
        )
        wa_trans = st.selectbox(
            "Transmission", ["Manual","Automatic","Semi-automatic"], key="wa_trans"
        )
        wa_year  = st.slider("Registration year", 2000, 2023, 2015, 1, key="wa_year")
        wa_power = st.number_input("Horsepower (PS)", 41, 799, 150, 5, key="wa_power")
        wa_mile  = st.number_input("Mileage (km)", 1, 500_000, 80_000, 1_000, key="wa_mile")

    with col_mid:
        st.markdown(
            "<div style='text-align:center;color:#b8960c;"
            "margin-top:5rem;font-size:1.4rem;font-weight:700'>vs</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("**Car B**")
        wb_brand = st.selectbox(
            "Brand", brand_options, key="wb_brand",
            index=brand_options.index("bmw") if "bmw" in brand_options else 0,
            format_func=lambda x: x.replace("-", " ").title(),
        )
        wb_avail = brand_models.get(wb_brand, [wb_brand.title() + " model"])
        wb_model = st.selectbox("Model", wb_avail, key="wb_model")
        wb_fuel  = st.selectbox(
            "Fuel type",
            ["Petrol","Diesel","Hybrid","Electric","LPG","CNG",
             "Diesel Hybrid","Ethanol","Hydrogen","Other"],
            key="wb_fuel",
        )
        wb_trans = st.selectbox(
            "Transmission", ["Manual","Automatic","Semi-automatic"], key="wb_trans"
        )
        wb_year  = st.slider("Registration year", 2000, 2023, 2018, 1, key="wb_year")
        wb_power = st.number_input("Horsepower (PS)", 41, 799, 200, 5, key="wb_power")
        wb_mile  = st.number_input("Mileage (km)", 1, 500_000, 40_000, 1_000, key="wb_mile")

    compare_btn = st.button(
        "Compare Cars", use_container_width=True, type="primary", key="compare_btn"
    )

    if compare_btn:
        in_a = build_input_row(wa_brand, wa_model, wa_fuel, wa_trans, wa_year, wa_power, wa_mile)
        in_b = build_input_row(wb_brand, wb_model, wb_fuel, wb_trans, wb_year, wb_power, wb_mile)

        pa  = float(reg_pipe.predict(in_a)[0])
        pb  = float(reg_pipe.predict(in_b)[0])
        sa  = "HIGH" if int(cls_pipe.predict(in_a)[0]) == 1 else "LOW"
        sb  = "HIGH" if int(cls_pipe.predict(in_b)[0]) == 1 else "LOW"
        ca  = cls_pipe.predict_proba(in_a)[0].max() * 100
        cb  = cls_pipe.predict_proba(in_b)[0].max() * 100
        diff     = pb - pa
        pct_diff = (diff / pa * 100) if pa > 0 else 0.0

        comp_df = pd.DataFrame({
            "": [
                "Brand", "Model", "Year", "Mileage", "Fuel",
                "Transmission", "Horsepower", "Estimated Price",
                "Price Segment", "Segment Confidence",
            ],
            "Car A": [
                wa_brand.replace("-", " ").title(), wa_model, str(wa_year),
                f"{wa_mile:,} km", wa_fuel, wa_trans, f"{wa_power} PS",
                f"€{pa:,.0f}", sa, f"{ca:.1f}%",
            ],
            "Car B": [
                wb_brand.replace("-", " ").title(), wb_model, str(wb_year),
                f"{wb_mile:,} km", wb_fuel, wb_trans, f"{wb_power} PS",
                f"€{pb:,.0f}", sb, f"{cb:.1f}%",
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if abs(diff) < 1:
            st.info("Both cars have the same estimated price.")
        elif diff > 0:
            st.success(
                f"Car B is estimated **€{abs(diff):,.0f} more expensive** than Car A"
                f" ({abs(pct_diff):.1f}% higher)"
            )
        else:
            st.success(
                f"Car A is estimated **€{abs(diff):,.0f} more expensive** than Car B"
                f" ({abs(pct_diff):.1f}% higher)"
            )


# ══════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">'
    'Model Performance — Test Set · 48,004 held-out cars'
    '</div>',
    unsafe_allow_html=True,
)

tab_cls, tab_reg, tab_about = st.tabs(["Classification", "Regression", "About"])

with tab_cls:
    st.caption(
        "Target: **price_segment** (HIGH ≥ €19,486 / LOW < €19,486) — "
        "threshold from training median only.  \n"
        "**F1_macro** is the primary metric. Dummy F1_binary=0.667 is misleading "
        "(always predicts HIGH); F1_macro=0.333 is the honest baseline."
    )
    display_cls = cls_lb[
        ["Model", "Accuracy", "F1_binary", "F1_macro",
         "ROC-AUC", "Train_Acc", "Train_F1_macro"]
    ].copy()
    display_cls.columns = [
        "Model", "Accuracy", "F1 (binary)", "F1 (macro)",
        "ROC-AUC", "Train Acc", "Train F1 (macro)",
    ]
    st.dataframe(display_cls, use_container_width=True, hide_index=True)

    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
| Metric | What it means | Good value |
|---|---|---|
| **Accuracy** | Out of all predictions, how many were correct? e.g. 0.93 = 93% correct | Close to 1 |
| **F1 (binary)** | Balances precision and recall for one class only. Can look good even if the model always guesses the same class — not fully reliable on its own | Close to 1 |
| **F1 (macro)** | Average F1 across both classes (HIGH and LOW). The honest metric here — a model that always guesses HIGH gets only 0.333, not 0.667 | Close to 1 |
| **ROC-AUC** | How well the model separates HIGH from LOW cars. 0.5 = guessing randomly, 1.0 = perfect separation | Close to 1 |
| **Train Acc / Train F1** | Same metrics but measured on the training data. If much higher than the test values, the model memorised instead of learning (overfitting) | Similar to test value |

**Our result:** XGBoost achieves F1_macro **0.934** and AUC **0.982** on 48,004 cars it never saw during training.
""")

    col_a, col_b = st.columns(2)
    with col_a:
        p = FIG_DIR / "roc_curve.png"
        if p.exists():
            st.image(str(p), caption="ROC Curve — all classifiers (test set)")
    with col_b:
        p = FIG_DIR / "feature_importance_cls.png"
        if p.exists():
            st.image(str(p), caption="Top 20 features — XGBoost Classifier")

    p = FIG_DIR / "confusion_matrix.png"
    if p.exists():
        c1, c2, c3 = st.columns([1, 1.4, 1])
        with c2:
            st.image(str(p), caption="Confusion Matrix — XGBoost (test set)")

with tab_reg:
    st.caption(
        "Target: **price_in_euro** (raw EUR).  \n"
        "Ridge R²=0.841 with near-zero train-test gap confirms strong additive structure. "
        "XGBoost adds +0.064 R² from non-linear brand × age × mileage interactions."
    )
    display_reg = reg_lb[
        ["Model", "R2", "MAE", "RMSE", "Train_R2", "Train_RMSE"]
    ].copy()
    display_reg.columns = [
        "Model", "R²", "MAE (€)", "RMSE (€)", "Train R²", "Train RMSE (€)",
    ]
    st.dataframe(display_reg, use_container_width=True, hide_index=True)

    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
| Metric | What it means | Good value |
|---|---|---|
| **R²** | How much of the price variation the model explains. 0 = no better than guessing the average, 1 = perfect | Close to 1 |
| **MAE (€)** | Mean Absolute Error — the average prediction error in euros. e.g. MAE €2,602 means the model is off by €2,602 on average | As low as possible |
| **RMSE (€)** | Root Mean Squared Error — similar to MAE but punishes large mistakes more heavily. Always ≥ MAE | As low as possible |
| **Train R² / Train RMSE** | Same metrics on training data. If Train R² is much higher than Test R², the model memorised instead of learning (overfitting) | Similar to test value |

**Our result:** XGBoost achieves R² **0.905** and MAE **€2,602** on 48,004 cars it never saw during training.
The Dummy baseline (always predict average price) has R² **0.000** and MAE **€10,915** — our model is roughly 4× more accurate.
""")

    col_a, col_b = st.columns(2)
    with col_a:
        p = FIG_DIR / "regression_predicted_vs_actual.png"
        if p.exists():
            st.image(str(p), caption="Predicted vs Actual — XGBoost (5k sample)")
    with col_b:
        p = FIG_DIR / "feature_importance_reg.png"
        if p.exists():
            st.image(str(p), caption="Top 20 features — XGBoost Regressor")

with tab_about:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Dataset**
Germany Used Cars 2023 — AutoScout24 scrape
~251k raw rows → 240k after general-market filtering

**Scope filter**
- Year ≥ 2000
- Price €500–€80,000
- Power 40–800 PS
- Mileage 1–500,000 km

**Price segment threshold**
€{threshold:,.0f} — derived from training-set median only (leakage-safe)

**Feature engineering**
- `car_age = {REFERENCE_YEAR} − year`
- `mileage_per_year = mileage_in_km ÷ max(car_age, 1)`
""")
    with c2:
        st.markdown(f"""
**Pipeline architecture**
`ColumnTransformer` (OHE + StandardScaler)
→ `VarianceThreshold(0.0)` → `XGBoost`

**High-cardinality handling**
`model` column (1,191 unique values):
`OneHotEncoder(min_frequency=50)` collapses rare
models into an infrequent bucket. `brand` retained
separately so rare Ferrari ≠ rare Opel.

**Output features after OHE**
559 (46 brand + 496 model + 13 fuel/trans + 4 numeric)

**Leakage proof**
Encoders, scaler, feature selector, and threshold all
fitted on training data only. Test set used exclusively
for final reported metrics.
""")
