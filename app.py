import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Rain Predictor",
    page_icon="🌦️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = "xgboost_model.pkl"
CSV_PATH = "state_city_mapping.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_locations():
    df = pd.read_csv(CSV_PATH)
    df["state"] = df["state"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df = df.dropna().drop_duplicates().sort_values(["state", "city"]).reset_index(drop=True)
    return df

model = load_model()
location_df = load_locations()

state_to_cities = (
    location_df.groupby("state")["city"]
    .apply(lambda s: sorted(s.unique().tolist()))
    .to_dict()
)

model_features = list(getattr(model, "feature_names_in_", []))

def build_feature_row(temperature, pressure, humidity, wind_speed, city):
    row = {feature: 0 for feature in model_features}
    for col_name, value in {
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,
        "wind_speed": wind_speed,
    }.items():
        if col_name in row:
            row[col_name] = value
    city_col = f"city_{city}"
    if city_col in row:
        row[city_col] = 1
    return pd.DataFrame([row], columns=model_features)

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return float(proba[1])
    pred = model.predict(X)[0]
    return float(pred)

def result_text(probability):
    pct = round(probability * 100, 1)
    if probability >= 0.70:
        return {"label": "🌧️ High chance of rain", "desc": "Rain is likely. Bringing an umbrella is a good idea.", "class_name": "rainy", "pct": pct}
    elif probability > 0.30:
        return {"label": "⛅ Moderate chance of rain", "desc": "Conditions are mixed. Light rain or showers may happen.", "class_name": "cloudy", "pct": pct}
    return {"label": "☀️ Low chance of rain", "desc": "Weather looks mostly dry based on the current inputs.", "class_name": "sunny", "pct": pct}

st.markdown("""
<style>
@keyframes fadeSlideIn {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeSlideOut {
    from {
        opacity: 1;
        transform: translateY(0);
    }
    to {
        opacity: 0;
        transform: translateY(-14px);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes progressFill {
    from { width: 0%; }
    to   { width: var(--target-width); }
}

@keyframes resultPop {
    0%   { opacity: 0; transform: scale(0.93) translateY(12px); }
    60%  { transform: scale(1.02) translateY(-2px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.page-enter {
    animation: fadeSlideIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.compare-enter {
    animation: fadeSlideIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.compare-enter:nth-child(2) {
    animation-delay: 0.07s;
}

.result-animate {
    animation: resultPop 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: #FFFFFF;
    animation: fadeSlideIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.sub-title {
    color: #6b7280;
    margin-bottom: 1.4rem;
    font-size: 1rem;
}

.badge-btn > button {
    display: inline-block !important;
    padding: 0.35rem 0.8rem !important;
    border-radius: 999px !important;
    background: #e0f2fe !important;
    color: #0369a1 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: none !important;
    width: auto !important;
    height: auto !important;
    margin-bottom: 0.8rem;
    margin-top: 10px;
    cursor: pointer;
    transition: background 0.2s ease, transform 0.15s ease;
}

.badge-btn > button:hover {
    background: #bae6fd !important;
    color: #0369a1 !important;
    transform: scale(1.04);
}

.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    animation: fadeSlideIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.result-card {
    border-radius: 20px;
    padding: 1.3rem;
    color: white;
    margin-top: 1rem;
    animation: resultPop 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.result-card.sunny  { background: linear-gradient(135deg, #f59e0b, #fbbf24); }
.result-card.cloudy { background: linear-gradient(135deg, #64748b, #94a3b8); }
.result-card.rainy  { background: linear-gradient(135deg, #0f766e, #14b8a6); }

.metric-box {
    background: #f8fafc;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    border: 1px solid #e2e8f0;
    text-align: center;
    margin: 5px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
}

.metric-label { color: #64748b; font-size: 0.85rem; margin-bottom: 0.3rem; }
.metric-value { color: #0f172a; font-size: 1.1rem; font-weight: 700; }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div { border-radius: 12px !important; }

.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 3rem;
    font-size: 1rem;
    font-weight: 700;
    border: none;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
    transition: transform 0.15s ease, opacity 0.15s ease;
}

.stButton > button:hover {
    color: white;
    border: none;
    transform: translateY(-1px);
    opacity: 0.92;
}

.stButton > button:active {
    transform: scale(0.97);
}

.small-note { color: #64748b; font-size: 0.9rem; }

.compare-btn > button {
    background: white !important;
    color: #0369a1 !important;
    border: 1.5px solid #0369a1 !important;
    border-radius: 999px !important;
    height: 2.2rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0 1rem !important;
    width: auto !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
}

.compare-btn > button:hover {
    background: #e0f2fe !important;
    color: #0369a1 !important;
    transform: scale(1.04) !important;
}

.compare-btn > button:active {
    transform: scale(0.97) !important;
}

.compare-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.compare-panel-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.3rem;
}

.compare-panel-sub {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 1.2rem;
}

.graph-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
    text-align: center;
}

.animated-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    margin: 1rem 0;
    animation: fadeIn 0.6s ease both;
}

[data-testid="stSelectbox"]:nth-child(1) { animation: fadeSlideIn 0.35s 0.05s both; }
[data-testid="stSelectbox"]:nth-child(2) { animation: fadeSlideIn 0.35s 0.10s both; }
[data-testid="stNumberInput"]            { animation: fadeSlideIn 0.35s 0.15s both; }
[data-testid="stSlider"]                 { animation: fadeSlideIn 0.35s 0.20s both; }

/* Progress bar smooth */
[data-testid="stProgress"] > div > div {
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1) !important;
}
</style>
""", unsafe_allow_html=True)

if "show_compare" not in st.session_state:
    st.session_state.show_compare = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown('<div class="badge-btn">', unsafe_allow_html=True)
    if st.button("Rain Predictor", key="badge_btn"):
        st.session_state.show_compare = False
        st.session_state.prediction_result = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with header_right:
    st.markdown('<div style="margin-top:6px"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="compare-btn">', unsafe_allow_html=True)
        compare_label = "✕ Close comparison" if st.session_state.show_compare else "📊 Compare with other model"
        if st.button(compare_label, key="compare_btn"):
            st.session_state.show_compare = not st.session_state.show_compare
            st.session_state.prediction_result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="main-title">Streamlit Rain Prediction App</div>', unsafe_allow_html=True)

if st.session_state.show_compare:
    st.markdown('<div class="compare-enter">', unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="graph-label compare-enter">Phase 1 — Random Forest vs XGBoost</div>', unsafe_allow_html=True)
        st.image("phase1.png", use_container_width=True)
    with g2:
        st.markdown('<div class="graph-label compare-enter" style="animation-delay:0.08s">Phase 2 — XGBoost vs AdaBoost</div>', unsafe_allow_html=True)
        st.image("phase2.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="page-enter">', unsafe_allow_html=True)
        st.divider()
        st.subheader("Location")
        states = sorted(state_to_cities.keys())
        selected_state = st.selectbox("State", states, index=0)
        cities = state_to_cities.get(selected_state, [])
        selected_city = st.selectbox("City", cities, index=0 if cities else None)

        st.divider()
        st.subheader("Weather Input")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=28.0, step=0.1)
            humidity = st.slider("Humidity (%)", min_value=50, max_value=100, value=75, step=1)
        with col2:
            pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1010.0, step=0.1)
            wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=60, value=10, step=1)

        predict_btn = st.button("Predict Rainfall")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="page-enter" style="animation-delay:0.06s">', unsafe_allow_html=True)
        st.subheader("Input Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">State</div><div class="metric-value">{selected_state}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><div class="metric-label">Temperature</div><div class="metric-value">{temperature:.1f} °C</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">City</div><div class="metric-value">{selected_city}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><div class="metric-label">Pressure</div><div class="metric-value">{pressure:.1f} hPa</div></div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Humidity</div><div class="metric-value">{humidity}%</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Wind Speed</div><div class="metric-value">{wind_speed} km/h</div></div>', unsafe_allow_html=True)

        if predict_btn:
            X = build_feature_row(
                temperature=temperature,
                pressure=pressure,
                humidity=humidity,
                wind_speed=wind_speed,
                city=selected_city
            )
            probability = safe_predict_proba(model, X)
            st.session_state.prediction_result = {
                "probability": probability,
                "prediction_class": int(model.predict(X)[0]) if hasattr(model, "predict") else None
            }

        if st.session_state.prediction_result:
            res = st.session_state.prediction_result
            info = result_text(res["probability"])
            st.markdown(
                f"""
                <div class="result-card {info['class_name']}">
                    <h2 style="margin-bottom:0.4rem;">{info['label']}</h2>
                    <div style="font-size:2rem; font-weight:800; margin-bottom:0.2rem;">{info['pct']}%</div>
                    <div style="font-size:1rem; opacity:0.95;">{info['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(min(max(res["probability"], 0.0), 1.0))
            if res["prediction_class"] is not None:
                st.caption(f"Predicted class: {res['prediction_class']}  |  1 = Rain, 0 = No Rain")
        else:
            st.markdown(
                '<div style="animation: fadeSlideIn 0.4s 0.15s both; display:block;">'
                '<div style="margin-top:1.5rem; padding:1.2rem; border-radius:14px; '
                'border: 1px dashed #cbd5e1; text-align:center; color:#94a3b8; font-size:0.95rem;">'
                'Set the values on the left, then click <strong>Predict Rainfall</strong>.'
                '</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)
