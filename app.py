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
    city_col = "city_" + city
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
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes resultPop {
    0%   { opacity: 0; transform: scale(0.93) translateY(12px); }
    60%  { transform: scale(1.02) translateY(-2px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}
.page-enter    { animation: fadeSlideIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) both; }
.compare-enter { animation: fadeSlideIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
.main-title {
    font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;
    color: #FFFFFF; animation: fadeSlideIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.badge-btn > button {
    display: inline-block !important; padding: 0.35rem 0.8rem !important;
    border-radius: 999px !important; background: #e0f2fe !important;
    color: #0369a1 !important; font-size: 0.85rem !important; font-weight: 600 !important;
    border: none !important; box-shadow: none !important; width: auto !important;
    height: auto !important; margin-bottom: 0.8rem; margin-top: 10px; cursor: pointer;
    transition: background 0.2s ease, transform 0.15s ease;
}
.badge-btn > button:hover { background: #bae6fd !important; color: #0369a1 !important; transform: scale(1.04); }
.result-card { border-radius: 20px; padding: 1.3rem; color: white; margin-top: 1rem; animation: resultPop 0.45s cubic-bezier(0.22, 1, 0.36, 1) both; }
.result-card.sunny  { background: linear-gradient(135deg, #f59e0b, #fbbf24); }
.result-card.cloudy { background: linear-gradient(135deg, #64748b, #94a3b8); }
.result-card.rainy  { background: linear-gradient(135deg, #0f766e, #14b8a6); }
.metric-box {
    background: #f8fafc; border-radius: 14px; padding: 0.85rem 1rem;
    border: 1px solid #e2e8f0; text-align: center; margin: 5px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-box:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.07); }
.metric-label { color: #64748b; font-size: 0.85rem; margin-bottom: 0.3rem; }
.metric-value { color: #0f172a; font-size: 1.1rem; font-weight: 700; }
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div { border-radius: 12px !important; }
.stButton > button {
    width: 100%; border-radius: 14px; height: 3rem; font-size: 1rem; font-weight: 700;
    border: none; background: linear-gradient(135deg, #111827, #374151); color: white;
    transition: transform 0.15s ease, opacity 0.15s ease;
}
.stButton > button:hover  { color: white; border: none; transform: translateY(-1px); opacity: 0.92; }
.stButton > button:active { transform: scale(0.97); }
.compare-btn > button {
    background: white !important; color: #0369a1 !important;
    border: 1.5px solid #0369a1 !important; border-radius: 999px !important;
    height: 2.2rem !important; font-size: 0.85rem !important; font-weight: 600 !important;
    padding: 0 1rem !important; width: auto !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
}
.compare-btn > button:hover  { background: #e0f2fe !important; color: #0369a1 !important; transform: scale(1.04) !important; }
.compare-btn > button:active { transform: scale(0.97) !important; }
[data-testid="stProgress"] > div > div { transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1) !important; }
.tbl-wrap {
    width: 100%; border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.10); box-sizing: border-box;
    animation: fadeSlideIn 0.4s 0.1s cubic-bezier(0.22,1,0.36,1) both;
    margin-top: 0.7rem;
}
.tbl-title {
    background: rgba(255,255,255,0.08); padding: 0.55rem 0.9rem;
    font-size: 11.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: rgba(255,255,255,0.55);
}
.perf-tbl { width: 100%; border-collapse: collapse; font-size: 11px; table-layout: fixed; }
.perf-tbl thead tr { background: rgba(255,255,255,0.05); }
.perf-tbl th {
    padding: 0.45rem 0.4rem; text-align: center; font-size: 10px;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    color: rgba(255,255,255,0.45); border-bottom: 1px solid rgba(255,255,255,0.08);
    overflow: hidden; word-break: break-word;
}
.perf-tbl th:first-child { text-align: left; }
.perf-tbl td {
    padding: 0.5rem 0.4rem; text-align: center;
    font-size: 11.5px; color: rgba(255,255,255,0.80);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    overflow: hidden; word-break: break-word;
}
.perf-tbl td:first-child { text-align: left; }
.perf-tbl tbody tr:last-child td { border-bottom: none; }
.perf-tbl tbody tr:hover td { background: rgba(255,255,255,0.03); }
.winner-row td { color: #ffffff !important; font-weight: 600; background: rgba(255,255,255,0.04); }
.winner-badge {
    display: inline-block; margin-left: 4px; padding: 1px 6px;
    border-radius: 999px; font-size: 9px; font-weight: 700;
    background: rgba(250,204,21,0.20); color: #facc15; vertical-align: middle;
}
.model-name { font-weight: 600; color: #ffffff !important; }
.rf-dot::before  { content: "● "; color: #f59e0b; font-size: 10px; }
.xgb-dot::before { content: "● "; color: #38bdf8; font-size: 10px; }
.ada-dot::before { content: "● "; color: #fb923c; font-size: 10px; }
.phase-divider-row td {
    background: rgba(255,255,255,0.03) !important;
    padding: 0.28rem 0.6rem !important;
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
    color: rgba(255,255,255,0.35) !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    text-align: left !important;
}
.fold-name { font-weight: 600; color: rgba(255,255,255,0.70) !important; font-size: 12px; }
.rf-col  { color: #f59e0b !important; font-weight: 500; }
.xgb-col { color: #38bdf8 !important; font-weight: 500; }
.ada-col { color: #fb923c !important; font-weight: 500; }
.best-marker { font-size: 8px; vertical-align: super; opacity: 0.85; }
.avg-row td {
    font-weight: 700 !important; background: rgba(255,255,255,0.06) !important;
    border-top: 1px solid rgba(255,255,255,0.12) !important; color: #ffffff !important;
}
.fold-explanation {
    margin-top: 1.6rem; padding: 1.1rem 1.4rem; border-radius: 16px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    display: flex; align-items: flex-start; gap: 0.8rem;
    animation: fadeSlideIn 0.5s 0.2s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.fold-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.fold-text { font-size: 13.5px; line-height: 1.65; color: rgba(255,255,255,0.75); }
.fold-text strong { color: #ffffff; font-weight: 700; }
.section-label {
    font-size: 11.5px; font-weight: 700; color: rgba(255,255,255,0.45);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.4rem; margin-top: 0.2rem;
}
.phase-sep {
    border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)

if "show_compare" not in st.session_state:
    st.session_state.show_compare = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

ada_cv = [0.69848577, 0.70001027, 0.69886609, 0.69854152, 0.69929924]
xgb_cv = [0.71444604, 0.71415083, 0.71471149, 0.71460569, 0.71456021]
rf_cv  = [0.68861236, 0.68816033, 0.68890710, 0.68939881, 0.68755748]

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

    g1, g2 = st.columns(2, gap="large")

    with g1:
        st.markdown('<div class="section-label">Phase 1 — Random Forest vs XGBoost</div>', unsafe_allow_html=True)
        st.image("phase1.png", use_container_width=True)

        perf_html = (
            '<div class="tbl-wrap">'
            '<div class="tbl-title">📋 Performance Comparison — All Phases</div>'
            '<table class="perf-tbl">'
            '<thead>'
            '<tr>'
            '<th>Model</th>'
            '<th>Accuracy</th>'
            '<th>Precision</th>'
            '<th>Recall</th>'
            '<th>F1</th>'
            '<th>ROC-AUC</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            '<tr class="phase-divider-row"><td colspan="6">Phase 1 — Random Forest vs XGBoost</td></tr>'
            '<tr>'
            '<td class="model-name rf-dot">Random Forest</td>'
            '<td>0.6873</td><td>0.74</td><td>0.03</td><td>0.06</td><td>0.6609</td>'
            '</tr>'
            '<tr class="winner-row">'
            '<td class="model-name xgb-dot">XGBoost <span class="winner-badge">Best</span></td>'
            '<td>0.7143</td><td>0.67</td><td>0.21</td><td>0.32</td><td>0.6985</td>'
            '</tr>'
            '<tr class="phase-divider-row"><td colspan="6">Phase 2 — XGBoost vs AdaBoost</td></tr>'
            '<tr class="winner-row">'
            '<td class="model-name xgb-dot">XGBoost <span class="winner-badge">Best</span></td>'
            '<td>0.7143</td><td>0.67</td><td>0.21</td><td>0.32</td><td>0.6985</td>'
            '</tr>'
            '<tr>'
            '<td class="model-name ada-dot">AdaBoost</td>'
            '<td>0.6990</td><td>0.56</td><td>0.08</td><td>0.14</td><td>0.5268</td>'
            '</tr>'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown('<div class="section-label" style="margin-top:1rem;">📊 5-Fold Cross-Validation Accuracy</div>', unsafe_allow_html=True)
        st.image("phase1CV.png", use_container_width=True)
        st.image("phase2CV.png", use_container_width=True)
        st.markdown(perf_html, unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-label">Phase 2 — XGBoost vs AdaBoost</div>', unsafe_allow_html=True)
        st.image("phase2.png", use_container_width=True)

        folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
        fold_rows_html = ""
        for i, f in enumerate(folds):
            best_p1 = max(rf_cv[i], xgb_cv[i])
            best_p2 = max(xgb_cv[i], ada_cv[i])
            rf_star  = '<span class="best-marker">▲</span>' if abs(rf_cv[i]  - best_p1) < 1e-9 else ""
            x1_star  = '<span class="best-marker">▲</span>' if abs(xgb_cv[i] - best_p1) < 1e-9 else ""
            x2_star  = '<span class="best-marker">▲</span>' if abs(xgb_cv[i] - best_p2) < 1e-9 else ""
            ada_star = '<span class="best-marker">▲</span>' if abs(ada_cv[i] - best_p2) < 1e-9 else ""
            fold_rows_html += (
                "<tr>"
                + '<td class="fold-name">' + f + "</td>"
                + '<td class="rf-col">'  + "{:.6f}".format(rf_cv[i])  + rf_star  + "</td>"
                + '<td class="xgb-col">' + "{:.6f}".format(xgb_cv[i]) + x1_star  + "</td>"
                + '<td class="xgb-col">' + "{:.6f}".format(xgb_cv[i]) + x2_star  + "</td>"
                + '<td class="ada-col">' + "{:.6f}".format(ada_cv[i]) + ada_star + "</td>"
                + "</tr>"
            )

        avg_rf  = sum(rf_cv)  / len(rf_cv)
        avg_xgb = sum(xgb_cv) / len(xgb_cv)
        avg_ada = sum(ada_cv)  / len(ada_cv)

        cv_html = (
            '<div class="tbl-wrap">'
            '<div class="tbl-title">🔁 5-Fold CV Accuracy — All Phases</div>'
            '<table class="perf-tbl">'
            '<thead>'
            '<tr>'
            '<th rowspan="2" style="width:18%; vertical-align:bottom;">Fold</th>'
            '<th colspan="2" style="border-left:1px solid rgba(255,255,255,0.10);">Phase 1</th>'
            '<th colspan="2" style="border-left:1px solid rgba(255,255,255,0.10);">Phase 2</th>'
            '</tr>'
            '<tr>'
            '<th style="border-left:1px solid rgba(255,255,255,0.10); border-top:1px solid rgba(255,255,255,0.08);">RF</th>'
            '<th style="border-top:1px solid rgba(255,255,255,0.08);">XGBoost</th>'
            '<th style="border-left:1px solid rgba(255,255,255,0.10); border-top:1px solid rgba(255,255,255,0.08);">XGBoost</th>'
            '<th style="border-top:1px solid rgba(255,255,255,0.08);">AdaBoost</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            + fold_rows_html
            + '<tr class="avg-row">'
            + '<td class="fold-name">Average</td>'
            + '<td class="rf-col">'  + "{:.6f}".format(avg_rf)  + "</td>"
            + '<td class="xgb-col">' + "{:.6f}".format(avg_xgb) + "</td>"
            + '<td class="xgb-col">' + "{:.6f}".format(avg_xgb) + "</td>"
            + '<td class="ada-col">' + "{:.6f}".format(avg_ada) + "</td>"
            + "</tr>"
            + "</tbody>"
            + "</table>"
            + "</div>"
        )
        st.markdown(cv_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="fold-explanation">'
        '<div class="fold-icon">🔁</div>'
        '<div class="fold-text">'
        '<strong>5-Fold Cross-Validation</strong> checks model reliability by training and testing '
        'the model on five different data splits and averaging the results to ensure '
        'consistent prediction performance.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="padding: 1.5rem 0 0.5rem;">'
        '<p style="font-size: 13px; color: #94a3b8; text-align: center; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 1rem; font-weight: 500;">Model Evaluation Metrics</p>'
        '<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">'
        '<div style="background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 14px; padding: 1rem 0.75rem; text-align: center;">'
        '<div style="font-size: 22px; margin-bottom: 6px;">🎯</div>'
        '<div style="font-size: 13px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">Accuracy</div>'
        '<div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">Percentage of total predictions classified correctly</div>'
        '</div>'
        '<div style="background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 14px; padding: 1rem 0.75rem; text-align: center;">'
        '<div style="font-size: 22px; margin-bottom: 6px;">✅</div>'
        '<div style="font-size: 13px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">Precision</div>'
        '<div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">How many predicted rain cases are actually correct</div>'
        '</div>'
        '<div style="background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 14px; padding: 1rem 0.75rem; text-align: center;">'
        '<div style="font-size: 22px; margin-bottom: 6px;">🔍</div>'
        '<div style="font-size: 13px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">Recall</div>'
        '<div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">How many actual rain events are correctly detected</div>'
        '</div>'
        '<div style="background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 14px; padding: 1rem 0.75rem; text-align: center;">'
        '<div style="font-size: 22px; margin-bottom: 6px;">⚖️</div>'
        '<div style="font-size: 13px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">F1-score</div>'
        '<div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">Balances precision and recall into one value</div>'
        '</div>'
        '<div style="background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 14px; padding: 1rem 0.75rem; text-align: center;">'
        '<div style="font-size: 22px; margin-bottom: 6px;">📈</div>'
        '<div style="font-size: 13px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">ROC-AUC</div>'
        '<div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">Distinguishes rain vs no rain across all thresholds</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

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
            st.markdown('<div class="metric-box"><div class="metric-label">State</div><div class="metric-value">' + selected_state + '</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-box"><div class="metric-label">Temperature</div><div class="metric-value">' + "{:.1f}".format(temperature) + ' \u00b0C</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-box"><div class="metric-label">City</div><div class="metric-value">' + selected_city + '</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-box"><div class="metric-label">Pressure</div><div class="metric-value">' + "{:.1f}".format(pressure) + ' hPa</div></div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="metric-box"><div class="metric-label">Humidity</div><div class="metric-value">' + str(humidity) + '%</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-box"><div class="metric-label">Wind Speed</div><div class="metric-value">' + str(wind_speed) + ' km/h</div></div>', unsafe_allow_html=True)

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
            card_html = (
                '<div class="result-card ' + info["class_name"] + '">'
                + '<h2 style="margin-bottom:0.4rem;">' + info["label"] + '</h2>'
                + '<div style="font-size:2rem; font-weight:800; margin-bottom:0.2rem;">' + str(info["pct"]) + '%</div>'
                + '<div style="font-size:1rem; opacity:0.95;">' + info["desc"] + '</div>'
                + '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
            st.progress(min(max(res["probability"], 0.0), 1.0))
            if res["prediction_class"] is not None:
                st.caption("Predicted class: " + str(res["prediction_class"]) + "  |  1 = Rain, 0 = No Rain")
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
