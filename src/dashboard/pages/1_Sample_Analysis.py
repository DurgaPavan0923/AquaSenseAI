"""AquaSense AI - Interactive Sample Analysis & Local SHAP Explainability."""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.common.constants import STATUTORY_DISCLAIMER, WHO_BIS_STANDARDS

st.set_page_config(page_title="Sample Analysis | AquaSense AI", page_icon="🔬", layout="wide")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.warning(f"⚠️ **REGULATORY NOTICE:** {STATUTORY_DISCLAIMER}")
st.title("🔬 Interactive Sample Analysis & Explainability")
st.markdown("Enter physical sensor readings to triage water quality risk, inspect conformal prediction sets, and audit local SHAP drivers.")

# Layout: Form on left, Outputs on right
col_input, col_output = st.columns([1, 1.2])

with col_input:
    st.subheader("1. Physicochemical Telemetry Inputs")
    with st.form("sample_inference_form"):
        station = st.selectbox("Monitoring Station", ["STATION_RIVER_NORTH_01", "STATION_LAKE_SOUTH_02", "STATION_INDUSTRIAL_03"])
        
        c1, c2 = st.columns(2)
        with c1:
            ph = st.slider("pH Level", 0.0, 14.0, 6.8, 0.05, help="WHO/BIS Desirable: 6.5 - 8.5")
            turbidity = st.slider("Turbidity (NTU)", 0.0, 100.0, 4.2, 0.1, help="WHO/BIS Max Limit: 5.0 NTU")
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 22.4, 0.5, help="Ambient baseline: 15 - 30 °C")
            do = st.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 5.8, 0.1, help="Potable minimum: >= 6.5 mg/L")
        with c2:
            bod = st.slider("BOD (mg/L)", 0.0, 30.0, 3.4, 0.1, help="Clean water baseline: <= 2.0 mg/L")
            lead = st.number_input("Lead (mg/L)", 0.0, 1.0, 0.012, 0.001, format="%.4f", help="Statutory Limit: <= 0.01 mg/L")
            mercury = st.number_input("Mercury (mg/L)", 0.0, 0.1, 0.0008, 0.0001, format="%.5f", help="Statutory Limit: <= 0.001 mg/L")
            arsenic = st.number_input("Arsenic (mg/L)", 0.0, 1.0, 0.009, 0.001, format="%.4f", help="Statutory Limit: <= 0.01 mg/L")

        confidence_level = st.select_slider(
            "Conformal Prediction Confidence Level (1 - α)",
            options=[0.80, 0.90, 0.95],
            value=0.90,
            help="Statistical coverage guarantee: ground truth label is guaranteed to be in set with probability >= (1 - α)"
        )

        submitted = st.form_submit_button("Run Risk Screening & Explain", use_container_width=True)

with col_output:
    st.subheader("2. Algorithmic Risk Triage & Uncertainty")
    
    payload = {
        "location_id": station,
        "sensor_id": "WEB_UI_PROBE",
        "source_type": "manual_lab",
        "features": {
            "pH": ph,
            "Turbidity": turbidity,
            "Temperature": temperature,
            "Dissolved Oxygen": do,
            "Biochemical Oxygen Demand": bod,
            "Lead": lead,
            "Mercury": mercury,
            "Arsenic": arsenic
        },
        "confidence_level": confidence_level
    }

    # Execute request
    result = None
    try:
        res = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=6)
        if res.status_code == 200:
            result = res.json()
    except Exception:
        pass

    # Fallback execution via local modules if API server is not running
    if result is None:
        try:
            from src.api.dependencies import ModelServicesContainer
            container = ModelServicesContainer.get_instance()
            if not container.is_ready:
                container.load_services()
            
            df_sample = pd.DataFrame([payload["features"]])
            probs = container.model.predict_proba(df_sample)[0]
            pred_class = int(probs.argmax())
            labels = ["Good", "Moderate", "Polluted"]
            _, c_set = container.conformal_predictor.predict_set(probs, alpha=1.0 - confidence_level)
            expl = container.explainer.explain_local_sample(df_sample)
            anom = container.anomaly_engine.score_sample(payload["features"])

            result = {
                "prediction": {
                    "predicted_label": labels[pred_class],
                    "confidence_score": float(probs[pred_class]),
                    "calibrated_probabilities": {labels[i]: float(probs[i]) for i in range(3)},
                    "conformal_prediction_set": c_set
                },
                "anomaly": anom,
                "explainability": expl,
                "regulatory_audit": {"who_bis_violations": []}
            }
        except Exception as e:
            st.error(f"Inference engine unavailable: {str(e)}")

    if result:
        pred_meta = result["prediction"]
        label = pred_meta["predicted_label"]
        color = "#2ca02c" if label == "Good" else "#ff7f0e" if label == "Moderate" else "#d62728"

        conformal_set = pred_meta.get("conformal_prediction_set", [label])
        is_ambiguous = len(conformal_set) > 1 or result.get("anomaly", {}).get("is_anomaly", False)

        st.markdown(
            f"""
            <div style="background-color: {color}22; border-left: 6px solid {color}; padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                <h3 style="color: {color}; margin: 0;">Predicted Risk: {label.upper()}</h3>
                <p style="margin: 4px 0 0 0; font-size: 14px;">Point Confidence: <b>{pred_meta['confidence_score']*100:.1f}%</b> | Conformal Set ({int(confidence_level*100)}% Coverage): <b>{', '.join(conformal_set)}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if is_ambiguous:
            st.error(
                "🛑 **AMBIGUOUS RISK / DECISION ABSTENTION PROTOCOL TRIGGERED:**\n\n"
                f"The conformal prediction set contains multiple classes: `{', '.join(conformal_set)}`. "
                "Because statistical coverage cannot isolate a single risk tier without exceeding statutory risk tolerances, "
                "automated triage is abstained. **Mandatory certified laboratory analytical assay required.**"
            )

        # Probabilities Bar Chart
        probs_df = pd.DataFrame(list(pred_meta["calibrated_probabilities"].items()), columns=["Class", "Probability"])
        fig_probs = px.bar(
            probs_df, x="Class", y="Probability", color="Class",
            color_discrete_map={"Good": "#2ca02c", "Moderate": "#ff7f0e", "Polluted": "#d62728"},
            text=probs_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
            range_y=[0, 1]
        )
        fig_probs.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_probs, use_container_width=True)

        # Anomaly Status
        anom = result.get("anomaly", {})
        if anom.get("is_anomaly"):
            st.error(f"🚨 **Sensor / Outlier Anomaly Flagged ({anom.get('severity')} Severity)**")
            for v in anom.get("violations", []):
                st.caption(f"• {v}")

        # Local SHAP Waterfall / Bar Chart
        st.subheader("3. Local TreeSHAP Feature Attributions")
        expl = result.get("explainability", {})
        attributions = expl.get("attributions", {}) or expl.get("shap_attributions", {})
        if attributions:
            shap_df = pd.DataFrame(list(attributions.items()), columns=["Feature", "Contribution"]).sort_values(by="Contribution")
            fig_shap = px.bar(
                shap_df, x="Contribution", y="Feature", orientation="h",
                color="Contribution",
                color_continuous_scale="RdBu_r",
                title="Force Impact on Pollution Risk (SHAP Values)"
            )
            fig_shap.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_shap, use_container_width=True)
            st.info(f"Key Driver Increasing Pollution Risk: **{expl.get('dominant_pollutant_driver', 'N/A')}**")
