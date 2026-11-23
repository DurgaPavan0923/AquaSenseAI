"""AquaSense AI - High-Throughput Batch CSV Screening & Data Enrichment."""

import io
import os

import pandas as pd
import requests
import streamlit as st

from src.common.constants import STATUTORY_DISCLAIMER

st.set_page_config(page_title="Batch Upload | AquaSense AI", page_icon="📁", layout="wide")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.warning(f"⚠️ **REGULATORY NOTICE:** {STATUTORY_DISCLAIMER}")
st.title("📁 High-Throughput Batch CSV Ingestion")
st.markdown("Upload environmental laboratory CSVs (up to 10MB) for batch risk screening, conformal prediction sets, and anomaly triage.")

# Uploader
uploaded_file = st.file_uploader(
    "Choose a water quality CSV file",
    type=["csv"],
    help="Must contain columns: pH, Turbidity, Temperature, Dissolved Oxygen, Biochemical Oxygen Demand, Lead, Mercury, Arsenic"
)

if uploaded_file is not None:
    # Size check
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > 10 * 1024 * 1024:
        st.error("File exceeds maximum allowable limit of 10MB.")
        st.stop()

    try:
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
        st.success(f"CSV uploaded successfully: **{len(raw_df)}** sample rows detected.")
        st.dataframe(raw_df.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Could not parse CSV: {str(e)}")
        st.stop()

    if st.button("Execute Batch Screening & Conformal Prediction", type="primary"):
        with st.spinner("Processing batch through XGBoost & Conformal Engine..."):
            # Execute batch via API or local pipeline fallback
            enriched_df = None
            try:
                files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
                res = requests.post(f"{API_BASE_URL}/predict/batch", files=files, timeout=60)
                if res.status_code == 200:
                    api_resp = res.json()
                    st.info(f"Batch completed: Flagged **{api_resp['anomalies_flagged']}** anomalies out of {api_resp['total_processed']} samples.")
            except Exception:
                pass

            # Local fallback execution
            try:
                from src.api.dependencies import ModelServicesContainer
                container = ModelServicesContainer.get_instance()
                if not container.is_ready:
                    container.load_services()

                numerical_cols = [
                    "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
                    "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"
                ]
                missing = [c for c in numerical_cols if c not in raw_df.columns]
                if missing:
                    st.error(f"Missing mandatory columns: {missing}")
                    st.stop()

                X = raw_df[numerical_cols]
                probs = container.model.predict_proba(X)
                preds = probs.argmax(axis=1)
                labels = ["Good", "Moderate", "Polluted"]
                conformal_sets = container.conformal_predictor.predict_sets_batch(probs, alpha=0.10)

                enriched_df = raw_df.copy()
                enriched_df["Predicted_Risk"] = [labels[p] for p in preds]
                enriched_df["Confidence_Score"] = [round(float(probs[i, preds[i]]), 3) for i in range(len(preds))]
                enriched_df["Conformal_Prediction_Set"] = [", ".join(c) for c in conformal_sets]

                # Check anomalies
                anom_flags = []
                for idx in range(len(raw_df)):
                    score_res = container.anomaly_engine.score_sample(raw_df.iloc[idx].to_dict())
                    anom_flags.append(score_res.get("is_anomaly", False))
                enriched_df["Is_Anomaly"] = anom_flags

            except Exception as e:
                st.error(f"Batch processing error: {str(e)}")
                st.stop()

        if enriched_df is not None:
            st.subheader("Enriched Risk Triage Results")
            st.dataframe(enriched_df, use_container_width=True)

            # Download Enriched CSV
            csv_output = enriched_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Enriched Prediction Dataset (CSV)",
                data=csv_output,
                file_name=f"aquasense_screened_{uploaded_file.name}",
                mime="text/csv"
            )
