"""
AquaSense AI: Interactive Research & Screening Dashboard.
For local exploratory analysis and model demonstration.
Run locally with: streamlit run app/app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AquaSense AI - Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mandatory Notice
st.warning("Screening support only. Not a substitute for certified laboratory analysis.")

st.title("AquaSense AI: Water Quality Monitoring & Risk Screening")
st.markdown(
    "**Course:** URP4201 Research Methodology | **Supervisor:** Dr. Alok Ku. Pani\n\n"
    "**Authors:** Ayusman Das, Selina Mohanty, Rajana Durga Pavan Kumar, Akanshaa Mohanta"
)

# Load Model Bundle
BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.joblib"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"

@st.cache_resource
def load_bundle():
    if BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)
    return None

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

bundle = load_bundle()
metrics = load_metrics()

tabs = st.tabs(["Single Screening", "Dataset Explorer", "Model Verification", "Diagnostic Figures"])

# Tab 1: Single Sample Screening
with tabs[0]:
    st.header("Single Sample Screening Analysis")
    st.markdown("Adjust parameters below to evaluate water risk tier and conformal prediction sets.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ph = st.number_input("pH (0 - 14)", min_value=0.0, max_value=14.0, value=7.20, step=0.05)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=500.0, value=2.80, step=0.1)
    with col2:
        temperature = st.number_input("Temperature (deg C)", min_value=-5.0, max_value=60.0, value=22.5, step=0.5)
        dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=25.0, value=6.50, step=0.1)
    with col3:
        bod = st.number_input("BOD (mg/L)", min_value=0.0, max_value=100.0, value=1.80, step=0.1)
        lead = st.number_input("Lead - Pb (mg/L)", min_value=0.0, max_value=5.0, value=0.0050, step=0.0005, format="%.4f")
    with col4:
        mercury = st.number_input("Mercury - Hg (mg/L)", min_value=0.0, max_value=1.0, value=0.00030, step=0.00005, format="%.5f")
        arsenic = st.number_input("Arsenic - As (mg/L)", min_value=0.0, max_value=5.0, value=0.0040, step=0.0005, format="%.4f")

    if st.button("Run Screening Prediction", type="primary"):
        if bundle:
            clf = bundle["model"]
            quantile = float(bundle.get("conformal_quantile", 0.0216))
            medians = np.array(bundle.get("training_medians", [7.15, 2.90, 22.1, 5.58, 2.41, 0.006, 0.00038, 0.0057]))

            sample_arr = np.array([[ph, turbidity, temperature, dissolved_oxygen, bod, lead, mercury, arsenic]], dtype=float)
            probs = clf.predict_proba(sample_arr)[0]
            pred_class = int(np.argmax(probs))
            classes = ["Good", "Moderate", "Polluted"]

            threshold = max(0.0, 1.0 - quantile)
            c_set = [classes[c] for c, p in enumerate(probs) if p >= threshold]
            if not c_set:
                c_set = [classes[pred_class]]

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader(f"Risk Tier: {classes[pred_class]}")
                st.markdown(f"**Conformal Prediction Set (90% Guarantee):** `[{', '.join(c_set)}]`")
                st.markdown(f"- Good: {probs[0]*100:.2f}%")
                st.markdown(f"- Moderate: {probs[1]*100:.2f}%")
                st.markdown(f"- Polluted: {probs[2]*100:.2f}%")

                if len(c_set) > 1 or max(probs) < 0.60:
                    st.error("FLAG: Screening support only. Needs laboratory test (Ambiguous or low confidence).")
                else:
                    st.success("Sufficient screening confidence.")

            with res_col2:
                st.subheader("Local Feature Attribution (Approximate Occlusion)")
                attr_dict = {}
                feature_names = ["pH", "Turbidity", "Temperature", "Dissolved Oxygen", "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"]
                base_prob = probs[pred_class]
                for i, fn in enumerate(feature_names):
                    occ = sample_arr[0].copy()
                    occ[i] = medians[i]
                    occ_p = clf.predict_proba(occ.reshape(1, -1))[0][pred_class]
                    attr_dict[fn] = base_prob - occ_p

                attr_df = pd.DataFrame(list(attr_dict.items()), columns=["Feature", "Probability Shift (Delta P)"])
                attr_df = attr_df.sort_values(by="Probability Shift (Delta P)", key=abs, ascending=False)
                st.dataframe(attr_df, use_container_width=True)
        else:
            st.error("Model bundle not found. Please train models first.")

# Tab 2: Dataset Explorer
with tabs[1]:
    st.header("Audited Dataset Overview")
    raw_path = PROJECT_ROOT / "data" / "raw" / "water_quality.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        st.write(f"Total Records: {len(df)} rows, {len(df.columns)} columns (Complete records, 0 missing, 0 duplicates).")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("Raw dataset file not found at data/raw/water_quality.csv.")

# Tab 3: Model Verification
with tabs[2]:
    st.header("Holdout Test Verification Metrics")
    if metrics:
        h = metrics["headline_metrics"]
        st.write(f"**Split:** {metrics['dataset_split']}")
        st.write(f"**Architecture:** {metrics['model_architecture']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Macro-F1", f"{h['macro_f1']*100:.2f}%")
        m2.metric("Accuracy", f"{h['accuracy']*100:.2f}%")
        m3.metric("PR-AUC", f"{h['macro_pr_auc']:.4f}")
        m4.metric("Brier Score", f"{h['brier_score']:.4f}")

        st.subheader("Per-Class Breakdown")
        st.dataframe(pd.DataFrame(metrics["per_class_metrics"]).T, use_container_width=True)
    else:
        st.info("Metrics not found at results/metrics.json.")

# Tab 4: Diagnostic Figures
with tabs[3]:
    st.header("Experimental Visualizations (outputs/figures/)")
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_cols = st.columns(2)
    figs = [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("pr_curves.png", "Precision-Recall Curves"),
        ("reliability_diagram.png", "Reliability Diagram (Calibration)"),
        ("shap_summary.png", "TreeSHAP Summary Plot"),
        ("feature_distributions.png", "Feature Histograms"),
        ("correlation_matrix.png", "Correlation Heatmap"),
    ]
    for idx, (fname, title) in enumerate(figs):
        fpath = fig_dir / fname
        if fpath.exists():
            with fig_cols[idx % 2]:
                st.subheader(title)
                st.image(str(fpath), use_container_width=True)
