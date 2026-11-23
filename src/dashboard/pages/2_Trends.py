"""AquaSense AI - Spatiotemporal Trends & Water Quality Dynamics."""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.common.constants import STATUTORY_DISCLAIMER

st.set_page_config(page_title="Trends | AquaSense AI", page_icon="📈", layout="wide")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.warning(f"⚠️ **REGULATORY NOTICE:** {STATUTORY_DISCLAIMER}")
st.title("📈 Spatiotemporal Water Quality Trends")
st.markdown("Monitor historical physicochemical shifts, rolling moving averages, and parameter breaches against statutory WHO and BIS limits.")

# Load historical data
@st.cache_data(ttl=15)
def load_trend_data():
    try:
        res = requests.get(f"{API_BASE_URL}/history?limit=200", timeout=4)
        if res.status_code == 200 and len(res.json()) > 10:
            df = pd.DataFrame(res.json())
            df["recorded_at"] = pd.to_datetime(df["recorded_at"])
            return df.sort_values(by="recorded_at")
    except Exception:
        pass

    from src.data.loader import generate_benchmark_dataset
    df = generate_benchmark_dataset(n_samples=150)
    df["recorded_at"] = pd.date_range(end=pd.Timestamp.now(), periods=150, freq="h")
    return df

df = load_trend_data()

# Station filter & metric selector
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    param = st.selectbox(
        "Select Parameter to Graph",
        ["pH", "Turbidity", "Dissolved Oxygen", "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"]
    )
with c2:
    window = st.slider("Moving Average Window (Samples)", 3, 30, 7)

# Parameter threshold lookup
threshold_val = None
threshold_label = None
if param == "pH":
    threshold_val = 8.5
    threshold_label = "WHO/BIS Max Desirable (8.5)"
elif param == "Turbidity":
    threshold_val = 5.0
    threshold_label = "WHO/BIS Acceptable Limit (5.0 NTU)"
elif param == "Dissolved Oxygen":
    threshold_val = 6.5
    threshold_label = "WHO/EPA Potable Minimum (6.5 mg/L)"
elif param == "Biochemical Oxygen Demand":
    threshold_val = 2.0
    threshold_label = "Clean Water Baseline (2.0 mg/L)"
elif param == "Lead":
    threshold_val = 0.01
    threshold_label = "Statutory Toxic Limit (0.01 mg/L)"
elif param == "Mercury":
    threshold_val = 0.001
    threshold_label = "Statutory Toxic Limit (0.001 mg/L)"
elif param == "Arsenic":
    threshold_val = 0.01
    threshold_label = "Statutory Toxic Limit (0.01 mg/L)"

# Compute rolling average
df["rolling_mean"] = df[param].rolling(window=window, min_periods=1).mean()

# Time Series Plot
fig = go.Figure()

# Raw readings scatter/line
fig.add_trace(go.Scatter(
    x=df["recorded_at"], y=df[param],
    mode="lines+markers",
    name="Raw Sensor Reading",
    line=dict(color="#1f77b4", width=1.5),
    opacity=0.6
))

# Rolling Average
fig.add_trace(go.Scatter(
    x=df["recorded_at"], y=df["rolling_mean"],
    mode="lines",
    name=f"{window}-Sample Rolling Mean",
    line=dict(color="#ff7f0e", width=2.5)
))

# Regulatory Threshold Line
if threshold_val is not None:
    fig.add_hline(
        y=threshold_val,
        line_dash="dash",
        line_color="red",
        annotation_text=threshold_label,
        annotation_position="top right"
    )

fig.update_layout(
    title=f"Temporal Dynamics for {param}",
    xaxis_title="Recorded Timestamp",
    yaxis_title=f"{param} Concentration / Value",
    height=450,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Parameter Cross-Correlation Heatmap
st.subheader("Multi-Parameter Cross-Correlation Dynamics")
numerical_cols = ["pH", "Turbidity", "Temperature", "Dissolved Oxygen", "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"]
present_num = [c for c in numerical_cols if c in df.columns]
corr_df = df[present_num].corr()

fig_corr = px.imshow(
    corr_df,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Covariance & Inter-Metric Correlation Matrix"
)
fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_corr, use_container_width=True)
