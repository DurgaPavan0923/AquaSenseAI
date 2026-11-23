"""AquaSense AI - Virtual IoT Sensor Telemetry & Fault Injection Simulator."""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.common.constants import STATUTORY_DISCLAIMER

st.set_page_config(page_title="IoT Live Demo | AquaSense AI", page_icon="📡", layout="wide")

st.warning(f"⚠️ **REGULATORY NOTICE:** {STATUTORY_DISCLAIMER}")
st.title("📡 Virtual IoT Edge Probe Telemetry Simulator")
st.markdown("Simulate high-frequency aquatic multi-probe sensor streams with real-time fault injection and plume contamination dynamics.")

# Session state initialization for streaming buffer
if "iot_stream_buffer" not in st.session_state:
    st.session_state.iot_stream_buffer = []

# Control Panel
st.subheader("1. Stream Telemetry Controls & Fault Injection")
c1, c2, c3 = st.columns(3)

with c1:
    stream_mode = st.selectbox(
        "Operational Telemetry Mode",
        ["Nominal Clean Stream", "Chemical Contamination Plume", "Sensor Fault (Air-Lock / Jump)"]
    )
with c2:
    n_ticks = st.slider("Samples to Generate", 1, 25, 5)
with c3:
    station_id = st.text_input("Station Tag", "STATION_RIVER_PROBE_09")

def generate_telemetry_tick(mode: str) -> dict:
    base = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "location_id": station_id,
        "pH": round(float(np.random.normal(7.3, 0.2)), 2),
        "Turbidity": round(float(np.random.exponential(1.5) + 0.5), 1),
        "Temperature": round(float(np.random.normal(21.0, 1.0)), 1),
        "Dissolved Oxygen": round(float(np.random.normal(8.2, 0.4)), 1),
        "Biochemical Oxygen Demand": round(float(np.random.exponential(1.0) + 0.5), 1),
        "Lead": round(float(np.random.exponential(0.002)), 4),
        "Mercury": round(float(np.random.exponential(0.0002)), 5),
        "Arsenic": round(float(np.random.exponential(0.002)), 4)
    }
    
    if mode == "Chemical Contamination Plume":
        base["pH"] = round(float(np.random.normal(5.4, 0.3)), 2)
        base["Turbidity"] = round(float(np.random.normal(35.0, 5.0)), 1)
        base["Dissolved Oxygen"] = round(float(np.random.normal(3.1, 0.4)), 1)  # Hypoxic
        base["Biochemical Oxygen Demand"] = round(float(np.random.normal(12.5, 2.0)), 1)
        base["Lead"] = round(float(np.random.normal(0.035, 0.005)), 4)  # 3.5x WHO limit
        base["status"] = "POLLUTION EVENT"
    elif mode == "Sensor Fault (Air-Lock / Jump)":
        base["pH"] = -1.5  # Impossible physical value
        base["Turbidity"] = 120.0  # Massive jump
        base["status"] = "HARDWARE FAULT"
    else:
        base["status"] = "NORMAL"

    return base

if st.button("Generate & Stream Telemetry Packets", type="primary"):
    for _ in range(n_ticks):
        pkt = generate_telemetry_tick(stream_mode)
        st.session_state.iot_stream_buffer.append(pkt)
        if len(st.session_state.iot_stream_buffer) > 50:
            st.session_state.iot_stream_buffer.pop(0)

st.divider()

# Visualization
st.subheader("2. Real-Time Telemetry Stream Buffer")
if st.session_state.iot_stream_buffer:
    buf_df = pd.DataFrame(st.session_state.iot_stream_buffer)
    buf_df["tick"] = range(1, len(buf_df) + 1)

    # Multi-trace line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=buf_df["tick"], y=buf_df["pH"], mode="lines+markers", name="pH Level", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=buf_df["tick"], y=buf_df["Dissolved Oxygen"], mode="lines+markers", name="Dissolved Oxygen (mg/L)", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=buf_df["tick"], y=buf_df["Turbidity"], mode="lines+markers", name="Turbidity (NTU)", line=dict(color="#ff7f0e")))

    # Thresholds
    fig.add_hline(y=6.5, line_dash="dash", line_color="blue", annotation_text="pH Min (6.5)")
    fig.add_hline(y=8.5, line_dash="dash", line_color="blue", annotation_text="pH Max (8.5)")

    fig.update_layout(
        title="Live Streamed Telemetry Parameters",
        xaxis_title="Packet Index",
        yaxis_title="Measured Magnitude",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Latest Packets Table
    st.subheader("Latest Received Telemetry Packets")
    st.dataframe(buf_df.tail(10), use_container_width=True)

    if st.button("Clear Buffer"):
        st.session_state.iot_stream_buffer = []
        st.rerun()
else:
    st.info("Buffer empty. Click 'Generate & Stream Telemetry Packets' to simulate incoming IoT edge data.")
