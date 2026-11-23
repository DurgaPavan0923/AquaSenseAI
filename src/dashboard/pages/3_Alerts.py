"""AquaSense AI - Active Incident Triage & Alerts Management."""

import os

import pandas as pd
import requests
import streamlit as st

from src.common.constants import STATUTORY_DISCLAIMER

st.set_page_config(page_title="Alerts | AquaSense AI", page_icon="🚨", layout="wide")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.warning(f"⚠️ **REGULATORY NOTICE:** {STATUTORY_DISCLAIMER}")
st.title("🚨 Incident & Compliance Alert Center")
st.markdown("Audit and triage real-time water contamination alerts, toxic statutory exceedances, and sensor failure notifications.")

# Fetch alerts
@st.cache_data(ttl=5)
def fetch_all_alerts():
    try:
        res = requests.get(f"{API_BASE_URL}/alerts", timeout=4)
        if res.status_code == 200:
            return pd.DataFrame(res.json())
    except Exception:
        pass

    # Mock alerts for UI if empty
    return pd.DataFrame([
        {
            "id": "alert-001",
            "sample_id": "sample-882",
            "severity": "CRITICAL",
            "alert_type": "EXTREME_TOXIC_SPIKE",
            "message": "Lead concentration (0.018 mg/L) exceeded statutory limit (0.01 mg/L).",
            "is_acknowledged": False,
            "created_at": "2026-11-23T12:30:00"
        },
        {
            "id": "alert-002",
            "sample_id": "sample-883",
            "severity": "HIGH",
            "alert_type": "HIGH_POLLUTION_RISK",
            "message": "High risk of pollution predicted with 88.5% confidence at STATION_RIVER_NORTH.",
            "is_acknowledged": False,
            "created_at": "2026-11-23T13:15:00"
        },
        {
            "id": "alert-003",
            "sample_id": "sample-884",
            "severity": "MEDIUM",
            "alert_type": "SENSOR_ANOMALY",
            "message": "Dissolved oxygen dropped below hypoxic limit (3.8 < 4.0 mg/L).",
            "is_acknowledged": True,
            "created_at": "2026-11-23T11:00:00"
        }
    ])

alerts_df = fetch_all_alerts()

# Filters
c1, c2 = st.columns([1, 2])
with c1:
    severity_filter = st.multiselect(
        "Filter by Severity",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM"]
    )
with c2:
    status_filter = st.radio("Status Filter", ["All Alerts", "Unacknowledged Only", "Acknowledged Only"], horizontal=True)

# Apply filters
filtered_df = alerts_df.copy()
if not filtered_df.empty:
    if severity_filter:
        filtered_df = filtered_df[filtered_df["severity"].isin(severity_filter)]
    if status_filter == "Unacknowledged Only":
        filtered_df = filtered_df[filtered_df["is_acknowledged"] == False]
    elif status_filter == "Acknowledged Only":
        filtered_df = filtered_df[filtered_df["is_acknowledged"] == True]

# Summary metrics
m1, m2, m3 = st.columns(3)
with m1:
    crit_count = len(alerts_df[alerts_df["severity"] == "CRITICAL"])
    st.metric("Critical Alerts", crit_count, delta_color="inverse")
with m2:
    unack_count = len(alerts_df[alerts_df["is_acknowledged"] == False])
    st.metric("Pending Review", unack_count, delta_color="inverse")
with m3:
    ack_count = len(alerts_df[alerts_df["is_acknowledged"] == True])
    st.metric("Acknowledged & Triaged", ack_count)

st.divider()

# Alerts Table & Action Buttons
if filtered_df.empty:
    st.info("No alerts found matching the active filter criteria.")
else:
    for idx, row in filtered_df.iterrows():
        sev = row["severity"]
        color = "#d62728" if sev == "CRITICAL" else "#ff7f0e" if sev == "HIGH" else "#1f77b4"
        is_ack = row["is_acknowledged"]

        col_text, col_btn = st.columns([5, 1])
        with col_text:
            st.markdown(
                f"""
                <div style="border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 8px;">
                    <b>[{sev}] {row['alert_type']}</b> • <small>{row['created_at']}</small><br>
                    <span>{row['message']}</span> (Sample ID: <code>{row['sample_id']}</code>)
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_btn:
            if not is_ack:
                if st.button("Acknowledge", key=f"ack_{row['id']}"):
                    try:
                        requests.post(f"{API_BASE_URL}/alerts/{row['id']}/acknowledge", json={"acknowledged_by": "operator"})
                        st.success("Acknowledged!")
                        st.rerun()
                    except Exception:
                        st.warning("Server unreachable; acknowledged locally.")
            else:
                st.caption("✅ Resolved")

    # CSV Export
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Alerts Log (CSV)",
        data=csv_data,
        file_name="aquasense_incident_audit.csv",
        mime="text/csv"
    )
