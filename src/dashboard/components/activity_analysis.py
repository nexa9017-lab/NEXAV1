"""
Activity Analysis component.
"""
import streamlit as st
import pandas as pd
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import format_pct, render_error_state

def render():
    st.title("Activity Analysis")

    try:
        activities_res = api.get_activities()
        activities = activities_res.get("data", [])
    except APIError as e:
        render_error_state("Unable to load activity analytics.")
        return
        
    if not activities:
        st.info("No activity data available.")
        return
        
    df = pd.DataFrame(activities)
    
    st.subheader("Activity Overview")
    
    # Sort by total reports descending
    df_sorted = df.sort_values("total_reports", ascending=False)
    
    display_df = df_sorted[["value", "total_reports", "sif_count", "sif_density", "average_sif_probability"]].copy()
    display_df.columns = ["Activity", "Total Reports", "SIF Count", "SIF Density", "Avg SIF Probability"]
    
    st.dataframe(
        display_df.style.format({
            "SIF Density": "{:.1%}", 
            "Avg SIF Probability": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("Activity Deep Dive")
    
    selected_activity = st.selectbox("Select Activity for deeper analysis", df_sorted["value"].tolist())
    
    detail = df_sorted[df_sorted["value"] == selected_activity].iloc[0].to_dict()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reports", detail.get("total_reports", 0))
    col2.metric("SIF Count", detail.get("sif_count", 0))
    col3.metric("SIF Density", format_pct(detail.get("sif_density", 0.0)))
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Associated Hazards**")
        st.dataframe(pd.DataFrame(detail.get("top_hazards", [])), hide_index=True, use_container_width=True)
        
        st.markdown("**Associated Barrier Failures**")
        st.dataframe(pd.DataFrame(detail.get("top_barrier_failures", [])), hide_index=True, use_container_width=True)
        
    with c2:
        st.markdown("**Life-Saving Rules**")
        st.dataframe(pd.DataFrame(detail.get("top_lsrs", [])), hide_index=True, use_container_width=True)
        
        st.markdown("**Affected Sites**")
        st.dataframe(pd.DataFrame(detail.get("top_sites", [])), hide_index=True, use_container_width=True)
