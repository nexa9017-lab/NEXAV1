"""
Emerging Trends Analysis component.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import apply_chart_style, render_error_state

def render():
    st.title("Emerging Trends")
    st.markdown("### Historical Trend Comparison")
    
    try:
        trends_res = api.get_emerging_trends()
        emerging_patterns = trends_res.get("data", [])
    except APIError as e:
        render_error_state("Unable to load emerging trends analytics.")
        return
        
    if not emerging_patterns:
        st.info("No emerging patterns detected in this reporting period.")
        return
        
    df = pd.DataFrame(emerging_patterns)
    
    # Overview
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patterns Evaluated", len(df))
    c2.metric("New Patterns", len(df[df["status"] == "NEW_PATTERN"]))
    c3.metric("Increasing Patterns", len(df[df["status"] == "INCREASING"]))
    
    st.markdown("---")
    
    # Filter by Status
    st.subheader("Trend Breakdown")
    status_filter = st.multiselect(
        "Filter by Status", 
        options=df["status"].unique(),
        default=[s for s in df["status"].unique() if s in ["NEW_PATTERN", "INCREASING", "EMERGING"]]
    )
    
    if not status_filter:
        filtered_df = df
    else:
        filtered_df = df[df["status"].isin(status_filter)]
        
    if filtered_df.empty:
        st.info("No patterns match the selected status.")
        return
        
    display_df = filtered_df[["pattern", "status", "previous_count", "current_count", "absolute_change", "percentage_change"]].copy()
    display_df.columns = ["Pattern", "Status", "Previous Period", "Current Period", "Absolute Change", "Percent Change"]
    
    # Format percent change
    def format_change(x):
        if pd.isna(x) or x is None:
            return "N/A"
        return f"+{x:.0%}" if x > 0 else f"{x:.0%}"
        
    display_df["Percent Change"] = display_df["Percent Change"].apply(format_change)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Highest Increasing Trends")
    
    # Visuals: Bar chart of top increases
    inc_df = df[df["absolute_change"] > 0].sort_values("absolute_change", ascending=False).head(10)
    
    if not inc_df.empty:
        fig = px.bar(
            inc_df,
            x="absolute_change",
            y="pattern",
            orientation="h",
            title="Patterns with Highest Absolute Increase",
            color="status",
            color_discrete_map={
                "NEW_PATTERN": "#d62728",
                "EMERGING": "#ff7f0e",
                "INCREASING": "#1f77b4"
            }
        )
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)
    else:
        st.write("No increasing trends to visualize.")
