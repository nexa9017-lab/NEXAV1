"""
Executive Overview dashboard component.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import format_pct, apply_chart_style, render_error_state

def render():
    st.title("Executive Overview")
    
    try:
        summary_res = api.get_summary()
        summary = summary_res.get("data", {})
        metadata = summary_res.get("metadata", {})
    except APIError as e:
        render_error_state("Unable to load overview analytics.")
        return

    # --- KPI CARDS ---
    st.subheader("Key Performance Indicators")
    
    total_reports = summary.get("total_reports", 0)
    sif_count = summary.get("total_predicted_sif_reports", 0)
    sif_density = summary.get("overall_sif_density", 0.0)
    avg_sif_prob = summary.get("avg_sif_probability", 0.0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reports", f"{total_reports:,}")
    with col2:
        st.metric("SIF Precursors", f"{sif_count:,}")
    with col3:
        st.metric("Overall SIF Density", format_pct(sif_density))
    with col4:
        st.metric("Avg SIF Probability", format_pct(avg_sif_prob))

    st.markdown("---")
    
    try:
        sites_res = api.get_sites()
        sites = sites_res.get("data", [])
        lsrs_res = api.get_lsr()
        lsrs = lsrs_res.get("data", [])
        hazards_res = api.get_hazards()
        hazards = hazards_res.get("data", [])
    except APIError:
        sites = []
        lsrs = []
        hazards = []

    # --- CHARTS ---
    st.subheader("Trends & Distributions")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # SIF vs Non-SIF
        non_sif_count = total_reports - sif_count
        fig_sif = px.pie(
            names=["SIF-Potential", "Non-SIF"],
            values=[sif_count, non_sif_count],
            title="SIF vs Non-SIF Distribution",
            hole=0.4,
            color_discrete_sequence=["#d62728", "#1f77b4"]
        )
        st.plotly_chart(apply_chart_style(fig_sif), use_container_width=True)
        
    with c2:
        # Top Sites by Count
        if sites:
            df_sites = pd.DataFrame(sites).head(10).sort_values("sif_count", ascending=True)
            fig_sites = px.bar(
                df_sites, 
                x="sif_count", 
                y="name", 
                orientation="h",
                title="Top Sites by SIF Count",
                color_discrete_sequence=["#d62728"]
            )
            st.plotly_chart(apply_chart_style(fig_sites), use_container_width=True)
        else:
            st.info("No site data available.")

    c3, c4 = st.columns(2)
    
    with c3:
        # Top LSRs
        if lsrs:
            df_lsrs = pd.DataFrame(lsrs).head(8).sort_values("sif_count", ascending=True)
            fig_lsr = px.bar(
                df_lsrs,
                x="sif_count",
                y="name",
                orientation="h",
                title="Top Life-Saving Rules Involved in SIFs",
                color_discrete_sequence=["#ff7f0e"]
            )
            st.plotly_chart(apply_chart_style(fig_lsr), use_container_width=True)
        else:
            st.info("No LSR data available.")
            
    with c4:
        # Top Hazards
        if hazards:
            df_haz = pd.DataFrame(hazards).head(8).sort_values("sif_count", ascending=True)
            fig_haz = px.bar(
                df_haz,
                x="sif_count",
                y="name",
                orientation="h",
                title="Most Frequent Hazards",
                color_discrete_sequence=["#2ca02c"]
            )
            st.plotly_chart(apply_chart_style(fig_haz), use_container_width=True)
        else:
            st.info("No hazard data available.")
