"""
LSR Analysis component.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import format_pct, apply_chart_style, render_error_state

def render():
    st.title("Life-Saving Rule Analysis")
    
    try:
        lsr_data = api.get_lsr()
    except APIError as e:
        render_error_state("Unable to load LSR analytics.")
        return
        
    lsr_list = lsr_data.get("data", [])
    if not lsr_list:
        st.info("No LSR data available.")
        return
        
    df = pd.DataFrame(lsr_list)
    
    # Table of all LSRs
    st.subheader("Life-Saving Rules Summary")
    display_df = df[["name", "total_reports", "sif_count", "sif_density", "avg_sif_probability"]].copy()
    display_df.columns = ["Life-Saving Rule", "Total Occurrences", "SIF Count", "SIF Density", "Avg SIF Probability"]
    
    st.dataframe(
        display_df.style.format({
            "SIF Density": "{:.1%}", 
            "Avg SIF Probability": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("LSR Distribution & Trends")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_freq = px.bar(
            df.sort_values("total_reports", ascending=True),
            x="total_reports",
            y="name",
            orientation="h",
            title="LSR Occurrence Frequency",
            color_discrete_sequence=["#1f77b4"]
        )
        st.plotly_chart(apply_chart_style(fig_freq), use_container_width=True)
        
    with col2:
        fig_dens = px.bar(
            df.sort_values("sif_density", ascending=True),
            x="sif_density",
            y="name",
            orientation="h",
            title="SIF Density by LSR",
            color_discrete_sequence=["#d62728"]
        )
        # Format x axis as percentage
        fig_dens.update_layout(xaxis=dict(tickformat=".1%"))
        st.plotly_chart(apply_chart_style(fig_dens), use_container_width=True)
        
    # Detail View
    st.markdown("---")
    st.subheader("Explore LSR Context")
    
    selected_lsr = st.selectbox("Select an LSR to view associated sites and activities", df["name"].unique())
    lsr_detail = df[df["name"] == selected_lsr].iloc[0]
    
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Top Sites for this LSR**")
        sites_list = lsr_detail.get("associated_sites", [])
        if sites_list:
            site_df = pd.DataFrame(sites_list)
            st.dataframe(site_df, use_container_width=True, hide_index=True)
        else:
            st.info("No site data.")
            
    with col4:
        st.write("**Top Activities for this LSR**")
        act_list = lsr_detail.get("associated_activities", [])
        if act_list:
            act_df = pd.DataFrame(act_list)
            st.dataframe(act_df, use_container_width=True, hide_index=True)
        else:
            st.info("No activity data.")
