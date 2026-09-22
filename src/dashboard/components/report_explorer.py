"""
Report Explorer component.
"""
import streamlit as st
import pandas as pd
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import render_error_state

def render():
    st.title("Report Explorer")
    st.write("Browse enriched reports. Use the export button to download the filtered dataset.")
    
    try:
        # Note: In production we'd implement real pagination against the backend.
        # For this prototype we grab a bulk set.
        reports_res = api.get_reports(limit=2000)
        reports = reports_res.get("data", [])
    except APIError as e:
        render_error_state("Unable to load reports.")
        return
        
    if not reports:
        st.info("No reports found.")
        return
        
    df = pd.DataFrame(reports)
    
    # Filter controls
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sites = ["All"] + sorted([s for s in df["site"].unique() if pd.notna(s)])
        sel_site = st.selectbox("Site", sites, key="explorer_site")
    with col2:
        types = ["All"] + sorted([t for t in df["report_type"].unique() if pd.notna(t)])
        sel_type = st.selectbox("Report Type", types, key="explorer_type")
    with col3:
        sif_status = ["All", "SIF-Potential", "Non-SIF"]
        sel_sif = st.selectbox("SIF Status", sif_status, key="explorer_sif")
    with col4:
        lsrs = ["All"] + sorted([r for r in df["primary_lsr"].unique() if pd.notna(r)])
        sel_lsr = st.selectbox("Primary LSR", lsrs, key="explorer_lsr")

    # Apply filters
    filtered_df = df.copy()
    
    # Safely handle column names that might differ between the backend and frontend
    sif_col = "sif_prediction" if "sif_prediction" in filtered_df.columns else "sif"
    
    if sel_site != "All":
        filtered_df = filtered_df[filtered_df["site"] == sel_site]
    if sel_type != "All":
        filtered_df = filtered_df[filtered_df["report_type"] == sel_type]
    if sel_sif != "All" and sif_col in filtered_df.columns:
        val = 1 if sel_sif == "SIF-Potential" else 0
        filtered_df = filtered_df[filtered_df[sif_col].astype(str) == str(val)]
    if sel_lsr != "All" and "primary_lsr" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["primary_lsr"] == sel_lsr]
        
    st.write(f"**{len(filtered_df)}** reports match your filters.")
    
    # Exclude internal/ground-truth columns for display and export
    drop_cols = [c for c in filtered_df.columns if "ground_truth" in c or c in ["synthetic_labels", "internal_debug"]]
    display_df = filtered_df.drop(columns=drop_cols, errors="ignore")
    
    # Display table
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # CSV Export
    if not display_df.empty:
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name='safety_reports_export.csv',
            mime='text/csv',
        )
