"""
SIF Analysis component.
Displays reports ranked by SIF probability and their details.
"""
import streamlit as st
import pandas as pd
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import format_pct, render_error_state

def render():
    st.title("SIF Analysis")
    
    try:
        # Fetch reports (simulating a list of reports or relying on summary API for now)
        # Note: In a real app with 10k rows, we'd paginate via the backend.
        reports_res = api.get_reports(limit=500)
        reports = reports_res.get("data", [])
    except APIError as e:
        render_error_state("Unable to load reports for SIF analysis.")
        return
        
    if not reports:
        st.info("No reports found.")
        return

    # Convert to DataFrame for easy filtering and display
    df = pd.DataFrame(reports)
    
    # We only want to show predicted fields, no ground truth
    display_cols = [
        "report_id", "date", "site", "report_type", 
        "description", "sif_prediction", "sif_probability", "primary_lsr"
    ]
    
    # Ensure columns exist
    for col in display_cols:
        if col not in df.columns:
            df[col] = None
            
    # Filter controls
    st.subheader("Filter Reports")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sites = ["All"] + sorted([s for s in df["site"].unique() if pd.notna(s)])
        sel_site = st.selectbox("Site", sites)
    with col2:
        types = ["All"] + sorted([t for t in df["report_type"].unique() if pd.notna(t)])
        sel_type = st.selectbox("Report Type", types)
    with col3:
        sif_status = ["All", "SIF-Potential", "Non-SIF"]
        sel_sif = st.selectbox("SIF Status", sif_status)
    with col4:
        lsrs = ["All"] + sorted([r for r in df["primary_lsr"].unique() if pd.notna(r)])
        sel_lsr = st.selectbox("Primary LSR", lsrs)

    # Apply filters
    filtered_df = df.copy()
    if sel_site != "All":
        filtered_df = filtered_df[filtered_df["site"] == sel_site]
    if sel_type != "All":
        filtered_df = filtered_df[filtered_df["report_type"] == sel_type]
    if sel_sif != "All":
        val = 1 if sel_sif == "SIF-Potential" else 0
        filtered_df = filtered_df[filtered_df["sif_prediction"] == val]
    if sel_lsr != "All":
        filtered_df = filtered_df[filtered_df["primary_lsr"] == sel_lsr]
        
    filtered_df = filtered_df.sort_values(by="sif_probability", ascending=False)
    
    st.write(f"Showing **{len(filtered_df)}** reports.")
    
    # Selection mechanism (using a simple dataframe display and a selectbox for detail view)
    # Streamlit dataframe selection is tricky without ag-grid, so we use a selectbox for ID
    st.dataframe(
        filtered_df[display_cols].style.format({"sif_probability": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
        height=300
    )
    
    st.markdown("---")
    st.subheader("Report Details")
    
    if not filtered_df.empty:
        report_ids = filtered_df["report_id"].tolist()
        sel_report_id = st.selectbox("Select Report ID to inspect", report_ids)
        
        if sel_report_id:
            try:
                detail_res = api.get_report_detail(sel_report_id)
                render_report_detail(detail_res.get("data", {}))
            except APIError:
                render_error_state(f"Could not load details for {sel_report_id}")
    else:
        st.info("No reports match the current filters.")

def render_report_detail(report_payload: dict):
    """Renders the detailed view of a single report."""
    orig = report_payload.get("original_report", {})
    pred = report_payload.get("predicted_fields", {})
    
    st.markdown(f"**Date:** {orig.get('date', 'N/A')} | **Site:** {orig.get('site', 'N/A')}")
    st.markdown(f"**Description:**")
    st.info(orig.get("description", "N/A"))
    
    col1, col2 = st.columns(2)
    with col1:
        sif_prob = pred.get("sif_probability", 0.0)
        try:
            sif_prob = float(sif_prob) if sif_prob else 0.0
        except ValueError:
            sif_prob = 0.0
            
        sif_val = pred.get("sif", "0")
        sif_status = "SIF-Potential" if str(sif_val) == "1" else "Non-SIF"
        color = "red" if sif_status == "SIF-Potential" else "green"
        st.markdown(f"**SIF Assessment:** :{color}[{sif_status}] ({format_pct(sif_prob)})")
        st.markdown(f"**Primary LSR:** {pred.get('primary_lsr', 'None')}")
        
    with col2:
        st.markdown("**Tags:**")
        tags = orig.get("precursor_tags", [])
        if tags:
            for tag in tags:
                st.markdown(f"- `{tag}`")
        else:
            st.write("None")
            
    st.markdown("### Extracted Intelligence")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Activities**")
        for x in pred.get("activities", []):
            st.write(f"- {x}")
        st.markdown("**Equipment**")
        for x in pred.get("equipment", []):
            st.write(f"- {x}")
    with c2:
        st.markdown("**Hazards**")
        for x in pred.get("hazards", []):
            st.write(f"- {x}")
        st.markdown("**Barrier Failures**")
        for x in pred.get("barrier_failures", []):
            st.write(f"- {x}")
    with c3:
        st.markdown("**Potential Consequences**")
        for x in pred.get("potential_consequences", []):
            st.write(f"- {x}")
            
    # Evidence
    evidence_fields = ["activities_evidence", "equipment_evidence", "hazards_evidence", "barrier_failures_evidence", "potential_consequences_evidence"]
    has_evidence = any(pred.get(f) for f in evidence_fields)
    
    if has_evidence:
        with st.expander("Show Extraction Evidence"):
            for field in evidence_fields:
                ev_list = pred.get(field, [])
                if ev_list:
                    name = field.replace("_evidence", "").replace("_", " ").title()
                    st.markdown(f"**{name}:**")
                    for ev in ev_list:
                        st.write(f"- *\"{ev}\"*")
