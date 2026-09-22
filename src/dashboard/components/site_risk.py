"""
Site Risk Analysis component.
"""
import streamlit as st
import pandas as pd
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import format_pct, render_error_state

def render():
    st.title("Site Risk Analysis")
    
    st.subheader("Filter Sites")
    sort_by = st.selectbox("Sort Sites By", ["sif_density", "sif_count", "total_reports", "average_sif_probability"])

    try:
        sites_res = api.get_sites(sort_by=sort_by)
        sites = sites_res.get("data", [])
    except APIError as e:
        render_error_state("Unable to load site analytics.")
        return
        
    if not sites:
        st.info("No site data available.")
        return
        
    df = pd.DataFrame(sites)
    
    st.write(f"Showing **{len(df)}** sites.")
    
    # Overview Table
    # Ensure all columns exist to prevent KeyError
    required_cols = ["name", "total_reports", "sif_count", "sif_density", "avg_sif_probability", "data_quality_warnings"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col != "name" and col != "data_quality_warnings" else None
            
    display_df = df[required_cols].copy()
    display_df.columns = ["Site", "Total Reports", "SIF Count", "SIF Density", "Avg SIF Probability", "Data Quality Warning"]
    
    # Format warnings list to string
    display_df["Data Quality Warning"] = display_df["Data Quality Warning"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) and x else ""
    )
    
    st.dataframe(
        display_df.style.format({
            "SIF Density": "{:.1%}", 
            "Avg SIF Probability": "{:.1%}"
        }).applymap(lambda x: 'color: orange;' if x else '', subset=['Data Quality Warning']),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("Site Detail Deep-Dive")
    
    selected_site_name = st.selectbox("Select a Site", df["name"].tolist())
    selected_site = df[df["name"] == selected_site_name].iloc[0].to_dict()
    
    # Find full detail from the list (the API returns full site profile including top hazards etc)
    detail = next((s for s in sites if s["name"] == selected_site_name), {})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reports", detail.get("total_reports", 0))
    col2.metric("SIF Count", detail.get("sif_count", 0))
    col3.metric("SIF Density", format_pct(detail.get("sif_density", 0.0)))
    
    warnings = detail.get("data_quality_warnings", [])
    if warnings:
        st.warning(f"⚠️ Data Quality Warning: {', '.join(warnings)}")
        
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Activities**")
        st.dataframe(pd.DataFrame(detail.get("top_activities", [])), hide_index=True, use_container_width=True)
        
        st.markdown("**Top Life-Saving Rules**")
        st.dataframe(pd.DataFrame(detail.get("top_lsr", [])), hide_index=True, use_container_width=True)

    with c2:
        st.markdown("**Top Hazards**")
        st.dataframe(pd.DataFrame(detail.get("top_hazards", [])), hide_index=True, use_container_width=True)
        
        st.markdown("**Top Barrier Failures**")
        st.dataframe(pd.DataFrame(detail.get("top_barriers", [])), hide_index=True, use_container_width=True)
        
    st.markdown("### Top Precursor Tags")
    st.dataframe(pd.DataFrame(detail.get("top_precursor_tags", [])), hide_index=True, use_container_width=True)
    
    # Note: Monthly trend is available in detail["reports_per_month"]
    st.markdown("### Reports per Month")
    trend = detail.get("reports_per_month", {})
    if trend:
        # Convert dict to dataframe format
        trend_list = [{"Month": k, "Reports": v} for k, v in trend.items()]
        trend_df = pd.DataFrame(trend_list)
        st.bar_chart(trend_df.set_index("Month")["Reports"])
    else:
        st.write("No monthly trend data.")
