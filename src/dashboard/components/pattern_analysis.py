"""
Precursor Patterns Analysis component.
"""
import streamlit as st
import pandas as pd
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import format_pct, format_score, render_error_state

def render():
    st.title("Precursor Patterns")
    
    # Filters
    st.subheader("Filter Patterns")
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.number_input("Minimum Support (Occurrences)", min_value=1, value=5)
    with col2:
        min_density = st.slider("Minimum SIF Density", 0.0, 1.0, 0.0, 0.05, format="%.2f")

    try:
        patterns_res = api.get_patterns(min_support=min_support)
        patterns = patterns_res.get("data", [])
    except APIError as e:
        render_error_state("Unable to load precursor patterns.")
        return
        
    if not patterns:
        st.info("No recurring patterns match the selected filters.")
        return
        
    # Map the JSON structure to dataframe columns expected
    mapped_patterns = []
    for p in patterns:
        p_copy = p.copy()
        # Ensure we have scalar values for the table
        p_copy["support"] = p.get("occurrences", 0)
        p_copy["site_count"] = p.get("affected_site_count", len(p.get("affected_sites", {})))
        
        # Format components nicely
        comps = p.get("components", {})
        parts = []
        for k, v in comps.items():
            try:
                import ast
                v_dict = ast.literal_eval(v)
                parts.append(v_dict.get("value", v))
            except:
                parts.append(str(v))
        p_copy["pattern"] = " + ".join(parts)
        mapped_patterns.append(p_copy)
        
    df = pd.DataFrame(mapped_patterns)
    
    # Ensure all required columns exist even if empty
    required_cols = ["pattern_id", "pattern", "support", "sif_count", "sif_density", "avg_sif_probability", "site_count", "priority_score"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col != "pattern" and col != "pattern_id" else ""
            
    # Apply density filter
    df = df[df["sif_density"] >= min_density]
    
    if df.empty:
        st.info("No recurring patterns match the selected filters.")
        return
        
    st.write(f"Showing **{len(df)}** patterns.")
    
    # Build display table
    display_df = df[required_cols].copy()
    display_df.columns = ["ID", "Pattern", "Occurrences", "SIF Count", "SIF Density", "Avg SIF Probability", "Affected Sites", "Prototype Prioritization Score"]
    
    st.dataframe(
        display_df.style.format({
            "SIF Density": "{:.1%}", 
            "Avg SIF Probability": "{:.1%}",
            "Prototype Prioritization Score": "{:.1f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("Pattern Detail")
    
    selected_id = st.selectbox("Select Pattern ID for deep dive", df["pattern_id"].tolist())
    
    if selected_id:
        try:
            detail_res = api.get_pattern_detail(selected_id)
            render_pattern_detail(detail_res.get("data", {}), df[df["pattern_id"] == selected_id].iloc[0].to_dict())
        except APIError:
            render_error_state("Could not load pattern details.")

def render_pattern_detail(detail: dict, table_row: dict):
    """Renders details for a specific pattern."""
    st.markdown(f"**Pattern Components:** `{table_row.get('pattern', 'N/A')}`")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Occurrences", detail.get("occurrences", 0))
    col2.metric("SIF Density", format_pct(detail.get("sif_density", 0.0)))
    
    sites_dict = detail.get("affected_sites", {})
    col3.metric("Affected Sites", len(sites_dict))
    
    score = detail.get("priority_score", 0.0)
    col4.metric("Prototype Prioritization Score", format_score(score), help="A prototype prioritization measure, not a validated industrial risk score.")
    
    st.markdown("### Score Breakdown")
    breakdown = detail.get("priority_components", {})
    st.json(breakdown)
    
    st.markdown("### Affected Sites")
    if sites_dict:
        # Convert dict to dataframe format
        site_list = [{"Site": k, "Occurrences": v} for k, v in sites_dict.items()]
        st.dataframe(pd.DataFrame(site_list), use_container_width=True, hide_index=True)
    else:
        st.write("No site data.")
