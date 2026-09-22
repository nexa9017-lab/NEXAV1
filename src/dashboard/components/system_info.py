"""
System / Model Info component.
"""
import streamlit as st
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import render_error_state

def render():
    st.title("System & Model Information")
    
    # 1. API Health
    st.subheader("Backend Health")
    is_healthy, health_data = api.check_health()
    
    if is_healthy:
        st.success("FastAPI Backend is ONLINE")
        health_payload = health_data.get("data", {})
        components = health_payload.get("components", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("SIF Model", components.get("sif_model", "unknown"))
        col2.metric("LSR Model", components.get("lsr_model", "unknown"))
        col3.metric("Extractor", components.get("precursor_extractor", "unknown"))
        col4.metric("Analytics", components.get("analytics", "unknown"))
    else:
        st.error("FastAPI Backend is OFFLINE")
        st.write("Ensure the backend is running: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`")
        return
        
    st.markdown("---")
    
    # 2. Model Info
    st.subheader("Model Configuration")
    try:
        model_info_res = api.get_models_info()
        model_info = model_info_res.get("data", {})
        st.json(model_info)
        
        st.info(f"**Prototype Notice:** {model_info.get('prototype_notice', 'N/A')}")
        st.write(f"**Synthetic Data:** {model_info.get('synthetic_data_notice', 'N/A')}")
    except APIError:
        render_error_state("Unable to load model metadata.")
        
    st.markdown("---")
    
    # 3. Analytics Metadata
    st.subheader("Analytics Cache Metadata")
    try:
        metadata_res = api.get_analytics_metadata()
        metadata = metadata_res.get("data", {})
        st.json(metadata)
    except APIError:
        render_error_state("Unable to load analytics metadata.")
