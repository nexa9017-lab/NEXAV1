"""
Main Streamlit Application entry point for Safety Analytics Dashboard.
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Ensure the project root is in the Python path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.api_client import api
from src.dashboard.components import (
    overview,
    sif_analysis,
    lsr_analysis,
    pattern_analysis,
    site_risk,
    activity_analysis,
    emerging_trends,
    report_explorer,
    new_report,
    data_upload,
    system_info
)
from src.dashboard.components.shared import render_prototype_banner

def main():
    st.set_page_config(
        page_title="SIF Precursor Analytics",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Global Banner
    render_prototype_banner()
    
    # Sidebar Navigation
    st.sidebar.title("HSE Analytics")
    st.sidebar.markdown("---")
    
    # API Health Check First
    is_healthy, _ = api.check_health()
    if not is_healthy:
        st.error("Backend unavailable. Start the FastAPI server before using the dashboard.")
        st.stop() # Halt execution if backend is down
        
    pages = {
        "Executive Overview": overview.render,
        "SIF Analysis": sif_analysis.render,
        "Life-Saving Rule Analysis": lsr_analysis.render,
        "Precursor Patterns": pattern_analysis.render,
        "Site Risk": site_risk.render,
        "Activity Analysis": activity_analysis.render,
        "Emerging Trends": emerging_trends.render,
        "Report Explorer": report_explorer.render,
        "Analyse New Report": new_report.render,
        "Data Upload": data_upload.render,
        "System / Model Info": system_info.render
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    st.sidebar.markdown("---")
    
    # 5. Sidebar Features: Refresh and Debug
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
        
    debug_mode = st.sidebar.checkbox("Developer Debug Mode (DASHBOARD_DEBUG)", value=os.getenv("DASHBOARD_DEBUG", "false").lower() == "true")
    os.environ["DASHBOARD_DEBUG"] = "true" if debug_mode else "false"
    
    
    # Render selected page
    page_func = pages[selection]
    page_func()

if __name__ == "__main__":
    main()
