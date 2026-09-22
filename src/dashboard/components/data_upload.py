"""
Data Upload dashboard component.
"""
import streamlit as st
import pandas as pd
import io
from src.dashboard.api_client import api, APIError

def render():
    st.title("Data Upload & Processing")
    st.write("Upload your own historical safety reports to generate insights. The backend AI pipeline will process your data and rebuild the analytics dashboard.")
    
    # Current Dataset Metadata
    st.markdown("### Active Dataset")
    try:
        metadata = api.get_analytics_metadata()
        c1, c2, c3 = st.columns(3)
        c1.metric("Source", metadata.get("dataset_source", "unknown").replace("_", " ").title())
        c2.metric("Version", metadata.get("dataset_version", "unknown"))
        c3.metric("Rows", metadata.get("row_count", "N/A"))
        
        if metadata.get("dataset_source") != "synthetic_baseline":
            if st.button("Restore Original Synthetic Dataset", type="secondary"):
                with st.spinner("Restoring baseline..."):
                    try:
                        api.reset_dataset()
                        st.success("Successfully restored original demo dataset. Insights have been refreshed.")
                        st.rerun()
                    except APIError as e:
                        st.error(f"Failed to restore: {str(e)}")
    except APIError:
        st.warning("Could not load current dataset metadata.")

    st.markdown("---")
    
    st.markdown("### Upload New Dataset")
    st.markdown("""
    **Required Column:**
    - `description`: The free-text narrative of the safety observation.
    
    **Optional Columns:**
    - `report_id`, `date`, `site`, `location`, `report_type`
    
    *(Max 5,000 rows. Processing may take a few minutes depending on file size).*
    """)
    
    uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Preview the file
            file_bytes = uploaded_file.getvalue()
            df_preview = pd.read_csv(io.BytesIO(file_bytes))
            
            st.markdown(f"**Filename:** `{uploaded_file.name}`")
            st.markdown(f"**Rows Found:** `{len(df_preview)}`")
            st.markdown(f"**Columns:** `{', '.join(df_preview.columns)}`")
            
            st.dataframe(df_preview.head(5), use_container_width=True)
            
            # Confirmation
            confirm = st.checkbox("I understand this will replace the currently active dashboard dataset.")
            
            if confirm:
                if st.button("Process Dataset", type="primary"):
                    with st.status("Processing Upload...", expanded=True) as status:
                        try:
                            st.write("Uploading dataset...")
                            st.write("Running AI inference... *(This may take a minute)*")
                            st.write("Rebuilding analytics...")
                            st.write("Refreshing dashboard cache...")
                            
                            result = api.upload_csv(file_bytes, uploaded_file.name)
                            
                            status.update(label="Upload and Processing Complete!", state="complete", expanded=False)
                            
                            st.success("Dashboard insights have been successfully refreshed!")
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Rows Processed", result.get("rows_processed", 0))
                            c2.metric("SIF Precursors Detected", result.get("sif_reports", 0))
                            c3.metric("SIF Density", f"{result.get('sif_density', 0):.1%}")
                            
                        except APIError as e:
                            status.update(label="Processing Failed", state="error", expanded=True)
                            st.error(str(e))
                            
        except Exception as e:
            st.error(f"Failed to read CSV preview: {e}")
