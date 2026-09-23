"""
Streamlit Application for OIL SIF Precursor Intelligence.
Strict one-page vertically flowing operational HSE dashboard.
Connects directly to the FastAPI unified safety backend.
"""

import io
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.api_client import api, APIError
from src.dashboard.transformers import (
    normalize_summary,
    normalize_metadata,
    normalize_sites,
    normalize_activities,
    normalize_lsr,
    normalize_patterns,
    normalize_unified_assessment,
)


def apply_hse_chart_style(fig):
    """Applies a clean, restrained styling to Plotly figures suitable for HSE dashboards."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="sans-serif", size=12, color="#212529"),
        title_font=dict(size=14, color="#1a202c"),
        height=320,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#edf2f7")
    fig.update_yaxes(showgrid=False)
    return fig


def main():
    st.set_page_config(
        page_title="OIL SIF Precursor Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -------------------------------------------------------------
    # Sidebar: Restrained to Status, Refresh, and Prototype Notice
    # -------------------------------------------------------------
    st.sidebar.title("System Status")
    is_healthy, health_data = api.check_health()
    if is_healthy:
        status_val = health_data.get("status", "ok")
        if status_val == "ok":
            st.sidebar.success("Backend: Online (Operational)")
        else:
            st.sidebar.warning(f"Backend: Degraded ({status_val})")
    else:
        st.sidebar.error("Backend: Offline")

    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**OIL Prototype Notice**:\nThis system is configured for HSE operations "
        "to evaluate SIF precursor patterns and support incident decision making."
    )

    # -------------------------------------------------------------
    # Backend Availability Check
    # -------------------------------------------------------------
    if not is_healthy:
        st.error("Backend unavailable. Start the FastAPI service and refresh the page.")
        st.stop()

    # -------------------------------------------------------------
    # Main Header
    # -------------------------------------------------------------
    st.title("OIL SIF Precursor Intelligence")
    st.caption("Operational safety intelligence for precursor detection, rule mapping, and incident analysis.")

    # -------------------------------------------------------------
    # Top Two-Column Layout: [ Upload Historical Dataset ] | [ Analyse New Incident ]
    # -------------------------------------------------------------
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Historical Dataset")
        st.write("Upload a CSV of safety reports to update vector retrieval and dataset analytics.")

        uploaded_file = st.file_uploader(
            "Upload Historical Reports (.csv)",
            type=["csv"],
            key="dataset_uploader",
            help="Select a CSV containing historical observations, near misses, and incidents.",
        )

        if uploaded_file is not None:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                df_preview = pd.read_csv(io.BytesIO(uploaded_bytes))
                num_rows = len(df_preview)
                cols_list = list(df_preview.columns)

                # Show ONLY required preview details
                st.markdown(f"**Filename:** `{uploaded_file.name}`")
                st.markdown(f"**Total Rows:** `{num_rows:,}` | **Columns:** `{len(cols_list)}`")
                st.caption(f"Detected columns: {', '.join(cols_list[:8])}{'...' if len(cols_list) > 8 else ''}")
                st.dataframe(df_preview.head(5), use_container_width=True, hide_index=True)

                if st.button("Process Dataset", type="primary", use_container_width=True):
                    with st.spinner("Processing safety reports..."):
                        try:
                            result = api.upload_dataset(
                                file_bytes=uploaded_bytes,
                                filename=uploaded_file.name,
                            )
                            indexed_count = result.get("records_processed", result.get("records_indexed", num_rows))
                            st.success(f"Dataset processed successfully — {indexed_count:,} reports available for analysis.")
                            st.cache_data.clear()
                            st.rerun()
                        except APIError as upload_err:
                            st.error(f"Processing failed: {str(upload_err)}")

            except Exception as read_err:
                st.error(f"Could not parse uploaded CSV: {str(read_err)}")

    with col_right:
        st.subheader("Analyse New Incident")
        st.write("Evaluate an incoming safety observation against historical intelligence.")

        # Quick example selector
        examples = {
            "Energy Isolation": "Technician opened pump discharge flange while residual pressure remained in line without verified zero energy state.",
            "Suspended Load": "Worker walked inside crane swing radius under suspended drill pipe during offloading operation.",
            "Hot Work": "Welder initiated pipe cutting without valid atmospheric gas test certificate in hazardous zone.",
        }

        ex_cols = st.columns(len(examples))
        for i, (ex_name, ex_text) in enumerate(examples.items()):
            if ex_cols[i].button(ex_name, use_container_width=True):
                st.session_state["incident_input"] = ex_text

        incident_text = st.text_area(
            "Incident Description",
            value=st.session_state.get("incident_input", ""),
            height=130,
            placeholder="Enter incident, near miss, or observation description...",
        )

        c_btn, _ = st.columns([1, 2])
        analyze_clicked = c_btn.button("Analyse Incident", type="primary", use_container_width=True)

        if analyze_clicked:
            if not incident_text.strip():
                st.warning("Please enter an incident description.")
            else:
                with st.spinner("Analyzing incident..."):
                    try:
                        raw_analysis = api.analyze_incident(description=incident_text.strip(), top_k=5)
                        st.session_state["active_analysis"] = normalize_unified_assessment(raw_analysis)
                    except APIError as e:
                        st.error(f"Analysis failed: {str(e)}")

        # -------------------------------------------------------------
        # Real-Time Incident Assessment Results (Underneath Input)
        # -------------------------------------------------------------
        if "active_analysis" in st.session_state:
            ass = st.session_state["active_analysis"]
            st.markdown("---")
            st.markdown("#### Incident Assessment")

            # 1. SIF Assessment (Legacy Model vs RAG)
            m_sif = ass.get("model_assessment", {})
            r_sif = ass.get("rag_assessment", {})

            col_m, col_r = st.columns(2)
            with col_m:
                pred_label = "SIF-Potential" if m_sif.get("sif_prediction") else "Non-SIF"
                pred_color = "#e53e3e" if m_sif.get("sif_prediction") else "#38a169"
                st.markdown(
                    f"<div style='border-left: 4px solid {pred_color}; padding: 8px 12px; background: #f7fafc;'>"
                    f"<div style='font-size: 11px; text-transform: uppercase; color: #718096;'>Legacy Model</div>"
                    f"<div style='font-size: 16px; font-weight: bold; color: {pred_color};'>{pred_label}</div>"
                    f"<div style='font-size: 12px; color: #4a5568;'>Probability: {m_sif.get('sif_probability', 0.0):.1%}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_r:
                rag_label = "SIF-Potential" if r_sif.get("sif_potential") else "Non-SIF"
                rag_color = "#e53e3e" if r_sif.get("sif_potential") else "#38a169"
                conf = r_sif.get("assessment_confidence", "Medium")
                st.markdown(
                    f"<div style='border-left: 4px solid {rag_color}; padding: 8px 12px; background: #f7fafc;'>"
                    f"<div style='font-size: 11px; text-transform: uppercase; color: #718096;'>RAG Reasoning</div>"
                    f"<div style='font-size: 16px; font-weight: bold; color: {rag_color};'>{rag_label}</div>"
                    f"<div style='font-size: 12px; color: #4a5568;'>Confidence: {conf}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Consistency Warning
            consistency = ass.get("consistency", {})
            if not consistency.get("sif_agreement", True):
                st.warning("⚠️ **Model & RAG Disagreement**: Legacy classifier and RAG reasoning produced conflicting SIF determinations.")

            # 2. Relevant Life-Saving Rules
            lsrs = r_sif.get("life_saving_rules", [])
            if lsrs:
                st.markdown(f"**Life-Saving Rules:** {', '.join(lsrs)}")

            # 3. Why This Was Flagged
            st.markdown("**Why This Was Flagged:**")
            st.markdown(f"> {r_sif.get('sif_explanation', 'No explanation provided.')}")

            # 4. Safety Factors
            sf = ass.get("safety_factors", {})
            df_sf = pd.DataFrame([{"Factor": k, "Extracted Intelligence": v} for k, v in sf.items()])
            st.dataframe(df_sf, use_container_width=True, hide_index=True)

            # 5. Similar Historical Incidents
            sim_cases = ass.get("similar_incidents", [])
            if sim_cases:
                st.markdown("**Similar Historical Incidents:**")
                st.dataframe(pd.DataFrame(sim_cases), use_container_width=True, hide_index=True)

            # 6. Uncertainties & Limitations
            uncert = r_sif.get("uncertainty_notes", [])
            if uncert:
                st.caption(f"**Limitations & Missing Details:** {'; '.join(uncert)}")

    # -------------------------------------------------------------
    # Active Dataset Status Banner
    # -------------------------------------------------------------
    st.markdown("---")
    try:
        raw_meta = api.get_metadata()
        meta = normalize_metadata(raw_meta)
    except Exception:
        meta = {"filename": "Historical Dataset", "report_count": None, "last_updated": None}

    report_count_display = f"{meta['report_count']:,}" if meta["report_count"] is not None else "Data unavailable"
    updated_display = meta["last_updated"] if meta["last_updated"] else "Unknown"

    st.markdown(
        f"<div style='background-color: #edf2f7; padding: 10px 16px; border-radius: 6px; font-size: 13px; color: #2d3748;'>"
        f"<strong>Active Dataset:</strong> <code>{meta['filename']}</code> &nbsp;|&nbsp; "
        f"<strong>Reports:</strong> <code>{report_count_display}</code> &nbsp;|&nbsp; "
        f"<strong>Last Updated:</strong> <code>{updated_display}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Executive Summary (Exactly 4 metrics, no extra cards)
    # -------------------------------------------------------------
    st.subheader("Executive Summary")

    try:
        raw_summary = api.get_summary()
        summary = normalize_summary(raw_summary)
    except APIError:
        summary = {"total_reports": None, "sif_reports": None, "non_sif_reports": None, "sif_density": None}

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        tot_val = f"{summary['total_reports']:,}" if summary["total_reports"] is not None else "Data unavailable"
        st.metric("Total Reports", tot_val)

    with m2:
        sif_val = f"{summary['sif_reports']:,}" if summary["sif_reports"] is not None else "Data unavailable"
        st.metric("SIF-Potential Reports", sif_val)

    with m3:
        non_val = f"{summary['non_sif_reports']:,}" if summary["non_sif_reports"] is not None else "Data unavailable"
        st.metric("Non-SIF Reports", non_val)

    with m4:
        dens_val = f"{summary['sif_density']:.1%}" if summary["sif_density"] is not None else "Data unavailable"
        st.metric("SIF Precursor Density", dens_val)

    st.markdown("---")

    # -------------------------------------------------------------
    # SIF Precursor Density by Site
    # -------------------------------------------------------------
    st.subheader("SIF Precursor Density by Site")

    try:
        raw_sites = api.get_sites()
        sites_data = normalize_sites(raw_sites)
    except APIError:
        sites_data = []

    if sites_data:
        df_sites = pd.DataFrame(sites_data)
        col_s_table, col_s_chart = st.columns([1, 1], gap="medium")
        with col_s_table:
            display_df_sites = df_sites.rename(columns={
                "site": "Site",
                "total_reports": "Total Reports",
                "sif_reports": "SIF Reports",
                "sif_density": "SIF Density",
            })
            display_df_sites["SIF Density"] = display_df_sites["SIF Density"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df_sites, use_container_width=True, hide_index=True)
        with col_s_chart:
            fig_sites = px.bar(
                df_sites.head(10).sort_values("sif_reports", ascending=True),
                x="sif_reports",
                y="site",
                orientation="h",
                title="Top Sites by SIF Report Count",
                color_discrete_sequence=["#e53e3e"],
                labels={"sif_reports": "SIF Reports", "site": "Site"},
            )
            st.plotly_chart(apply_hse_chart_style(fig_sites), use_container_width=True)
    else:
        st.info("No site analytics data available.")

    st.markdown("---")

    # -------------------------------------------------------------
    # SIF Precursor Density by Activity
    # -------------------------------------------------------------
    st.subheader("SIF Precursor Density by Activity")

    try:
        raw_activities = api.get_activities()
        activities_data = normalize_activities(raw_activities)
    except APIError:
        activities_data = []

    if activities_data:
        df_acts = pd.DataFrame(activities_data)
        col_a_table, col_a_chart = st.columns([1, 1], gap="medium")
        with col_a_table:
            display_df_acts = df_acts.rename(columns={
                "activity": "Activity",
                "total_reports": "Total Reports",
                "sif_reports": "SIF Reports",
                "sif_density": "SIF Density",
            })
            display_df_acts["SIF Density"] = display_df_acts["SIF Density"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df_acts.head(15), use_container_width=True, hide_index=True)
        with col_a_chart:
            fig_acts = px.bar(
                df_acts.head(10).sort_values("sif_reports", ascending=True),
                x="sif_reports",
                y="activity",
                orientation="h",
                title="Top Activities Associated with SIF Precursors",
                color_discrete_sequence=["#dd6b20"],
                labels={"sif_reports": "SIF Reports", "activity": "Activity"},
            )
            st.plotly_chart(apply_hse_chart_style(fig_acts), use_container_width=True)
    else:
        st.info("No activity analytics data available.")

    st.markdown("---")

    # -------------------------------------------------------------
    # Life-Saving Rule Mapping
    # -------------------------------------------------------------
    st.subheader("Life-Saving Rule Mapping")

    try:
        raw_lsr = api.get_lsr()
        lsr_data = normalize_lsr(raw_lsr)
    except APIError:
        lsr_data = []

    if lsr_data:
        df_lsr = pd.DataFrame(lsr_data)
        col_l_table, col_l_chart = st.columns([1, 1], gap="medium")
        with col_l_table:
            display_df_lsr = df_lsr.rename(columns={
                "rule": "Life-Saving Rule",
                "mapped_reports": "Mapped Reports",
                "sif_reports": "SIF Reports",
            })
            st.dataframe(display_df_lsr, use_container_width=True, hide_index=True)
        with col_l_chart:
            fig_lsr = px.bar(
                df_lsr.head(10).sort_values("mapped_reports", ascending=True),
                x="mapped_reports",
                y="rule",
                orientation="h",
                title="Reports Mapped to Life-Saving Rules",
                color_discrete_sequence=["#3182ce"],
                labels={"mapped_reports": "Mapped Reports", "rule": "Rule"},
            )
            st.plotly_chart(apply_hse_chart_style(fig_lsr), use_container_width=True)
    else:
        st.info("No Life-Saving Rule data available.")

    st.markdown("---")

    # -------------------------------------------------------------
    # Recurring SIF Precursor Patterns
    # -------------------------------------------------------------
    st.subheader("Recurring SIF Precursor Patterns")

    try:
        raw_patterns = api.get_patterns()
        patterns_data = normalize_patterns(raw_patterns)
    except APIError:
        patterns_data = []

    if patterns_data:
        df_patterns = pd.DataFrame(patterns_data).rename(columns={
            "pattern": "Precursor Pattern",
            "occurrences": "Occurrences",
            "sif_reports": "SIF Reports",
            "sif_density": "SIF Density",
            "affected_sites": "Affected Sites",
        })
        df_patterns["SIF Density"] = df_patterns["SIF Density"].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_patterns, use_container_width=True, hide_index=True)
    else:
        st.info("No recurring precursor patterns identified for current threshold.")


if __name__ == "__main__":
    main()
