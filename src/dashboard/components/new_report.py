"""
Analyse New Report component.
"""
import streamlit as st
from src.dashboard.api_client import api, APIError
from src.dashboard.components.shared import format_pct

def render():
    st.title("Analyse New Report")
    st.write("Submit a safety observation or incident description to evaluate SIF precursor potential and extract safety intelligence in real-time.")
    
    # Pre-fill examples
    examples = {
        "Energy Isolation": "Technician began opening the pump discharge flange before isolation was verified. Residual pressure remained in the line.",
        "Hot Work": "Welder continued cutting after the gas test validity period had expired.",
        "Lifting": "Worker entered the crane swing radius while a suspended load was being moved.",
        "Non-SIF": "The extinguisher inspection label is due for routine renewal next week."
    }
    
    st.markdown("**Example Inputs:**")
    cols = st.columns(len(examples))
    for i, (name, text) in enumerate(examples.items()):
        if cols[i].button(name):
            st.session_state["new_report_text"] = text
            
    # Form
    with st.form("new_report_form"):
        desc = st.text_area(
            "Describe the safety observation / near miss / incident",
            value=st.session_state.get("new_report_text", ""),
            height=150
        )
        
        c1, c2 = st.columns(2)
        with c1:
            site = st.text_input("Site (Optional)")
        with c2:
            report_type = st.selectbox("Report Type (Optional)", ["Unsafe Condition", "Unsafe Act", "Near Miss", "Incident"])
            
        submit = st.form_submit_button("Analyse Report")
        
    if submit:
        if not desc.strip():
            st.warning("Please enter a report description.")
            return
            
        with st.spinner("Analysing report..."):
            try:
                result_res = api.predict_single(description=desc, site=site, report_type=report_type)
                render_prediction_result(result_res.get("data", {}))
            except APIError as e:
                st.error(f"Prediction failed: {str(e)}")

def render_prediction_result(result: dict):
    st.markdown("---")
    st.subheader("Analysis Results")
    
    # SIF Assessment
    sif_prob = result.get("sif", {}).get("probability", 0.0)
    sif_label = result.get("sif", {}).get("label", "Unknown")
    
    color = "#dc3545" if sif_label == "SIF-Potential" else "#28a745"
    
    st.markdown(f"### SIF Assessment")
    st.markdown(f"<div style='padding:15px; border-radius:5px; background-color:#f8f9fa; border-left: 5px solid {color}'>"
                f"<h4 style='margin:0; color:{color}'>{sif_label}</h4>"
                f"<p style='margin:0'>Probability: <strong>{format_pct(sif_prob)}</strong></p>"
                f"</div>", unsafe_allow_html=True)
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    # LSRs
    st.markdown("### Life-Saving Rules")
    lsrs = result.get("life_saving_rules", [])
    if lsrs:
        df_lsr = pd.DataFrame(lsrs)
        st.dataframe(df_lsr.style.format({"score": "{:.1%}"}), hide_index=True)
        st.write(f"**Primary LSR:** {result.get('primary_lsr')}")
    else:
        st.write("No specific Life-Saving Rules detected above threshold.")
        
    st.markdown("### Extracted Safety Intelligence")
    
    def display_list(title, items):
        if items:
            st.markdown(f"**{title}:**")
            for item in items:
                # If it's a dict (like equipment), format it nicely
                if isinstance(item, dict):
                    val = item.get("value", "")
                    method = item.get("match_method", "")
                    st.write(f"- {val} _(via {method})_")
                else:
                    st.write(f"- {item}")
                    
    c1, c2 = st.columns(2)
    with c1:
        display_list("Activities", result.get("activities", []))
        display_list("Equipment", result.get("equipment", []))
        display_list("Hazards", result.get("hazards", []))
    with c2:
        display_list("Barrier Failures", result.get("barrier_failures", []))
        display_list("Potential Consequences", result.get("potential_consequences", []))
        display_list("Precursor Tags", result.get("precursor_tags", []))
        
    st.markdown("### Evidence")
    evidence_found = False
    for key in ["activities_evidence", "equipment_evidence", "hazards_evidence", "barrier_failures_evidence", "potential_consequences_evidence"]:
        ev = result.get(key, [])
        if ev:
            evidence_found = True
            name = key.replace("_evidence", "").replace("_", " ").title()
            st.markdown(f"**{name}:**")
            for e in ev:
                st.markdown(f"> *{e}*")
                
    if not evidence_found:
        st.write("No direct phrase evidence extracted.")
        
    with st.expander("Show Metadata"):
        st.json(result.get("inference_metadata", {}))

# Need pandas in this scope
import pandas as pd
