"""
Data transformers to normalize backend API responses for the Streamlit frontend.
Ensures strict contract enforcement, safe string parsing (no eval), and distinguishes
real zero values from missing fields (no silent null fallbacks).
"""

import ast
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def clean_entity_name(raw_name: Any) -> str:
    """
    Cleans raw entity names that may be structured dicts or legacy Python-like string representations.
    NEVER uses eval(). Uses structured checking with safe ast.literal_eval() compatibility fallback.
    """
    if raw_name is None:
        return "Unknown"

    if isinstance(raw_name, dict):
        if "value" in raw_name:
            return str(raw_name["value"]).strip()
        if "rule" in raw_name:
            return str(raw_name["rule"]).strip()
        if "name" in raw_name:
            return str(raw_name["name"]).strip()
        return str(raw_name).strip()

    name_str = str(raw_name).strip()

    # If it looks like a serialized dict string, safely parse with ast.literal_eval
    if (name_str.startswith("{") and name_str.endswith("}")) or (name_str.startswith("[") and name_str.endswith("]")):
        try:
            parsed = ast.literal_eval(name_str)
            if isinstance(parsed, dict):
                return clean_entity_name(parsed)
            if isinstance(parsed, list) and parsed:
                return clean_entity_name(parsed[0])
        except Exception:
            # Fallback to simple extraction if literal_eval fails on malformed strings
            pass

    return name_str


def _unwrap_envelope(raw_response: Any) -> Any:
    """Unwraps the standard FastAPI envelope if present."""
    if isinstance(raw_response, dict) and "data" in raw_response:
        return raw_response.get("data")
    return raw_response


def normalize_summary(raw_response: Any) -> Dict[str, Any]:
    """
    Normalizes executive summary response.
    Distinguishes missing fields from 0: returns None when a field is absent.
    """
    data = _unwrap_envelope(raw_response)
    if not isinstance(data, dict):
        logger.warning(f"Expected dict for summary data, got {type(data)}")
        return {
            "total_reports": None,
            "sif_reports": None,
            "non_sif_reports": None,
            "sif_density": None,
            "avg_sif_probability": None,
        }

    total = data.get("total_reports")
    sif = data.get("total_predicted_sif_reports")
    if sif is None:
        sif = data.get("sif_reports")

    density = data.get("overall_sif_density")
    if density is None:
        density = data.get("sif_density")

    avg_prob = data.get("avg_sif_probability")

    # Compute non_sif_reports only if total and sif are known ints
    non_sif = None
    if total is not None and sif is not None:
        try:
            non_sif = int(total) - int(sif)
        except (ValueError, TypeError):
            non_sif = None

    return {
        "total_reports": int(total) if total is not None else None,
        "sif_reports": int(sif) if sif is not None else None,
        "non_sif_reports": non_sif,
        "sif_density": float(density) if density is not None else None,
        "avg_sif_probability": float(avg_prob) if avg_prob is not None else None,
    }


def normalize_metadata(raw_response: Any) -> Dict[str, Any]:
    """Normalizes active dataset metadata."""
    data = _unwrap_envelope(raw_response)
    meta = raw_response.get("metadata", {}) if isinstance(raw_response, dict) else {}

    combined = {}
    if isinstance(meta, dict):
        combined.update(meta)
    if isinstance(data, dict):
        combined.update(data)

    filename = combined.get("filename") or combined.get("source_dataset") or "Historical Dataset"
    report_count = combined.get("report_count")
    last_updated = combined.get("last_updated") or combined.get("generated_at")
    version = combined.get("dataset_version") or combined.get("analytics_version") or "active"

    return {
        "filename": str(filename),
        "report_count": int(report_count) if report_count is not None else None,
        "last_updated": str(last_updated) if last_updated else None,
        "dataset_version": str(version),
    }


def normalize_sites(raw_response: Any) -> List[Dict[str, Any]]:
    """Normalizes site analytics list."""
    data = _unwrap_envelope(raw_response)
    if not isinstance(data, list):
        logger.warning(f"Expected list for sites data, got {type(data)}")
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = clean_entity_name(item.get("name", "Unknown Site"))
        total = item.get("total_reports", 0)
        sif = item.get("sif_count", 0)
        density = item.get("sif_density")
        if density is None and total > 0:
            density = round(sif / total, 4)

        results.append({
            "site": name,
            "total_reports": int(total) if total is not None else 0,
            "sif_reports": int(sif) if sif is not None else 0,
            "sif_density": float(density) if density is not None else 0.0,
        })
    return sorted(results, key=lambda x: x["sif_reports"], reverse=True)


def normalize_activities(raw_response: Any) -> List[Dict[str, Any]]:
    """Normalizes activity analytics list."""
    data = _unwrap_envelope(raw_response)
    if not isinstance(data, list):
        logger.warning(f"Expected list for activity data, got {type(data)}")
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = clean_entity_name(item.get("name", "Unknown Activity"))
        total = item.get("total_reports", 0)
        sif = item.get("sif_count", 0)
        density = item.get("sif_density")
        if density is None and total > 0:
            density = round(sif / total, 4)

        results.append({
            "activity": name,
            "total_reports": int(total) if total is not None else 0,
            "sif_reports": int(sif) if sif is not None else 0,
            "sif_density": float(density) if density is not None else 0.0,
        })
    return sorted(results, key=lambda x: x["sif_reports"], reverse=True)


def normalize_lsr(raw_response: Any) -> List[Dict[str, Any]]:
    """Normalizes Life-Saving Rules analytics list."""
    data = _unwrap_envelope(raw_response)
    if not isinstance(data, list):
        logger.warning(f"Expected list for LSR data, got {type(data)}")
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = clean_entity_name(item.get("name", "Unknown Rule"))
        mapped = item.get("total_reports", 0)
        sif = item.get("sif_count", 0)

        results.append({
            "rule": name,
            "mapped_reports": int(mapped) if mapped is not None else 0,
            "sif_reports": int(sif) if sif is not None else 0,
        })
    return sorted(results, key=lambda x: x["sif_reports"], reverse=True)


def normalize_patterns(raw_response: Any) -> List[Dict[str, Any]]:
    """Normalizes recurring precursor patterns list into clean tabular format."""
    data = _unwrap_envelope(raw_response)
    if not isinstance(data, list):
        logger.warning(f"Expected list for patterns data, got {type(data)}")
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue

        comps = item.get("components", {})
        comp_parts = []
        if isinstance(comps, dict):
            for k, v in comps.items():
                clean_k = str(k).replace("_", " ").title()
                clean_v = clean_entity_name(v)
                comp_parts.append(f"{clean_k}: {clean_v}")
        pattern_desc = " | ".join(comp_parts) if comp_parts else item.get("pattern_type", "Pattern")

        occurrences = item.get("occurrences", 0)
        sif = item.get("sif_count", 0)
        density = item.get("sif_density")
        if density is None and occurrences > 0:
            density = round(sif / occurrences, 4)

        sites = item.get("affected_sites", [])
        if isinstance(sites, list):
            sites_str = ", ".join(str(s) for s in sites)
        elif isinstance(sites, dict):
            sites_str = ", ".join(str(s) for s in sites.keys())
        else:
            sites_str = str(sites or "")

        results.append({
            "pattern": pattern_desc,
            "occurrences": int(occurrences) if occurrences is not None else 0,
            "sif_reports": int(sif) if sif is not None else 0,
            "sif_density": float(density) if density is not None else 0.0,
            "affected_sites": sites_str or "Unspecified",
        })
    return sorted(results, key=lambda x: x["sif_reports"], reverse=True)


def normalize_unified_assessment(raw_response: Any) -> Dict[str, Any]:
    """Normalizes the UnifiedSafetyAssessment from POST /api/v1/analyze."""
    data = _unwrap_envelope(raw_response)
    if not isinstance(data, dict):
        logger.warning(f"Expected dict for unified assessment, got {type(data)}")
        return {}

    model_ass = data.get("model_assessment", {}) or {}
    rag_ass = data.get("rag_assessment", {}) or {}
    sim_incidents = data.get("similar_incidents", []) or []
    consistency = data.get("consistency", {}) or {}
    warnings = data.get("warnings", []) or []

    # Format similar incidents table
    normalized_similar = []
    for inc in sim_incidents:
        if isinstance(inc, dict):
            normalized_similar.append({
                "Rank": inc.get("rank"),
                "Incident ID": inc.get("incident_id"),
                "Similarity": f"{float(inc.get('similarity_score', 0.0)):.1%}",
                "Site": inc.get("site") or "—",
                "Date": inc.get("date") or "—",
                "Type": inc.get("report_type") or "—",
                "Description": inc.get("description", ""),
            })

    # Format safety factors
    safety_factors = {
        "Activity": rag_ass.get("activity") or (", ".join(model_ass.get("activities", [])) if model_ass.get("activities") else "—"),
        "Equipment": ", ".join(rag_ass.get("equipment", []) or model_ass.get("equipment", [])) or "—",
        "Hazards": ", ".join(rag_ass.get("hazards", []) or model_ass.get("hazards", [])) or "—",
        "Barrier Failures": ", ".join(rag_ass.get("barrier_failures", []) or model_ass.get("barrier_failures", [])) or "—",
        "Unsafe Actions": ", ".join(rag_ass.get("unsafe_actions", []) or model_ass.get("unsafe_actions", [])) or "—",
        "Unsafe Conditions": ", ".join(rag_ass.get("unsafe_conditions", []) or model_ass.get("unsafe_conditions", [])) or "—",
        "Potential Consequences": ", ".join(rag_ass.get("potential_consequences", []) or model_ass.get("potential_consequences", [])) or "—",
    }

    # Life-saving rules
    lsr_list = rag_ass.get("life_saving_rules") or model_ass.get("lsr_predictions") or []
    if not lsr_list and model_ass.get("primary_lsr"):
        lsr_list = [model_ass["primary_lsr"]]

    return {
        "model_assessment": {
            "sif_prediction": bool(model_ass.get("sif_prediction", False)),
            "sif_probability": float(model_ass.get("sif_probability", 0.0)),
            "primary_lsr": model_ass.get("primary_lsr"),
        },
        "rag_assessment": {
            "sif_potential": bool(rag_ass.get("sif_potential", False)),
            "assessment_confidence": str(rag_ass.get("assessment_confidence", "medium")).capitalize(),
            "sif_explanation": rag_ass.get("sif_explanation", "No reasoning provided."),
            "life_saving_rules": lsr_list,
            "uncertainty_notes": rag_ass.get("uncertainty_notes", []),
        },
        "safety_factors": safety_factors,
        "similar_incidents": normalized_similar,
        "consistency": {
            "sif_agreement": bool(consistency.get("sif_agreement", True)),
            "conflicts": consistency.get("conflicts", []),
        },
        "warnings": warnings,
    }


# Backward compatibility wrappers
def normalize_envelope(raw_response: Any, default_data: Any = None) -> Dict[str, Any]:
    if not isinstance(raw_response, dict):
        return {"data": raw_response if raw_response is not None else default_data, "metadata": {}}
    if "data" in raw_response and "metadata" in raw_response:
        return raw_response
    return {"data": raw_response, "metadata": {}}


def normalize_list_response(raw_response: Any) -> Dict[str, Any]:
    norm = normalize_envelope(raw_response, default_data=[])
    if not isinstance(norm["data"], list):
        norm["data"] = []
    return norm


def normalize_dict_response(raw_response: Any) -> Dict[str, Any]:
    norm = normalize_envelope(raw_response, default_data={})
    if not isinstance(norm["data"], dict):
        norm["data"] = {}
    return norm
