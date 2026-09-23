"""
Comprehensive integration tests for the Streamlit dashboard data pipeline (Phase R7).
Verifies: FastAPI responses -> API Client -> Transformers -> Render-ready dataframes/widgets.
Ensures strict contract enforcement and no silent null-to-zero fallbacks.
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.dashboard.api_client import SafetyAPIClient, APIError
from src.dashboard.transformers import (
    clean_entity_name,
    normalize_summary,
    normalize_metadata,
    normalize_sites,
    normalize_activities,
    normalize_lsr,
    normalize_patterns,
    normalize_unified_assessment,
)
from src.api.dependencies import (
    get_analytics_cache,
    get_rag_pipeline,
    get_embedder,
    _components_status,
)


@pytest.fixture
def mock_analytics_cache():
    """Provides standard mock analytics data matching backend schemas."""
    return {
        "overall_summary": {
            "metadata": {
                "generated_at": "2026-09-23T12:00:00Z",
                "source_dataset": "reports_test.csv",
                "report_count": 2500,
                "analytics_version": "1.0.0",
            },
            "data": {
                "total_reports": 2500,
                "total_predicted_sif_reports": 614,
                "overall_sif_density": 0.2456,
                "avg_sif_probability": 0.2926,
            },
        },
        "site_analytics": {
            "metadata": {},
            "data": [
                {
                    "name": "Refinery Alpha",
                    "total_reports": 500,
                    "sif_count": 140,
                    "sif_density": 0.28,
                },
                {
                    "name": "Platform Charlie",
                    "total_reports": 300,
                    "sif_count": 60,
                    "sif_density": 0.20,
                },
            ],
        },
        "activity_analytics": {
            "metadata": {},
            "data": [
                {
                    "name": "{'value': 'Maintenance', 'evidence': 'Maintenance', 'confidence': 'HIGH'}",
                    "total_reports": 210,
                    "sif_count": 65,
                    "sif_density": 0.3095,
                },
                {
                    "name": "Welding Operations",
                    "total_reports": 90,
                    "sif_count": 25,
                    "sif_density": 0.2778,
                },
            ],
        },
        "lsr_analytics": {
            "metadata": {},
            "data": [
                {
                    "name": "{'rule': 'Energy Isolation', 'score': 0.95}",
                    "total_reports": 180,
                    "sif_count": 72,
                },
                {
                    "name": "Line of Fire",
                    "total_reports": 120,
                    "sif_count": 45,
                },
            ],
        },
        "recurring_patterns": {
            "metadata": {},
            "data": [
                {
                    "pattern_id": "act_haz_1",
                    "pattern_type": "activity_hazard",
                    "components": {
                        "activity": "{'value': 'Operations'}",
                        "hazards": "{'value': 'Loss of Containment'}",
                    },
                    "occurrences": 12,
                    "sif_count": 10,
                    "sif_density": 0.8333,
                    "affected_sites": ["Site Alpha", "Platform Charlie"],
                }
            ],
        },
    }


@pytest.fixture
def integration_client(mock_analytics_cache):
    """TestClient wired with mock dependencies."""
    app.dependency_overrides[get_analytics_cache] = lambda: mock_analytics_cache
    _components_status["embedder"] = "ready"
    _components_status["vector_store"] = "ready"
    _components_status["retriever"] = "ready"
    _components_status["legacy_pipeline"] = "ready"
    _components_status["llm"] = "ready"
    _components_status["rag_pipeline"] = "ready"

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# -------------------------------------------------------------
# 1. Entity Name Cleaning Unit Tests
# -------------------------------------------------------------

def test_clean_entity_name_variations():
    """Verifies that clean_entity_name cleanly extracts strings without eval()."""
    # Plain string
    assert clean_entity_name("Maintenance") == "Maintenance"

    # Dict object
    assert clean_entity_name({"value": "Confined Space"}) == "Confined Space"
    assert clean_entity_name({"rule": "Energy Isolation"}) == "Energy Isolation"
    assert clean_entity_name({"name": "Hot Work"}) == "Hot Work"

    # String representation of dict (legacy compatibility fallback)
    legacy_act = "{'value': 'Maintenance', 'evidence': 'Maintenance', 'confidence': 'HIGH'}"
    assert clean_entity_name(legacy_act) == "Maintenance"

    legacy_rule = "{'rule': 'Bypassing Safety Controls', 'score': 0.97}"
    assert clean_entity_name(legacy_rule) == "Bypassing Safety Controls"

    # None and empty
    assert clean_entity_name(None) == "Unknown"
    assert clean_entity_name("") == ""


# -------------------------------------------------------------
# 2. Executive Summary Normalization Tests
# -------------------------------------------------------------

def test_normalize_summary_valid(mock_analytics_cache):
    """Verifies proper calculation of executive metrics from backend envelope."""
    norm = normalize_summary(mock_analytics_cache["overall_summary"])
    assert norm["total_reports"] == 2500
    assert norm["sif_reports"] == 614
    assert norm["non_sif_reports"] == 2500 - 614
    assert norm["sif_density"] == 0.2456


def test_normalize_summary_missing_fields_not_zero():
    """Verifies that missing fields yield None rather than fake zeros."""
    partial_payload = {"data": {"total_reports": 100}}  # sif_reports and density missing
    norm = normalize_summary(partial_payload)
    assert norm["total_reports"] == 100
    assert norm["sif_reports"] is None
    assert norm["non_sif_reports"] is None
    assert norm["sif_density"] is None


# -------------------------------------------------------------
# 3. Dimensional Analytics Normalization Tests
# -------------------------------------------------------------

def test_normalize_sites(mock_analytics_cache):
    """Verifies sites analytics normalization and sorting."""
    sites = normalize_sites(mock_analytics_cache["site_analytics"])
    assert len(sites) == 2
    assert sites[0]["site"] == "Refinery Alpha"
    assert sites[0]["total_reports"] == 500
    assert sites[0]["sif_reports"] == 140
    assert sites[0]["sif_density"] == 0.28


def test_normalize_activities(mock_analytics_cache):
    """Verifies activity analytics clean name unwrapping."""
    activities = normalize_activities(mock_analytics_cache["activity_analytics"])
    assert len(activities) == 2
    # First item had string dict representation; must be unwrapped to 'Maintenance'
    assert activities[0]["activity"] == "Maintenance"
    assert activities[0]["sif_reports"] == 65
    assert activities[1]["activity"] == "Welding Operations"


def test_normalize_lsr(mock_analytics_cache):
    """Verifies Life-Saving Rule analytics normalization."""
    lsr = normalize_lsr(mock_analytics_cache["lsr_analytics"])
    assert len(lsr) == 2
    assert lsr[0]["rule"] == "Energy Isolation"
    assert lsr[0]["mapped_reports"] == 180
    assert lsr[0]["sif_reports"] == 72


def test_normalize_patterns(mock_analytics_cache):
    """Verifies recurring precursor patterns normalization."""
    patterns = normalize_patterns(mock_analytics_cache["recurring_patterns"])
    assert len(patterns) == 1
    p = patterns[0]
    assert "Activity: Operations" in p["pattern"]
    assert "Hazards: Loss of Containment" in p["pattern"]
    assert p["occurrences"] == 12
    assert p["sif_reports"] == 10
    assert p["sif_density"] == 0.8333
    assert "Site Alpha" in p["affected_sites"]


# -------------------------------------------------------------
# 4. Unified Incident Assessment Normalization Tests
# -------------------------------------------------------------

def test_normalize_unified_assessment():
    """Verifies transformation of UnifiedSafetyAssessment for Streamlit display."""
    raw_assessment = {
        "incident_text": "Sample incident",
        "model_assessment": {
            "sif_prediction": True,
            "sif_probability": 0.82,
            "primary_lsr": "Energy Isolation",
            "activities": ["Maintenance"],
            "hazards": ["Pressure"],
        },
        "rag_assessment": {
            "sif_potential": True,
            "assessment_confidence": "high",
            "sif_explanation": "Opening pressurized flange presents severe SIF risk.",
            "life_saving_rules": ["Energy Isolation"],
            "activity": "Maintenance",
            "equipment": ["Discharge pump"],
            "hazards": ["Pressurized line"],
            "barrier_failures": ["Lockout tagout absent"],
            "unsafe_actions": ["Unbolting flange before verification"],
            "unsafe_conditions": ["Residual pressure"],
            "potential_consequences": ["High pressure fluid release"],
            "uncertainty_notes": ["Line pressure not stated in PSI"],
        },
        "similar_incidents": [
            {
                "rank": 1,
                "incident_id": "INC-001",
                "similarity_score": 0.88,
                "site": "Refinery Beta",
                "date": "2024-01-15",
                "report_type": "Near Miss",
                "description": "Technician cracked flange before zero energy check.",
            }
        ],
        "consistency": {
            "sif_agreement": True,
            "conflicts": [],
        },
        "warnings": [],
    }

    norm = normalize_unified_assessment(raw_assessment)

    # Legacy model check
    assert norm["model_assessment"]["sif_prediction"] is True
    assert norm["model_assessment"]["sif_probability"] == 0.82

    # RAG reasoning check
    assert norm["rag_assessment"]["sif_potential"] is True
    assert norm["rag_assessment"]["assessment_confidence"] == "High"
    assert "pressurized flange" in norm["rag_assessment"]["sif_explanation"]

    # Safety factors check
    sf = norm["safety_factors"]
    assert sf["Activity"] == "Maintenance"
    assert "Discharge pump" in sf["Equipment"]
    assert "Pressurized line" in sf["Hazards"]
    assert "Lockout tagout absent" in sf["Barrier Failures"]

    # Similar incidents table columns check
    sim = norm["similar_incidents"]
    assert len(sim) == 1
    assert sim[0]["Rank"] == 1
    assert sim[0]["Incident ID"] == "INC-001"
    assert sim[0]["Similarity"] == "88.0%"
    assert sim[0]["Site"] == "Refinery Beta"

    # Consistency check
    assert norm["consistency"]["sif_agreement"] is True


# -------------------------------------------------------------
# 5. Full End-to-End API Client to Transformers Contract Tests
# -------------------------------------------------------------

def test_api_client_analytics_flow(integration_client, monkeypatch):
    """Tests that SafetyAPIClient methods return real API responses and feed transformers."""
    api_client = SafetyAPIClient()

    # Monkeypatch api_client base methods to talk to TestClient
    monkeypatch.setattr(api_client, "_get", lambda endpoint, params=None, timeout=10.0: integration_client.get(endpoint, params=params).json())

    # Summary
    summary_raw = api_client.get_summary()
    summary = normalize_summary(summary_raw)
    assert summary["total_reports"] == 2500
    assert summary["sif_reports"] == 614

    # Sites
    sites_raw = api_client.get_sites()
    sites = normalize_sites(sites_raw)
    assert len(sites) == 2
    assert sites[0]["site"] == "Refinery Alpha"

    # Activities
    acts_raw = api_client.get_activities()
    acts = normalize_activities(acts_raw)
    assert len(acts) == 2
    assert acts[0]["activity"] == "Maintenance"

    # LSR
    lsr_raw = api_client.get_lsr()
    lsr = normalize_lsr(lsr_raw)
    assert len(lsr) == 2
    assert lsr[0]["rule"] == "Energy Isolation"

    # Patterns
    patterns_raw = api_client.get_patterns()
    patterns = normalize_patterns(patterns_raw)
    assert len(patterns) == 1
    assert "Operations" in patterns[0]["pattern"]

    # Metadata
    meta_raw = api_client.get_metadata()
    meta = normalize_metadata(meta_raw)
    assert meta["report_count"] == 2500
