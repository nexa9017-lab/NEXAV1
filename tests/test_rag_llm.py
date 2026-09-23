"""
Unit tests for LLM clients and assessment parser in src/rag/llm_client.py.
Uses MockLLMClient exclusively; makes zero external network calls.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
from src.rag.llm_client import MockLLMClient, GroqLLMClient, LLMResponseError, parse_assessment_response
from src.api.schemas import LLMIncidentAssessment

@pytest.fixture
def valid_assessment_payload():
    return {
        "sif_potential": True,
        "assessment_confidence": "high",
        "sif_explanation": "Operating unisolated equipment under pressure poses severe fatality risk.",
        "life_saving_rules": ["Energy Isolation", "Line of Fire"],
        "activity": "Maintenance",
        "equipment": ["discharge flange", "pump"],
        "hazards": ["pressurized hydrocarbon", "stored energy"],
        "barrier_failures": ["isolation not verified", "padlock omitted"],
        "unsafe_actions": ["loosened bolts before zero energy verification"],
        "unsafe_conditions": ["residual pressure in pipe"],
        "potential_consequences": ["high pressure spray", "fatal strike"],
        "recurring_precursor_pattern": "Maintenance on unisolated pressurized lines",
        "incident_evidence": ["Technician opened flange before isolation was verified."],
        "historical_pattern_observations": [
            {
                "observation": "Historical cases show frequent flange unbolting before isolation checks.",
                "supporting_incident_ids": ["REP-12345678", "REP-87654321"]
            }
        ],
        "uncertainty_notes": ["Operating temperature not documented."]
    }

def test_mock_llm_client_generation(valid_assessment_payload):
    json_str = json.dumps(valid_assessment_payload)
    client = MockLLMClient(response=json_str)
    
    output = client.generate("test prompt", system_prompt="test system")
    assert output == json_str
    assert client.last_prompt == "test prompt"
    assert client.last_system_prompt == "test system"

def test_parse_valid_json(valid_assessment_payload):
    json_str = json.dumps(valid_assessment_payload)
    assessment = parse_assessment_response(json_str)
    
    assert isinstance(assessment, LLMIncidentAssessment)
    assert assessment.sif_potential is True
    assert assessment.assessment_confidence == "high"
    assert "Energy Isolation" in assessment.life_saving_rules
    assert len(assessment.equipment) == 2
    assert len(assessment.historical_pattern_observations) == 1
    assert assessment.historical_pattern_observations[0].supporting_incident_ids == ["REP-12345678", "REP-87654321"]

def test_parse_markdown_wrapped_json(valid_assessment_payload):
    json_str = json.dumps(valid_assessment_payload)
    wrapped = f"```json\n{json_str}\n```"
    
    assessment = parse_assessment_response(wrapped)
    assert isinstance(assessment, LLMIncidentAssessment)
    assert assessment.sif_potential is True

def test_parse_invalid_json_raises_error():
    bad_json = "This is not valid JSON at all."
    with pytest.raises(LLMResponseError, match="Failed to parse LLM response as JSON"):
        parse_assessment_response(bad_json)

def test_parse_empty_string_raises_error():
    with pytest.raises(LLMResponseError, match="Received empty response from LLM"):
        parse_assessment_response("")

def test_missing_required_field_raises_error(valid_assessment_payload):
    del valid_assessment_payload["sif_potential"]
    json_str = json.dumps(valid_assessment_payload)
    
    with pytest.raises(LLMResponseError, match="failed schema validation"):
        parse_assessment_response(json_str)

def test_invalid_confidence_literal_raises_error(valid_assessment_payload):
    valid_assessment_payload["assessment_confidence"] = "very_high" # Not in ('high', 'medium', 'low')
    json_str = json.dumps(valid_assessment_payload)
    
    with pytest.raises(LLMResponseError, match="failed schema validation"):
        parse_assessment_response(json_str)

def test_invalid_life_saving_rule_raises_error(valid_assessment_payload):
    valid_assessment_payload["life_saving_rules"] = ["General Safety Guidelines", "Energy Isolation"]
    json_str = json.dumps(valid_assessment_payload)
    
    with pytest.raises(LLMResponseError, match="Unsupported Life-Saving Rule 'General Safety Guidelines'"):
        parse_assessment_response(json_str)

def test_null_optional_activity_succeeds(valid_assessment_payload):
    valid_assessment_payload["activity"] = None
    json_str = json.dumps(valid_assessment_payload)
    
    assessment = parse_assessment_response(json_str)
    assert assessment.activity is None

def test_groq_client_missing_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Groq API key not found"):
        GroqLLMClient(api_key=None)
