"""
Unit tests for SafetyPromptBuilder in src/rag/prompts.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.rag.prompts import SafetyPromptBuilder
from src.preprocessing.safety_taxonomy import LIFE_SAVING_RULES

FORBIDDEN_SYNTHETIC_LABELS = [
    "sif_label", "sif_potential", "primary_lsr", "lsr_tags",
    "actual_consequence", "severity", "precursor_tags"
]

@pytest.fixture
def prompt_builder():
    return SafetyPromptBuilder()

@pytest.fixture
def sample_retrieved_cases():
    return [
        {
            "rank": 1,
            "incident_id": "REP-12345678",
            "similarity_score": 0.8421,
            "site": "Refinery Alpha",
            "location": "Cracker Unit",
            "date": "2023-05-12",
            "report_type": "Near Miss",
            "description": "Technician cracked flange before depressurization was complete."
        },
        {
            "rank": 2,
            "incident_id": "REP-87654321",
            "similarity_score": 0.7915,
            "site": "Platform Bravo",
            "location": "Deck 2",
            "date": "2023-06-20",
            "report_type": "Unsafe Act",
            "description": "Worker loosened bolts on pressurized line without LOTO padlock."
        }
    ]

def test_system_prompt_contains_canonical_lsr_list(prompt_builder):
    sys_prompt = prompt_builder.build_system_prompt()
    for rule in LIFE_SAVING_RULES.keys():
        assert rule in sys_prompt, f"Canonical LSR '{rule}' missing from system prompt."

def test_system_prompt_asserts_new_incident_primary_truth(prompt_builder):
    sys_prompt = prompt_builder.build_system_prompt()
    assert "PRIMARY SOURCE OF TRUTH: The NEW INCIDENT is the primary source of truth" in sys_prompt
    assert "SUPPORTING CONTEXT ONLY" in sys_prompt

def test_system_prompt_prohibits_fact_copying(prompt_builder):
    sys_prompt = prompt_builder.build_system_prompt()
    assert "Never copy facts" in sys_prompt or "PROHIBITION ON FACT COPYING" in sys_prompt
    assert "must never overwrite or fabricate facts" in sys_prompt

def test_system_prompt_requests_json_only(prompt_builder):
    sys_prompt = prompt_builder.build_system_prompt()
    assert "Output valid, parseable JSON ONLY" in sys_prompt
    assert "Do not enclose in markdown blocks" in sys_prompt

def test_system_prompt_uncertainty_instructions(prompt_builder):
    sys_prompt = prompt_builder.build_system_prompt()
    assert "uncertainty_notes" in sys_prompt
    assert "ambiguous or absent" in sys_prompt

def test_analysis_prompt_includes_new_incident(prompt_builder, sample_retrieved_cases):
    new_inc = "Operator bypassed high pressure interlock on boiler B-12."
    prompt = prompt_builder.build_analysis_prompt(new_inc, sample_retrieved_cases)
    assert new_inc in prompt
    assert "NEW INCIDENT" in prompt

def test_analysis_prompt_includes_retrieved_ids_and_narratives(prompt_builder, sample_retrieved_cases):
    new_inc = "Worker unbolted flange under residual pressure."
    prompt = prompt_builder.build_analysis_prompt(new_inc, sample_retrieved_cases)
    
    assert "REP-12345678" in prompt
    assert "REP-87654321" in prompt
    assert "Technician cracked flange before depressurization was complete." in prompt
    assert "Worker loosened bolts on pressurized line without LOTO padlock." in prompt

def test_analysis_prompt_no_forbidden_labels(prompt_builder, sample_retrieved_cases):
    new_inc = "Technician identified leak at manifold."
    prompt = prompt_builder.build_analysis_prompt(new_inc, sample_retrieved_cases)
    
    prompt_lower = prompt.lower()
    for forbidden in FORBIDDEN_SYNTHETIC_LABELS:
        assert f"{forbidden}:" not in prompt_lower, f"Forbidden label '{forbidden}' leaked into prompt!"

def test_analysis_prompt_empty_new_incident_raises_error(prompt_builder):
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        prompt_builder.build_analysis_prompt("")
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        prompt_builder.build_analysis_prompt("   \t  ")
