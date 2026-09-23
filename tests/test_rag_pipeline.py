"""
Unit tests for the Unified Safety RAG Pipeline (Phase R5).
All tests execute offline using deterministic mocks; no network calls or Groq API keys required.
"""
import json
import pytest

from src.api.schemas import (
    LegacyModelAssessment,
    RetrievedIncident,
    AssessmentConsistency,
    UnifiedSafetyAssessment,
    LLMIncidentAssessment,
)
from src.rag.prompts import SafetyPromptBuilder
from src.rag.llm_client import MockLLMClient
from src.rag.pipeline import SafetyRAGPipeline


class DummyLegacyPipeline:
    """Mock legacy inference pipeline returning structured prediction output."""
    def __init__(self, sif_label="SIF-Potential", primary_lsr="Energy Isolation", should_fail=False):
        self.sif_label = sif_label
        self.primary_lsr = primary_lsr
        self.should_fail = should_fail

    def predict_single(self, report):
        if self.should_fail:
            raise RuntimeError("Legacy model inference engine encountered internal error.")

        return {
            "normalized_text": report.get("description", ""),
            "sif": {
                "label": self.sif_label,
                "probability": 0.82 if self.sif_label == "SIF-Potential" else 0.15,
            },
            "life_saving_rules": [
                {"rule": self.primary_lsr, "score": 0.79}
            ],
            "primary_lsr": self.primary_lsr,
            "activities": [{"value": "Maintenance"}],
            "equipment": [{"value": "Pump"}, {"value": "Flange"}],
            "hazards": [{"value": "Stored Energy"}],
            "barrier_failures": [{"value": "ENERGY_ISOLATION_FAILURE"}],
            "unsafe_actions": [{"value": "Opened flange prematurely"}],
            "unsafe_conditions": [{"value": "Residual pressure in line"}],
            "potential_consequences": [{"value": "Major Process Safety Event"}],
            "precursor_tags": ["stored-energy", "missing-loto"],
            "latency_ms": 15.2,
        }


class DummyRetriever:
    """Mock retriever returning deterministic historical cases."""
    def __init__(self, cases=None, should_fail=False):
        self.should_fail = should_fail
        if cases is not None:
            self.cases = cases
        else:
            self.cases = [
                {
                    "rank": 1,
                    "vector_id": 101,
                    "incident_id": "REP-HIST-001",
                    "similarity_score": 0.88,
                    "description": "Historical case 1: pipe pressurized during flange loosening.",
                    "site": "Plant North",
                    "location": "Unit 2",
                    "date": "2023-05-12",
                    "report_type": "Near Miss",
                },
                {
                    "rank": 2,
                    "vector_id": 102,
                    "incident_id": "REP-HIST-002",
                    "similarity_score": 0.81,
                    "description": "Historical case 2: valve opened before line isolation verified.",
                    "site": "Plant South",
                    "location": "Pump Bay",
                    "date": "2023-09-18",
                    "report_type": "Incident",
                },
            ]

    def retrieve_similar_incidents(self, query, top_k=5):
        if self.should_fail:
            raise RuntimeError("FAISS index segmentation fault or missing vector store.")
        return self.cases[:top_k]


def make_valid_llm_payload(
    sif_potential=True,
    rules=None,
    supporting_ids=None,
):
    """Generates valid JSON string payload matching LLMIncidentAssessment schema."""
    if rules is None:
        rules = ["Energy Isolation"]
    if supporting_ids is None:
        supporting_ids = ["REP-HIST-001", "REP-HIST-002"]

    return json.dumps({
        "sif_potential": sif_potential,
        "assessment_confidence": "high",
        "sif_explanation": "Opening unisolated pressurized line creates serious energy release potential.",
        "life_saving_rules": rules,
        "activity": "Maintenance",
        "equipment": ["pump", "flange"],
        "hazards": ["pressurized fluid"],
        "barrier_failures": ["energy isolation unverified"],
        "unsafe_actions": ["flange loosening without verification"],
        "unsafe_conditions": ["residual pressure"],
        "potential_consequences": ["fluid spray", "eye injury"],
        "recurring_precursor_pattern": "Flange loosening under line pressure",
        "incident_evidence": [
            "Technician began opening pump discharge flange.",
            "Residual pressure remained in the line."
        ],
        "historical_pattern_observations": [
            {
                "observation": "Historical cases consistently involve opening equipment before zero-energy verification.",
                "supporting_incident_ids": supporting_ids
            }
        ],
        "uncertainty_notes": ["Fluid toxicity is unspecified."]
    })


@pytest.fixture
def sample_incident():
    return "Technician began opening the pump discharge flange before isolation was verified. Residual pressure remained in the line."


def test_pipeline_full_success(sample_incident):
    """Verifies end-to-end execution of SafetyRAGPipeline with all components functioning."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload()),
    )

    assessment = pipeline.analyze(sample_incident, top_k=2)

    assert isinstance(assessment, UnifiedSafetyAssessment)
    assert assessment.incident_text == sample_incident
    assert assessment.model_assessment is not None
    assert assessment.rag_assessment is not None
    assert len(assessment.similar_incidents) == 2
    assert assessment.consistency is not None
    assert assessment.consistency.sif_agreement is True
    assert "Energy Isolation" in assessment.consistency.lsr_overlap
    assert len(assessment.consistency.conflicts) == 0
    assert len(assessment.warnings) == 0


def test_legacy_output_preserved(sample_incident):
    """Verifies that legacy classification and extraction outputs are cleanly preserved."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(sif_label="SIF-Potential", primary_lsr="Energy Isolation"),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload()),
    )

    assessment = pipeline.analyze(sample_incident)
    leg = assessment.model_assessment

    assert leg.sif_prediction is True
    assert leg.sif_probability == 0.82
    assert leg.primary_lsr == "Energy Isolation"
    assert "Energy Isolation" in leg.lsr_predictions
    assert "Pump" in leg.equipment
    assert "Stored Energy" in leg.hazards
    assert "stored-energy" in leg.precursor_tags


def test_retrieval_output_preserved(sample_incident):
    """Verifies deterministic retrieval output and properties."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload()),
    )

    assessment = pipeline.analyze(sample_incident, top_k=2)
    hits = assessment.similar_incidents

    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[0].incident_id == "REP-HIST-001"
    assert hits[0].similarity_score == 0.88
    assert hits[0].site == "Plant North"
    assert hits[1].rank == 2
    assert hits[1].incident_id == "REP-HIST-002"


def test_llm_output_preserved(sample_incident):
    """Verifies structured LLM reasoning output preserved in rag_assessment."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload(sif_potential=True)),
    )

    assessment = pipeline.analyze(sample_incident)
    rag = assessment.rag_assessment

    assert rag.sif_potential is True
    assert rag.assessment_confidence == "high"
    assert "Energy Isolation" in rag.life_saving_rules
    assert len(rag.incident_evidence) == 2
    assert len(rag.historical_pattern_observations) == 1


def test_invalid_historical_ids_filtered(sample_incident):
    """Verifies that hallucinated or un-retrieved incident IDs are stripped and warned."""
    # LLM hallucinates an ID REP-HALLUCINATED-999
    raw_json = make_valid_llm_payload(
        supporting_ids=["REP-HIST-001", "REP-HALLUCINATED-999"]
    )
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=raw_json),
    )

    assessment = pipeline.analyze(sample_incident)

    # Hallucinated ID must be filtered out
    obs = assessment.rag_assessment.historical_pattern_observations[0]
    assert "REP-HALLUCINATED-999" not in obs.supporting_incident_ids
    assert "REP-HIST-001" in obs.supporting_incident_ids

    # Warning must be present
    warning_texts = " ".join(assessment.warnings)
    assert "REP-HALLUCINATED-999" in warning_texts
    assert "not present in retrieved context" in warning_texts


def test_conflict_detection_sif_disagreement(sample_incident):
    """Verifies conflict detection when legacy classifier and LLM disagree on SIF."""
    # Legacy says Non-SIF, LLM says SIF-Potential
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(sif_label="Non-SIF"),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload(sif_potential=True)),
    )

    assessment = pipeline.analyze(sample_incident)
    assert assessment.consistency.sif_agreement is False
    assert len(assessment.consistency.conflicts) > 0
    assert any("Legacy classifier predicted Non-SIF" in c for c in assessment.consistency.conflicts)


def test_conflict_detection_lsr_overlap(sample_incident):
    """Verifies computation of LSR overlap and detection of disjoint LSR predictions."""
    # Legacy predicts Line of Fire; LLM predicts Work at Height
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(sif_label="SIF-Potential", primary_lsr="Line of Fire"),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload(rules=["Work at Height"])),
    )

    assessment = pipeline.analyze(sample_incident)
    consistency = assessment.consistency

    assert consistency.sif_agreement is True
    assert consistency.lsr_overlap == []
    assert any("LSR mismatch" in c for c in consistency.conflicts)


def test_legacy_failure_fallback(sample_incident):
    """Verifies pipeline returns RAG reasoning and warnings if legacy inference raises an exception."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(should_fail=True),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload()),
    )

    assessment = pipeline.analyze(sample_incident)

    assert assessment.model_assessment is None
    assert assessment.rag_assessment is not None
    assert len(assessment.similar_incidents) == 2
    assert assessment.consistency is None
    assert any("Legacy model inference failed" in w for w in assessment.warnings)


def test_llm_failure_fallback(sample_incident):
    """Verifies pipeline returns legacy outputs and retrieval if LLM generation/parsing fails."""
    # Malformed JSON from LLM
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response="MALFORMED_NON_JSON_RESPONSE"),
    )

    assessment = pipeline.analyze(sample_incident)

    assert assessment.model_assessment is not None
    assert assessment.rag_assessment is None
    assert len(assessment.similar_incidents) == 2
    assert assessment.consistency is None
    assert any("RAG LLM assessment failed" in w for w in assessment.warnings)


def test_retrieval_failure_fallback(sample_incident):
    """Verifies that retrieval failure skips LLM and returns legacy result with warning."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(should_fail=True),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload()),
    )

    assessment = pipeline.analyze(sample_incident)

    assert assessment.model_assessment is not None
    assert assessment.rag_assessment is None
    assert len(assessment.similar_incidents) == 0
    assert any("Retrieval failed" in w for w in assessment.warnings)


def test_empty_retrieval_handling(sample_incident):
    """Verifies that empty retrieval result skips LLM reasoning and records warning."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(cases=[]),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(response=make_valid_llm_payload()),
    )

    assessment = pipeline.analyze(sample_incident)

    assert assessment.model_assessment is not None
    assert assessment.rag_assessment is None
    assert len(assessment.similar_incidents) == 0
    assert any("No historical incidents retrieved" in w for w in assessment.warnings)


def test_empty_query_rejected():
    """Verifies that empty or whitespace query raises ValueError."""
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=DummyLegacyPipeline(),
        retriever=DummyRetriever(),
        prompt_builder=SafetyPromptBuilder(),
        llm_client=MockLLMClient(),
    )

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        pipeline.analyze("")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        pipeline.analyze("   \t\n  ")
