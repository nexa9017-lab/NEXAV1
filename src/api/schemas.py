"""
Pydantic schemas for the FastAPI backend (Pydantic v2).
"""

from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Core Inference Requests
# ---------------------------------------------------------------------------

class SafetyReportRequest(BaseModel):
    description: str = Field(..., description="The free-text description of the safety event.")
    site: Optional[str] = Field(None, description="Optional metadata: facility or site.")
    location: Optional[str] = Field(None, description="Optional metadata: specific location.")
    report_type: Optional[str] = Field(None, description="Optional metadata: e.g. Near Miss, Incident.")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Technician started opening a flange while the line remained pressurized. LOTO had not been applied.",
                "site": "North Processing Facility",
                "location": "Pump Bay",
                "report_type": "Near Miss"
            }
        }
    )

class BatchSafetyReportRequest(BaseModel):
    reports: List[SafetyReportRequest] = Field(..., max_length=100)

# ---------------------------------------------------------------------------
# Core Inference Responses
# ---------------------------------------------------------------------------

class SIFPrediction(BaseModel):
    label: str
    probability: float

class LSRPrediction(BaseModel):
    rule: str
    score: float

class ExtractionItem(BaseModel):
    value: str
    evidence: str
    confidence: str
    match_method: str

class SafetyAnalysisResponse(BaseModel):
    normalized_text: Optional[str] = None
    sif: Optional[SIFPrediction] = None
    life_saving_rules: Optional[List[LSRPrediction]] = None
    primary_lsr: Optional[str] = None
    activities: Optional[List[ExtractionItem]] = None
    equipment: Optional[List[ExtractionItem]] = None
    hazards: Optional[List[ExtractionItem]] = None
    barrier_failures: Optional[List[ExtractionItem]] = None
    unsafe_actions: Optional[List[ExtractionItem]] = None
    unsafe_conditions: Optional[List[ExtractionItem]] = None
    potential_consequences: Optional[List[ExtractionItem]] = None
    precursor_tags: Optional[List[str]] = None
    inference_metadata: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    request_id: Optional[str] = None

class BatchSafetyAnalysisResponse(BaseModel):
    results: List[SafetyAnalysisResponse]
    request_id: Optional[str] = None

# ---------------------------------------------------------------------------
# System / Analytics Responses
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    sif_model: str
    lsr_model: str
    precursor_extractor: str
    analytics: str

class HealthResponse(BaseModel):
    status: str
    components: ComponentHealth
    request_id: Optional[str] = None

class ModelInfoResponse(BaseModel):
    production_sif_model_type: str
    production_lsr_model_type: str
    sentence_embedding_model: str
    sif_threshold: float
    lsr_threshold: float
    prototype_notice: str
    synthetic_data_notice: str
    request_id: Optional[str] = None

class AnalyticsSummaryResponse(BaseModel):
    metadata: Dict[str, Any]
    data: Dict[str, Any]
    request_id: Optional[str] = None

class ReportLookupResponse(BaseModel):
    report_id: str
    original_report: Dict[str, Any]
    predicted_fields: Dict[str, Any]
    evaluation_ground_truth: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

# ---------------------------------------------------------------------------
# RAG LLM Reasoning Schemas (Phase R4)
# ---------------------------------------------------------------------------

class HistoricalPatternObservation(BaseModel):
    observation: str = Field(..., description="Synthesis of a recurring safety risk or vulnerability pattern observed in retrieved cases.")
    supporting_incident_ids: List[str] = Field(default_factory=list, description="List of historical incident IDs (from retrieved cases) that support this observation.")

class LLMIncidentAssessment(BaseModel):
    sif_potential: bool = Field(..., description="True if the incident exhibits potential for serious injury or fatality.")
    assessment_confidence: Literal["high", "medium", "low"] = Field(..., description="Qualitative LLM confidence in the assessment (high, medium, low). This is NOT a statistical or calibrated probability.")
    sif_explanation: str = Field(..., description="Causal reasoning for the SIF potential determination based on physical and operational energy.")
    life_saving_rules: List[str] = Field(default_factory=list, description="Canonical Life-Saving Rules relevant to the incident.")
    activity: Optional[str] = Field(None, description="Operational activity underway during the event.")
    equipment: List[str] = Field(default_factory=list, description="Specific industrial equipment, machinery, or tools involved.")
    hazards: List[str] = Field(default_factory=list, description="Physical, chemical, or environmental hazards present.")
    barrier_failures: List[str] = Field(default_factory=list, description="Procedural, engineered, or human barriers that failed or were omitted.")
    unsafe_actions: List[str] = Field(default_factory=list, description="Specific unsafe acts or deviations committed by personnel.")
    unsafe_conditions: List[str] = Field(default_factory=list, description="Unsafe physical conditions or environmental factors.")
    potential_consequences: List[str] = Field(default_factory=list, description="Plausible worst-case consequences that could have occurred.")
    recurring_precursor_pattern: Optional[str] = Field(None, description="Synthesized multi-factor precursor pattern derived from the new incident and historical context.")
    incident_evidence: List[str] = Field(default_factory=list, description="Short factual statements extracted strictly from the NEW incident description.")
    historical_pattern_observations: List[HistoricalPatternObservation] = Field(default_factory=list, description="Pattern observations grounded in retrieved historical incidents.")
    uncertainty_notes: List[str] = Field(default_factory=list, description="Caveats, missing details, or ambiguous information in the incident description.")


# ---------------------------------------------------------------------------
# Unified Safety Intelligence Schemas (Phase R5)
# ---------------------------------------------------------------------------

class LegacyModelAssessment(BaseModel):
    sif_prediction: bool = Field(..., description="Boolean SIF prediction from legacy classifier (True for SIF-Potential, False for Non-SIF).")
    sif_probability: Optional[float] = Field(None, description="Calibrated legacy probability of SIF potential.")
    primary_lsr: Optional[str] = Field(None, description="Top-ranked canonical Life-Saving Rule from legacy model.")
    lsr_predictions: List[str] = Field(default_factory=list, description="Rules identified above threshold by legacy model.")
    activities: List[str] = Field(default_factory=list, description="Extracted activity values.")
    equipment: List[str] = Field(default_factory=list, description="Extracted equipment values.")
    hazards: List[str] = Field(default_factory=list, description="Extracted hazard values.")
    barrier_failures: List[str] = Field(default_factory=list, description="Extracted barrier failure values.")
    unsafe_actions: List[str] = Field(default_factory=list, description="Extracted unsafe actions.")
    unsafe_conditions: List[str] = Field(default_factory=list, description="Extracted unsafe conditions.")
    potential_consequences: List[str] = Field(default_factory=list, description="Extracted potential consequences.")
    precursor_tags: List[str] = Field(default_factory=list, description="Extracted precursor tags.")


class RetrievedIncident(BaseModel):
    rank: int = Field(..., description="1-indexed similarity rank.")
    vector_id: int = Field(..., description="FAISS internal vector index.")
    incident_id: str = Field(..., description="Historical report ID.")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0).")
    description: str = Field(..., description="Original incident narrative.")
    site: Optional[str] = Field(None, description="Site or facility.")
    location: Optional[str] = Field(None, description="Specific location within facility.")
    date: Optional[str] = Field(None, description="Date of occurrence.")
    report_type: Optional[str] = Field(None, description="Type of report (e.g. Near Miss, Incident).")


class AssessmentConsistency(BaseModel):
    sif_agreement: bool = Field(..., description="True if legacy classifier and LLM agree on SIF potential.")
    lsr_overlap: List[str] = Field(default_factory=list, description="Intersection of Life-Saving Rules identified by both subsystems.")
    conflicts: List[str] = Field(default_factory=list, description="Human-readable explanations of disagreements between subsystems.")


class UnifiedSafetyAssessment(BaseModel):
    incident_text: str = Field(..., description="Original query or incident narrative.")
    model_assessment: Optional[LegacyModelAssessment] = Field(None, description="Legacy model prediction and precursor extraction outputs.")
    rag_assessment: Optional[LLMIncidentAssessment] = Field(None, description="Qualitative safety intelligence and reasoning from Groq LLM.")
    similar_incidents: List[RetrievedIncident] = Field(default_factory=list, description="Top-k retrieved historical cases from FAISS.")
    consistency: Optional[AssessmentConsistency] = Field(None, description="Comparison and conflict detection between legacy and RAG subsystems.")
    warnings: List[str] = Field(default_factory=list, description="System warnings, fallback notifications, or sanitized hallucinated IDs.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata, latencies, model versions, and parameters.")


# ---------------------------------------------------------------------------
# REST API Request/Response Schemas (Phase R6)
# ---------------------------------------------------------------------------

class AnalyzeIncidentRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Incident narrative text to analyze.")
    top_k: int = Field(default=5, ge=1, description="Number of historical incidents to retrieve as context.")
    site: Optional[str] = Field(None, description="Optional facility or site metadata.")
    report_type: Optional[str] = Field(None, description="Optional report type metadata.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Technician began opening the pump discharge flange before isolation was verified. Residual pressure remained in the line.",
                "top_k": 5,
                "site": "Refinery Alpha",
                "report_type": "Near Miss"
            }
        }
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text safety incident narrative or hazard query.")
    top_k: int = Field(default=5, ge=1, description="Number of similar incidents to retrieve.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "technician worked at height with harness disconnected",
                "top_k": 5
            }
        }
    )


class SearchResponse(BaseModel):
    query: str = Field(..., description="The original search query.")
    count: int = Field(..., description="Number of results returned.")
    results: List[RetrievedIncident] = Field(default_factory=list, description="Ranked similar historical incident hits.")


class IndexInfoResponse(BaseModel):
    loaded: bool = Field(..., description="True if FAISS index and metadata catalog are loaded in memory.")
    total_records: int = Field(..., description="Total number of historical incidents indexed.")
    dimension: Optional[int] = Field(None, description="Dense vector embedding dimension.")
    index_type: str = Field(..., description="FAISS index type (e.g. IndexFlatIP).")
    embedding_model: str = Field(..., description="Dense sentence embedding model name.")


class ReindexResponse(BaseModel):
    status: str = Field(..., description="Status of the reindexing operation (e.g. 'success').")
    records_received: int = Field(..., description="Total rows parsed from uploaded CSV.")
    records_indexed: int = Field(..., description="Valid records embedded and indexed into FAISS.")
    records_skipped: int = Field(default=0, description="Skipped records due to empty or invalid descriptions.")
    records_processed: int = Field(..., description="Total records enriched and processed through analytics.")
    sif_reports: int = Field(..., description="Number of predicted SIF reports in uploaded dataset.")
    sif_density: float = Field(..., description="Predicted SIF density in uploaded dataset.")
    dimension: int = Field(..., description="Embedding dimension of indexed vectors.")
    index_type: str = Field(default="IndexFlatIP", description="FAISS index type.")
    vector_index_updated: bool = Field(default=True, description="True if FAISS index was successfully refreshed.")
    analytics_updated: bool = Field(default=True, description="True if analytics artifacts were refreshed.")
    dataset_version: str = Field(..., description="Dataset version identifier.")


class DatasetMetadataResponse(BaseModel):
    filename: str = Field(..., description="Active dataset filename.")
    report_count: int = Field(..., description="Total reports in active dataset.")
    last_updated: Optional[str] = Field(None, description="ISO timestamp of last dataset refresh.")
    dataset_version: str = Field(..., description="Version identifier of active dataset.")


class RAGHealthResponse(BaseModel):
    status: str = Field(..., description="Overall system health status ('ok' or 'degraded').")
    components: Dict[str, str] = Field(default_factory=dict, description="Component readiness statuses.")
    request_id: Optional[str] = Field(None, description="Tracking request UUID.")



