"""
Pydantic schemas for the FastAPI backend (Pydantic v2).
"""

from typing import List, Dict, Any, Optional
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
