"""
Unit tests for the FastAPI RAG endpoints (Phase R6).
All tests execute offline using deterministic mocks and FastAPI TestClient; no network or Groq API calls.
"""
import io
import json
import pytest
from fastapi.testclient import TestClient
import numpy as np

from src.api.main import app
from src.api.dependencies import (
    get_rag_pipeline,
    get_retriever,
    get_vector_store,
    get_embedder,
    _components_status,
)
from src.api.schemas import (
    UnifiedSafetyAssessment,
    LegacyModelAssessment,
    RetrievedIncident,
    AssessmentConsistency,
    LLMIncidentAssessment,
)


class MockRetriever:
    """Deterministic mock retriever for API tests."""
    def __init__(self, cases=None):
        self.cases = cases or [
            {
                "rank": 1,
                "vector_id": 101,
                "incident_id": "REP-TEST-001",
                "similarity_score": 0.85,
                "description": "Historical incident 1 description",
                "site": "Refinery Alpha",
                "location": "Unit 1",
                "date": "2024-01-01",
                "report_type": "Near Miss",
            },
            {
                "rank": 2,
                "vector_id": 102,
                "incident_id": "REP-TEST-002",
                "similarity_score": 0.78,
                "description": "Historical incident 2 description",
                "site": "Refinery Beta",
                "location": "Unit 2",
                "date": "2024-01-02",
                "report_type": "Incident",
            },
        ]

    def retrieve_similar_incidents(self, query, top_k=5):
        return self.cases[:top_k]


class MockVectorStore:
    """Deterministic mock vector store for API tests."""
    def __init__(self):
        self.records = [{"report_id": "REP-1"}, {"report_id": "REP-2"}]

    def get_info(self):
        return {
            "loaded": True,
            "total_records": 2500,
            "dimension": 384,
            "index_type": "IndexFlatIP",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }

    def load_index(self):
        return True


class MockPipeline:
    """Deterministic mock SafetyRAGPipeline."""
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def analyze(self, incident_text, top_k=5):
        if self.should_fail:
            raise RuntimeError("Pipeline execution failure.")

        return UnifiedSafetyAssessment(
            incident_text=incident_text,
            model_assessment=LegacyModelAssessment(
                sif_prediction=True,
                sif_probability=0.75,
                primary_lsr="Energy Isolation",
                lsr_predictions=["Energy Isolation"],
                activities=["Maintenance"],
                equipment=["Pump"],
                hazards=["Stored Energy"],
                barrier_failures=["ENERGY_ISOLATION_FAILURE"],
                unsafe_actions=[],
                unsafe_conditions=["Residual pressure"],
                potential_consequences=["Major Process Safety Event"],
                precursor_tags=["stored-energy"],
            ),
            rag_assessment=LLMIncidentAssessment(
                sif_potential=True,
                assessment_confidence="high",
                sif_explanation="High stored energy creates SIF potential.",
                life_saving_rules=["Energy Isolation"],
                activity="Maintenance",
                equipment=["Pump discharge flange"],
                hazards=["Pressurized fluid"],
                barrier_failures=["Isolation not confirmed"],
                unsafe_actions=["Opened flange before isolation verification"],
                unsafe_conditions=["Residual pressure"],
                potential_consequences=["Fluid spray"],
                recurring_precursor_pattern="Opening equipment before isolation",
                incident_evidence=["Technician opened flange before isolation."],
                historical_pattern_observations=[],
                uncertainty_notes=[],
            ),
            similar_incidents=[
                RetrievedIncident(
                    rank=1,
                    vector_id=101,
                    incident_id="REP-TEST-001",
                    similarity_score=0.85,
                    description="Historical case 1",
                    site="Site A",
                    location="Loc 1",
                    date="2024-01-01",
                    report_type="Near Miss",
                )
            ],
            consistency=AssessmentConsistency(
                sif_agreement=True,
                lsr_overlap=["Energy Isolation"],
                conflicts=[],
            ),
            warnings=[],
            metadata={"top_k": top_k, "retrieved_count": 1},
        )


class MockEmbedder:
    """Deterministic mock embedder for reindex tests."""
    dimension = 4
    def embed_batch(self, texts, batch_size=64):
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    """TestClient fixture with dependencies overridden for fast offline testing."""
    monkeypatch.setattr("src.api.main.init_rag_dependencies", lambda: None)

    app.dependency_overrides[get_retriever] = lambda: MockRetriever()
    app.dependency_overrides[get_vector_store] = lambda: MockVectorStore()
    app.dependency_overrides[get_rag_pipeline] = lambda: MockPipeline()
    app.dependency_overrides[get_embedder] = lambda: MockEmbedder()

    # Set mock statuses
    _components_status["embedder"] = "ready"
    _components_status["vector_store"] = "ready"
    _components_status["retriever"] = "ready"
    _components_status["legacy_pipeline"] = "ready"
    _components_status["llm"] = "ready"
    _components_status["rag_pipeline"] = "ready"

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


def test_health_healthy(client):
    """Test 1: /health endpoint returns ok when all components ready."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["components"]["llm"] == "ready"
    assert data["components"]["vector_store"] == "ready"


def test_health_degraded_when_llm_unavailable(client):
    """Test 2: /health endpoint returns degraded when LLM is unavailable."""
    _components_status["llm"] = "unavailable"
    _components_status["rag_pipeline"] = "unavailable"

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["components"]["llm"] == "unavailable"

    # Reset
    _components_status["llm"] = "ready"
    _components_status["rag_pipeline"] = "ready"


def test_analyze_valid_request(client):
    """Test 3: POST /api/v1/analyze with valid payload returns 200."""
    payload = {
        "description": "Technician opened the flange while residual pressure remained.",
        "top_k": 3,
        "site": "Refinery Alpha",
        "report_type": "Near Miss",
    }
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["incident_text"] == payload["description"]
    assert data["model_assessment"]["sif_prediction"] is True
    assert data["rag_assessment"]["sif_potential"] is True
    assert len(data["similar_incidents"]) == 1
    assert data["consistency"]["sif_agreement"] is True


def test_analyze_empty_description_rejected(client):
    """Test 4: POST /api/v1/analyze rejects empty or whitespace-only description."""
    resp = client.post("/api/v1/analyze", json={"description": ""})
    assert resp.status_code == 422

    resp2 = client.post("/api/v1/analyze", json={"description": "   \n\t  "})
    assert resp2.status_code == 422


def test_analyze_invalid_top_k_rejected(client):
    """Test 5: POST /api/v1/analyze rejects invalid top_k (< 1)."""
    resp = client.post("/api/v1/analyze", json={"description": "Valid incident", "top_k": 0})
    assert resp.status_code == 422

    resp2 = client.post("/api/v1/analyze", json={"description": "Valid incident", "top_k": -5})
    assert resp2.status_code == 422


def test_analyze_pipeline_failure_handled(client):
    """Test 6: POST /api/v1/analyze returns 500 when pipeline raises an exception."""
    app.dependency_overrides[get_rag_pipeline] = lambda: MockPipeline(should_fail=True)
    resp = client.post("/api/v1/analyze", json={"description": "Technician incident report."})
    assert resp.status_code == 500
    assert "Safety analysis execution failed" in resp.json()["detail"]


def test_search_returns_ranked_results(client):
    """Test 7: POST /api/v1/search returns ranked historical incidents."""
    resp = client.post("/api/v1/search", json={"query": "pump flange pressure", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "pump flange pressure"
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["rank"] == 1
    assert data["results"][0]["incident_id"] == "REP-TEST-001"


def test_search_forbidden_labels_absent(client):
    """Test 8: POST /api/v1/search results contain NO synthetic/ground-truth labels."""
    resp = client.post("/api/v1/search", json={"query": "test query", "top_k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]

    forbidden = ["sif_label", "sif_potential", "primary_lsr", "hazard", "barrier_failure"]
    for item in results:
        for f in forbidden:
            assert f not in item, f"Forbidden label {f} leaked into search response."


def test_index_info_returns_metadata(client):
    """Test 9: GET /api/v1/index/info returns vector store metadata without paths."""
    resp = client.get("/api/v1/index/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["loaded"] is True
    assert data["total_records"] == 2500
    assert data["dimension"] == 384
    assert data["index_type"] == "IndexFlatIP"
    assert "all-MiniLM-L6-v2" in data["embedding_model"]
    # Path fields must not be present
    assert "index_path" not in data
    assert "metadata_path" not in data


def test_reindex_accepts_valid_csv(client, tmp_path, monkeypatch):
    """Test 10: POST /api/v1/reindex processes a valid uploaded CSV, enriches analytics, and swaps index."""
    csv_content = (
        "report_id,description,site,location,report_type\n"
        "REP-991,Technician opened pump flange under pressure,Site A,Bay 1,Near Miss\n"
        "REP-992,Worker observed loose scaffolding clamp,Site B,Tower 2,Incident\n"
    )
    files = {"file": ("test_reports.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}

    test_idx = tmp_path / "test_active.faiss"
    test_meta = tmp_path / "test_active.json"
    test_analytics = tmp_path / "test_analytics"
    test_enriched = tmp_path / "test_enriched.csv"
    monkeypatch.setattr("src.api.routers.rag.VECTOR_INDEX_PATH", test_idx)
    monkeypatch.setattr("src.api.routers.rag.VECTOR_METADATA_PATH", test_meta)
    monkeypatch.setattr("src.api.routers.rag.ANALYTICS_OUTPUT_DIR", test_analytics)
    monkeypatch.setattr("src.api.routers.rag.ENRICHED_REPORTS_PATH", test_enriched)

    def mock_enrich(df, output_enriched_path, output_analytics_dir, legacy_pipeline=None, source_filename=""):
        output_enriched_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_enriched_path, index=False)
        output_analytics_dir.mkdir(parents=True, exist_ok=True)
        for rj in [
            "overall_summary.json",
            "site_analytics.json",
            "activity_analytics.json",
            "lsr_analytics.json",
            "recurring_patterns.json",
        ]:
            with open(output_analytics_dir / rj, "w") as f:
                json.dump({"metadata": {}, "data": []}, f)
        return {
            "records_processed": len(df),
            "sif_reports": 1,
            "sif_density": 0.5,
            "runtime_seconds": 0.05,
        }

    monkeypatch.setattr("src.api.routers.rag.process_dataset_enrichment_and_analytics", mock_enrich)

    resp = client.post("/api/v1/reindex", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["records_received"] == 2
    assert data["records_indexed"] == 2
    assert data["records_processed"] == 2
    assert data["sif_reports"] == 1
    assert data["sif_density"] == 0.5
    assert data["vector_index_updated"] is True
    assert data["analytics_updated"] is True
    assert "upload-" in data["dataset_version"]


def test_reindex_failure_preserves_active_artifacts(client, tmp_path, monkeypatch):
    """Test 10b: Reindexing failure rolls back and preserves existing active artifacts."""
    active_idx = tmp_path / "active.faiss"
    active_meta = tmp_path / "active.json"
    active_idx.write_text("old_index_data")
    active_meta.write_text("old_meta_data")

    monkeypatch.setattr("src.api.routers.rag.VECTOR_INDEX_PATH", active_idx)
    monkeypatch.setattr("src.api.routers.rag.VECTOR_METADATA_PATH", active_meta)

    def failing_enrich(*args, **kwargs):
        raise RuntimeError("Simulated analytics pipeline crash")

    monkeypatch.setattr("src.api.routers.rag.process_dataset_enrichment_and_analytics", failing_enrich)

    csv_content = "report_id,description\nREP-1,Valid description here\n"
    files = {"file": ("reports.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}

    resp = client.post("/api/v1/reindex", files=files)
    assert resp.status_code == 500
    # Verify active artifacts were NOT replaced or wiped
    assert active_idx.read_text() == "old_index_data"
    assert active_meta.read_text() == "old_meta_data"


def test_reindex_rejects_missing_description(client):
    """Test 11: POST /api/v1/reindex rejects CSV without required 'description' column."""
    bad_csv = "report_id,site,location\nREP-1,Site A,Bay 1\n"
    files = {"file": ("bad_reports.csv", io.BytesIO(bad_csv.encode("utf-8")), "text/csv")}

    resp = client.post("/api/v1/reindex", files=files)
    assert resp.status_code == 400
    assert "Required column 'description' is missing" in resp.json()["detail"]


def test_reindex_rejects_non_csv_file(client):
    """Test 12: POST /api/v1/reindex rejects non-CSV upload."""
    files = {"file": ("document.txt", io.BytesIO(b"Hello world"), "text/plain")}
    resp = client.post("/api/v1/reindex", files=files)
    assert resp.status_code == 400
    assert "Invalid file format" in resp.json()["detail"]


def test_missing_groq_key_does_not_break_search(client):
    """Test 13: In degraded mode (missing Groq), /search and /index/info still work."""
    _components_status["llm"] = "unavailable"
    _components_status["rag_pipeline"] = "unavailable"

    search_resp = client.post("/api/v1/search", json={"query": "height harness", "top_k": 1})
    assert search_resp.status_code == 200
    assert search_resp.json()["count"] == 1

    info_resp = client.get("/api/v1/index/info")
    assert info_resp.status_code == 200

    # Reset
    _components_status["llm"] = "ready"
    _components_status["rag_pipeline"] = "ready"


def test_unified_response_schema_preserved(client):
    """Test 14: /api/v1/analyze preserves all required R5 response sections."""
    resp = client.post("/api/v1/analyze", json={"description": "Technician incident."})
    assert resp.status_code == 200
    data = resp.json()

    # Check that required top-level keys exist and separate subsystems
    required_keys = [
        "incident_text",
        "model_assessment",
        "rag_assessment",
        "similar_incidents",
        "consistency",
        "warnings",
        "metadata",
    ]
    for k in required_keys:
        assert k in data, f"Required top-level key {k} missing from analyze response."

    # Verify no fabricated single-truth fields
    assert "final_sif" not in data
    assert "final_lsr" not in data
    assert "final_risk" not in data
