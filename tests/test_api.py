"""
API integration tests for FastAPI backend.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import json

from src.api.main import app
from src.api.dependencies import get_pipeline, get_analytics_cache, load_analytics_artifacts
from src.pipeline.inference import SafetyInferencePipeline

# Mock Pipeline
class MockPipeline:
    def __init__(self):
        self.loaded = True
        self.sif_clf = True
        self.lsr_clf = True
        self.extractor = True

    def predict_single(self, report):
        if not report.get("description", "").strip():
            raise ValueError("Empty description provided.")
        return {
            "normalized_text": report["description"],
            "sif": {"label": "SIF-Potential", "probability": 0.8},
            "life_saving_rules": [],
            "primary_lsr": "None",
            "activities": [],
            "inference_metadata": {"mocked": True}
        }
        
    def predict_batch(self, reports):
        results = []
        for r in reports:
            if not r.get("description", "").strip():
                results.append({"error": "Empty or whitespace-only description"})
            else:
                results.append(self.predict_single(r))
        return results

@pytest.fixture
def client():
    # Override dependencies
    app.dependency_overrides[get_pipeline] = lambda: MockPipeline()
    # Let analytics use the real artifacts if they exist
    real_cache = load_analytics_artifacts()
    app.dependency_overrides[get_analytics_cache] = lambda: real_cache
    
    with TestClient(app) as c:
        yield c
        
    app.dependency_overrides.clear()

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["app_name"] == "Safety Analytics API"

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["components"]["sif_model"] == "loaded"
    assert "request_id" in data

def test_predict_single(client):
    response = client.post("/api/v1/predict", json={"description": "Test report"})
    assert response.status_code == 200
    data = response.json()
    assert data["sif"]["label"] == "SIF-Potential"
    assert "x-request-id" in response.headers

def test_predict_single_invalid(client):
    response = client.post("/api/v1/predict", json={"description": "   "})
    assert response.status_code == 422

def test_predict_batch(client):
    response = client.post(
        "/api/v1/predict/batch", 
        json={"reports": [{"description": "First"}, {"description": "Second"}]}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["normalized_text"] == "First"

def test_predict_batch_oversized(client):
    reports = [{"description": f"Test {i}"} for i in range(101)]
    response = client.post("/api/v1/predict/batch", json={"reports": reports})
    assert response.status_code == 422 # Pydantic max_length validation triggers 422

def test_analytics_summary(client):
    response = client.get("/api/v1/analytics/summary")
    if response.status_code == 503:
        pytest.skip("Analytics artifacts not generated yet")
    assert response.status_code == 200
    assert "data" in response.json()

def test_analytics_sites_sorting(client):
    response = client.get("/api/v1/analytics/sites?sort_by=sif_density")
    if response.status_code == 503:
        pytest.skip("Analytics artifacts not generated yet")
    assert response.status_code == 200
    
def test_analytics_sites_invalid_sort(client):
    response = client.get("/api/v1/analytics/sites?sort_by=invalid_col")
    if response.status_code == 503:
        pytest.skip("Analytics artifacts not generated yet")
    assert response.status_code == 400

def test_analytics_patterns_filter(client):
    response = client.get("/api/v1/analytics/patterns?min_support=10")
    if response.status_code == 503:
        pytest.skip("Analytics artifacts not generated yet")
    assert response.status_code == 200

def test_pattern_lookup_missing(client):
    response = client.get("/api/v1/analytics/patterns/invalid_pattern_id")
    if response.status_code == 503:
        pytest.skip("Analytics artifacts not generated yet")
    assert response.status_code == 404

def test_models_info(client):
    response = client.get("/api/v1/models/info")
    assert response.status_code == 200
    assert "production_sif_model_type" in response.json()
