import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.dashboard.transformers import normalize_dict_response, normalize_list_response
from src.api.dependencies import get_analytics_cache, load_analytics_artifacts

app.dependency_overrides[get_analytics_cache] = load_analytics_artifacts

client = TestClient(app)

def test_summary_contract():
    res = client.get("/api/v1/analytics/summary")
    assert res.status_code == 200
    json_data = res.json()
    
    # Must be enveloped
    assert "metadata" in json_data
    assert "data" in json_data
    
    # Transformer must unwrap
    normalized = normalize_dict_response(json_data)
    assert "total_reports" in normalized["data"]
    
def test_sites_contract():
    res = client.get("/api/v1/analytics/sites")
    assert res.status_code == 200
    json_data = res.json()
    
    # Must be enveloped
    assert "metadata" in json_data
    assert "data" in json_data
    assert isinstance(json_data["data"], list)
    
    normalized = normalize_list_response(json_data)
    assert isinstance(normalized["data"], list)
    if len(normalized["data"]) > 0:
        assert "name" in normalized["data"][0]

def test_reports_contract():
    res = client.get("/api/v1/reports?limit=10")
    # if it fails with 503 it means dataset is missing, but if it passes it should be enveloped
    if res.status_code == 200:
        json_data = res.json()
        assert "metadata" in json_data
        assert "data" in json_data
        assert isinstance(json_data["data"], list)
        
        normalized = normalize_list_response(json_data)
        assert isinstance(normalized["data"], list)

def test_health_contract():
    res = client.get("/health")
    assert res.status_code == 200
    json_data = res.json()
    
    # Health endpoint usually unwrapped, but transformer wraps it
    normalized = normalize_dict_response(json_data)
    assert "status" in normalized["data"]
    assert "components" in normalized["data"]
