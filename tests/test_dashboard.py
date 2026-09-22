"""
Tests for Streamlit Dashboard components and API Client.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.dashboard.api_client import SafetyAPIClient, APIError
from src.dashboard.components.shared import format_pct, format_score

# --- FORMATTING TESTS ---

def test_format_pct():
    assert format_pct(0.248) == "24.8%"
    assert format_pct(0.9134) == "91.3%"
    assert format_pct(0.0) == "0.0%"
    assert format_pct(1.0) == "100.0%"

def test_format_score():
    assert format_score(82.36) == "82.4"
    assert format_score(0.0) == "0.0"
    assert format_score(100.11) == "100.1"

# --- API CLIENT TESTS ---

@pytest.fixture
def api_client():
    return SafetyAPIClient(base_url="http://test-server")

@patch("src.dashboard.api_client.httpx.get")
def test_api_client_health_ok(mock_get, api_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok", "components": {"sif_model": "loaded"}}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response
    
    is_healthy, data = api_client.check_health()
    assert is_healthy is True
    assert data["status"] == "ok"
    assert mock_get.called

@patch("src.dashboard.api_client.httpx.get")
def test_api_client_health_fail(mock_get, api_client):
    import httpx
    # Simulate connection error
    mock_get.side_effect = httpx.RequestError("Failed to connect")
    
    is_healthy, data = api_client.check_health()
    assert is_healthy is False
    assert data == {}

@patch("src.dashboard.api_client.httpx.post")
def test_api_client_predict_single(mock_post, api_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"sif": {"label": "SIF-Potential", "probability": 0.8}}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response
    
    res = api_client.predict_single("test description")
    assert res["sif"]["label"] == "SIF-Potential"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["description"] == "test description"

@patch("src.dashboard.api_client.httpx.get")
def test_api_client_handles_http_errors(mock_get, api_client):
    import httpx
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    
    mock_get.side_effect = httpx.HTTPStatusError("404 Client Error", request=MagicMock(), response=mock_response)
    
    with pytest.raises(APIError) as exc_info:
        # Need to call _get directly to bypass streamlit cache for testing
        api_client._get("/test")
    
    assert "HTTP 404" in str(exc_info.value)
