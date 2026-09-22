"""
Streamlit API Client for Safety Analytics FastAPI Backend.
Handles HTTP requests, caching, and error wrapping.
"""
import os
import httpx
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple

from src.dashboard.transformers import normalize_dict_response, normalize_list_response, normalize_envelope

API_BASE_URL = os.getenv("SIF_API_BASE_URL", "http://localhost:8000")

class APIError(Exception):
    pass

class SafetyAPIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Base GET method with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = httpx.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise APIError(f"Network error: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Base POST method with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = httpx.post(url, json=payload, timeout=15.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise APIError(f"Network error: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")

    @st.cache_data(ttl=300, show_spinner=False)
    def check_health(_self) -> Tuple[bool, Dict[str, Any]]:
        """Check API health status."""
        try:
            data = _self._get("/health")
            return True, normalize_dict_response(data)
        except APIError:
            return False, normalize_dict_response({})

    @st.cache_data(ttl=300, show_spinner=False)
    def get_summary(_self) -> Dict[str, Any]:
        return normalize_dict_response(_self._get("/api/v1/analytics/summary"))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_sites(_self, sort_by: str = None) -> Dict[str, Any]:
        params = {"sort_by": sort_by} if sort_by else None
        return normalize_list_response(_self._get("/api/v1/analytics/sites", params=params))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_lsr(_self) -> Dict[str, Any]:
        return normalize_list_response(_self._get("/api/v1/analytics/lsr"))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_hazards(_self) -> Dict[str, Any]:
        return normalize_list_response(_self._get("/api/v1/analytics/hazards"))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_activities(_self) -> Dict[str, Any]:
        return normalize_list_response(_self._get("/api/v1/analytics/activities"))
        
    @st.cache_data(ttl=300, show_spinner=False)
    def get_patterns(_self, min_support: int = 5) -> Dict[str, Any]:
        return normalize_list_response(_self._get("/api/v1/analytics/patterns", params={"min_support": min_support}))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_pattern_detail(_self, pattern_id: str) -> Dict[str, Any]:
        return normalize_dict_response(_self._get(f"/api/v1/analytics/patterns/{pattern_id}"))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_emerging_trends(_self) -> Dict[str, Any]:
        return normalize_list_response(_self._get("/api/v1/analytics/emerging"))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_models_info(_self) -> Dict[str, Any]:
        return normalize_dict_response(_self._get("/api/v1/models/info"))
        
    @st.cache_data(ttl=300, show_spinner=False)
    def get_analytics_metadata(_self) -> Dict[str, Any]:
        return normalize_dict_response(_self._get("/api/v1/analytics/metadata"))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_reports(_self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        # Using pagination matching backend params `page` and `page_size`
        page = (skip // limit) + 1 if limit > 0 else 1
        return normalize_list_response(_self._get("/api/v1/reports", params={"page": page, "page_size": limit}))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_report_detail(_self, report_id: str) -> Dict[str, Any]:
        return normalize_dict_response(_self._get(f"/api/v1/reports/{report_id}"))

    # DO NOT CACHE PREDICTIONS
    def predict_single(self, description: str, site: str = None, location: str = None, report_type: str = None) -> Dict[str, Any]:
        payload = {"description": description}
        if site: payload["site"] = site
        if location: payload["location"] = location
        if report_type: payload["report_type"] = report_type
        return normalize_dict_response(self._post("/api/v1/predict", payload))

    def upload_csv(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Uploads a dataset CSV for processing."""
        try:
            url = f"{self.base_url}/api/v1/upload"
            files = {"file": (filename, file_bytes, "text/csv")}
            response = httpx.post(url, files=files, timeout=300.0) # Longer timeout for batch processing
            response.raise_for_status()
            
            # Clear cache upon success
            st.cache_data.clear()
            return normalize_dict_response(response.json())
        except httpx.RequestError as e:
            raise APIError(f"Network error: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")
            
    def reset_dataset(self) -> Dict[str, Any]:
        """Restores the baseline dataset and clears cache."""
        res = self._post("/api/v1/upload/reset", payload={})
        st.cache_data.clear()
        return res

# Global singleton instance
api = SafetyAPIClient()
