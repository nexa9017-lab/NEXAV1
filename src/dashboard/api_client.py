"""
Streamlit API Client for Safety Analytics FastAPI Backend.
Handles HTTP requests, caching for read-only analytics, high timeouts for uploads,
and strict error handling.
"""

import os
from typing import Any, Dict, Optional, Tuple
import httpx
import streamlit as st

API_BASE_URL = os.getenv("SIF_API_BASE_URL", "http://localhost:8000")


class APIError(Exception):
    pass


class SafetyAPIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _get(self, endpoint: str, params: Dict[str, Any] = None, timeout: float = 10.0) -> Dict[str, Any]:
        """Base GET method with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = httpx.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise APIError(f"Network error: {str(e)}")
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text
            raise APIError(f"HTTP {e.response.status_code}: {detail}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")

    def _post(self, endpoint: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Base POST method with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = httpx.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise APIError(f"Network error: {str(e)}")
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text
            raise APIError(f"HTTP {e.response.status_code}: {detail}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Check API health status (not cached for immediate status)."""
        try:
            url = f"{self.base_url}/health"
            response = httpx.get(url, timeout=5.0)
            status = getattr(response, "status_code", 200)
            if status == 200 or not isinstance(status, int):
                return True, response.json()
            return False, {}
        except Exception:
            return False, {}

    @st.cache_data(ttl=300, show_spinner=False)
    def get_metadata(_self) -> Dict[str, Any]:
        """Retrieves active dataset metadata."""
        return _self._get("/api/v1/analytics/metadata")

    @st.cache_data(ttl=300, show_spinner=False)
    def get_summary(_self) -> Dict[str, Any]:
        """Retrieves executive summary metrics."""
        return _self._get("/api/v1/analytics/summary")

    @st.cache_data(ttl=300, show_spinner=False)
    def get_sites(_self) -> Dict[str, Any]:
        """Retrieves site analytics."""
        return _self._get("/api/v1/analytics/sites")

    @st.cache_data(ttl=300, show_spinner=False)
    def get_activities(_self) -> Dict[str, Any]:
        """Retrieves activity analytics."""
        return _self._get("/api/v1/analytics/activities")

    @st.cache_data(ttl=300, show_spinner=False)
    def get_lsr(_self) -> Dict[str, Any]:
        """Retrieves Life-Saving Rule analytics."""
        return _self._get("/api/v1/analytics/lsr")

    @st.cache_data(ttl=300, show_spinner=False)
    def get_patterns(_self, min_support: Optional[int] = None) -> Dict[str, Any]:
        """Retrieves recurring precursor patterns."""
        params = {"min_support": min_support} if min_support is not None else None
        return _self._get("/api/v1/analytics/patterns", params=params)

    # DO NOT CACHE ANALYZE CALLS
    def analyze_incident(self, description: str, top_k: int = 5) -> Dict[str, Any]:
        """Runs unified RAG incident analysis."""
        payload = {"description": description, "top_k": top_k}
        return self._post("/api/v1/analyze", payload=payload, timeout=60.0)

    # DO NOT CACHE DATASET UPLOADS
    def upload_dataset(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Uploads a CSV dataset for reindexing and analytics regeneration.
        Uses a high timeout (300 seconds) to accommodate batch inference and indexing.
        """
        try:
            url = f"{self.base_url}/api/v1/reindex"
            files = {"file": (filename, file_bytes, "text/csv")}
            response = httpx.post(url, files=files, timeout=300.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise APIError(f"Network error during upload: {str(e)}")
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text
            raise APIError(f"Upload failed (HTTP {e.response.status_code}): {detail}")
        except Exception as e:
            raise APIError(f"Unexpected upload error: {str(e)}")

    # Backward compatibility aliases
    def predict_single(self, description: str, **kwargs) -> Dict[str, Any]:
        return self.analyze_incident(description=description)

    def upload_csv(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        return self.upload_dataset(file_bytes=file_bytes, filename=filename)


# Global client singleton
api = SafetyAPIClient()
