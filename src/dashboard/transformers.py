"""
Data transformers to normalize backend API responses for the Streamlit frontend.
Ensures stable contracts and prevents silent nulls.
"""
from typing import Any, Dict, List

def normalize_envelope(raw_response: Any, default_data: Any = None) -> Dict[str, Any]:
    """
    Normalizes any response into a guaranteed {"data": ..., "metadata": ...} structure.
    If the response isn't wrapped, it wraps it.
    """
    if not isinstance(raw_response, dict):
        return {"data": raw_response if raw_response is not None else default_data, "metadata": {}}

    # If it's already an envelope (has 'data' and 'metadata')
    if "data" in raw_response and "metadata" in raw_response:
        data = raw_response.get("data")
        return {
            "data": data if data is not None else default_data,
            "metadata": raw_response.get("metadata", {})
        }
        
    # Unwrapped response (e.g., /health or /models/info or error payload)
    if "detail" in raw_response and len(raw_response) == 1:
        # Error payload, shouldn't really reach here if raise_for_status is used, but just in case
        return {"data": default_data, "metadata": {"error": raw_response["detail"]}}
        
    return {"data": raw_response, "metadata": {}}

def normalize_list_response(raw_response: Any) -> Dict[str, Any]:
    """Ensures data is a list."""
    norm = normalize_envelope(raw_response, default_data=[])
    if not isinstance(norm["data"], list):
        norm["data"] = []
    return norm

def normalize_dict_response(raw_response: Any) -> Dict[str, Any]:
    """Ensures data is a dict."""
    norm = normalize_envelope(raw_response, default_data={})
    if not isinstance(norm["data"], dict):
        norm["data"] = {}
    return norm
