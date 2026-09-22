# API to Frontend Data Contract

This document outlines the standard data contract between the FastAPI backend and the Streamlit frontend.

## Standard Envelope

All endpoints consumed by the Streamlit frontend must wrap their responses in a standard envelope to preserve metadata and maintain consistency.

```json
{
  "metadata": {
    "generated_at": "2023-10-27T10:00:00Z",
    "version": "1.0.0",
    "query_params": {}
  },
  "data": <payload>
}
```

The `<payload>` can be either an array of objects (`list`) or a single dictionary object (`dict`).

## Frontend Transformers

The `src/dashboard/transformers.py` module normalizes the API responses, un-wrapping the envelope so the Streamlit components can easily bind to data frames and widgets, while keeping `metadata` available in the returned structure for diagnostics.

- `normalize_list_response(response: Dict) -> Dict[str, Any]`
  Returns `{"data": [...], "metadata": {...}}`
  
- `normalize_dict_response(response: Dict) -> Dict[str, Any]`
  Returns `{"data": {...}, "metadata": {...}}`

## Normalized Key Schemas

### Site Risk (`/api/v1/analytics/sites`)
- Expected in List:
  - `name`: str
  - `total_reports`: int
  - `sif_count`: int
  - `sif_density`: float
  - `avg_sif_probability`: float
  - `data_quality_warnings`: list[str]

### Precursor Patterns (`/api/v1/analytics/patterns`)
- Expected in List:
  - `pattern_id`: str
  - `components`: dict
  - `occurrences`: int
  - `sif_count`: int
  - `sif_density`: float
  - `average_sif_probability`: float
  - `affected_sites`: dict
  - `priority_score`: float

### SIF Analysis (`/api/v1/reports`)
- Expected in List:
  - `report_id`: str
  - `original_report`: dict
  - `predicted_fields`: dict

### Prediction (`/api/v1/predict`)
- Expected in Dict:
  - `sif`: dict
  - `life_saving_rules`: list
  - `activities`: list
  - `equipment`: list
  - `hazards`: list
  - `barrier_failures`: list
  - `potential_consequences`: list
  - `precursor_tags`: list
