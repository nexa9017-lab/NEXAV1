# API to Frontend Data Contract (Phase R7)

This document specifies the exact contract between the FastAPI backend and the Streamlit frontend.
All frontend code MUST adhere to these schemas through the centralized transformers in `src/dashboard/transformers.py`.

---

## 1. System Health

### `GET /health`
- **Frontend Consumer**: Sidebar status indicator, initial page health gate.
- **Response Schema**:
  ```json
  {
    "status": "ok" | "degraded",
    "components": {
      "embedder": "ready",
      "vector_store": "ready",
      "retriever": "ready",
      "legacy_pipeline": "ready",
      "llm": "ready" | "unavailable",
      "rag_pipeline": "ready" | "unavailable"
    },
    "request_id": "uuid-string"
  }
  ```
- **Error Behavior**: If request fails or status is not 200, frontend displays:
  `Backend unavailable. Start the FastAPI service and refresh the page.` and stops further rendering.

---

## 2. Active Dataset Status

### `GET /api/v1/analytics/metadata`
- **Frontend Consumer**: Active Dataset Status header badge.
- **Response Envelope**:
  ```json
  {
    "metadata": { ... },
    "data": {
      "filename": "all_reports.csv",
      "report_count": 2500,
      "last_updated": "2026-09-23T12:00:00Z",
      "dataset_version": "v1.0"
    }
  }
  ```
- **Required Fields**: `filename`, `report_count`, `last_updated`.

---

## 3. Executive Summary

### `GET /api/v1/analytics/summary`
- **Frontend Consumer**: Top KPI Executive Summary (exactly 4 metric cards).
- **Response Envelope**:
  ```json
  {
    "metadata": { ... },
    "data": {
      "total_reports": 2500,
      "total_predicted_sif_reports": 614,
      "overall_sif_density": 0.2456,
      "avg_sif_probability": 0.2926
    }
  }
  ```
- **Normalized Fields (Transformer)**:
  - `total_reports`: `int` (or `None` if missing)
  - `sif_reports`: `int` (`total_predicted_sif_reports` or `None`)
  - `non_sif_reports`: `int` (`total_reports - total_predicted_sif_reports` or `None`)
  - `sif_density`: `float` (`overall_sif_density` or `None`)
- **Null Safety**: If any field is missing or None, frontend displays `"Data unavailable"`. Never substitute missing values with `0`.

---

## 4. SIF Precursor Density by Site

### `GET /api/v1/analytics/sites`
- **Frontend Consumer**: Site analytics table and horizontal bar chart.
- **Response Envelope**:
  ```json
  {
    "metadata": { ... },
    "data": [
      {
        "name": "Refinery Alpha",
        "total_reports": 500,
        "sif_count": 140,
        "sif_density": 0.2800
      }
    ]
  }
  ```
- **Normalized Fields per Item**:
  - `site`: clean string name
  - `total_reports`: `int`
  - `sif_reports`: `int` (mapped from `sif_count`)
  - `sif_density`: `float`

---

## 5. SIF Precursor Density by Activity

### `GET /api/v1/analytics/activities`
- **Frontend Consumer**: Activity analytics table and horizontal bar chart.
- **Response Envelope**:
  ```json
  {
    "metadata": { ... },
    "data": [
      {
        "name": "{'value': 'Maintenance', ...}" | "Maintenance",
        "total_reports": 210,
        "sif_count": 65,
        "sif_density": 0.3095
      }
    ]
  }
  ```
- **Entity Name Cleaning**:
  If `name` is a Python-like serialized dict string, parse using safe `ast.literal_eval()` (never `eval()`) to extract the clean `'value'` string (e.g. `"Maintenance"`).
- **Normalized Fields per Item**:
  - `activity`: clean string name
  - `total_reports`: `int`
  - `sif_reports`: `int` (from `sif_count`)
  - `sif_density`: `float`

---

## 6. Life-Saving Rule Mapping

### `GET /api/v1/analytics/lsr`
- **Frontend Consumer**: Life-Saving Rule table and horizontal bar chart.
- **Response Envelope**:
  ```json
  {
    "metadata": { ... },
    "data": [
      {
        "name": "{'rule': 'Energy Isolation', ...}" | "Energy Isolation",
        "total_reports": 180,
        "sif_count": 72
      }
    ]
  }
  ```
- **Entity Name Cleaning**:
  Parse dict string safely via `ast.literal_eval()` to extract the clean `'rule'` string (e.g. `"Energy Isolation"`).
- **Normalized Fields per Item**:
  - `rule`: clean string name
  - `mapped_reports`: `int` (from `total_reports`)
  - `sif_reports`: `int` (from `sif_count`)

---

## 7. Recurring SIF Precursor Patterns

### `GET /api/v1/analytics/patterns`
- **Frontend Consumer**: Recurring SIF Precursor Patterns table (one unified table, NOT cards).
- **Response Envelope**:
  ```json
  {
    "metadata": { ... },
    "data": [
      {
        "pattern_id": "...",
        "pattern_type": "activity_hazard",
        "components": {
          "activity": "{'value': 'Operations', ...}",
          "hazards": "{'value': 'Loss of Containment', ...}"
        },
        "occurrences": 12,
        "sif_count": 10,
        "sif_density": 0.8333,
        "affected_sites": ["Site Alpha", "Platform Charlie"]
      }
    ]
  }
  ```
- **Normalized Fields per Item**:
  - `pattern`: formatted readable string e.g. `"Activity: Operations | Hazards: Loss of Containment"`
  - `occurrences`: `int`
  - `sif_reports`: `int` (from `sif_count`)
  - `sif_density`: `float`
  - `affected_sites`: comma-separated string of sites

---

## 8. Single Incident Analysis

### `POST /api/v1/analyze`
- **Request Payload**:
  ```json
  {
    "description": "Technician opened pump flange under pressure without energy isolation.",
    "top_k": 5
  }
  ```
- **Response Schema** (`UnifiedSafetyAssessment`):
  - `model_assessment`:
    - `sif_prediction`: `bool`
    - `sif_probability`: `float`
    - `primary_lsr`: `str`
  - `rag_assessment`:
    - `sif_potential`: `bool`
    - `assessment_confidence`: `"high" | "medium" | "low"`
    - `sif_explanation`: `str` (Why this was flagged)
    - `life_saving_rules`: `List[str]`
    - `uncertainty_notes`: `List[str]`
  - `safety_factors`:
    - `activity`: `str`
    - `equipment`: `List[str]`
    - `hazards`: `List[str]`
    - `barrier_failures`: `List[str]`
    - `unsafe_actions`: `List[str]`
    - `unsafe_conditions`: `List[str]`
    - `potential_consequences`: `List[str]`
  - `similar_incidents`:
    - `rank`: `int`
    - `incident_id`: `str`
    - `similarity_score`: `float`
    - `site`: `str`
    - `date`: `str`
    - `report_type`: `str`
    - `description`: `str`
  - `consistency`:
    - `sif_agreement`: `bool`
    - `conflicts`: `List[str]`
  - `warnings`: `List[str]`

---

## 9. Dataset Upload & Reindexing

### `POST /api/v1/reindex`
- **Request**: Multipart form data with `.csv` file.
- **Timeout**: Up to 300 seconds (handles vector embedding + batch inference + analytics generation).
- **Response Schema** (`ReindexResponse`):
  ```json
  {
    "status": "success",
    "records_received": 2500,
    "records_indexed": 2500,
    "records_processed": 2500,
    "sif_reports": 614,
    "sif_density": 0.2456,
    "dimension": 384,
    "index_type": "IndexFlatIP",
    "vector_index_updated": true,
    "analytics_updated": true,
    "dataset_version": "upload-..."
  }
  ```
