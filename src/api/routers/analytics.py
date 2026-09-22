"""
Analytics endpoints serving precomputed JSON artifacts.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from typing import Dict, Any, Optional
import pandas as pd
import json
from pathlib import Path

from src.api.dependencies import get_analytics_cache
from src.config import ANALYTICS_OUTPUT_DIR

router = APIRouter(prefix="/api/v1", tags=["analytics"])

def _get_artifact(cache: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in cache:
        raise HTTPException(status_code=503, detail=f"Analytics artifact '{name}' is missing or unavailable.")
    return cache[name]

@router.get("/analytics/summary")
async def get_summary(
    request: Request,
    cache: Dict[str, Any] = Depends(get_analytics_cache)
):
    data = _get_artifact(cache, "overall_summary")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", {}), "request_id": request.state.request_id}

@router.get("/analytics/sites")
async def get_sites(
    request: Request,
    sort_by: Optional[str] = Query(None, description="Field to sort by, e.g. sif_density"),
    cache: Dict[str, Any] = Depends(get_analytics_cache)
):
    data = _get_artifact(cache, "site_analytics")
    items = data.get("data", [])
    
    if sort_by:
        valid_sort_keys = ["sif_density", "total_reports", "sif_count", "avg_sif_probability"]
        if sort_by not in valid_sort_keys:
            raise HTTPException(status_code=400, detail=f"Invalid sort_by parameter. Must be one of {valid_sort_keys}")
        items = sorted(items, key=lambda x: x.get(sort_by, 0), reverse=True)
        
    return {"metadata": data.get("metadata", {}), "data": items, "request_id": request.state.request_id}

@router.get("/analytics/activities")
async def get_activities(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "activity_analytics")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", []), "request_id": request.state.request_id}

@router.get("/analytics/lsr")
async def get_lsr(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "lsr_analytics")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", []), "request_id": request.state.request_id}

@router.get("/analytics/hazards")
async def get_hazards(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "hazard_analytics")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", []), "request_id": request.state.request_id}

@router.get("/analytics/barriers")
async def get_barriers(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "barrier_analytics")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", []), "request_id": request.state.request_id}

@router.get("/analytics/consequences")
async def get_consequences(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "consequence_analytics")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", []), "request_id": request.state.request_id}

@router.get("/analytics/emerging")
async def get_emerging(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "emerging_patterns")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", []), "request_id": request.state.request_id}

@router.get("/analytics/temporal")
async def get_temporal(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "temporal_analytics")
    return {"metadata": data.get("metadata", {}), "data": data.get("data", {}), "request_id": request.state.request_id}

@router.get("/analytics/metadata")
async def get_metadata(request: Request, cache: Dict[str, Any] = Depends(get_analytics_cache)):
    data = _get_artifact(cache, "overall_summary")
    return {"metadata": data.get("metadata", {}), "request_id": request.state.request_id}

@router.get("/analytics/patterns")
async def get_patterns(
    request: Request,
    min_support: Optional[int] = Query(None, ge=1),
    site: Optional[str] = Query(None),
    pattern_type: Optional[str] = Query(None),
    cache: Dict[str, Any] = Depends(get_analytics_cache)
):
    data = _get_artifact(cache, "recurring_patterns")
    patterns = data.get("data", [])
    
    # Apply filters
    if min_support is not None:
        patterns = [p for p in patterns if p.get("occurrences", 0) >= min_support]
    if site is not None:
        patterns = [p for p in patterns if site in p.get("affected_sites", [])]
    if pattern_type is not None:
        patterns = [p for p in patterns if p.get("pattern_type") == pattern_type]
        
    return {"metadata": data.get("metadata", {}), "data": patterns, "request_id": request.state.request_id}

@router.get("/analytics/patterns/{pattern_id}")
async def get_pattern_by_id(
    request: Request,
    pattern_id: str,
    cache: Dict[str, Any] = Depends(get_analytics_cache)
):
    data = _get_artifact(cache, "recurring_patterns")
    patterns = data.get("data", [])
    
    for p in patterns:
        if p.get("pattern_id") == pattern_id:
            return {"metadata": data.get("metadata", {}), "data": p, "request_id": request.state.request_id}
            
    raise HTTPException(status_code=404, detail=f"Pattern with ID '{pattern_id}' not found.")

@router.get("/reports")
async def list_reports(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    cache: Dict[str, Any] = Depends(get_analytics_cache) # ensures cache is loaded
):
    enriched_file = Path(ANALYTICS_OUTPUT_DIR) / "enriched_reports.csv"
    if not enriched_file.exists():
        raise HTTPException(status_code=503, detail="Enriched dataset not available.")
        
    try:
        df = pd.read_csv(enriched_file, dtype=str)
        
        # Paginate
        total_items = len(df)
        total_pages = (total_items + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        page_df = df.iloc[start_idx:end_idx]
        
        # Clean dataframe for JSON response
        records = page_df.where(pd.notnull(page_df), None).to_dict(orient="records")
        
        metadata = {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages
        }
        
        return {"metadata": metadata, "data": records, "request_id": request.state.request_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading reports: {str(e)}")

@router.get("/reports/{report_id}")
async def get_report(
    request: Request,
    report_id: str,
    include_ground_truth: bool = Query(False, description="Include ground truth data for evaluation.")
):
    enriched_file = Path(ANALYTICS_OUTPUT_DIR) / "enriched_reports.csv"
    if not enriched_file.exists():
        raise HTTPException(status_code=503, detail="Enriched dataset not available.")
        
    try:
        # Load specifically for this request - doing this efficiently is hard without a DB
        # In a real app we would use SQLite or a search index. 
        # Here we just iterate or read in chunks to find the row.
        df = pd.read_csv(enriched_file, dtype=str)
        report_row = df[df["report_id"] == report_id]
        
        if report_row.empty:
            raise HTTPException(status_code=404, detail=f"Report with ID '{report_id}' not found.")
            
        row = report_row.iloc[0].to_dict()
        
        def safe_json(val):
            if pd.isna(val) or val == "":
                return None
            try:
                return json.loads(val.replace("'", '"'))
            except:
                return val
                
        # Parse fields
        original_report = {
            "description": row.get("description", ""),
            "site": row.get("site", ""),
            "date": row.get("date", "")
        }
        
        predicted_fields = {
            "sif": row.get("predicted_sif", ""),
            "sif_probability": row.get("sif_probability", ""),
            "primary_lsr": row.get("predicted_lsr", ""),
            "activities": safe_json(row.get("predicted_activity", "[]")),
            "hazards": safe_json(row.get("predicted_hazards", "[]")),
            "barrier_failures": safe_json(row.get("predicted_barrier_failures", "[]"))
        }
        
        response = {
            "report_id": report_id,
            "original_report": original_report,
            "predicted_fields": predicted_fields,
            "request_id": request.state.request_id
        }
        
        if include_ground_truth:
            response["evaluation_ground_truth"] = {
                "sif": row.get("sif", ""),
                "lsr": row.get("lsr", ""),
                "activities": safe_json(row.get("activity", "[]")),
                "hazards": safe_json(row.get("hazard", "[]")),
                "barrier_failures": safe_json(row.get("barrier_failure", "[]"))
            }
            
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {str(e)}")

@router.get("/models/info")
async def get_models_info(
    request: Request,
    pipeline = Depends(get_analytics_cache) # just a dependency for standard shape, we can pull config
):
    from src.config import LSR_DEFAULT_GLOBAL_THRESHOLD, EMBEDDING_MODEL_NAME, ANALYTICS_PROTOTYPE_NOTICE
    
    return {
        "production_sif_model_type": "embedding_logistic_regression",
        "production_lsr_model_type": "embedding_ovr_logistic_regression",
        "sentence_embedding_model": EMBEDDING_MODEL_NAME,
        "sif_threshold": 0.5,
        "lsr_threshold": LSR_DEFAULT_GLOBAL_THRESHOLD,
        "prototype_notice": ANALYTICS_PROTOTYPE_NOTICE,
        "synthetic_data_notice": "All models trained and evaluated on generated synthetic data.",
        "request_id": request.state.request_id
    }
