"""
Inference endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any

from src.api.schemas import (
    SafetyReportRequest, BatchSafetyReportRequest, 
    SafetyAnalysisResponse, BatchSafetyAnalysisResponse
)
from src.api.dependencies import get_pipeline
from src.pipeline.inference import SafetyInferencePipeline

router = APIRouter(prefix="/api/v1/predict", tags=["inference"])

@router.post("", response_model=SafetyAnalysisResponse)
async def predict_single(
    request: Request,
    report: SafetyReportRequest,
    pipeline: SafetyInferencePipeline = Depends(get_pipeline)
):
    if not pipeline.loaded:
        raise HTTPException(status_code=503, detail="Inference models are not loaded or unavailable.")
    
    if not report.description or not report.description.strip():
        raise HTTPException(status_code=422, detail="Empty description provided.")
        
    try:
        req_id = request.state.request_id
        result = pipeline.predict_single(report.model_dump(exclude_none=True))
        result["request_id"] = req_id
        return result
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal inference error.")


@router.post("/batch", response_model=BatchSafetyAnalysisResponse)
async def predict_batch(
    request: Request,
    batch: BatchSafetyReportRequest,
    pipeline: SafetyInferencePipeline = Depends(get_pipeline)
):
    if not pipeline.loaded:
        raise HTTPException(status_code=503, detail="Inference models are not loaded or unavailable.")
        
    if len(batch.reports) == 0:
        raise HTTPException(status_code=422, detail="Empty batch provided.")
        
    if len(batch.reports) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 100.")
        
    try:
        req_id = request.state.request_id
        raw_reports = [r.model_dump(exclude_none=True) for r in batch.reports]
        results = pipeline.predict_batch(raw_reports)
        
        # Format the output to ensure type safety
        formatted_results = []
        for r in results:
            if "error" in r:
                formatted_results.append(SafetyAnalysisResponse(error=r["error"], request_id=req_id))
            else:
                r["request_id"] = req_id
                formatted_results.append(SafetyAnalysisResponse(**r))
                
        return {"results": formatted_results, "request_id": req_id}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal batch inference error.")
