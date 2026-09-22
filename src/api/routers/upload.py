"""
File upload and data processing endpoints.
"""

import os
import csv
import json
import uuid
import datetime
import shutil
import pandas as pd
from pathlib import Path
from io import StringIO
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from src.pipeline.inference import SafetyInferencePipeline
from src.api.dependencies import get_pipeline, reload_analytics_cache
from src.analytics.pipeline import AnalyticsPipeline
from src.config import ENRICHED_REPORTS_PATH, ANALYTICS_OUTPUT_DIR

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

MAX_UPLOAD_ROWS = 15000
DATA_DIR = Path("data/analytics")
UPLOADS_DIR = DATA_DIR / "uploads"
ARTIFACTS_DIR = Path(ANALYTICS_OUTPUT_DIR)
CANDIDATE_DATA_PATH = DATA_DIR / "enriched_reports_upload_candidate.csv"
CANDIDATE_ARTIFACTS_DIR = Path("artifacts/analytics_upload_candidate")

def _safe_json_str(obj):
    try:
        return json.dumps(obj)
    except:
        return "[]"

@router.post("")
async def upload_dataset(
    file: UploadFile = File(...),
    pipeline: SafetyInferencePipeline = Depends(get_pipeline)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        csv_str = contents.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV encoding: {e}")
        
    try:
        df = pd.read_csv(StringIO(csv_str))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
        
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
        
    if len(df) > MAX_UPLOAD_ROWS:
        raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed rows ({MAX_UPLOAD_ROWS}).")
        
    if "description" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'description' column.")
        
    # Drop rows with empty descriptions
    df = df.dropna(subset=['description'])
    if df.empty:
        raise HTTPException(status_code=400, detail="All descriptions were empty.")
        
    # Standardize columns
    if "report_id" not in df.columns:
        df["report_id"] = [f"UPLOAD_{str(uuid.uuid4())[:8].upper()}" for _ in range(len(df))]
    if "site" not in df.columns:
        df["site"] = "Unknown"
    if "location" not in df.columns:
        df["location"] = "Unknown"
    if "report_type" not in df.columns:
        df["report_type"] = "Unspecified"
    if "date" not in df.columns:
        df["date"] = datetime.datetime.now().strftime("%Y-%m-%d")
        
    # Run Inference
    raw_reports = df.to_dict('records')
    try:
        predictions = pipeline.predict_batch(raw_reports)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Inference failed: {e}")
        
    # Build enriched schema
    enriched_rows = []
    sif_count = 0
    for row, pred in zip(raw_reports, predictions):
        if "error" in pred:
            continue
            
        sif_prob = pred.get("sif", {}).get("probability", 0.0)
        sif_label = pred.get("sif", {}).get("label", "Unknown")
        if sif_label == "SIF-Potential":
            sif_count += 1
            
        enriched_row = {
            "report_id": row.get("report_id"),
            "date": row.get("date"),
            "site": row.get("site"),
            "location": row.get("location"),
            "report_type": row.get("report_type"),
            "description": row.get("description"),
            
            "sif_prediction": 1 if sif_label == "SIF-Potential" else 0,
            "sif_probability": sif_prob,
            "primary_lsr": pred.get("primary_lsr"),
            
            # Safe JSON strings
            "predicted_activity": _safe_json_str(pred.get("activities", [])),
            "predicted_equipment": _safe_json_str(pred.get("equipment", [])),
            "predicted_hazards": _safe_json_str(pred.get("hazards", [])),
            "predicted_barrier_failures": _safe_json_str(pred.get("barrier_failures", [])),
            "predicted_potential_consequences": _safe_json_str(pred.get("potential_consequences", [])),
            "predicted_precursor_tags": _safe_json_str(pred.get("precursor_tags", [])),
            "predicted_lsr": _safe_json_str(pred.get("life_saving_rules", []))
        }
        enriched_rows.append(enriched_row)
        
    if not enriched_rows:
        raise HTTPException(status_code=500, detail="Inference succeeded but yielded zero valid rows.")

    # Save Candidate
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CANDIDATE_ARTIFACTS_DIR, exist_ok=True)
    
    candidate_df = pd.DataFrame(enriched_rows)
    candidate_df.to_csv(CANDIDATE_DATA_PATH, index=False)
    
    # Run Analytics Candidate
    try:
        analytics_pipe = AnalyticsPipeline(
            enriched_path=CANDIDATE_DATA_PATH,
            output_dir=str(CANDIDATE_ARTIFACTS_DIR)
        )
        analytics_pipe.load_data()
        analytics_pipe.run()
        analytics_pipe.save()
    except Exception as e:
        # Cleanup
        if CANDIDATE_DATA_PATH.exists():
            CANDIDATE_DATA_PATH.unlink()
        raise HTTPException(status_code=500, detail=f"Analytics rebuild failed: {e}")
        
    # Promote Candidate (Atomic-ish)
    try:
        # Version dataset
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"upload_{timestamp}"
        
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        versioned_path = UPLOADS_DIR / f"{version_name}_enriched.csv"
        
        # Copy to active and versioned
        shutil.copy2(CANDIDATE_DATA_PATH, ENRICHED_REPORTS_PATH)
        shutil.copy2(CANDIDATE_DATA_PATH, versioned_path)
        
        # Copy JSON artifacts
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        for json_file in CANDIDATE_ARTIFACTS_DIR.glob("*.json"):
            shutil.copy2(json_file, ARTIFACTS_DIR / json_file.name)
            
        # Clean up candidates
        if CANDIDATE_DATA_PATH.exists():
            CANDIDATE_DATA_PATH.unlink()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to promote candidate files: {e}")
        
    # Reload Cache
    cache_reloaded = reload_analytics_cache()
    if not cache_reloaded:
        raise HTTPException(status_code=500, detail="Failed to reload memory cache after promotion.")
        
    # Save Metadata
    metadata = {
        "dataset_source": "upload",
        "filename": file.filename,
        "upload_timestamp": datetime.datetime.now().isoformat(),
        "row_count": len(enriched_rows),
        "dataset_version": version_name
    }
    with open(DATA_DIR / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f)
        
    return {
        "status": "success",
        "rows_processed": len(enriched_rows),
        "sif_reports": sif_count,
        "sif_density": sif_count / len(enriched_rows) if len(enriched_rows) > 0 else 0,
        "dataset_version": version_name,
        "analytics_rebuilt": True,
        "cache_reloaded": True
    }


@router.post("/reset")
async def reset_dataset():
    """Restore the default synthetic dataset."""
    default_data = DATA_DIR / "default" / "default_enriched_reports.csv"
    default_artifacts = ARTIFACTS_DIR / "default"
    
    if not default_data.exists() or not default_artifacts.exists():
        raise HTTPException(status_code=500, detail="Default baseline copies not found.")
        
    try:
        shutil.copy2(default_data, ENRICHED_REPORTS_PATH)
        
        for json_file in default_artifacts.glob("*.json"):
            shutil.copy2(json_file, ARTIFACTS_DIR / json_file.name)
            
        # Reload cache
        reload_analytics_cache()
        
        # Update metadata
        metadata = {
            "dataset_source": "synthetic_baseline",
            "filename": "default_enriched_reports.csv",
            "upload_timestamp": datetime.datetime.now().isoformat(),
            "dataset_version": "baseline"
        }
        with open(DATA_DIR / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f)
            
        return {"status": "success", "message": "Baseline restored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@router.get("/metadata")
async def get_metadata():
    metadata_path = DATA_DIR / "dataset_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {
        "dataset_source": "synthetic_baseline",
        "dataset_version": "baseline"
    }
