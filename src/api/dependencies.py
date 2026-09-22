"""
FastAPI dependencies and global state.
"""

import json
from pathlib import Path
from typing import Dict, Any

from src.pipeline.inference import SafetyInferencePipeline
from src.config import ANALYTICS_OUTPUT_DIR

# Global State
pipeline = SafetyInferencePipeline()
analytics_cache: Dict[str, Any] = {}

def get_pipeline() -> SafetyInferencePipeline:
    return pipeline

def get_analytics_cache() -> Dict[str, Any]:
    return analytics_cache

def load_analytics_artifacts():
    """Load precomputed analytics JSON files into memory cache."""
    cache = {}
    analytics_dir = Path(ANALYTICS_OUTPUT_DIR)
    
    if not analytics_dir.exists():
        print(f"Warning: Analytics directory not found at {analytics_dir}")
        return cache

    # List of expected files
    files_to_load = [
        "overall_summary", "site_analytics", "activity_analytics",
        "lsr_analytics", "hazard_analytics", "barrier_analytics",
        "consequence_analytics", "temporal_analytics", "recurring_patterns",
        "emerging_patterns", "cross_site_patterns", "analytics_validation",
        "full_analytics"
    ]
    
    loaded_count = 0
    for name in files_to_load:
        file_path = analytics_dir / f"{name}.json"
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    cache[name] = json.load(f)
                loaded_count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: Analytics artifact {file_path} missing.")
            
    # Also load the enriched dataset for report lookup (if needed, but memory heavy)
    # We will do a lazy read or load only essential columns if needed.
    
    print(f"Loaded {loaded_count} analytics artifacts into cache.")
    return cache

def init_globals():
    """Initialize models and analytics on app startup."""
    try:
        pipeline.load_models()
    except Exception as e:
        print(f"ERROR: Failed to load inference models: {e}")
        
    try:
        global analytics_cache
        analytics_cache = load_analytics_artifacts()
    except Exception as e:
        print(f"ERROR: Failed to load analytics cache: {e}")

def reload_analytics_cache() -> bool:
    """Reloads the analytics cache atomically after validating."""
    try:
        new_cache = load_analytics_artifacts()
        if "overall_summary" in new_cache and "recurring_patterns" in new_cache:
            global analytics_cache
            analytics_cache = new_cache
            return True
        else:
            print("ERROR: New cache missing critical artifacts. Aborting reload.")
            return False
    except Exception as e:
        print(f"ERROR: Failed to reload analytics cache: {e}")
        return False
