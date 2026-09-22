"""
Check Readiness script for MVP demo.
"""
import sys
from pathlib import Path

def check_readiness():
    print("Checking Demo Readiness...\n")
    
    critical_paths = [
        "requirements.txt",
        "artifacts/models/sif_embedding.joblib",
        "artifacts/models/lsr_embedding.joblib",
        "data/analytics/enriched_reports.csv",
        "artifacts/analytics/overall_summary.json",
        "artifacts/demo/demo_predictions.json"
    ]
    
    missing = False
    for p in critical_paths:
        path = Path(p)
        if path.exists():
            print(f" [OK] {p}")
        else:
            print(f" [MISSING] {p}")
            missing = True
            
    if missing:
        print("\nSome critical artifacts are missing. Run the preparation pipeline first.")
        return False
        
    print("\nChecking Imports...")
    try:
        from src.pipeline.inference import SafetyInferencePipeline
        print(" [OK] Unified Pipeline imports successfully.")
    except Exception as e:
        print(f" [ERROR] Unified Pipeline import failed: {e}")
        return False
        
    try:
        from src.api.main import app
        print(" [OK] FastAPI Backend imports successfully.")
    except Exception as e:
        print(f" [ERROR] FastAPI Backend import failed: {e}")
        return False
        
    try:
        from src.dashboard.app import main
        print(" [OK] Streamlit Dashboard imports successfully.")
    except Exception as e:
        print(f" [ERROR] Streamlit Dashboard import failed: {e}")
        return False
        
    print("\nDEMO_READY = true")
    return True

if __name__ == "__main__":
    success = check_readiness()
    sys.exit(0 if success else 1)
