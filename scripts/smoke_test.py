"""
Smoke test to programmatically verify the end-to-end MVP.
"""
import sys
import pytest
from fastapi.testclient import TestClient

from src.pipeline.inference import SafetyInferencePipeline
from src.api.main import app

def run_smoke_test():
    print("Starting End-to-End Smoke Test...\n")
    
    # 1. Pipeline Test
    print("1. Testing SafetyInferencePipeline loading...")
    pipeline = SafetyInferencePipeline()
    try:
        pipeline.load_models()
        print("   -> Success")
    except Exception as e:
        print(f"   -> Failed: {e}")
        return False
        
    print("2. Testing Pipeline inference...")
    try:
        report = {"description": "Worker was grinding near a confined space without a valid permit."}
        res = pipeline.predict_single(report)
        if "sif" not in res or "life_saving_rules" not in res:
            print("   -> Failed: Missing key outputs in prediction.")
            return False
        print("   -> Success")
    except Exception as e:
        print(f"   -> Failed: {e}")
        return False
        
    # 3. API Test using TestClient (this triggers lifespan events which loads pipeline & cache)
    print("3. Testing FastAPI Endpoints...")
    try:
        with TestClient(app) as client:
            print("   -> Checking health...")
            health_res = client.get("/health")
            if health_res.status_code != 200:
                print("   -> Failed: Health check returned non-200")
                return False
                
            print("   -> Checking predict endpoint...")
            predict_res = client.post("/api/v1/predict", json={"description": "Test report"})
            if predict_res.status_code != 200:
                print("   -> Failed: Predict endpoint returned non-200")
                return False
                
            print("   -> Checking analytics summary...")
            summary_res = client.get("/api/v1/analytics/summary")
            if summary_res.status_code != 200:
                print("   -> Failed: Analytics summary returned non-200")
                return False
                
            print("   -> Checking pattern lookup...")
            patterns_res = client.get("/api/v1/analytics/patterns")
            if patterns_res.status_code == 200:
                patterns_data = patterns_res.json()
                if patterns_data.get("patterns"):
                    pattern_id = patterns_data["patterns"][0]["pattern_id"]
                    detail_res = client.get(f"/api/v1/analytics/patterns/{pattern_id}")
                    if detail_res.status_code != 200:
                        print(f"   -> Failed: Pattern detail for {pattern_id} returned non-200")
                        return False
            print("   -> Success")
    except Exception as e:
        print(f"   -> Failed: {e}")
        return False
        
    print("\nSMOKE_TEST_PASSED = true")
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
