"""
Prepare Demo script.
Generates demo_predictions.json by running demo_reports.json through the pipeline.
"""
import json
import os
from pathlib import Path
from src.pipeline.inference import SafetyInferencePipeline

DEMO_REPORTS_PATH = Path("data/demo/demo_reports.json")
DEMO_PREDICTIONS_PATH = Path("artifacts/demo/demo_predictions.json")

def generate_demo_predictions():
    print("Preparing demo predictions...")
    
    if not DEMO_REPORTS_PATH.exists():
        print(f"Error: {DEMO_REPORTS_PATH} not found.")
        return False
        
    os.makedirs(DEMO_PREDICTIONS_PATH.parent, exist_ok=True)
    
    with open(DEMO_REPORTS_PATH, "r") as f:
        demo_reports = json.load(f)
        
    pipeline = SafetyInferencePipeline()
    pipeline.load_models()
    
    print(f"Running inference on {len(demo_reports)} demo reports...")
    predictions = pipeline.predict_batch(demo_reports)
    
    with open(DEMO_PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f, indent=2)
        
    print(f"Demo predictions saved to {DEMO_PREDICTIONS_PATH}")
    return True

if __name__ == "__main__":
    success = generate_demo_predictions()
    if not success:
        import sys
        sys.exit(1)
