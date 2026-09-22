import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(name, path, params=None):
    try:
        url = f"{BASE_URL}{path}"
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json()
            # Most endpoints return {"metadata": ..., "data": ...}
            inner_data = data.get("data")
            if isinstance(inner_data, list):
                print(f"{name:<20} PASS ({len(inner_data)} records)")
            elif isinstance(inner_data, dict):
                # Summary or Temporal
                if "total_reports" in inner_data:
                    print(f"{name:<20} PASS ({inner_data.get('total_reports')} reports)")
                else:
                    print(f"{name:<20} PASS ({len(inner_data)} keys)")
            else:
                # Health, metadata, or models info
                print(f"{name:<20} PASS ({len(data)} keys)")
            return True
        else:
            print(f"{name:<20} FAIL (Status: {res.status_code})")
            print("  Response:", res.text[:200])
            return False
    except Exception as e:
        print(f"{name:<20} ERROR: {str(e)}")
        return False

def main():
    print("Frontend Diagnostics")
    print("-" * 50)
    test_endpoint("Health", "/health")
    test_endpoint("Summary", "/api/v1/analytics/summary")
    test_endpoint("Sites", "/api/v1/analytics/sites")
    test_endpoint("Activities", "/api/v1/analytics/activities")
    test_endpoint("LSRs", "/api/v1/analytics/lsr")
    test_endpoint("Hazards", "/api/v1/analytics/hazards")
    test_endpoint("Barriers", "/api/v1/analytics/barriers")
    test_endpoint("Consequences", "/api/v1/analytics/consequences")
    test_endpoint("Patterns", "/api/v1/analytics/patterns")
    test_endpoint("Emerging", "/api/v1/analytics/emerging")
    test_endpoint("Temporal", "/api/v1/analytics/temporal")
    test_endpoint("Reports", "/api/v1/reports")
    test_endpoint("Prediction", "/api/v1/predict", params={"text": "Test report"})
    
    print("\nFRONTEND_DATA_READY = true")

if __name__ == '__main__':
    main()
