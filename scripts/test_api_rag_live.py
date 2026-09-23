"""
Live smoke-test script for FastAPI RAG endpoints (Phase R6).
Executes real requests through FastAPI TestClient with full lifespan initialization:
1. GET  /health
2. GET  /api/v1/index/info
3. POST /api/v1/search
4. POST /api/v1/analyze
"""
import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 80)
    print("PHASE R6: FASTAPI RAG ENDPOINTS LIVE SMOKE TEST")
    print("=" * 80)

    print("\nStarting FastAPI TestClient with real lifespan initialization...")
    t0 = time.time()
    with TestClient(app) as client:
        print(f"Server started in {round((time.time() - t0) * 1000, 2)} ms.\n")

        # 1. Health check
        print("-" * 80)
        print("[1/4] GET /health")
        print("-" * 80)
        t_req = time.time()
        resp = client.get("/health")
        elapsed = round((time.time() - t_req) * 1000, 2)
        print(f"Status: {resp.status_code} ({elapsed} ms)")
        print(f"Payload: {resp.json()}")

        # 2. Index info
        print("\n" + "-" * 80)
        print("[2/4] GET /api/v1/index/info")
        print("-" * 80)
        t_req = time.time()
        resp = client.get("/api/v1/index/info")
        elapsed = round((time.time() - t_req) * 1000, 2)
        print(f"Status: {resp.status_code} ({elapsed} ms)")
        print(f"Payload: {resp.json()}")

        # 3. Semantic search
        print("\n" + "-" * 80)
        print("[3/4] POST /api/v1/search")
        print("-" * 80)
        search_query = "Technician began opening the pump discharge flange before isolation was verified"
        t_req = time.time()
        resp = client.post(
            "/api/v1/search",
            json={"query": search_query, "top_k": 3},
        )
        elapsed = round((time.time() - t_req) * 1000, 2)
        print(f"Status: {resp.status_code} ({elapsed} ms)")
        search_data = resp.json()
        print(f"Results Count: {search_data.get('count')}")
        for r in search_data.get("results", []):
            print(f"  Rank {r['rank']} | ID: {r['incident_id']} | Score: {r['similarity_score']} | Site: {r['site']}")
            print(f"    Desc: {r['description'][:90]}...")

        # 4. Full unified analysis
        print("\n" + "-" * 80)
        print("[4/4] POST /api/v1/analyze")
        print("-" * 80)
        analyze_payload = {
            "description": (
                "Technician began opening the pump discharge flange before isolation was verified. "
                "Residual pressure remained in the line."
            ),
            "top_k": 5,
            "site": "Refinery Alpha",
            "report_type": "Near Miss",
        }
        t_req = time.time()
        resp = client.post("/api/v1/analyze", json=analyze_payload)
        elapsed = round((time.time() - t_req) * 1000, 2)
        print(f"Status: {resp.status_code} ({elapsed} ms)")

        if resp.status_code == 200:
            analysis = resp.json()
            leg = analysis.get("model_assessment") or {}
            rag = analysis.get("rag_assessment") or {}
            cons = analysis.get("consistency") or {}
            sim = analysis.get("similar_incidents") or []

            print("\n  Subsystem 1 - Legacy Model:")
            print(f"    SIF Prediction: {leg.get('sif_prediction')} (prob: {leg.get('sif_probability')})")
            print(f"    Primary LSR:    {leg.get('primary_lsr')}")
            print(f"    Equipment:      {leg.get('equipment')}")
            print(f"    Hazards:        {leg.get('hazards')}")

            print("\n  Subsystem 2 - Semantic Retrieval:")
            print(f"    Retrieved Count: {len(sim)} historical incidents")
            for inc in sim[:2]:
                print(f"    - Rank {inc['rank']}: ID={inc['incident_id']} (score: {inc['similarity_score']})")

            print("\n  Subsystem 3 - Groq LLM Reasoning:")
            print(f"    SIF Potential:         {rag.get('sif_potential')}")
            print(f"    Assessment Confidence: {rag.get('assessment_confidence')}")
            print(f"    Life-Saving Rules:     {rag.get('life_saving_rules')}")
            print(f"    Recurring Precursor:   {rag.get('recurring_precursor_pattern')}")

            print("\n  Subsystem 4 - Consistency & Conflicts:")
            print(f"    SIF Agreement: {cons.get('sif_agreement')}")
            print(f"    LSR Overlap:   {cons.get('lsr_overlap')}")
            print(f"    Conflicts:     {cons.get('conflicts')}")

            print("\n  Execution Metadata:")
            print(f"    Warnings:  {analysis.get('warnings')}")
            print(f"    Latencies: {analysis.get('metadata', {}).get('latencies')}")
        else:
            print(f"Analysis failed: {resp.text}")

    print("\n" + "=" * 80)
    print("LIVE API SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
