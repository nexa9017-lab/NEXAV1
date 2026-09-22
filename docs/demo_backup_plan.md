# Demo Backup Plan

During live demonstrations, technical failures can occur. This document outlines backup procedures to ensure the presentation can continue smoothly regardless of infrastructure issues.

## Scenario 1: FastAPI Backend Fails to Start
**Symptoms:** The Streamlit dashboard shows "Backend unavailable. Start the FastAPI server before using the dashboard."
**Action:**
1. Do not attempt to debug live. 
2. Immediately switch to the pre-captured screenshots located in `docs/screenshots/`.
3. Walk through the screenshots as if navigating the live dashboard, following the speaking points in `docs/demo_script.md`.

## Scenario 2: Streamlit Dashboard Fails
**Symptoms:** The dashboard crashes, freezes, or fails to start.
**Action:**
1. Switch to the FastAPI Swagger UI available at `http://localhost:8000/docs`.
2. Demonstrate the API functionality directly. 
3. Open the `POST /api/v1/predict` endpoint, click "Try it out", and paste one of the examples from `data/demo/demo_reports.json`.
4. Show the rich JSON response, explaining how it extracts SIF probability, Life-Saving Rules, and evidence.
5. Use `GET /api/v1/analytics/patterns` to show how recurring patterns are served to the frontend.

## Scenario 3: Internet Connection is Lost
**Symptoms:** Network timeouts.
**Action:**
- The entire application (FastAPI + Streamlit) is designed to run locally. 
- As long as the `sentence-transformers/all-MiniLM-L6-v2` model weights are cached locally (which happens upon first run), the system requires zero internet access.
- Continue the demo as normal.

## Scenario 4: Live Prediction Fails
**Symptoms:** Hitting "Analyse Report" on the new report page throws an error.
**Action:**
1. Pivot to explaining that while live inference encountered an error, the system pre-computes predictions in batch for historical analytics.
2. Open `artifacts/demo/demo_predictions.json` in an IDE or text editor.
3. Show the pre-calculated outputs for the canonical demo reports to prove the pipeline's capabilities.
