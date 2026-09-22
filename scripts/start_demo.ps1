# Start Demo PowerShell Script
# Starts the FastAPI backend and Streamlit dashboard in the background

Write-Host "Checking readiness..."
python -m scripts.check_readiness
if ($LASTEXITCODE -ne 0) {
    Write-Host "Readiness check failed. Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "Starting FastAPI Backend..."
Start-Process "uvicorn" -ArgumentList "src.api.main:app --host 0.0.0.0 --port 8000" -WindowStyle Normal

Write-Host "Starting Streamlit Dashboard..."
Start-Process "streamlit" -ArgumentList "run src\dashboard\app.py" -WindowStyle Normal

Write-Host "Demo applications launched in separate windows." -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000/docs"
Write-Host "Dashboard: http://localhost:8501"
