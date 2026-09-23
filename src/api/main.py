"""
FastAPI application for Safety Analytics Unified API.
"""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import init_rag_dependencies, get_dependency_health
from src.api.schemas import RAGHealthResponse
from src.api.routers import analytics, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Initializing RAG dependencies...")
    start_time = time.time()
    
    init_rag_dependencies()
    
    elapsed = time.time() - start_time
    print(f"Startup complete in {elapsed:.2f} seconds.")
    yield
    print("Application shutdown: Cleaning up resources...")

app = FastAPI(
    title="Safety Analytics Unified API",
    description="Unified inference and analytics API for HSE reporting.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    req_id = str(uuid.uuid4())
    request.state.request_id = req_id
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
    except Exception as exc:
        print(f"[{req_id}] ERROR {request.method} {request.url.path} - {str(exc)}")
        raise
        
    process_time = (time.time() - start_time) * 1000
    
    print(f"[{req_id}] {request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response

# Root & Health
@app.get("/", tags=["system"])
async def read_root():
    return {
        "app_name": "Safety Intelligence RAG API",
        "version": "1.0.0",
        "phase": "R6",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.get("/health", response_model=RAGHealthResponse, tags=["system"])
async def health_check(request: Request):
    req_id = getattr(request.state, "request_id", None)
    health_info = get_dependency_health()
    return RAGHealthResponse(
        status=health_info["status"],
        components=health_info["components"],
        request_id=req_id
    )

# Include Routers (Phase R6 & R7)
app.include_router(rag.router)
app.include_router(analytics.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
