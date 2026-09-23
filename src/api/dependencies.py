"""
FastAPI dependencies and global application state for the Safety RAG MVP (Phase R6).
Manages one-time initialization of heavy models, vector stores, retrievers, and LLM clients.
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import HTTPException

import json
from src.config import (
    VECTOR_INDEX_PATH,
    VECTOR_METADATA_PATH,
    ANALYTICS_OUTPUT_DIR,
    GROQ_MODEL,
    RAG_EMBEDDING_MODEL,
)
from src.preprocessing.text_preprocessor import SafetyTextPreprocessor
from src.pipeline.inference import SafetyInferencePipeline
from src.rag.embeddings import IncidentEmbedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import IncidentRetriever
from src.rag.prompts import SafetyPromptBuilder
from src.rag.llm_client import GroqLLMClient, BaseLLMClient
from src.rag.pipeline import SafetyRAGPipeline

# ---------------------------------------------------------------------------
# Module-level Production Singletons
# ---------------------------------------------------------------------------
_embedder: Optional[IncidentEmbedder] = None
_vector_store: Optional[VectorStore] = None
_retriever: Optional[IncidentRetriever] = None
_legacy_pipeline: Optional[SafetyInferencePipeline] = None
_prompt_builder: Optional[SafetyPromptBuilder] = None
_llm_client: Optional[BaseLLMClient] = None
_rag_pipeline: Optional[SafetyRAGPipeline] = None
_preprocessor: Optional[SafetyTextPreprocessor] = None

_components_status: Dict[str, str] = {
    "embedder": "uninitialized",
    "vector_store": "uninitialized",
    "retriever": "uninitialized",
    "legacy_pipeline": "uninitialized",
    "llm": "uninitialized",
    "rag_pipeline": "uninitialized",
}


def load_environment_keys():
    """Load GROQ_API_KEY from environment or .env if present."""
    if not os.getenv("GROQ_API_KEY"):
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and not val.startswith("your_"):
                            os.environ["GROQ_API_KEY"] = val
                    elif line.startswith("GROQ_MODEL="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ["GROQ_MODEL"] = val


def init_rag_dependencies(
    skip_legacy: bool = False,
    override_llm_client: Optional[BaseLLMClient] = None
) -> None:
    """
    Initializes all production components ONCE during FastAPI lifespan startup.
    Does NOT reload them per request.
    Does NOT crash if GROQ_API_KEY is missing; enters graceful degraded mode.
    """
    global _embedder, _vector_store, _retriever, _legacy_pipeline
    global _prompt_builder, _llm_client, _rag_pipeline, _preprocessor
    global _components_status

    load_environment_keys()
    print("Initializing production RAG dependencies...")
    t_start = time.time()

    # 1. Preprocessor & Embedder
    try:
        _preprocessor = SafetyTextPreprocessor()
        _embedder = IncidentEmbedder(model_name=RAG_EMBEDDING_MODEL)
        _components_status["embedder"] = "ready"
    except Exception as e:
        print(f"ERROR initializing embedder: {e}")
        _components_status["embedder"] = "error"

    # 2. Vector Store
    try:
        store = VectorStore(
            index_path=VECTOR_INDEX_PATH,
            metadata_path=VECTOR_METADATA_PATH,
            embedding_model=RAG_EMBEDDING_MODEL,
        )
        if store.load_index():
            _vector_store = store
            _components_status["vector_store"] = "ready"
        else:
            print(f"WARNING: Vector store index missing at {VECTOR_INDEX_PATH}.")
            _components_status["vector_store"] = "unavailable"
    except Exception as e:
        print(f"ERROR loading vector store: {e}")
        _components_status["vector_store"] = "error"

    # 3. Retriever
    if _embedder is not None and _vector_store is not None:
        try:
            _retriever = IncidentRetriever(
                embedder=_embedder,
                vector_store=_vector_store,
                preprocessor=_preprocessor,
            )
            _components_status["retriever"] = "ready"
        except Exception as e:
            print(f"ERROR initializing retriever: {e}")
            _components_status["retriever"] = "error"
    else:
        _components_status["retriever"] = "unavailable"

    # 4. Legacy Pipeline
    if not skip_legacy:
        try:
            p = SafetyInferencePipeline()
            p.load_models()
            _legacy_pipeline = p
            _components_status["legacy_pipeline"] = "ready"
        except Exception as e:
            print(f"ERROR loading legacy pipeline: {e}")
            _components_status["legacy_pipeline"] = "error"
    else:
        _components_status["legacy_pipeline"] = "bypassed"

    # 5. Prompt Builder
    try:
        _prompt_builder = SafetyPromptBuilder()
    except Exception as e:
        print(f"ERROR initializing prompt builder: {e}")

    # 6. Groq LLM Client
    if override_llm_client is not None:
        _llm_client = override_llm_client
        _components_status["llm"] = "ready"
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                _llm_client = GroqLLMClient(api_key=api_key, model=GROQ_MODEL)
                _components_status["llm"] = "ready"
            except Exception as e:
                print(f"WARNING: Groq client failed to initialize: {e}")
                _llm_client = None
                _components_status["llm"] = "error"
        else:
            print("INFO: GROQ_API_KEY not found. LLM reasoning set to unavailable (degraded mode).")
            _llm_client = None
            _components_status["llm"] = "unavailable"

    # 7. Unified Safety RAG Pipeline
    if _llm_client is not None and _retriever is not None:
        _rag_pipeline = SafetyRAGPipeline(
            legacy_pipeline=_legacy_pipeline,
            retriever=_retriever,
            prompt_builder=_prompt_builder,
            llm_client=_llm_client,
        )
        _components_status["rag_pipeline"] = "ready"
    else:
        _rag_pipeline = None
        _components_status["rag_pipeline"] = "unavailable"

    # 8. Analytics Cache
    try:
        reload_analytics_cache()
    except Exception as e:
        print(f"WARNING: Initial analytics cache load failed: {e}")

    elapsed = round((time.time() - t_start) * 1000, 2)
    print(f"RAG dependencies initialized in {elapsed} ms. Component statuses: {_components_status}")


def update_vector_store_dependencies(new_vector_store: VectorStore) -> None:
    """
    Atomically updates the active vector store, retriever, and pipeline references
    after a successful reindexing operation.
    Keeps the active Groq client and legacy pipeline singletons unchanged.
    """
    global _vector_store, _retriever, _rag_pipeline, _components_status

    _vector_store = new_vector_store
    _components_status["vector_store"] = "ready"

    if _embedder is not None:
        _retriever = IncidentRetriever(
            embedder=_embedder,
            vector_store=_vector_store,
            preprocessor=_preprocessor,
        )
        _components_status["retriever"] = "ready"

        if _llm_client is not None:
            _rag_pipeline = SafetyRAGPipeline(
                legacy_pipeline=_legacy_pipeline,
                retriever=_retriever,
                prompt_builder=_prompt_builder,
                llm_client=_llm_client,
            )
            _components_status["rag_pipeline"] = "ready"


# ---------------------------------------------------------------------------
# Dependency Accessors
# ---------------------------------------------------------------------------

def get_rag_pipeline() -> SafetyRAGPipeline:
    """Returns the initialized SafetyRAGPipeline or raises HTTP 503."""
    if _rag_pipeline is None:
        if _components_status.get("llm") == "unavailable":
            raise HTTPException(
                status_code=503,
                detail="Groq LLM is not configured. Set GROQ_API_KEY environment variable to enable /analyze."
            )
        raise HTTPException(
            status_code=503,
            detail=f"Safety RAG Pipeline unavailable (status: {_components_status.get('rag_pipeline')})."
        )
    return _rag_pipeline


def get_retriever() -> IncidentRetriever:
    """Returns the initialized IncidentRetriever or raises HTTP 503."""
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Incident Retriever unavailable. Vector store may not be loaded."
        )
    return _retriever


def get_vector_store() -> VectorStore:
    """Returns the initialized VectorStore or raises HTTP 503."""
    if _vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not loaded."
        )
    return _vector_store


def get_embedder() -> IncidentEmbedder:
    """Returns the initialized IncidentEmbedder or raises HTTP 503."""
    if _embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Incident Embedder is not initialized."
        )
    return _embedder


def get_dependency_health() -> Dict[str, Any]:
    """Returns health and readiness status for all components."""
    # System status is 'ok' only if core components and LLM are ready; otherwise 'degraded'
    core_ready = (
        _components_status.get("embedder") == "ready"
        and _components_status.get("vector_store") == "ready"
        and _components_status.get("retriever") == "ready"
    )
    llm_ready = _components_status.get("llm") == "ready"

    if core_ready and llm_ready:
        overall_status = "ok"
    elif core_ready:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "components": dict(_components_status),
    }


# ---------------------------------------------------------------------------
# Analytics Cache & Dataset Metadata Management
# ---------------------------------------------------------------------------

_analytics_cache: Dict[str, Any] = {}
_dataset_metadata: Dict[str, Any] = {
    "filename": "all_reports.csv",
    "report_count": 0,
    "last_updated": None,
    "dataset_version": "baseline",
}


def load_analytics_artifacts(analytics_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Loads all JSON files from the analytics directory into an in-memory dictionary."""
    target_dir = Path(analytics_dir or ANALYTICS_OUTPUT_DIR)
    cache: Dict[str, Any] = {}
    if target_dir.exists():
        for file in target_dir.glob("*.json"):
            stem = file.stem
            try:
                with open(file, "r", encoding="utf-8") as f:
                    cache[stem] = json.load(f)
            except Exception as e:
                print(f"Error loading artifact {file.name}: {e}")
    return cache


def reload_analytics_cache(
    analytics_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Reloads the in-memory analytics cache from disk and refreshes active metadata."""
    global _analytics_cache, _dataset_metadata
    _analytics_cache = load_analytics_artifacts(analytics_dir)

    if metadata:
        _dataset_metadata.update(metadata)
    elif "overall_summary" in _analytics_cache:
        meta = _analytics_cache["overall_summary"].get("metadata", {})
        _dataset_metadata["report_count"] = meta.get("report_count", 0)
        _dataset_metadata["last_updated"] = meta.get("generated_at")
        _dataset_metadata["filename"] = meta.get("source_dataset", "all_reports.csv")


def get_analytics_cache() -> Dict[str, Any]:
    """FastAPI dependency for accessing the active analytics cache."""
    global _analytics_cache
    if not _analytics_cache:
        reload_analytics_cache()
    return _analytics_cache


def get_dataset_metadata() -> Dict[str, Any]:
    """Returns metadata for the currently active dataset."""
    global _dataset_metadata
    if not _dataset_metadata.get("last_updated") and "overall_summary" in _analytics_cache:
        meta = _analytics_cache["overall_summary"].get("metadata", {})
        _dataset_metadata["report_count"] = meta.get("report_count", 0)
        _dataset_metadata["last_updated"] = meta.get("generated_at")
        _dataset_metadata["filename"] = meta.get("source_dataset", "all_reports.csv")
    return dict(_dataset_metadata)


def get_pipeline() -> Optional[SafetyInferencePipeline]:
    """Legacy accessor stub for historical test compatibility."""
    return _legacy_pipeline


def get_legacy_pipeline() -> Optional[SafetyInferencePipeline]:
    """Returns the production legacy inference pipeline instance."""
    return _legacy_pipeline


