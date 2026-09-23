"""
RAG Safety API Router (Phase R6).
Exposes /analyze, /search, /index/info, and /reindex endpoints.
"""
import io
import shutil
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

import uuid
import time
from src.config import (
    DEFAULT_TOP_K,
    VECTOR_INDEX_PATH,
    VECTOR_METADATA_PATH,
    RAG_EMBEDDING_MODEL,
    ANALYTICS_OUTPUT_DIR,
    ENRICHED_REPORTS_PATH,
)
from src.api.schemas import (
    AnalyzeIncidentRequest,
    UnifiedSafetyAssessment,
    SearchRequest,
    SearchResponse,
    RetrievedIncident,
    IndexInfoResponse,
    ReindexResponse,
)
from src.api.dependencies import (
    get_rag_pipeline,
    get_retriever,
    get_vector_store,
    get_embedder,
    get_legacy_pipeline,
    update_vector_store_dependencies,
    reload_analytics_cache,
)
from src.preprocessing.text_preprocessor import SafetyTextPreprocessor
from src.analytics.dataset_processor import process_dataset_enrichment_and_analytics
from src.rag.pipeline import SafetyRAGPipeline
from src.rag.retriever import IncidentRetriever
from src.rag.vector_store import VectorStore
from src.rag.embeddings import IncidentEmbedder

router = APIRouter(prefix="/api/v1", tags=["rag"])

ALLOWED_METADATA_COLS = ["report_id", "date", "site", "location", "report_type"]


@router.post(
    "/analyze",
    response_model=UnifiedSafetyAssessment,
    summary="Analyze new incident using unified pipeline",
    description="Runs legacy ML classifiers/extractors, FAISS historical retrieval, and Groq LLM reasoning into a unified assessment.",
)
async def analyze_incident(
    request: AnalyzeIncidentRequest,
    pipeline: SafetyRAGPipeline = Depends(get_rag_pipeline),
):
    desc = request.description.strip()
    if not desc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Incident description cannot be empty or whitespace-only.",
        )

    if request.top_k < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_k must be greater than or equal to 1.",
        )

    try:
        assessment = pipeline.analyze(incident_text=desc, top_k=request.top_k)
        return assessment
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Safety analysis execution failed: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search similar historical incidents",
    description="Dense semantic vector similarity search via FAISS. Does not invoke LLM.",
)
async def search_incidents(
    request: SearchRequest,
    retriever: IncidentRetriever = Depends(get_retriever),
):
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Search query cannot be empty or whitespace-only.",
        )

    if request.top_k < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_k must be greater than or equal to 1.",
        )

    try:
        hits = retriever.retrieve_similar_incidents(query=query, top_k=request.top_k)
        items = [
            RetrievedIncident(
                rank=h["rank"],
                vector_id=h["vector_id"],
                incident_id=str(h["incident_id"]),
                similarity_score=float(h["similarity_score"]),
                description=h["description"],
                site=h.get("site"),
                location=h.get("location"),
                date=h.get("date"),
                report_type=h.get("report_type"),
            )
            for h in hits
        ]
        return SearchResponse(query=query, count=len(items), results=items)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval failed: {str(e)}",
        )


@router.get(
    "/index/info",
    response_model=IndexInfoResponse,
    summary="Retrieve FAISS vector store status and metadata",
    description="Returns record count, embedding dimensions, model name, and loaded status without exposing server filesystem paths.",
)
async def get_index_info(
    store: VectorStore = Depends(get_vector_store),
):
    info = store.get_info()
    return IndexInfoResponse(
        loaded=bool(info.get("loaded", False)),
        total_records=int(info.get("total_records", 0)),
        dimension=info.get("dimension"),
        index_type=str(info.get("index_type", "IndexFlatIP")),
        embedding_model=str(info.get("embedding_model", RAG_EMBEDDING_MODEL)),
    )


@router.post(
    "/reindex",
    response_model=ReindexResponse,
    summary="Rebuild FAISS index and analytics from uploaded CSV",
    description="Builds candidate FAISS vector index, runs batch legacy inference, computes dataset analytics, validates all candidate artifacts, and performs safe candidate promotion with rollback.",
)
async def reindex_dataset(
    file: UploadFile = File(...),
    embedder: IncidentEmbedder = Depends(get_embedder),
    legacy_pipeline=Depends(get_legacy_pipeline),
):
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Uploaded file must be a CSV (.csv).",
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    if "description" not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Required column 'description' is missing from uploaded CSV.",
        )

    total_rows = len(df)
    valid_mask = df["description"].fillna("").astype(str).str.strip() != ""
    skipped_count = int((~valid_mask).sum())
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    valid_count = len(valid_df)

    if valid_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV contains no valid non-empty incident descriptions.",
        )

    candidate_vector_dir = Path("artifacts/vector_store_candidate")
    candidate_analytics_dir = Path("artifacts/analytics_candidate")
    candidate_enriched_csv = Path("data/analytics/enriched_reports_candidate.csv")
    backup_dir = Path("artifacts/backup_temp")

    candidate_index_path = candidate_vector_dir / "index.faiss"
    candidate_metadata_path = candidate_vector_dir / "index_metadata.json"

    dataset_version = f"upload-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    try:
        # Prepare candidate directories
        candidate_vector_dir.mkdir(parents=True, exist_ok=True)
        candidate_analytics_dir.mkdir(parents=True, exist_ok=True)
        candidate_enriched_csv.parent.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # 1. Candidate FAISS Vector Indexing
        # -------------------------------------------------------------
        preprocessor = SafetyTextPreprocessor()
        records: List[Dict[str, Any]] = []
        cleaned_texts: List[str] = []

        for _, row in valid_df.iterrows():
            raw_desc = str(row["description"]).strip()
            cleaned_desc = preprocessor.preprocess(raw_desc)
            cleaned_texts.append(cleaned_desc)

            record = {
                "description": raw_desc,
                "clean_description": cleaned_desc,
            }
            for col in ALLOWED_METADATA_COLS:
                if col in row and pd.notna(row[col]):
                    record[col] = str(row[col])
                else:
                    record[col] = None
            records.append(record)

        embeddings = embedder.embed_batch(cleaned_texts)

        candidate_store = VectorStore(
            index_path=candidate_index_path,
            metadata_path=candidate_metadata_path,
            embedding_model=RAG_EMBEDDING_MODEL,
        )
        candidate_store.build_index(records=records, embeddings=embeddings)
        candidate_store.save_index()

        # Validate candidate vector store
        verify_store = VectorStore(
            index_path=candidate_index_path,
            metadata_path=candidate_metadata_path,
        )
        if not verify_store.load_index() or verify_store.index.ntotal != valid_count:
            raise RuntimeError("Verification reload of candidate index failed.")

        # -------------------------------------------------------------
        # 2. Candidate Batch Inference & Analytics Generation
        # -------------------------------------------------------------
        analytics_stats = process_dataset_enrichment_and_analytics(
            df=valid_df,
            output_enriched_path=candidate_enriched_csv,
            output_analytics_dir=candidate_analytics_dir,
            legacy_pipeline=legacy_pipeline,
            source_filename=filename,
        )

        # Validate required candidate analytics JSONs
        required_jsons = [
            "overall_summary.json",
            "site_analytics.json",
            "activity_analytics.json",
            "lsr_analytics.json",
            "recurring_patterns.json",
        ]
        for rj in required_jsons:
            target_json = candidate_analytics_dir / rj
            if not target_json.exists() or target_json.stat().st_size == 0:
                raise RuntimeError(f"Candidate analytics validation failed: missing {rj}")

        # -------------------------------------------------------------
        # 3. Safe Candidate Promotion with Rollback
        # -------------------------------------------------------------
        active_index = Path(VECTOR_INDEX_PATH)
        active_meta = Path(VECTOR_METADATA_PATH)
        active_analytics = Path(ANALYTICS_OUTPUT_DIR)
        active_enriched = Path(ENRICHED_REPORTS_PATH)

        # Backup active artifacts if they exist
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_vector_index = backup_dir / "index.faiss"
        backup_vector_meta = backup_dir / "index_metadata.json"
        backup_enriched = backup_dir / "enriched_reports.csv"
        backup_analytics_dir = backup_dir / "analytics"

        if active_index.exists():
            shutil.copy2(active_index, backup_vector_index)
        if active_meta.exists():
            shutil.copy2(active_meta, backup_vector_meta)
        if active_enriched.exists():
            shutil.copy2(active_enriched, backup_enriched)
        if active_analytics.exists():
            shutil.copytree(active_analytics, backup_analytics_dir, dirs_exist_ok=True)

        try:
            # Promote vector store
            active_index.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_index_path, active_index)
            shutil.copy2(candidate_metadata_path, active_meta)

            # Promote enriched reports
            active_enriched.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_enriched_csv, active_enriched)

            # Promote analytics JSONs
            active_analytics.mkdir(parents=True, exist_ok=True)
            for f in candidate_analytics_dir.glob("*.json"):
                shutil.copy2(f, active_analytics / f.name)

        except Exception as promo_err:
            # Rollback to previous active state
            if backup_vector_index.exists():
                shutil.copy2(backup_vector_index, active_index)
            if backup_vector_meta.exists():
                shutil.copy2(backup_vector_meta, active_meta)
            if backup_enriched.exists():
                shutil.copy2(backup_enriched, active_enriched)
            if backup_analytics_dir.exists():
                shutil.copytree(backup_analytics_dir, active_analytics, dirs_exist_ok=True)
            raise RuntimeError(f"Promotion failed; rolled back to previous dataset: {str(promo_err)}")

        # -------------------------------------------------------------
        # 4. In-Memory Hot Swap & Cache Refresh
        # -------------------------------------------------------------
        new_active_store = VectorStore(
            index_path=active_index,
            metadata_path=active_meta,
            embedding_model=RAG_EMBEDDING_MODEL,
        )
        new_active_store.load_index()
        update_vector_store_dependencies(new_active_store)

        meta_info = {
            "filename": filename,
            "report_count": valid_count,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_version": dataset_version,
        }
        reload_analytics_cache(analytics_dir=active_analytics, metadata=meta_info)

        return ReindexResponse(
            status="success",
            records_received=total_rows,
            records_indexed=valid_count,
            records_skipped=skipped_count,
            records_processed=analytics_stats["records_processed"],
            sif_reports=analytics_stats["sif_reports"],
            sif_density=analytics_stats["sif_density"],
            dimension=int(embeddings.shape[1]),
            index_type="IndexFlatIP",
            vector_index_updated=True,
            analytics_updated=True,
            dataset_version=dataset_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reindexing failed: {str(e)}",
        )
    finally:
        # Clean up candidate and backup staging directories
        for cleanup_path in [candidate_vector_dir, candidate_analytics_dir, backup_dir]:
            if cleanup_path.exists():
                try:
                    shutil.rmtree(cleanup_path)
                except Exception:
                    pass
        if candidate_enriched_csv.exists():
            try:
                candidate_enriched_csv.unlink()
            except Exception:
                pass

