"""
Semantic retriever module for the RAG safety MVP.
Preprocesses queries, generates dense embeddings, executes FAISS similarity searches,
and maps vector hits back to clean historical incident records.
"""
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import DEFAULT_TOP_K
from src.preprocessing.text_preprocessor import SafetyTextPreprocessor
from src.rag.embeddings import IncidentEmbedder
from src.rag.vector_store import VectorStore

class IncidentRetriever:
    """
    Retrieves top-k semantically similar historical safety incidents.
    Reuses existing IncidentEmbedder, VectorStore, and SafetyTextPreprocessor.
    """
    def __init__(
        self,
        embedder: IncidentEmbedder,
        vector_store: VectorStore,
        preprocessor: Optional[SafetyTextPreprocessor] = None
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.preprocessor = preprocessor or SafetyTextPreprocessor()

    def retrieve_similar_incidents(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_incident_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most semantically similar historical incidents.

        Args:
            query: Free-text safety incident narrative.
            top_k: Number of similar incidents to return (>= 1).
            exclude_incident_id: Optional incident ID to exclude from results.

        Returns:
            List of ranked result dictionaries with rank (1-indexed), vector_id,
            incident_id, similarity_score (raw float cosine similarity),
            description (original historical narrative), site, location, date, report_type.
        """
        # 1. Query validation
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        raw_query = query.strip()

        # 2. top_k validation
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k}.")

        # 3. Load validation
        if self.vector_store.index is None or not self.vector_store.records:
            raise RuntimeError("Vector store is not loaded. Load or build index before retrieving.")

        total_records = len(self.vector_store.records)
        effective_k = min(top_k, total_records)
        fetch_k = min(effective_k + (1 if exclude_incident_id else 0), total_records)

        # 4. Preprocess query
        clean_query = self.preprocessor.preprocess(raw_query)
        if not clean_query.strip():
            clean_query = raw_query

        # 5. Embed query
        query_vector = self.embedder.embed_text(clean_query)

        # 6. FAISS search
        scores, indices = self.vector_store.search_vector(query_vector, top_k=fetch_k)
        row_scores = scores[0]
        row_indices = indices[0]

        # 7. Map vector IDs to records and format output
        results: List[Dict[str, Any]] = []
        rank = 1

        for score, vec_idx in zip(row_scores, row_indices):
            if vec_idx < 0 or vec_idx >= total_records:
                continue

            record = self.vector_store.records[vec_idx]
            inc_id = record.get("report_id")

            if exclude_incident_id and inc_id == exclude_incident_id:
                continue

            res_item = {
                "rank": rank,
                "vector_id": int(vec_idx),
                "incident_id": str(inc_id) if inc_id is not None else None,
                "similarity_score": float(round(float(score), 4)),
                "description": record.get("description"),
                "site": record.get("site"),
                "location": record.get("location"),
                "date": record.get("date"),
                "report_type": record.get("report_type")
            }
            results.append(res_item)
            rank += 1

            if len(results) >= effective_k:
                break

        return results

def format_retrieved_context(results: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved historical cases into clean text context for downstream prompting.
    Does NOT include any ground-truth labels.
    """
    if not results:
        return "No similar historical incidents found."

    cases = []
    for item in results:
        lines = [
            f"Case {item.get('rank', 1)}",
            f"Incident ID: {item.get('incident_id', 'N/A')}",
            f"Site: {item.get('site', 'Unknown')}",
            f"Location: {item.get('location', 'Unknown')}",
            f"Date: {item.get('date', 'Unknown')}",
            f"Report Type: {item.get('report_type', 'Unknown')}",
            f"Description: {item.get('description', '')}",
            f"Similarity Score: {item.get('similarity_score', 0.0)}"
        ]
        cases.append("\n".join(lines))

    return "\n\n".join(cases)
