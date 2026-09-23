"""
Unit and integration tests for IncidentRetriever in src/rag/retriever.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.config import VECTOR_INDEX_PATH, VECTOR_METADATA_PATH, DEFAULT_TOP_K
from src.rag.embeddings import IncidentEmbedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import IncidentRetriever, format_retrieved_context

FORBIDDEN_LABELS = {
    "sif_label", "sif_potential", "primary_lsr", "lsr_tags",
    "hazard", "barrier_failure", "equipment", "potential_consequence",
    "actual_consequence", "severity", "precursor_tags", "activity"
}

# ---------------------------------------------------------------------------
# Unit tests using Mock VectorStore & Embedder
# ---------------------------------------------------------------------------

class MockEmbedder:
    def __init__(self, dimension=4):
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        # Simple deterministic vector
        vec = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float32)
        return vec / np.linalg.norm(vec)

@pytest.fixture
def mock_store():
    store = VectorStore()
    records = [
        {"report_id": f"REP-00{i}", "description": f"Incident description {i}", "site": "Site A", "date": "2023-01-01", "location": "Area 1", "report_type": "Near Miss"}
        for i in range(1, 6)
    ]
    # Synthetic 4-dimensional embeddings
    np.random.seed(42)
    raw = np.random.randn(5, 4).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms
    store.build_index(records=records, embeddings=embeddings)
    return store

def test_empty_query_raises_value_error(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        retriever.retrieve_similar_incidents("")

def test_blank_query_raises_value_error(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        retriever.retrieve_similar_incidents("   \t\n ")

def test_top_k_zero_raises_value_error(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    with pytest.raises(ValueError, match="top_k must be an integer >= 1"):
        retriever.retrieve_similar_incidents("test query", top_k=0)

def test_top_k_negative_raises_value_error(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    with pytest.raises(ValueError, match="top_k must be an integer >= 1"):
        retriever.retrieve_similar_incidents("test query", top_k=-3)

def test_unloaded_vector_store_raises_runtime_error():
    empty_store = VectorStore()
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=empty_store)
    with pytest.raises(RuntimeError, match="Vector store is not loaded"):
        retriever.retrieve_similar_incidents("test query", top_k=3)

def test_returned_list_length_and_order(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    results = retriever.retrieve_similar_incidents("valid query", top_k=3)
    
    assert len(results) == 3
    # Check descending similarity scores
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_rank_values_sequence(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    results = retriever.retrieve_similar_incidents("valid query", top_k=4)
    
    ranks = [r["rank"] for r in results]
    assert ranks == [1, 2, 3, 4]

def test_metadata_fields_and_no_forbidden_labels(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    results = retriever.retrieve_similar_incidents("valid query", top_k=2)
    
    required_keys = {"rank", "vector_id", "incident_id", "similarity_score", "description", "site", "location", "date", "report_type"}
    for r in results:
        assert required_keys.issubset(r.keys())
        assert isinstance(r["similarity_score"], float)
        assert isinstance(r["description"], str)
        # Verify no forbidden ground truth labels exist
        for forbidden in FORBIDDEN_LABELS:
            assert forbidden not in r, f"Forbidden label {forbidden} was found in result!"

def test_format_retrieved_context(mock_store):
    retriever = IncidentRetriever(embedder=MockEmbedder(), vector_store=mock_store)
    results = retriever.retrieve_similar_incidents("valid query", top_k=2)
    context_str = format_retrieved_context(results)
    
    assert "Case 1" in context_str
    assert "Case 2" in context_str
    assert "Incident ID:" in context_str
    assert "Description:" in context_str
    assert "Similarity Score:" in context_str
    for forbidden in FORBIDDEN_LABELS:
        assert forbidden not in context_str.lower()

# ---------------------------------------------------------------------------
# Integration tests using real FAISS index & SentenceTransformer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_retriever():
    if not VECTOR_INDEX_PATH.exists() or not VECTOR_METADATA_PATH.exists():
        pytest.skip("Real vector store artifacts do not exist. Run scripts/index_data.py first.")
    
    store = VectorStore(index_path=VECTOR_INDEX_PATH, metadata_path=VECTOR_METADATA_PATH)
    assert store.load_index() is True
    embedder = IncidentEmbedder()
    return IncidentRetriever(embedder=embedder, vector_store=store)

def test_real_retrieval_energy_isolation(real_retriever):
    query = "worker opened a flange before isolation was verified and residual pressure remained"
    results = real_retriever.retrieve_similar_incidents(query, top_k=5)
    
    assert len(results) == 5
    # Inspect that at least one top result contains domain keywords
    top_texts = " ".join([r["description"].lower() for r in results])
    assert any(term in top_texts for term in ["flange", "isolation", "pressure", "valve", "loto", "line"])

def test_real_retrieval_work_at_height(real_retriever):
    query = "technician working at height with harness disconnected"
    results = real_retriever.retrieve_similar_incidents(query, top_k=5)
    
    assert len(results) == 5
    top_texts = " ".join([r["description"].lower() for r in results])
    assert any(term in top_texts for term in ["height", "harness", "scaffold", "fall", "lanyard", "ladder"])

def test_real_retrieval_lifting_line_of_fire(real_retriever):
    query = "worker entered crane swing radius while suspended load was moving"
    results = real_retriever.retrieve_similar_incidents(query, top_k=5)
    
    assert len(results) == 5
    top_texts = " ".join([r["description"].lower() for r in results])
    assert any(term in top_texts for term in ["crane", "load", "suspended", "lift", "swing", "radius"])
