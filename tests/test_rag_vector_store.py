"""
Unit tests for VectorStore in src/rag/vector_store.py.
Uses synthetic numeric mock vectors (no model download or SentenceTransformer loading required).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pytest
from src.rag.vector_store import VectorStore

@pytest.fixture
def mock_embeddings():
    """Generates 3 synthetic 384-dimensional L2-normalized vectors."""
    np.random.seed(42)
    raw = np.random.randn(3, 384).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms

@pytest.fixture
def mock_records():
    return [
        {
            "report_id": "REP-001",
            "date": "2023-01-15",
            "site": "Platform Alpha",
            "location": "Deck B",
            "report_type": "Near Miss",
            "description": "Technician cracked flange with residual pressure."
        },
        {
            "report_id": "REP-002",
            "date": "2023-02-10",
            "site": "Facility Bravo",
            "location": "Tank 4",
            "report_type": "Unsafe Act",
            "description": "Worker entered vessel before atmospheric test completed."
        },
        {
            "report_id": "REP-003",
            "date": "2023-03-05",
            "site": "Logistics Hub",
            "location": "Crane Bay",
            "report_type": "Incident",
            "description": "Load swung into pedestrian walkway."
        }
    ]

def test_build_index(mock_records, mock_embeddings):
    store = VectorStore()
    store.build_index(records=mock_records, embeddings=mock_embeddings)
    
    assert store.index is not None
    assert store.index.ntotal == 3
    assert store.dimension == 384
    assert len(store.records) == 3
    assert store.records[0]["vector_id"] == 0
    assert store.records[1]["vector_id"] == 1
    assert store.records[2]["vector_id"] == 2
    assert store.records[0]["report_id"] == "REP-001"

def test_save_and_load_roundtrip(tmp_path, mock_records, mock_embeddings):
    index_file = tmp_path / "test_index.faiss"
    meta_file = tmp_path / "test_meta.json"
    
    store = VectorStore(index_path=index_file, metadata_path=meta_file)
    store.build_index(records=mock_records, embeddings=mock_embeddings)
    store.save_index()
    
    assert index_file.exists()
    assert meta_file.exists()
    
    # Verify metadata JSON structure
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["version"] == 1
    assert meta_data["dimension"] == 384
    assert meta_data["total_records"] == 3
    assert len(meta_data["records"]) == 3
    
    # Reload in a clean store instance
    restored_store = VectorStore(index_path=index_file, metadata_path=meta_file)
    success = restored_store.load_index()
    
    assert success is True
    assert restored_store.index.ntotal == 3
    assert restored_store.dimension == 384
    assert len(restored_store.records) == 3
    assert restored_store.records[0]["report_id"] == "REP-001"
    assert restored_store.records[2]["description"] == "Load swung into pedestrian walkway."

def test_load_nonexistent_returns_false(tmp_path):
    store = VectorStore(
        index_path=tmp_path / "missing.faiss",
        metadata_path=tmp_path / "missing.json"
    )
    assert store.load_index() is False

def test_mismatched_records_and_vectors_raises_error(mock_records, mock_embeddings):
    store = VectorStore()
    with pytest.raises(ValueError, match="does not match"):
        # 2 records vs 3 embeddings
        store.build_index(records=mock_records[:2], embeddings=mock_embeddings)

def test_1d_embedding_matrix_raises_error(mock_records):
    store = VectorStore()
    flat_vector = np.ones(384, dtype=np.float32)
    with pytest.raises(ValueError, match="must be a 2D numpy array"):
        store.build_index(records=mock_records[:1], embeddings=flat_vector)

def test_nan_embeddings_raises_error(mock_records):
    store = VectorStore()
    bad_vectors = np.ones((len(mock_records), 384), dtype=np.float32)
    bad_vectors[0, 5] = np.nan
    with pytest.raises(ValueError, match="contains NaN or Inf values"):
        store.build_index(records=mock_records, embeddings=bad_vectors)

def test_get_info(mock_records, mock_embeddings):
    store = VectorStore()
    info_before = store.get_info()
    assert info_before["loaded"] is False
    assert info_before["total_records"] == 0
    
    store.build_index(records=mock_records, embeddings=mock_embeddings)
    info_after = store.get_info()
    assert info_after["loaded"] is True
    assert info_after["total_records"] == 3
    assert info_after["dimension"] == 384
    assert info_after["index_type"] == "IndexFlatIP"

def test_search_vector(mock_records, mock_embeddings):
    store = VectorStore()
    store.build_index(records=mock_records, embeddings=mock_embeddings)
    
    # Query with exact first vector: similarity should be approx 1.0 at index 0
    query = mock_embeddings[0]
    scores, indices = store.search_vector(query, top_k=2)
    
    assert scores.shape == (1, 2)
    assert indices.shape == (1, 2)
    assert indices[0][0] == 0
    assert np.isclose(scores[0][0], 1.0, atol=1e-4)
