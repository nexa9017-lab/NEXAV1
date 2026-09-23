"""
Unit tests for IncidentEmbedder in src/rag/embeddings.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.rag.embeddings import IncidentEmbedder

@pytest.fixture(scope="module")
def embedder():
    """Load IncidentEmbedder once for this test module to avoid redundant model loads."""
    return IncidentEmbedder()

def test_dimension_property(embedder):
    assert embedder.dimension == 384

def test_embed_text_shape_and_dtype(embedder):
    text = "Technician identified residual pressure in the piping spool."
    vec = embedder.embed_text(text)
    
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

def test_embed_text_normalization(embedder):
    text = "Hot work permit expired before cutting commenced."
    vec = embedder.embed_text(text)
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-3)

def test_embed_text_blank_raises_error(embedder):
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        embedder.embed_text("")
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        embedder.embed_text("   \n\t  ")

def test_embed_batch_shape_and_dtype(embedder):
    texts = [
        "Worker entered confined space without atmospheric test.",
        "Scaffolding guardrail was missing on platform level 3.",
        "Crane sling showed signs of excessive mechanical wear."
    ]
    matrix = embedder.embed_batch(texts)
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 384)
    assert matrix.dtype == np.float32

def test_embed_batch_normalization(embedder):
    texts = [
        "Chemical leak observed at pump seal.",
        "Forklift speed exceeded site limit in warehouse."
    ]
    matrix = embedder.embed_batch(texts)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

def test_embed_batch_empty_list_raises_error(embedder):
    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        embedder.embed_batch([])

def test_embed_batch_blank_item_raises_error(embedder):
    with pytest.raises(ValueError, match="Encountered empty or blank text at index 1"):
        embedder.embed_batch(["Valid description", "   ", "Another description"])
