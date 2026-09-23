"""
Embedding utilities for the RAG safety MVP.
Loads SentenceTransformer once and provides single and batch embedding methods.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import RAG_EMBEDDING_MODEL

class IncidentEmbedder:
    """
    Generates dense, L2-normalized sentence embeddings for safety incident reports.
    """
    def __init__(self, model_name: str = RAG_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        """Returns the dense embedding vector dimension retrieved from the model."""
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        Validates non-empty string, generates L2-normalized float32 vector.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 2:
            embedding = embedding[0]
        
        # Verify normalization
        norm = np.linalg.norm(embedding)
        if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
            embedding = embedding / norm

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Batch-embed a list of text strings.
        Validates non-empty list and non-blank entries.
        Returns 2D float32 matrix of shape (N, dimension).
        """
        if not isinstance(texts, (list, tuple)) or len(texts) == 0:
            raise ValueError("Texts list cannot be empty.")

        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Encountered empty or blank text at index {i}.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Verify normalization for every row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid divide-by-zero if any norm is 0
        norms = np.where(norms == 0, 1.0, norms)
        if not np.allclose(norms, 1.0, atol=1e-3):
            embeddings = embeddings / norms

        return embeddings
