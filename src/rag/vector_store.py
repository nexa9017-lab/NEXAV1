"""
Vector store implementation using FAISS IndexFlatIP.
Stores dense vectors in FAISS and companion metadata in structured JSON.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import faiss
import numpy as np

from src.config import VECTOR_INDEX_PATH, VECTOR_METADATA_PATH, RAG_EMBEDDING_MODEL

class VectorStore:
    """
    FAISS-backed vector store for safety incident embeddings.
    Uses IndexFlatIP with normalized vectors for exact cosine similarity search.
    """
    def __init__(
        self,
        index_path: Union[str, Path] = VECTOR_INDEX_PATH,
        metadata_path: Union[str, Path] = VECTOR_METADATA_PATH,
        embedding_model: str = RAG_EMBEDDING_MODEL
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model = embedding_model
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.records: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None

    def build_index(
        self,
        records: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> None:
        """
        Builds a FAISS IndexFlatIP index and stores metadata records.
        """
        if not isinstance(records, (list, tuple)) or len(records) == 0:
            raise ValueError("Records list must be a non-empty list of dictionaries.")
        
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D numpy array.")
            
        if len(records) != embeddings.shape[0]:
            raise ValueError(
                f"Record count ({len(records)}) does not match embeddings row count ({embeddings.shape[0]})."
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings matrix contains NaN or Inf values.")

        vectors = np.asarray(embeddings, dtype=np.float32)
        dim = int(vectors.shape[1])
        
        # Create FAISS IndexFlatIP
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        # Store state
        self.index = index
        self.dimension = dim
        self.records = []
        for i, rec in enumerate(records):
            item = dict(rec)
            item["vector_id"] = i
            self.records.append(item)

    def save_index(self) -> None:
        """
        Persists FAISS index to index_path and metadata to metadata_path.
        """
        if self.index is None or len(self.records) == 0:
            raise RuntimeError("Cannot save an empty or uninitialized index.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary index
        faiss.write_index(self.index, str(self.index_path))

        # Save companion metadata JSON
        metadata_payload = {
            "version": 1,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "total_records": len(self.records),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "records": self.records
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

    def load_index(self) -> bool:
        """
        Loads the FAISS binary index and metadata JSON from disk.
        Returns True if successfully loaded, or False if files do not exist.
        Raises ValueError if index and metadata are inconsistent.
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        try:
            index = faiss.read_index(str(self.index_path))
        except Exception as e:
            raise ValueError(f"Failed to read FAISS index from {self.index_path}: {e}")

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse metadata JSON from {self.metadata_path}: {e}")

        records = data.get("records", [])
        dim = data.get("dimension")

        if index.ntotal != len(records):
            raise ValueError(
                f"Index count ({index.ntotal}) does not match metadata record count ({len(records)})."
            )

        if index.d != dim:
            raise ValueError(
                f"FAISS index dimension ({index.d}) does not match metadata dimension ({dim})."
            )

        for i, rec in enumerate(records):
            if rec.get("vector_id") != i:
                raise ValueError(
                    f"Vector ID alignment error at position {i}: expected {i}, got {rec.get('vector_id')}."
                )

        self.index = index
        self.records = records
        self.dimension = index.d
        self.embedding_model = data.get("embedding_model", self.embedding_model)

        return True

    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the active index.
        """
        return {
            "loaded": self.index is not None and len(self.records) > 0,
            "total_records": self.index.ntotal if self.index is not None else 0,
            "dimension": self.dimension if self.dimension is not None else 0,
            "index_type": "IndexFlatIP",
            "index_path": str(self.index_path.name),
            "metadata_path": str(self.metadata_path.name),
            "embedding_model": self.embedding_model
        }

    def search_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Low-level vector search on raw numeric vector.
        Accepts numeric vector only (no text processing).
        Returns:
            (scores, indices): tuple of cosine similarity scores and integer index positions.
        """
        if self.index is None:
            raise RuntimeError("Index is not loaded or built.")

        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, top_k)
        return scores, indices
