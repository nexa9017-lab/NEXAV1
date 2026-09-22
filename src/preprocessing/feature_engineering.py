import joblib
from pathlib import Path
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Used to prevent target leakage in the baseline models.
FORBIDDEN_FEATURE_COLUMNS = [
    "sif_label", "sif_potential", "primary_lsr", "lsr_tags",
    "hazard", "barrier_failure", "potential_consequence",
    "actual_consequence", "severity", "precursor_tags"
]

class TfidfFeatureExtractor:
    """
    Reusable TF-IDF feature pipeline for safety text.
    Fitted ONLY on training data.
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.95, ngram_range: tuple = (1, 2), lowercase: bool = True):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=lowercase
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Fits the TF-IDF vectorizer."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]):
        """Transforms text using the fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Vectorizer is not fitted! Call fit() on training data first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fits and transforms the training text."""
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts)

    def save(self, path: Union[str, Path]):
        """Saves the fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted vectorizer.")
        joblib.dump(self.vectorizer, path)

    def load(self, path: Union[str, Path]):
        """Loads a fitted vectorizer."""
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        return self


class SentenceEmbeddingFeatures:
    """
    Optional sentence embedding interface using sentence-transformers.
    Defaults to all-MiniLM-L6-v2.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            print("Warning: `sentence-transformers` is not installed. Embeddings are disabled.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{self.model_name}': {e}")
            self.model = None

    def transform(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts."""
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded or sentence-transformers is missing.")
        return self.model.encode(texts, show_progress_bar=False)
