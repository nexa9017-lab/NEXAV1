import joblib
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, average_precision_score
)

class SIFClassifier:
    """
    Binary classification model for Serious Injury and Fatality (SIF) potential.
    Uses Logistic Regression over provided feature vectors.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: str = "balanced", random_state: int = 42, feature_mode: str = "tfidf"):
        self.model = LogisticRegression(
            C=C, 
            max_iter=max_iter, 
            class_weight=class_weight, 
            random_state=random_state,
            solver='lbfgs'
        )
        self.threshold = 0.5
        self.is_fitted = False
        self.feature_names_out = None
        self.feature_mode = feature_mode

    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """Fits the logistic regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
        if feature_names is not None:
            self.feature_names_out = np.array(feature_names)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Returns probability of SIF (class 1)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = None) -> np.ndarray:
        """Predicts binary labels based on a threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def evaluate(self, X, y, threshold: float = None) -> Dict[str, Any]:
        """Evaluates the model and computes standard binary classification metrics."""
        y_pred = self.predict(X, threshold=threshold)
        y_prob = self.predict_proba(X)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "roc_auc": roc_auc_score(y, y_prob),
            "pr_auc": average_precision_score(y, y_prob),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

    def explain_prediction(self, feature_vector, top_k: int = 5) -> Dict[str, float]:
        """
        Returns the top classification evidence (influential model terms).
        Requires feature_names to have been set during fit().
        feature_vector must be a 1D sparse matrix or array (single sample).
        """
        if not self.is_fitted or self.feature_names_out is None:
            raise ValueError("Model must be fitted with feature_names to explain predictions.")
            
        coefs = self.model.coef_[0]
        if hasattr(feature_vector, "toarray"):
            feature_vector = feature_vector.toarray()[0]
        elif len(feature_vector.shape) == 2:
            feature_vector = feature_vector[0]
            
        contributions = feature_vector * coefs
        non_zero_indices = np.where(contributions != 0)[0]
        
        term_contributions = {
            self.feature_names_out[idx]: contributions[idx] 
            for idx in non_zero_indices
        }
        
        # Sort by absolute contribution magnitude
        sorted_terms = sorted(term_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        return dict(sorted_terms[:top_k])

    def save(self, path: Union[str, Path]):
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        state = {
            "model": self.model,
            "threshold": self.threshold,
            "feature_names": self.feature_names_out,
            "feature_mode": self.feature_mode
        }
        joblib.dump(state, path)

    def load(self, path: Union[str, Path]):
        state = joblib.load(path)
        self.model = state["model"]
        self.threshold = state.get("threshold", 0.5)
        self.feature_names_out = state.get("feature_names")
        self.feature_mode = state.get("feature_mode", "tfidf")
        self.is_fitted = True
        return self
