import joblib
import numpy as np
import json
import ast
from pathlib import Path
from typing import Union, Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, hamming_loss
)
from src.preprocessing.safety_taxonomy import LIFE_SAVING_RULES

class LSRClassifier:
    """
    Multi-label Life-Saving Rule classifier.
    Uses One-Vs-Rest Logistic Regression.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: str = "balanced", random_state: int = 42, feature_mode: str = "tfidf"):
        base_lr = LogisticRegression(
            C=C, 
            max_iter=max_iter, 
            class_weight=class_weight, 
            random_state=random_state,
            solver='lbfgs'
        )
        self.model = OneVsRestClassifier(base_lr)
        self.mlb = MultiLabelBinarizer(classes=list(LIFE_SAVING_RULES.keys()))
        self.global_threshold = 0.35
        self.is_fitted = False
        self.feature_mode = feature_mode

    @staticmethod
    def safely_parse_lsr_tags(tag_str: str) -> List[str]:
        """Safely parses lsr_tags which might be joined by '|' or JSON arrays."""
        if pd.isna(tag_str) or tag_str == "None" or not tag_str:
            return []
        
        # It's joined by '|' in our generation
        if "|" in tag_str:
            tags = [t.strip() for t in tag_str.split("|")]
        # Try JSON parsing
        elif tag_str.startswith("["):
            try:
                tags = json.loads(tag_str)
            except json.JSONDecodeError:
                try:
                    tags = ast.literal_eval(tag_str)
                except Exception:
                    tags = []
        else:
            tags = [tag_str.strip()]
            
        # Normalize/Filter against taxonomy
        valid_rules = list(LIFE_SAVING_RULES.keys())
        filtered_tags = [t for t in tags if t in valid_rules]
        return filtered_tags

    def fit(self, X, y_strings: List[str]):
        """Fits the model using raw tag strings."""
        y_parsed = [self.safely_parse_lsr_tags(s) for s in y_strings]
        y_encoded = self.mlb.fit_transform(y_parsed)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.predict_proba(X)

    def predict(self, X, threshold: float = None) -> np.ndarray:
        thresh = threshold if threshold is not None else self.global_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def predict_with_scores(self, X, threshold: float = None) -> List[Dict[str, Any]]:
        thresh = threshold if threshold is not None else self.global_threshold
        probs = self.predict_proba(X)
        classes = self.mlb.classes_
        
        results = []
        for i in range(probs.shape[0]):
            sample_probs = probs[i]
            active_indices = np.where(sample_probs >= thresh)[0]
            
            rules = []
            for idx in active_indices:
                rules.append({"rule": classes[idx], "score": float(sample_probs[idx])})
                
            # Fallback: if no rule exceeds threshold, take the max probability rule
            # provided its probability is not completely zero, but mark it below threshold.
            if len(rules) == 0:
                max_idx = np.argmax(sample_probs)
                rules.append({"rule": classes[max_idx], "score": float(sample_probs[max_idx])})
                
            # Sort by score descending
            rules.sort(key=lambda x: x["score"], reverse=True)
            primary_rule = rules[0]["rule"] if rules else "None"
            
            results.append({
                "rules": rules,
                "primary_rule": primary_rule
            })
            
        return results

    def evaluate(self, X, y_strings: List[str], threshold: float = None) -> Dict[str, Any]:
        y_parsed = [self.safely_parse_lsr_tags(s) for s in y_strings]
        y_true = self.mlb.transform(y_parsed)
        y_pred = self.predict(X, threshold)
        
        metrics = {
            "micro_precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
            "micro_recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
            "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "exact_match_ratio": accuracy_score(y_true, y_pred)
        }
        
        # Per class
        classes = self.mlb.classes_
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        support = y_true.sum(axis=0)
        
        per_class_metrics = {}
        for idx, cls in enumerate(classes):
            per_class_metrics[cls] = {
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
                "support": int(support[idx])
            }
            
        metrics["per_class"] = per_class_metrics
        return metrics

    def save(self, path: Union[str, Path]):
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        state = {
            "model": self.model,
            "mlb": self.mlb,
            "global_threshold": self.global_threshold,
            "feature_mode": self.feature_mode
        }
        joblib.dump(state, path)

    def load(self, path: Union[str, Path]):
        state = joblib.load(path)
        self.model = state["model"]
        self.mlb = state["mlb"]
        self.global_threshold = state.get("global_threshold", 0.35)
        self.feature_mode = state.get("feature_mode", "tfidf")
        self.is_fitted = True
        return self

import pandas as pd
