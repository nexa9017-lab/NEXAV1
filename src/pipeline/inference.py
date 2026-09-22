"""
Unified Inference Pipeline.
Loads production models once and provides single/batch prediction interfaces.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    SIF_MODEL_EMBEDDING_PATH, LSR_MODEL_EMBEDDING_PATH,
    EMBEDDING_MODEL_NAME, ANALYTICS_PROTOTYPE_NOTICE,
    ANALYTICS_VERSION, LSR_DEFAULT_GLOBAL_THRESHOLD
)
from src.models.sif_classifier import SIFClassifier
from src.models.lsr_classifier import LSRClassifier
from src.preprocessing.feature_engineering import SentenceEmbeddingFeatures
from src.extraction.precursor_extractor import SafetyPrecursorExtractor
from src.extraction.normalization import (
    normalize_activity, normalize_equipment,
    normalize_hazard, normalize_barrier_failure, normalize_consequence,
    normalize_precursor_tag
)

class SafetyInferencePipeline:
    def __init__(self):
        self.embedder = None
        self.sif_clf = None
        self.lsr_clf = None
        self.extractor = None
        self.loaded = False

    def load_models(self):
        """Load all production models and initialize embeddings into memory."""
        if self.loaded:
            return

        print("Loading Sentence Embedding Model...")
        self.embedder = SentenceEmbeddingFeatures(model_name=EMBEDDING_MODEL_NAME)
        
        print("Loading SIF Classifier...")
        self.sif_clf = SIFClassifier(feature_mode="embedding")
        self.sif_clf.load(SIF_MODEL_EMBEDDING_PATH)

        print("Loading LSR Classifier...")
        self.lsr_clf = LSRClassifier(feature_mode="embedding")
        self.lsr_clf.load(LSR_MODEL_EMBEDDING_PATH)

        print("Loading Precursor Extractor...")
        self.extractor = SafetyPrecursorExtractor()

        self.loaded = True
        print("Safety Inference Pipeline fully loaded.")

    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate common metadata for the inference request."""
        return {
            "model_versions": {
                "sif": "production_embedding",
                "lsr": "production_embedding",
                "extractor": "hybrid_rule_semantic",
                "embedder": EMBEDDING_MODEL_NAME
            },
            "feature_mode": "embedding",
            "inference_timestamp": datetime.utcnow().isoformat() + "Z",
            "thresholds": {
                "sif_threshold": float(self.sif_clf.threshold),
                "lsr_threshold": float(LSR_DEFAULT_GLOBAL_THRESHOLD)
            },
            "prototype_notice": ANALYTICS_PROTOTYPE_NOTICE
        }
        
    def _sanitize_confidence(self, val: Any) -> float:
        if isinstance(val, (np.float32, np.float64)):
            return float(val)
        return val

    def predict_single(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Run full inference on a single report."""
        if not self.loaded:
            self.load_models()

        description = report.get("description", "").strip()
        if not description:
            raise ValueError("Empty or whitespace-only description provided.")

        start_time = time.time()

        # 1. Embed text
        embedding = self.embedder.transform([description])

        # 2. SIF Prediction
        sif_prob = float(self.sif_clf.predict_proba(embedding)[0])
        sif_pred = int(self.sif_clf.predict(embedding)[0])
        sif_label = "SIF-Potential" if sif_pred == 1 else "Non-SIF"

        # 3. LSR Prediction
        lsr_result = self.lsr_clf.predict_with_scores(embedding)[0]
        lsr_rules = [
            {"rule": r["rule"], "score": float(r["score"])}
            for r in lsr_result["rules"]
        ]
        primary_lsr = lsr_result["primary_rule"]

        # 4. Extraction
        extraction = self.extractor.extract(description)

        # Helper to format and normalize extractions
        def format_extractions(items, norm_func=None):
            result = []
            seen = set()
            for item in items:
                raw_val = item["value"]
                norm_val = norm_func(raw_val) if norm_func else raw_val
                
                if norm_val and norm_val != "CONSEQUENCE_IGNORE" and norm_val not in seen:
                    seen.add(norm_val)
                    result.append({
                        "value": norm_val,
                        "evidence": item.get("evidence", ""),
                        "confidence": item.get("confidence", "UNKNOWN"),
                        "match_method": item.get("match_method", "unknown")
                    })
            return result

        activities = format_extractions(extraction.get("activities", []), normalize_activity)
        equipment = format_extractions(extraction.get("equipment", []), normalize_equipment)
        hazards = format_extractions(extraction.get("hazards", []), normalize_hazard)
        barrier_failures = format_extractions(extraction.get("barrier_failures", []), normalize_barrier_failure)
        unsafe_actions = format_extractions(extraction.get("unsafe_actions", []), None) # No norm for these
        unsafe_conditions = format_extractions(extraction.get("unsafe_conditions", []), None)
        potential_consequences = format_extractions(extraction.get("potential_consequences", []), normalize_consequence)
        
        # Precursor tags are plain strings from the extractor
        raw_tags = extraction.get("precursor_tags", [])
        if raw_tags and isinstance(raw_tags[0], dict):
            raw_tags = [t["value"] for t in raw_tags]
        precursor_tags = list(set([normalize_precursor_tag(t) for t in raw_tags if t]))

        elapsed = round((time.time() - start_time) * 1000, 2) # in ms

        output = {
            "normalized_text": description,
            "sif": {
                "label": sif_label,
                "probability": round(sif_prob, 4)
            },
            "life_saving_rules": lsr_rules,
            "primary_lsr": primary_lsr,
            "activities": activities,
            "equipment": equipment,
            "hazards": hazards,
            "barrier_failures": barrier_failures,
            "unsafe_actions": unsafe_actions,
            "unsafe_conditions": unsafe_conditions,
            "potential_consequences": potential_consequences,
            "precursor_tags": precursor_tags,
            "inference_metadata": self._generate_metadata(),
            "latency_ms": elapsed
        }

        return output

    def predict_batch(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run inference on a batch of reports, preserving order."""
        if not self.loaded:
            self.load_models()

        if not reports:
            return []

        # We will do sequential processing for extraction, but batched for embeddings & classification
        # for better performance.
        
        valid_indices = []
        descriptions = []
        for i, report in enumerate(reports):
            desc = report.get("description", "").strip()
            if desc:
                valid_indices.append(i)
                descriptions.append(desc)
        
        results = [None] * len(reports)
        
        if not descriptions:
            for i in range(len(reports)):
                results[i] = {"error": "Empty or whitespace-only description"}
            return results

        # 1. Embed batch
        embeddings = self.embedder.transform(descriptions)

        # 2. Classify batch
        sif_probs = self.sif_clf.predict_proba(embeddings)
        sif_preds = self.sif_clf.predict(embeddings)
        lsr_results = self.lsr_clf.predict_with_scores(embeddings)

        # 3. Extract and assemble
        for idx_in_batch, original_idx in enumerate(valid_indices):
            desc = descriptions[idx_in_batch]
            sif_prob = float(sif_probs[idx_in_batch])
            sif_pred = int(sif_preds[idx_in_batch])
            sif_label = "SIF-Potential" if sif_pred == 1 else "Non-SIF"
            
            lsr_result = lsr_results[idx_in_batch]
            lsr_rules = [
                {"rule": r["rule"], "score": float(r["score"])}
                for r in lsr_result["rules"]
            ]
            primary_lsr = lsr_result["primary_rule"]

            extraction = self.extractor.extract(desc)

            def format_extractions(items, norm_func=None):
                res = []
                seen = set()
                for item in items:
                    raw_val = item["value"]
                    norm_val = norm_func(raw_val) if norm_func else raw_val
                    if norm_val and norm_val != "CONSEQUENCE_IGNORE" and norm_val not in seen:
                        seen.add(norm_val)
                        res.append({
                            "value": norm_val,
                            "evidence": item.get("evidence", ""),
                            "confidence": item.get("confidence", "UNKNOWN"),
                            "match_method": item.get("match_method", "unknown")
                        })
                return res

            activities = format_extractions(extraction.get("activities", []), normalize_activity)
            equipment = format_extractions(extraction.get("equipment", []), normalize_equipment)
            hazards = format_extractions(extraction.get("hazards", []), normalize_hazard)
            barrier_failures = format_extractions(extraction.get("barrier_failures", []), normalize_barrier_failure)
            unsafe_actions = format_extractions(extraction.get("unsafe_actions", []), None)
            unsafe_conditions = format_extractions(extraction.get("unsafe_conditions", []), None)
            potential_consequences = format_extractions(extraction.get("potential_consequences", []), normalize_consequence)
            
            raw_tags = extraction.get("precursor_tags", [])
            if raw_tags and isinstance(raw_tags[0], dict):
                raw_tags = [t["value"] for t in raw_tags]
            precursor_tags = list(set([normalize_precursor_tag(t) for t in raw_tags if t]))

            output = {
                "normalized_text": desc,
                "sif": {
                    "label": sif_label,
                    "probability": round(sif_prob, 4)
                },
                "life_saving_rules": lsr_rules,
                "primary_lsr": primary_lsr,
                "activities": activities,
                "equipment": equipment,
                "hazards": hazards,
                "barrier_failures": barrier_failures,
                "unsafe_actions": unsafe_actions,
                "unsafe_conditions": unsafe_conditions,
                "potential_consequences": potential_consequences,
                "precursor_tags": precursor_tags,
                "inference_metadata": self._generate_metadata()
            }
            results[original_idx] = output
            
        # Fill in errors for invalid reports
        for i, res in enumerate(results):
            if res is None:
                results[i] = {"error": "Empty or whitespace-only description"}

        return results
