"""
Build enriched analytics dataset.

Loads all synthetic reports and enriches each one with production
model predictions (SIF classifier, LSR classifier, precursor extractor).

Uses ONLY embedding-based models (not TF-IDF baseline).
Saves to data/analytics/enriched_reports.csv.
Does NOT overwrite original datasets.

Lists are serialized as JSON strings. Never use eval() to read them.
"""

import json
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.config import (
    ALL_REPORTS_FILE, ENRICHED_REPORTS_PATH,
    SIF_MODEL_EMBEDDING_PATH, LSR_MODEL_EMBEDDING_PATH,
    LSR_ENCODER_PATH, EMBEDDING_MODEL_NAME,
)
from src.models.sif_classifier import SIFClassifier
from src.models.lsr_classifier import LSRClassifier
from src.preprocessing.feature_engineering import SentenceEmbeddingFeatures
from src.extraction.precursor_extractor import SafetyPrecursorExtractor
from src.extraction.normalization import (
    normalize_activity, normalize_equipment,
    normalize_hazard, normalize_barrier_failure, normalize_consequence,
)


def build_enriched_dataset():
    """Build the enriched analytics dataset using production models."""
    print("=" * 60)
    print("BUILDING ENRICHED ANALYTICS DATASET")
    print("=" * 60)

    start_time = time.time()

    # Load all reports
    print(f"\nLoading reports from {ALL_REPORTS_FILE}...")
    df = pd.read_csv(ALL_REPORTS_FILE)
    print(f"  Loaded {len(df)} reports")

    # Validate required columns
    required = ["report_id", "date", "site", "location", "report_type",
                 "description", "sif_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ----- Load Production Models -----
    print("\nLoading production embedding model...")
    embedder = SentenceEmbeddingFeatures(model_name=EMBEDDING_MODEL_NAME)

    print("Loading SIF classifier (embedding-based)...")
    sif_clf = SIFClassifier(feature_mode="embedding")
    sif_clf.load(SIF_MODEL_EMBEDDING_PATH)
    print(f"  Feature mode: {sif_clf.feature_mode}")

    print("Loading LSR classifier (embedding-based)...")
    lsr_clf = LSRClassifier(feature_mode="embedding")
    lsr_clf.load(LSR_MODEL_EMBEDDING_PATH)
    print(f"  Feature mode: {lsr_clf.feature_mode}")

    print("Loading precursor extractor...")
    extractor = SafetyPrecursorExtractor()

    # ----- Generate Embeddings -----
    print("\nGenerating embeddings for all descriptions...")
    descriptions = df["description"].fillna("").tolist()
    embeddings = embedder.transform(descriptions)
    print(f"  Embedding shape: {embeddings.shape}")

    # ----- SIF Predictions -----
    print("\nRunning SIF classification...")
    sif_probs = sif_clf.predict_proba(embeddings)
    sif_preds = sif_clf.predict(embeddings)
    df["sif_prediction"] = sif_preds
    df["sif_probability"] = np.round(sif_probs, 4)
    print(f"  SIF predicted: {sif_preds.sum()} / {len(sif_preds)}")

    # ----- LSR Predictions -----
    print("Running LSR classification...")
    lsr_results = lsr_clf.predict_with_scores(embeddings)
    predicted_lsr = []
    primary_lsr = []
    for result in lsr_results:
        rules = [r["rule"] for r in result["rules"]]
        predicted_lsr.append(json.dumps(rules))
        primary_lsr.append(result["primary_rule"])
    df["predicted_lsr"] = predicted_lsr
    df["primary_lsr_predicted"] = primary_lsr

    # ----- Precursor Extraction -----
    print("Running precursor extraction (this may take a few minutes)...")
    extract_start = time.time()

    pred_activities = []
    pred_equipment = []
    pred_hazards = []
    pred_barriers = []
    pred_precursor_tags = []
    pred_consequences = []

    batch_size = 100
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        if (i // batch_size) % 5 == 0:
            print(f"  Processing reports {i}-{min(i + batch_size, len(descriptions))}...")

        for text in batch:
            result = extractor.extract(text)

            # Activity
            acts = [a["value"] for a in result.get("activities", [])]
            norm_acts = list(set([normalize_activity(a) for a in acts]))
            pred_activities.append(json.dumps(norm_acts))

            # Equipment
            equips = [e["value"] for e in result.get("equipment", [])]
            pred_equipment.append(json.dumps(equips))

            # Hazards
            hazs = [h["value"] for h in result.get("hazards", [])]
            norm_hazs = list(set([normalize_hazard(h) for h in hazs]))
            pred_hazards.append(json.dumps(norm_hazs))

            # Barriers
            bars = [b["value"] for b in result.get("barrier_failures", [])]
            norm_bars = list(set([normalize_barrier_failure(b) for b in bars]))
            pred_barriers.append(json.dumps(norm_bars))

            # Precursor tags (returned as plain strings, not dicts)
            tags = result.get("precursor_tags", [])
            if tags and isinstance(tags[0], dict):
                tags = [t["value"] for t in tags]
            pred_precursor_tags.append(json.dumps(list(set(tags))))

            # Consequences
            cons = [c["value"] for c in result.get("potential_consequences", [])]
            norm_cons = list(set([normalize_consequence(c) for c in cons]))
            pred_consequences.append(json.dumps(norm_cons))

    df["predicted_activity"] = pred_activities
    df["predicted_equipment"] = pred_equipment
    df["predicted_hazards"] = pred_hazards
    df["predicted_barrier_failures"] = pred_barriers
    df["predicted_precursor_tags"] = pred_precursor_tags
    df["predicted_potential_consequences"] = pred_consequences

    extract_elapsed = round(time.time() - extract_start, 2)
    print(f"  Extraction completed in {extract_elapsed}s")

    # ----- Select Output Columns -----
    output_cols = [
        "report_id", "date", "site", "location", "report_type", "description",
        "activity",  # ground-truth activity metadata
        "predicted_activity", "predicted_equipment", "predicted_hazards",
        "predicted_barrier_failures", "predicted_precursor_tags",
        "predicted_potential_consequences",
        "sif_prediction", "sif_probability",
        "predicted_lsr", "primary_lsr_predicted",
        # Ground-truth columns for validation (not used in production analytics)
        "sif_label", "primary_lsr", "lsr_tags", "hazard", "barrier_failure",
        "potential_consequence", "precursor_tags", "severity",
    ]

    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    enriched = df[output_cols]

    # ----- Save -----
    ENRICHED_REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(ENRICHED_REPORTS_PATH, index=False)

    total_elapsed = round(time.time() - start_time, 2)

    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"  Output: {ENRICHED_REPORTS_PATH}")
    print(f"  Reports: {len(enriched)}")
    print(f"  Columns: {len(enriched.columns)}")
    print(f"  Predicted SIF: {int(enriched['sif_prediction'].sum())}")
    print(f"  SIF density: {enriched['sif_prediction'].mean():.4f}")
    print(f"  Enrichment runtime: {total_elapsed}s")
    print(f"    (extraction: {extract_elapsed}s)")

    return enriched


if __name__ == "__main__":
    build_enriched_dataset()
