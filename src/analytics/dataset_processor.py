"""
Reusable batch enrichment and analytics generation service.
Coordinates batch model inference and analytics pipeline execution.
Used both by CLI scripts and the /reindex upload endpoint.
"""

import json
import time
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    SIF_MODEL_EMBEDDING_PATH,
    LSR_MODEL_EMBEDDING_PATH,
    EMBEDDING_MODEL_NAME,
    ENRICHED_REPORTS_PATH,
    ANALYTICS_OUTPUT_DIR,
)
from src.models.sif_classifier import SIFClassifier
from src.models.lsr_classifier import LSRClassifier
from src.preprocessing.feature_engineering import SentenceEmbeddingFeatures
from src.extraction.precursor_extractor import SafetyPrecursorExtractor
from src.extraction.normalization import (
    normalize_activity,
    normalize_equipment,
    normalize_hazard,
    normalize_barrier_failure,
    normalize_consequence,
    normalize_precursor_tag,
)
from src.analytics.pipeline import AnalyticsPipeline


def process_dataset_enrichment_and_analytics(
    df: pd.DataFrame,
    output_enriched_path: Path,
    output_analytics_dir: Path,
    legacy_pipeline: Optional[Any] = None,
    source_filename: str = "uploaded_reports.csv",
) -> Dict[str, Any]:
    """
    Enriches an incident reports dataframe with SIF, LSR, and precursor extractions,
    saves the enriched dataset to output_enriched_path, and generates analytics
    JSON artifacts in output_analytics_dir.

    Args:
        df: Input DataFrame containing at least 'description'.
        output_enriched_path: Target CSV path for enriched reports.
        output_analytics_dir: Target directory for generated analytics JSONs.
        legacy_pipeline: Optional loaded SafetyInferencePipeline to reuse in-memory models.
        source_filename: Name of the source file for metadata recording.

    Returns:
        Dict containing summary stats: records_processed, sif_count, sif_density.
    """
    start_time = time.time()
    work_df = df.copy()

    # Ensure required structural columns exist with reasonable defaults
    if "report_id" not in work_df.columns:
        work_df["report_id"] = [f"REP-{i+1:05d}" for i in range(len(work_df))]
    else:
        work_df["report_id"] = work_df["report_id"].fillna("").astype(str)
        # Fill any empty IDs
        empty_ids = work_df["report_id"].str.strip() == ""
        if empty_ids.any():
            work_df.loc[empty_ids, "report_id"] = [f"REP-{i+1:05d}" for i in range(empty_ids.sum())]

    if "date" not in work_df.columns:
        work_df["date"] = time.strftime("%Y-%m-%d")
    else:
        work_df["date"] = work_df["date"].fillna(time.strftime("%Y-%m-%d")).astype(str)

    if "site" not in work_df.columns:
        work_df["site"] = "General Site"
    else:
        work_df["site"] = work_df["site"].fillna("General Site").astype(str)

    if "location" not in work_df.columns:
        work_df["location"] = "Unspecified Location"
    else:
        work_df["location"] = work_df["location"].fillna("Unspecified Location").astype(str)

    if "report_type" not in work_df.columns:
        work_df["report_type"] = "Incident"
    else:
        work_df["report_type"] = work_df["report_type"].fillna("Incident").astype(str)

    descriptions = work_df["description"].fillna("").astype(str).tolist()

    # 1. Models Initialization or Reuse
    if legacy_pipeline is not None and getattr(legacy_pipeline, "loaded", False):
        embedder = legacy_pipeline.embedder
        sif_clf = legacy_pipeline.sif_clf
        lsr_clf = legacy_pipeline.lsr_clf
        extractor = legacy_pipeline.extractor
    else:
        embedder = SentenceEmbeddingFeatures(model_name=EMBEDDING_MODEL_NAME)
        sif_clf = SIFClassifier(feature_mode="embedding")
        sif_clf.load(SIF_MODEL_EMBEDDING_PATH)
        lsr_clf = LSRClassifier(feature_mode="embedding")
        lsr_clf.load(LSR_MODEL_EMBEDDING_PATH)
        extractor = SafetyPrecursorExtractor()

    # 2. Embedding Generation
    embeddings = embedder.transform(descriptions)

    # 3. SIF Classification
    sif_probs = sif_clf.predict_proba(embeddings)
    sif_preds = sif_clf.predict(embeddings)
    work_df["sif_prediction"] = [int(p) for p in sif_preds]
    work_df["sif_probability"] = np.round(sif_probs, 4)

    # 4. LSR Classification
    lsr_results = lsr_clf.predict_with_scores(embeddings)
    predicted_lsr = []
    primary_lsr = []
    for res in lsr_results:
        rules = [r["rule"] for r in res["rules"]]
        predicted_lsr.append(json.dumps(rules))
        primary_lsr.append(res["primary_rule"])
    work_df["predicted_lsr"] = predicted_lsr
    work_df["primary_lsr_predicted"] = primary_lsr

    # 5. Precursor Entity Extraction
    pred_activities = []
    pred_equipment = []
    pred_hazards = []
    pred_barriers = []
    pred_precursor_tags = []
    pred_consequences = []

    for text in descriptions:
        ext = extractor.extract(text)

        # Activities
        acts = [a["value"] for a in ext.get("activities", []) if "value" in a]
        norm_acts = list(set([normalize_activity(a) for a in acts if normalize_activity(a)]))
        pred_activities.append(json.dumps(norm_acts))

        # Equipment
        equips = [e["value"] for e in ext.get("equipment", []) if "value" in e]
        pred_equipment.append(json.dumps(equips))

        # Hazards
        hazs = [h["value"] for h in ext.get("hazards", []) if "value" in h]
        norm_hazs = list(set([normalize_hazard(h) for h in hazs if normalize_hazard(h)]))
        pred_hazards.append(json.dumps(norm_hazs))

        # Barriers
        bars = [b["value"] for b in ext.get("barrier_failures", []) if "value" in b]
        norm_bars = list(set([normalize_barrier_failure(b) for b in bars if normalize_barrier_failure(b)]))
        pred_barriers.append(json.dumps(norm_bars))

        # Tags
        tags = ext.get("precursor_tags", [])
        if tags and isinstance(tags[0], dict):
            tags = [t.get("value", "") for t in tags]
        norm_tags = list(set([normalize_precursor_tag(t) for t in tags if t]))
        pred_precursor_tags.append(json.dumps(norm_tags))

        # Consequences
        cons = [c["value"] for c in ext.get("potential_consequences", []) if "value" in c]
        norm_cons = list(set([normalize_consequence(c) for c in cons if normalize_consequence(c)]))
        pred_consequences.append(json.dumps(norm_cons))

    work_df["predicted_activity"] = pred_activities
    work_df["predicted_equipment"] = pred_equipment
    work_df["predicted_hazards"] = pred_hazards
    work_df["predicted_barrier_failures"] = pred_barriers
    work_df["predicted_precursor_tags"] = pred_precursor_tags
    work_df["predicted_potential_consequences"] = pred_consequences

    # 6. Save Enriched Reports CSV
    output_enriched_path = Path(output_enriched_path)
    output_enriched_path.parent.mkdir(parents=True, exist_ok=True)
    work_df.to_csv(output_enriched_path, index=False)

    # 7. Run Analytics Pipeline
    output_analytics_dir = Path(output_analytics_dir)
    output_analytics_dir.mkdir(parents=True, exist_ok=True)

    analytics_pipe = AnalyticsPipeline(
        enriched_path=output_enriched_path,
        output_dir=output_analytics_dir,
    )
    analytics_pipe.load_data()
    results = analytics_pipe.run()
    analytics_pipe.save()

    total_records = len(work_df)
    sif_count = int(work_df["sif_prediction"].sum())
    sif_density = round(float(sif_count / total_records), 4) if total_records > 0 else 0.0

    return {
        "records_processed": total_records,
        "sif_reports": sif_count,
        "sif_density": sif_density,
        "runtime_seconds": round(time.time() - start_time, 2),
    }
