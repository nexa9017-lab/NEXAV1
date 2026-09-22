"""
Analytics orchestration pipeline.

Coordinates all analytics stages and produces a complete, JSON-serializable
output. CLI scripts call this pipeline rather than reimplementing logic.
"""

import json
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from src.analytics.aggregation import (
    AnalyticsAggregator, sif_density, safe_parse_json_list
)
from src.analytics.pattern_mining import PatternMiner
from src.analytics.risk_scoring import PriorityScorer
from src.analytics.temporal import TemporalAnalyzer
from src.config import (
    ENRICHED_REPORTS_PATH, ANALYTICS_OUTPUT_DIR,
    MIN_PATTERN_SUPPORT, LOW_SAMPLE_SIZE_THRESHOLD,
    ANALYTICS_VERSION, ANALYTICS_PROTOTYPE_NOTICE,
    EMERGING_PERIOD, EMERGING_PATTERN_THRESHOLD,
)


# JSON list columns in the enriched dataset
JSON_LIST_COLUMNS = [
    "predicted_activity",
    "predicted_equipment",
    "predicted_hazards",
    "predicted_barrier_failures",
    "predicted_precursor_tags",
    "predicted_potential_consequences",
    "predicted_lsr",
]


def _sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _make_metadata(report_count: int, source: str = "enriched_reports.csv") -> Dict[str, Any]:
    """Create standard analytics metadata block."""
    return {
        "generated_at": datetime.now().isoformat(),
        "source_dataset": source,
        "report_count": report_count,
        "analytics_version": ANALYTICS_VERSION,
        "uses_predicted_fields": True,
        "prototype_notice": ANALYTICS_PROTOTYPE_NOTICE,
    }


class AnalyticsPipeline:
    """
    Orchestrates full analytics computation from enriched reports.

    Responsibilities:
    - Load and parse enriched dataset
    - Run aggregation, temporal, pattern mining, scoring
    - Run predicted-vs-ground-truth validation
    - Serialize all outputs to JSON
    """

    def __init__(self, enriched_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.enriched_path = enriched_path or ENRICHED_REPORTS_PATH
        self.output_dir = Path(output_dir or ANALYTICS_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.results = {}
        self.warnings = []

    def load_data(self) -> pd.DataFrame:
        """Load and validate enriched reports CSV."""
        print(f"Loading enriched reports from {self.enriched_path}...")
        self.df = pd.read_csv(self.enriched_path)

        required = [
            "report_id", "sif_prediction", "sif_probability",
            "predicted_activity", "predicted_lsr"
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"  Loaded {len(self.df)} reports with {len(self.df.columns)} columns")
        return self.df

    def run(self) -> Dict[str, Any]:
        """Execute the full analytics pipeline. Returns consolidated results."""
        start_time = time.time()

        if self.df is None:
            self.load_data()

        metadata = _make_metadata(len(self.df))

        # 1. Overall Summary
        print("Computing overall summary...")
        aggregator = AnalyticsAggregator(self.df)
        overall = aggregator.overall_summary()

        # 2. Dimensional Analytics
        print("Computing site analytics...")
        sites = aggregator.site_analytics()

        print("Computing activity analytics...")
        activities = aggregator.activity_analytics()

        print("Computing LSR analytics...")
        lsr = aggregator.lsr_analytics()

        print("Computing hazard analytics...")
        hazards = aggregator.hazard_analytics()

        print("Computing barrier analytics...")
        barriers = aggregator.barrier_analytics()

        print("Computing consequence analytics...")
        consequences = aggregator.consequence_analytics()

        # 3. Temporal Analytics
        print("Computing temporal analytics...")
        temporal_analyzer = TemporalAnalyzer(self.df)
        temporal = temporal_analyzer.full_temporal_analytics()

        # 4. Pattern Mining
        print("Mining recurring patterns...")
        miner = PatternMiner(self.df, min_support=MIN_PATTERN_SUPPORT)
        pair_patterns = miner.mine_pair_patterns()
        triple_patterns = miner.mine_triple_patterns()
        all_patterns = pair_patterns + triple_patterns

        # 5. Prioritization Scoring
        print("Computing prioritization scores...")
        scorer = PriorityScorer()
        scored_patterns = scorer.score_patterns(all_patterns)

        # Add top pattern to overall summary
        if scored_patterns:
            overall["highest_ranked_recurring_pattern"] = {
                "pattern_id": scored_patterns[0]["pattern_id"],
                "priority_score": scored_patterns[0]["priority_score"],
                "components": scored_patterns[0]["components"],
            }
        else:
            overall["highest_ranked_recurring_pattern"] = None

        # 6. Emerging Pattern Detection
        print("Detecting emerging patterns...")
        emerging = miner.detect_emerging_patterns(
            all_patterns, period=EMERGING_PERIOD, threshold=EMERGING_PATTERN_THRESHOLD
        )

        # 7. Cross-Site Patterns
        print("Identifying cross-site patterns...")
        cross_site = miner.cross_site_patterns(scored_patterns, min_sites=2)

        # 8. Predicted vs Ground-Truth Validation
        print("Running predicted-vs-ground-truth validation...")
        validation = self._run_validation()

        # Assemble results
        elapsed = round(time.time() - start_time, 2)
        print(f"\nAnalytics pipeline completed in {elapsed}s")

        self.results = {
            "metadata": metadata,
            "overall_summary": overall,
            "sites": sites,
            "activities": activities,
            "lsr": lsr,
            "hazards": hazards,
            "barriers": barriers,
            "consequences": consequences,
            "temporal": temporal,
            "recurring_patterns": scored_patterns,
            "emerging_patterns": emerging,
            "cross_site_patterns": cross_site,
            "validation_summary": validation,
            "analytics_runtime_seconds": elapsed,
        }

        return self.results

    def _run_validation(self) -> Dict[str, Any]:
        """Compare predicted analytics vs ground-truth analytics."""
        gt_cols = {
            "activity": "activity",
            "hazard": "hazard",
            "barrier_failure": "barrier_failure",
        }
        pred_cols = {
            "activity": "predicted_activity",
            "hazard": "predicted_hazards",
            "barrier_failure": "predicted_barrier_failures",
        }

        validation = {}

        for dim, gt_col in gt_cols.items():
            if gt_col not in self.df.columns:
                continue
            pred_col = pred_cols[dim]
            if pred_col not in self.df.columns:
                continue

            # Get top-5 predicted
            pred_density = sif_density(self.df, pred_col, explode=True)
            pred_top5 = [r["name"] for r in pred_density[:5]]

            # Get top-5 ground truth
            gt_density = sif_density(self.df, gt_col, explode=False)
            gt_top5 = [r["name"] for r in gt_density[:5]]

            overlap = set(pred_top5) & set(gt_top5)
            validation[dim] = {
                "predicted_top5": pred_top5,
                "ground_truth_top5": gt_top5,
                "overlap_count": len(overlap),
                "overlap_ratio": round(len(overlap) / 5, 2) if len(pred_top5) >= 5 else round(len(overlap) / max(1, len(pred_top5)), 2),
                "overlapping_items": list(overlap),
            }

        # Site SIF density comparison
        if "site" in self.df.columns and "sif_label" in self.df.columns:
            pred_site = sorted(
                sif_density(self.df, "site"),
                key=lambda x: x["sif_density"], reverse=True
            )
            pred_top5_sites = [r["name"] for r in pred_site[:5]]

            # GT SIF density
            gt_df = self.df.copy()
            gt_df["sif_prediction_gt"] = gt_df["sif_label"]
            gt_grouped = gt_df.groupby("site").agg(
                total=("sif_label", "count"),
                sif_count=("sif_label", "sum"),
            ).reset_index()
            gt_grouped["density"] = gt_grouped["sif_count"] / gt_grouped["total"]
            gt_grouped = gt_grouped.sort_values("density", ascending=False)
            gt_top5_sites = gt_grouped["site"].head(5).tolist()

            overlap = set(pred_top5_sites) & set(gt_top5_sites)
            validation["site_sif_density"] = {
                "predicted_top5": pred_top5_sites,
                "ground_truth_top5": gt_top5_sites,
                "overlap_count": len(overlap),
                "overlap_ratio": round(len(overlap) / max(1, min(5, len(pred_top5_sites))), 2),
                "overlapping_items": list(overlap),
            }

        return validation

    def save(self) -> Dict[str, str]:
        """Save all analytics outputs to JSON files."""
        if not self.results:
            raise ValueError("No results to save. Run pipeline first.")

        saved_files = {}

        # Individual files
        file_mapping = {
            "overall_summary.json": {"metadata": self.results["metadata"],
                                      "data": self.results["overall_summary"]},
            "site_analytics.json": {"metadata": self.results["metadata"],
                                     "data": self.results["sites"]},
            "activity_analytics.json": {"metadata": self.results["metadata"],
                                         "data": self.results["activities"]},
            "lsr_analytics.json": {"metadata": self.results["metadata"],
                                    "data": self.results["lsr"]},
            "hazard_analytics.json": {"metadata": self.results["metadata"],
                                       "data": self.results["hazards"]},
            "barrier_analytics.json": {"metadata": self.results["metadata"],
                                        "data": self.results["barriers"]},
            "consequence_analytics.json": {"metadata": self.results["metadata"],
                                            "data": self.results["consequences"]},
            "temporal_analytics.json": {"metadata": self.results["metadata"],
                                         "data": self.results["temporal"]},
            "recurring_patterns.json": {"metadata": self.results["metadata"],
                                         "data": self.results["recurring_patterns"]},
            "emerging_patterns.json": {"metadata": self.results["metadata"],
                                        "data": self.results["emerging_patterns"]},
            "cross_site_patterns.json": {"metadata": self.results["metadata"],
                                          "data": self.results["cross_site_patterns"]},
            "analytics_validation.json": {"metadata": self.results["metadata"],
                                           "data": self.results["validation_summary"]},
        }

        for filename, content in file_mapping.items():
            path = self.output_dir / filename
            sanitized = _sanitize_for_json(content)
            with open(path, "w") as f:
                json.dump(sanitized, f, indent=2, default=str)
            saved_files[filename] = str(path)

        # Consolidated full analytics
        full_path = self.output_dir / "full_analytics.json"
        sanitized_full = _sanitize_for_json(self.results)
        with open(full_path, "w") as f:
            json.dump(sanitized_full, f, indent=2, default=str)
        saved_files["full_analytics.json"] = str(full_path)

        print(f"\nSaved {len(saved_files)} analytics files to {self.output_dir}")
        return saved_files
