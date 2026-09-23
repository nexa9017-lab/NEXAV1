"""
Build enriched analytics dataset using the reusable dataset processor.
Saves to data/analytics/enriched_reports.csv and generates analytics in artifacts/analytics/.
"""

import sys
import os
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ALL_REPORTS_FILE, ENRICHED_REPORTS_PATH, ANALYTICS_OUTPUT_DIR
from src.analytics.dataset_processor import process_dataset_enrichment_and_analytics


def build_enriched_dataset():
    """Build the enriched analytics dataset using production models."""
    print("=" * 60)
    print("BUILDING ENRICHED ANALYTICS DATASET")
    print("=" * 60)

    if not ALL_REPORTS_FILE.exists():
        raise FileNotFoundError(f"Source file not found at {ALL_REPORTS_FILE}")

    df = pd.read_csv(ALL_REPORTS_FILE)
    print(f"Loaded {len(df)} reports from {ALL_REPORTS_FILE}")

    stats = process_dataset_enrichment_and_analytics(
        df=df,
        output_enriched_path=ENRICHED_REPORTS_PATH,
        output_analytics_dir=ANALYTICS_OUTPUT_DIR,
        source_filename=ALL_REPORTS_FILE.name,
    )

    print("\n" + "=" * 60)
    print("ENRICHMENT & ANALYTICS COMPLETE")
    print("=" * 60)
    print(f"  Reports processed: {stats['records_processed']}")
    print(f"  Predicted SIF:     {stats['sif_reports']}")
    print(f"  SIF density:       {stats['sif_density']:.4f}")
    print(f"  Runtime:           {stats['runtime_seconds']}s")
    return stats


if __name__ == "__main__":
    build_enriched_dataset()
