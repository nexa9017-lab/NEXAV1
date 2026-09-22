"""
Run the full analytics pipeline on the enriched dataset.

This script only CONSUMES the already-enriched dataset.
It does NOT re-run any embedding models or classifiers.

Usage:
    python scripts/run_analytics.py
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.pipeline import AnalyticsPipeline
from src.config import ENRICHED_REPORTS_PATH, ANALYTICS_OUTPUT_DIR


def main():
    print("=" * 60)
    print("RUNNING ANALYTICS PIPELINE")
    print("=" * 60)

    if not ENRICHED_REPORTS_PATH.exists():
        print(f"\nERROR: Enriched dataset not found at {ENRICHED_REPORTS_PATH}")
        print("Run `python scripts/build_analytics_dataset.py` first.")
        sys.exit(1)

    start_time = time.time()

    # Initialize and run pipeline
    pipeline = AnalyticsPipeline(
        enriched_path=ENRICHED_REPORTS_PATH,
        output_dir=ANALYTICS_OUTPUT_DIR,
    )
    pipeline.load_data()
    results = pipeline.run()
    saved_files = pipeline.save()

    elapsed = round(time.time() - start_time, 2)

    # Print summary
    overall = results["overall_summary"]
    print("\n" + "=" * 60)
    print("ANALYTICS SUMMARY")
    print("=" * 60)

    print(f"\n--- Overall ---")
    print(f"  Total reports:          {overall['total_reports']}")
    print(f"  Predicted SIF reports:  {overall['total_predicted_sif_reports']}")
    print(f"  Overall SIF density:    {overall['overall_sif_density']:.4f}")
    print(f"  Avg SIF probability:    {overall['avg_sif_probability']:.4f}")
    print(f"  Highest volume site:    {overall['highest_volume_site']}")
    print(f"  Highest SIF count site: {overall['highest_sif_count_site']}")
    print(f"  Highest SIF density (adequate sample): {overall['highest_sif_density_site_with_adequate_sample']}")
    print(f"  Most frequent activity: {overall['most_frequent_activity']}")
    print(f"  Most frequent LSR:      {overall['most_frequent_lsr']}")
    print(f"  Most frequent hazard:   {overall['most_frequent_hazard']}")
    print(f"  Most frequent barrier:  {overall['most_frequent_barrier_failure']}")

    # Top Patterns
    patterns = results["recurring_patterns"]
    if patterns:
        print(f"\n--- Top 5 Recurring Patterns ---")
        for i, p in enumerate(patterns[:5]):
            comps = ", ".join(f"{k}={v}" for k, v in p["components"].items())
            print(f"  {i+1}. [{p['pattern_type']}] {comps}")
            print(f"     Occurrences: {p['occurrences']} | SIF: {p['sif_count']} | "
                  f"Density: {p['sif_density']:.3f} | Score: {p['priority_score']}")

    # Top Pattern Score Breakdown
    if patterns:
        top = patterns[0]
        print(f"\n--- Priority Score Breakdown (Top Pattern) ---")
        print(f"  Priority Score: {top['priority_score']}")
        for comp, val in top["priority_components"].items():
            weight = top["priority_weights"].get(comp, 0)
            print(f"    {comp}: {val:.4f} (weight: {weight})")

    # Emerging Patterns
    emerging = results["emerging_patterns"]
    if emerging:
        print(f"\n--- Emerging Patterns (top 5) ---")
        for ep in emerging[:5]:
            comps = ", ".join(f"{k}={v}" for k, v in ep["components"].items())
            pct = f"{ep['percentage_change']:.0%}" if ep["percentage_change"] is not None else "N/A"
            print(f"  [{ep['status']}] {ep['pattern_type']}: {comps}")
            print(f"    Current: {ep['current_count']} | Previous: {ep['previous_count']} | Change: {pct}")

    # Temporal Trends
    temporal = results["temporal"]
    quarterly = temporal.get("quarterly", [])
    if len(quarterly) >= 2:
        print(f"\n--- Temporal Trends (last 2 quarters) ---")
        for q in quarterly[-2:]:
            print(f"  {q['period']}: {q['report_volume']} reports, "
                  f"{q['sif_count']} SIF, density={q['sif_density']:.4f}")
            if q.get("trends"):
                for metric, trend in q["trends"].items():
                    if trend.get("percentage_change") is not None:
                        print(f"    {metric}: {trend['absolute_change']:+.4f} "
                              f"({trend['percentage_change']:+.1%})")

    # Validation
    validation = results.get("validation_summary", {})
    if validation:
        print(f"\n--- Predicted vs Ground-Truth Validation ---")
        for dim, val in validation.items():
            print(f"  {dim}: overlap {val['overlap_count']}/5 "
                  f"(ratio: {val['overlap_ratio']:.2f})")

    # Cross-site patterns
    cross_site = results["cross_site_patterns"]
    print(f"\n--- Cross-Site Patterns ---")
    print(f"  Total cross-site patterns: {len(cross_site)}")

    # Files
    print(f"\n--- Saved Files ---")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")

    print(f"\n  Analytics runtime: {elapsed}s")
    print(f"  (Enrichment runtime is separate — see build_analytics_dataset.py)")


if __name__ == "__main__":
    main()
