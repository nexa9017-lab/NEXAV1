"""
Prototype Prioritization Score (0–100).

Ranks recurring patterns using a transparent, explainable formula.
NOT a validated industrial risk score.

Formula:
    priority_score = 100 * (
        w_freq * norm_frequency
      + w_density * sif_density
      + w_prob * avg_sif_probability
      + w_cross_site * norm_cross_site
    )

All components are normalized to [0, 1] before weighting.
"""

from typing import List, Dict, Any
import numpy as np

from src.config import (
    PRIORITY_WEIGHT_FREQUENCY,
    PRIORITY_WEIGHT_SIF_DENSITY,
    PRIORITY_WEIGHT_AVG_PROBABILITY,
    PRIORITY_WEIGHT_CROSS_SITE,
)


def _safe_min_max_normalize(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalize a value to [0, 1]. Safe for min == max."""
    if max_val == min_val:
        # All values identical — assign 0.5 as neutral midpoint
        return 0.5
    return (value - min_val) / (max_val - min_val)


class PriorityScorer:
    """
    Computes Prototype Prioritization Score for recurring patterns.
    Score range: 0–100, deterministic, explainable.
    """

    def __init__(
        self,
        weight_frequency: float = PRIORITY_WEIGHT_FREQUENCY,
        weight_sif_density: float = PRIORITY_WEIGHT_SIF_DENSITY,
        weight_avg_probability: float = PRIORITY_WEIGHT_AVG_PROBABILITY,
        weight_cross_site: float = PRIORITY_WEIGHT_CROSS_SITE,
    ):
        self.weights = {
            "frequency": weight_frequency,
            "sif_density": weight_sif_density,
            "avg_sif_probability": weight_avg_probability,
            "cross_site": weight_cross_site,
        }
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def score_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score and rank patterns. Returns patterns with added priority_score
        and priority_components fields.
        """
        if not patterns:
            return []

        # Extract raw values for normalization
        occurrences = [p["occurrences"] for p in patterns]
        site_counts = [p.get("affected_site_count", 1) for p in patterns]

        min_occ, max_occ = min(occurrences), max(occurrences)
        min_sites, max_sites = min(site_counts), max(site_counts)

        scored = []
        for pattern in patterns:
            # Normalize components to [0, 1]
            freq_norm = _safe_min_max_normalize(pattern["occurrences"], min_occ, max_occ)
            density_norm = min(1.0, max(0.0, pattern.get("sif_density", 0.0)))
            prob_norm = min(1.0, max(0.0, pattern.get("avg_sif_probability", 0.0)))
            cross_site_norm = _safe_min_max_normalize(
                pattern.get("affected_site_count", 1), min_sites, max_sites
            )

            # Weighted sum
            raw_score = (
                self.weights["frequency"] * freq_norm
                + self.weights["sif_density"] * density_norm
                + self.weights["avg_sif_probability"] * prob_norm
                + self.weights["cross_site"] * cross_site_norm
            )

            # Scale to 0–100
            priority_score = round(min(100.0, max(0.0, raw_score * 100)), 1)

            scored_pattern = {
                **pattern,
                "priority_score": priority_score,
                "priority_components": {
                    "frequency": round(freq_norm, 4),
                    "sif_density": round(density_norm, 4),
                    "avg_sif_probability": round(prob_norm, 4),
                    "cross_site": round(cross_site_norm, 4),
                },
                "priority_weights": dict(self.weights),
                "priority_label": "Prototype Prioritization Score",
            }
            scored.append(scored_pattern)

        # Sort by priority score descending
        scored.sort(key=lambda x: x["priority_score"], reverse=True)
        return scored
