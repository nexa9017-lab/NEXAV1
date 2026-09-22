"""
Recurring-pattern mining engine.

Discovers pair and triple combinations across safety dimensions,
with cross-site detection and emerging-pattern analysis.
All patterns use predicted fields only.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

from src.analytics.aggregation import safe_parse_json_list
from src.config import (
    MIN_PATTERN_SUPPORT, EMERGING_PATTERN_THRESHOLD, EMERGING_PERIOD
)


def _make_pattern_id(pattern_type: str, components: Dict[str, str]) -> str:
    """Generate deterministic pattern ID from type and component values."""
    parts = [pattern_type]
    for key in sorted(components.keys()):
        val = components[key].lower().replace(" ", "_").replace("/", "_")
        parts.append(val)
    return "__".join(parts)


def _get_report_excerpts(df: pd.DataFrame, report_ids: List[str],
                         max_excerpts: int = 3, max_len: int = 120) -> List[Dict[str, str]]:
    """Get short description excerpts for example reports."""
    excerpts = []
    desc_col = "description" if "description" in df.columns else None
    for rid in report_ids[:max_excerpts]:
        row = df[df["report_id"] == rid]
        if row.empty:
            continue
        desc = ""
        if desc_col:
            raw = str(row.iloc[0][desc_col])
            desc = raw[:max_len] + ("..." if len(raw) > max_len else "")
        excerpts.append({"report_id": rid, "description_excerpt": desc})
    return excerpts


class PatternMiner:
    """
    Discovers recurring combinations across safety dimensions.
    Supports pair and triple patterns with minimum-support filtering.
    """

    # Pattern definitions: (pattern_type, col_a, col_b, [col_c], explode_a, explode_b, [explode_c])
    PAIR_PATTERNS = [
        ("activity_barrier",   "predicted_activity", "predicted_barrier_failures", True, True),
        ("activity_hazard",    "predicted_activity", "predicted_hazards",          True, True),
        ("activity_lsr",       "predicted_activity", "predicted_lsr",              True, True),
        ("site_lsr",           "site",               "predicted_lsr",              False, True),
        ("site_barrier",       "site",               "predicted_barrier_failures", False, True),
        ("site_hazard",        "site",               "predicted_hazards",          False, True),
        ("hazard_barrier",     "predicted_hazards",   "predicted_barrier_failures", True, True),
        ("location_precursor", "location",            "predicted_precursor_tags",   False, True),
        ("lsr_barrier",        "predicted_lsr",       "predicted_barrier_failures", True, True),
    ]

    TRIPLE_PATTERNS = [
        ("activity_hazard_barrier", "predicted_activity", "predicted_hazards",
         "predicted_barrier_failures", True, True, True),
    ]

    def __init__(self, df: pd.DataFrame, min_support: int = MIN_PATTERN_SUPPORT):
        self.df = df.copy()
        self.min_support = min_support
        self.total_reports = len(df)

    def _explode_col(self, df: pd.DataFrame, col: str, should_explode: bool) -> pd.DataFrame:
        """Explode a column if it contains JSON lists."""
        if should_explode:
            df = df.copy()
            df[col] = df[col].apply(safe_parse_json_list)
            df = df.explode(col)
            df = df[df[col].notna() & (df[col] != "")]
        return df

    def mine_pair_patterns(self) -> List[Dict[str, Any]]:
        """Mine all configured pair patterns."""
        all_patterns = []
        for pattern_type, col_a, col_b, explode_a, explode_b in self.PAIR_PATTERNS:
            if col_a not in self.df.columns or col_b not in self.df.columns:
                continue
            patterns = self._mine_pair(pattern_type, col_a, col_b, explode_a, explode_b)
            all_patterns.extend(patterns)
        return all_patterns

    def mine_triple_patterns(self) -> List[Dict[str, Any]]:
        """Mine all configured triple patterns."""
        all_patterns = []
        for (pattern_type, col_a, col_b, col_c,
             explode_a, explode_b, explode_c) in self.TRIPLE_PATTERNS:
            if any(c not in self.df.columns for c in [col_a, col_b, col_c]):
                continue
            patterns = self._mine_triple(
                pattern_type, col_a, col_b, col_c,
                explode_a, explode_b, explode_c
            )
            all_patterns.extend(patterns)
        return all_patterns

    def _mine_pair(self, pattern_type: str, col_a: str, col_b: str,
                   explode_a: bool, explode_b: bool) -> List[Dict[str, Any]]:
        """Mine patterns for a single pair definition."""
        needed_cols = list(dict.fromkeys([
            "report_id", "site", "date", "sif_prediction",
            "sif_probability", col_a, col_b
        ]))
        work = self.df[needed_cols].copy()
        if "description" in self.df.columns:
            work["description"] = self.df["description"].values

        work = self._explode_col(work, col_a, explode_a)
        work = self._explode_col(work, col_b, explode_b)

        # Deduplicate per report per combination
        work = work.drop_duplicates(subset=["report_id", col_a, col_b])

        if work.empty:
            return []

        grouped = work.groupby([col_a, col_b]).agg(
            occurrences=("report_id", "count"),
            sif_count=("sif_prediction", "sum"),
            avg_sif_probability=("sif_probability", "mean"),
            report_ids=("report_id", lambda x: list(x.unique())),
            sites=("site", lambda x: list(x.unique())),
            min_date=("date", "min"),
            max_date=("date", "max"),
        ).reset_index()

        results = []
        for _, row in grouped.iterrows():
            occ = int(row["occurrences"])
            if occ < self.min_support:
                continue

            sif_c = int(row["sif_count"])
            components = {
                col_a.replace("predicted_", ""): row[col_a],
                col_b.replace("predicted_", ""): row[col_b],
            }
            pattern_id = _make_pattern_id(pattern_type, components)
            sites = row["sites"] if isinstance(row["sites"], list) else [row["sites"]]
            report_ids = row["report_ids"] if isinstance(row["report_ids"], list) else [row["report_ids"]]

            excerpt_df = work if "description" not in work.columns else work
            example_reports = _get_report_excerpts(
                self.df, report_ids[:5]
            )

            results.append({
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "components": components,
                "occurrences": occ,
                "sif_count": sif_c,
                "non_sif_count": occ - sif_c,
                "sif_density": round(sif_c / occ, 4) if occ > 0 else 0.0,
                "avg_sif_probability": round(float(row["avg_sif_probability"]), 4),
                "support_ratio": round(occ / self.total_reports, 4) if self.total_reports > 0 else 0.0,
                "affected_site_count": len(sites),
                "affected_sites": sites,
                "date_range": {
                    "min": str(row["min_date"]) if pd.notna(row["min_date"]) else None,
                    "max": str(row["max_date"]) if pd.notna(row["max_date"]) else None,
                },
                "example_report_ids": report_ids[:5],
                "example_reports": example_reports,
                "data_quality_warnings": (
                    ["LOW_PATTERN_SUPPORT"] if occ < 10 else []
                ),
            })

        results.sort(key=lambda x: x["occurrences"], reverse=True)
        return results

    def _mine_triple(self, pattern_type: str, col_a: str, col_b: str,
                     col_c: str, explode_a: bool, explode_b: bool,
                     explode_c: bool) -> List[Dict[str, Any]]:
        """Mine patterns for a triple definition."""
        needed_cols = list(dict.fromkeys([
            "report_id", "site", "date", "sif_prediction",
            "sif_probability", col_a, col_b, col_c
        ]))
        work = self.df[needed_cols].copy()
        if "description" in self.df.columns:
            work["description"] = self.df["description"].values

        work = self._explode_col(work, col_a, explode_a)
        work = self._explode_col(work, col_b, explode_b)
        work = self._explode_col(work, col_c, explode_c)
        work = work.drop_duplicates(subset=["report_id", col_a, col_b, col_c])

        if work.empty:
            return []

        grouped = work.groupby([col_a, col_b, col_c]).agg(
            occurrences=("report_id", "count"),
            sif_count=("sif_prediction", "sum"),
            avg_sif_probability=("sif_probability", "mean"),
            report_ids=("report_id", lambda x: list(x.unique())),
            sites=("site", lambda x: list(x.unique())),
            min_date=("date", "min"),
            max_date=("date", "max"),
        ).reset_index()

        results = []
        for _, row in grouped.iterrows():
            occ = int(row["occurrences"])
            if occ < self.min_support:
                continue
            sif_c = int(row["sif_count"])
            components = {
                col_a.replace("predicted_", ""): row[col_a],
                col_b.replace("predicted_", ""): row[col_b],
                col_c.replace("predicted_", ""): row[col_c],
            }
            pattern_id = _make_pattern_id(pattern_type, components)
            sites = row["sites"] if isinstance(row["sites"], list) else [row["sites"]]
            report_ids = row["report_ids"] if isinstance(row["report_ids"], list) else [row["report_ids"]]

            example_reports = _get_report_excerpts(self.df, report_ids[:5])

            results.append({
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "components": components,
                "occurrences": occ,
                "sif_count": sif_c,
                "non_sif_count": occ - sif_c,
                "sif_density": round(sif_c / occ, 4) if occ > 0 else 0.0,
                "avg_sif_probability": round(float(row["avg_sif_probability"]), 4),
                "support_ratio": round(occ / self.total_reports, 4) if self.total_reports > 0 else 0.0,
                "affected_site_count": len(sites),
                "affected_sites": sites,
                "date_range": {
                    "min": str(row["min_date"]) if pd.notna(row["min_date"]) else None,
                    "max": str(row["max_date"]) if pd.notna(row["max_date"]) else None,
                },
                "example_report_ids": report_ids[:5],
                "example_reports": example_reports,
                "data_quality_warnings": (
                    ["LOW_PATTERN_SUPPORT"] if occ < 10 else []
                ),
            })

        results.sort(key=lambda x: x["occurrences"], reverse=True)
        return results

    def cross_site_patterns(self, patterns: List[Dict[str, Any]],
                            min_sites: int = 2) -> List[Dict[str, Any]]:
        """Filter patterns occurring across multiple sites."""
        return [
            p for p in patterns
            if p["affected_site_count"] >= min_sites
        ]

    def detect_emerging_patterns(
        self,
        patterns: List[Dict[str, Any]],
        period: str = None,
        threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect emerging patterns by comparing recent vs previous period.

        Returns patterns with status: STABLE, INCREASING, DECREASING, EMERGING,
        NEW_PATTERN.
        """
        period = period or EMERGING_PERIOD
        threshold = threshold if threshold is not None else EMERGING_PATTERN_THRESHOLD

        if "date" not in self.df.columns:
            return []

        dates = pd.to_datetime(self.df["date"], errors="coerce")
        valid_dates = dates.dropna()
        if valid_dates.empty:
            return []

        max_date = valid_dates.max()

        if period == "quarter":
            current_start = max_date - pd.DateOffset(months=3)
            previous_start = current_start - pd.DateOffset(months=3)
        else:  # month
            current_start = max_date - pd.DateOffset(months=1)
            previous_start = current_start - pd.DateOffset(months=1)

        current_mask = (dates >= current_start) & (dates <= max_date)
        previous_mask = (dates >= previous_start) & (dates < current_start)

        current_df = self.df[current_mask]
        previous_df = self.df[previous_mask]

        # Re-mine patterns for each period
        current_miner = PatternMiner(current_df, min_support=1) if len(current_df) > 0 else None
        previous_miner = PatternMiner(previous_df, min_support=1) if len(previous_df) > 0 else None

        # Build pattern counts per period
        def _count_patterns(miner):
            if miner is None:
                return {}
            counts = {}
            for p in miner.mine_pair_patterns():
                counts[p["pattern_id"]] = p["occurrences"]
            for p in miner.mine_triple_patterns():
                counts[p["pattern_id"]] = p["occurrences"]
            return counts

        current_counts = _count_patterns(current_miner)
        previous_counts = _count_patterns(previous_miner)

        results = []
        for pattern in patterns:
            pid = pattern["pattern_id"]
            curr = current_counts.get(pid, 0)
            prev = previous_counts.get(pid, 0)

            if curr < self.min_support:
                continue

            abs_change = curr - prev
            if prev == 0 and curr > 0:
                status = "NEW_PATTERN"
                pct_change = None
            elif prev == 0:
                continue
            else:
                pct_change = round(abs_change / prev, 4)
                if pct_change >= threshold:
                    status = "EMERGING"
                elif pct_change > 0:
                    status = "INCREASING"
                elif pct_change < 0:
                    status = "DECREASING"
                else:
                    status = "STABLE"

            results.append({
                "pattern_id": pid,
                "pattern_type": pattern["pattern_type"],
                "components": pattern["components"],
                "status": status,
                "current_count": curr,
                "previous_count": prev,
                "absolute_change": abs_change,
                "percentage_change": pct_change,
                "comparison_period": period,
                "current_period": f"{current_start.date()} to {max_date.date()}",
                "previous_period": f"{previous_start.date()} to {current_start.date()}",
            })

        # Sort: EMERGING and NEW_PATTERN first, then by absolute change
        status_priority = {"EMERGING": 0, "NEW_PATTERN": 1, "INCREASING": 2, "STABLE": 3, "DECREASING": 4}
        results.sort(key=lambda x: (status_priority.get(x["status"], 5), -x["absolute_change"]))
        return results
