"""
Aggregation engine for safety analytics.

Computes SIF density, weighted SIF probability, and dimensional summaries
for sites, activities, LSRs, hazards, barriers, consequences, and precursors.
All analytics use predicted fields only — ground truth is never mixed in.
"""

import json
import warnings
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config import LOW_SAMPLE_SIZE_THRESHOLD


# ---------------------------------------------------------------------------
# Utility: Safe JSON list parsing
# ---------------------------------------------------------------------------

def safe_parse_json_list(val) -> List[str]:
    """Parse a JSON-serialized list from a CSV cell. Never uses eval()."""
    if isinstance(val, list):
        return [str(x) for x in val]
    if val is None:
        return []
    try:
        if pd.isna(val):
            return []
    except (ValueError, TypeError):
        pass
    if val == "" or val == "null":
        return []
    val = str(val).strip()
    if not val:
        return []
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        warnings.warn(f"MALFORMED_LIST_FIELD: could not parse '{val[:80]}...'")
        return []


def deduplicated_list(items: List[str]) -> List[str]:
    """Return deduplicated list preserving order."""
    seen = set()
    result = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


# ---------------------------------------------------------------------------
# SIF Density
# ---------------------------------------------------------------------------

def sif_density(df: pd.DataFrame, group_col: str, explode: bool = False) -> List[Dict[str, Any]]:
    """
    Compute SIF density for a given dimension column.

    Parameters
    ----------
    df : DataFrame with 'sif_prediction' (0/1) and 'sif_probability' columns.
    group_col : Column name to group by.
    explode : If True, the column contains JSON lists that need exploding.

    Returns list of dicts with total_reports, sif_count, non_sif_count,
    sif_density, avg_sif_probability, and data_quality_warnings.
    """
    work = df.copy()

    if explode:
        work[group_col] = work[group_col].apply(safe_parse_json_list)
        work = work.explode(group_col)
        work = work[work[group_col].notna() & (work[group_col] != "")]
        # Deduplicate: one report should count once per unique value
        work = work.drop_duplicates(subset=["report_id", group_col])

    results = []
    if work.empty:
        return results

    grouped = work.groupby(group_col).agg(
        total_reports=("sif_prediction", "count"),
        sif_count=("sif_prediction", "sum"),
        avg_sif_probability=("sif_probability", "mean"),
    ).reset_index()

    for _, row in grouped.iterrows():
        total = int(row["total_reports"])
        sif_c = int(row["sif_count"])
        density = sif_c / total if total > 0 else 0.0
        warnings_list = []
        if total < LOW_SAMPLE_SIZE_THRESHOLD:
            warnings_list.append("LOW_SAMPLE_SIZE")

        results.append({
            "name": row[group_col],
            "total_reports": total,
            "sif_count": sif_c,
            "non_sif_count": total - sif_c,
            "sif_density": round(density, 4),
            "avg_sif_probability": round(float(row["avg_sif_probability"]), 4),
            "data_quality_warnings": warnings_list,
        })

    results.sort(key=lambda x: x["total_reports"], reverse=True)
    return results


def weighted_sif_probability(df: pd.DataFrame, group_col: str, explode: bool = False) -> List[Dict[str, Any]]:
    """Average SIF probability per dimension. Separate from binary SIF density."""
    work = df.copy()
    if explode:
        work[group_col] = work[group_col].apply(safe_parse_json_list)
        work = work.explode(group_col)
        work = work[work[group_col].notna() & (work[group_col] != "")]
        work = work.drop_duplicates(subset=["report_id", group_col])

    if work.empty:
        return []

    grouped = work.groupby(group_col).agg(
        count=("sif_probability", "count"),
        avg_probability=("sif_probability", "mean"),
        max_probability=("sif_probability", "max"),
    ).reset_index()

    results = []
    for _, row in grouped.iterrows():
        results.append({
            "name": row[group_col],
            "count": int(row["count"]),
            "avg_sif_probability": round(float(row["avg_probability"]), 4),
            "max_sif_probability": round(float(row["max_probability"]), 4),
        })

    results.sort(key=lambda x: x["avg_sif_probability"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Dimensional Analytics
# ---------------------------------------------------------------------------

class AnalyticsAggregator:
    """Core aggregation engine for safety analytics."""

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : Enriched reports DataFrame with predicted fields.
        """
        self.df = df.copy()
        self._validate()

    def _validate(self):
        required = [
            "report_id", "sif_prediction", "sif_probability",
            "predicted_activity", "predicted_lsr"
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _top_items(self, density_results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Return top-k items from density results by total_reports."""
        return density_results[:top_k]

    def _top_by_sif_density(self, density_results: List[Dict], top_k: int = 5,
                            min_sample: int = None) -> List[Dict]:
        """Return top-k by SIF density, optionally filtering by sample size."""
        filtered = density_results
        if min_sample is not None:
            filtered = [r for r in filtered if r["total_reports"] >= min_sample]
        return sorted(filtered, key=lambda x: x["sif_density"], reverse=True)[:top_k]

    # ---- Site Analytics ----

    def site_analytics(self) -> List[Dict[str, Any]]:
        """Full site-level analytics."""
        densities = sif_density(self.df, "site")
        site_results = []

        for site_info in densities:
            site_name = site_info["name"]
            site_df = self.df[self.df["site"] == site_name]

            # Top activities
            act_density = sif_density(site_df, "predicted_activity", explode=True)
            # Top hazards
            haz_density = sif_density(site_df, "predicted_hazards", explode=True)
            # Top barriers
            bar_density = sif_density(site_df, "predicted_barrier_failures", explode=True)
            # Top LSRs
            lsr_density = sif_density(site_df, "predicted_lsr", explode=True)
            # Top precursor tags
            pre_density = sif_density(site_df, "predicted_precursor_tags", explode=True)

            # Reports per month
            if "date" in site_df.columns:
                site_dates = pd.to_datetime(site_df["date"], errors="coerce")
                valid_dates = site_dates.dropna()
                if len(valid_dates) > 0:
                    months_span = max(1, (valid_dates.max() - valid_dates.min()).days / 30.0)
                    reports_per_month = round(len(valid_dates) / months_span, 2)
                else:
                    reports_per_month = 0.0
            else:
                reports_per_month = 0.0

            site_results.append({
                **site_info,
                "reports_per_month": reports_per_month,
                "top_activities": self._top_items(act_density),
                "top_hazards": self._top_items(haz_density),
                "top_barriers": self._top_items(bar_density),
                "top_lsr": self._top_items(lsr_density),
                "top_precursor_tags": self._top_items(pre_density),
            })

        return site_results

    # ---- Activity Analytics ----

    def activity_analytics(self) -> List[Dict[str, Any]]:
        """Per-activity analytics with associated dimensions."""
        densities = sif_density(self.df, "predicted_activity", explode=True)
        results = []
        for act_info in densities:
            act_name = act_info["name"]
            # Filter to reports containing this activity
            mask = self.df["predicted_activity"].apply(
                lambda x: act_name in safe_parse_json_list(x)
            )
            act_df = self.df[mask]

            haz = sif_density(act_df, "predicted_hazards", explode=True)
            bar = sif_density(act_df, "predicted_barrier_failures", explode=True)
            lsr = sif_density(act_df, "predicted_lsr", explode=True)
            sites = act_df["site"].unique().tolist() if "site" in act_df.columns else []

            results.append({
                **act_info,
                "affected_sites": sites,
                "affected_site_count": len(sites),
                "common_hazards": self._top_items(haz),
                "common_barriers": self._top_items(bar),
                "common_lsr": self._top_items(lsr),
            })
        return results

    # ---- LSR Analytics ----

    def lsr_analytics(self) -> List[Dict[str, Any]]:
        """Per-LSR analytics (multi-label aware)."""
        densities = sif_density(self.df, "predicted_lsr", explode=True)
        results = []
        for lsr_info in densities:
            lsr_name = lsr_info["name"]
            mask = self.df["predicted_lsr"].apply(
                lambda x: lsr_name in safe_parse_json_list(x)
            )
            lsr_df = self.df[mask]

            acts = sif_density(lsr_df, "predicted_activity", explode=True)
            bars = sif_density(lsr_df, "predicted_barrier_failures", explode=True)
            sites = lsr_df["site"].unique().tolist() if "site" in lsr_df.columns else []

            results.append({
                **lsr_info,
                "associated_sites": sites,
                "associated_site_count": len(sites),
                "associated_activities": self._top_items(acts),
                "common_barriers": self._top_items(bars),
            })
        return results

    # ---- Hazard Analytics ----

    def hazard_analytics(self) -> List[Dict[str, Any]]:
        """Per-hazard analytics."""
        densities = sif_density(self.df, "predicted_hazards", explode=True)
        results = []
        for haz_info in densities:
            haz_name = haz_info["name"]
            mask = self.df["predicted_hazards"].apply(
                lambda x: haz_name in safe_parse_json_list(x)
            )
            haz_df = self.df[mask]

            acts = sif_density(haz_df, "predicted_activity", explode=True)
            bars = sif_density(haz_df, "predicted_barrier_failures", explode=True)
            sites = haz_df["site"].unique().tolist() if "site" in haz_df.columns else []

            results.append({
                **haz_info,
                "associated_sites": sites,
                "associated_site_count": len(sites),
                "associated_activities": self._top_items(acts),
                "associated_barriers": self._top_items(bars),
            })
        return results

    # ---- Barrier Analytics ----

    def barrier_analytics(self) -> List[Dict[str, Any]]:
        """Per-barrier-failure analytics."""
        densities = sif_density(self.df, "predicted_barrier_failures", explode=True)
        results = []
        for bar_info in densities:
            bar_name = bar_info["name"]
            mask = self.df["predicted_barrier_failures"].apply(
                lambda x: bar_name in safe_parse_json_list(x)
            )
            bar_df = self.df[mask]

            acts = sif_density(bar_df, "predicted_activity", explode=True)
            hazs = sif_density(bar_df, "predicted_hazards", explode=True)
            sites = bar_df["site"].unique().tolist() if "site" in bar_df.columns else []

            results.append({
                **bar_info,
                "affected_sites": sites,
                "affected_site_count": len(sites),
                "associated_activities": self._top_items(acts),
                "associated_hazards": self._top_items(hazs),
            })
        return results

    # ---- Consequence Analytics ----

    def consequence_analytics(self) -> List[Dict[str, Any]]:
        """Per-potential-consequence analytics (labeled as Potential Consequences)."""
        if "predicted_potential_consequences" not in self.df.columns:
            return []
        densities = sif_density(self.df, "predicted_potential_consequences", explode=True)
        results = []
        for con_info in densities:
            con_name = con_info["name"]
            mask = self.df["predicted_potential_consequences"].apply(
                lambda x: con_name in safe_parse_json_list(x)
            )
            con_df = self.df[mask]

            hazs = sif_density(con_df, "predicted_hazards", explode=True)
            bars = sif_density(con_df, "predicted_barrier_failures", explode=True)
            acts = sif_density(con_df, "predicted_activity", explode=True)

            results.append({
                **con_info,
                "label": "Potential Consequence",
                "associated_hazards": self._top_items(hazs),
                "associated_barriers": self._top_items(bars),
                "affected_activities": self._top_items(acts),
            })
        return results

    # ---- Overall Summary ----

    def overall_summary(self) -> Dict[str, Any]:
        """Top-level analytics summary."""
        total = len(self.df)
        sif_total = int(self.df["sif_prediction"].sum())
        density = sif_total / total if total > 0 else 0.0
        avg_prob = float(self.df["sif_probability"].mean()) if total > 0 else 0.0

        # Site rankings
        site_densities = sif_density(self.df, "site")
        sites_by_volume = sorted(site_densities, key=lambda x: x["total_reports"], reverse=True)
        sites_by_sif_count = sorted(site_densities, key=lambda x: x["sif_count"], reverse=True)
        sites_by_density_adequate = sorted(
            [s for s in site_densities if s["total_reports"] >= LOW_SAMPLE_SIZE_THRESHOLD],
            key=lambda x: x["sif_density"], reverse=True
        )
        sites_by_density_all = sorted(site_densities, key=lambda x: x["sif_density"], reverse=True)

        # Top dimensions
        act_densities = sif_density(self.df, "predicted_activity", explode=True)
        lsr_densities = sif_density(self.df, "predicted_lsr", explode=True)
        haz_densities = sif_density(self.df, "predicted_hazards", explode=True)
        bar_densities = sif_density(self.df, "predicted_barrier_failures", explode=True)

        return {
            "total_reports": total,
            "total_predicted_sif_reports": sif_total,
            "overall_sif_density": round(density, 4),
            "avg_sif_probability": round(avg_prob, 4),
            "highest_volume_site": sites_by_volume[0]["name"] if sites_by_volume else None,
            "highest_sif_count_site": sites_by_sif_count[0]["name"] if sites_by_sif_count else None,
            "highest_sif_density_site_with_adequate_sample": (
                sites_by_density_adequate[0]["name"] if sites_by_density_adequate else None
            ),
            "highest_sif_density_site_unrestricted": (
                sites_by_density_all[0]["name"] if sites_by_density_all else None
            ),
            "most_frequent_activity": act_densities[0]["name"] if act_densities else None,
            "most_frequent_lsr": lsr_densities[0]["name"] if lsr_densities else None,
            "most_frequent_hazard": haz_densities[0]["name"] if haz_densities else None,
            "most_frequent_barrier_failure": bar_densities[0]["name"] if bar_densities else None,
        }
