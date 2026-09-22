"""
Temporal analytics for safety reports.

Computes monthly/quarterly/yearly aggregations, trend calculations,
and period-over-period comparisons.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.analytics.aggregation import safe_parse_json_list


def _safe_pct_change(current: float, previous: float) -> Optional[float]:
    """Calculate percentage change, handling division by zero."""
    if previous == 0:
        return None  # Cannot compute percentage change from zero
    return round((current - previous) / previous, 4)


class TemporalAnalyzer:
    """Time-series analytics engine for safety report data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_dates()

    def _prepare_dates(self):
        """Parse dates and create period columns."""
        if "date" in self.df.columns:
            self.df["_date"] = pd.to_datetime(self.df["date"], errors="coerce")
            self.df["_year"] = self.df["_date"].dt.year
            self.df["_month"] = self.df["_date"].dt.to_period("M")
            self.df["_quarter"] = self.df["_date"].dt.to_period("Q")
            self._has_dates = self.df["_date"].notna().any()
        else:
            self._has_dates = False

    def monthly_summary(self) -> List[Dict[str, Any]]:
        """Monthly report volume, SIF count, SIF density, avg SIF probability."""
        if not self._has_dates:
            return []

        valid = self.df[self.df["_date"].notna()].copy()
        grouped = valid.groupby("_month").agg(
            report_volume=("report_id", "count"),
            sif_count=("sif_prediction", "sum"),
            avg_sif_probability=("sif_probability", "mean"),
        ).reset_index()

        grouped = grouped.sort_values("_month")

        results = []
        for _, row in grouped.iterrows():
            total = int(row["report_volume"])
            sif_c = int(row["sif_count"])
            density = sif_c / total if total > 0 else 0.0

            results.append({
                "period": str(row["_month"]),
                "period_type": "month",
                "report_volume": total,
                "sif_count": sif_c,
                "sif_density": round(density, 4),
                "avg_sif_probability": round(float(row["avg_sif_probability"]), 4),
            })

        return results

    def quarterly_summary(self) -> List[Dict[str, Any]]:
        """Quarterly aggregation."""
        if not self._has_dates:
            return []

        valid = self.df[self.df["_date"].notna()].copy()
        grouped = valid.groupby("_quarter").agg(
            report_volume=("report_id", "count"),
            sif_count=("sif_prediction", "sum"),
            avg_sif_probability=("sif_probability", "mean"),
        ).reset_index()

        grouped = grouped.sort_values("_quarter")

        results = []
        for _, row in grouped.iterrows():
            total = int(row["report_volume"])
            sif_c = int(row["sif_count"])
            density = sif_c / total if total > 0 else 0.0

            results.append({
                "period": str(row["_quarter"]),
                "period_type": "quarter",
                "report_volume": total,
                "sif_count": sif_c,
                "sif_density": round(density, 4),
                "avg_sif_probability": round(float(row["avg_sif_probability"]), 4),
            })

        return results

    def yearly_summary(self) -> List[Dict[str, Any]]:
        """Yearly aggregation."""
        if not self._has_dates:
            return []

        valid = self.df[self.df["_date"].notna()].copy()
        grouped = valid.groupby("_year").agg(
            report_volume=("report_id", "count"),
            sif_count=("sif_prediction", "sum"),
            avg_sif_probability=("sif_probability", "mean"),
        ).reset_index()

        grouped = grouped.sort_values("_year")

        results = []
        for _, row in grouped.iterrows():
            total = int(row["report_volume"])
            sif_c = int(row["sif_count"])
            density = sif_c / total if total > 0 else 0.0

            results.append({
                "period": str(int(row["_year"])),
                "period_type": "year",
                "report_volume": total,
                "sif_count": sif_c,
                "sif_density": round(density, 4),
                "avg_sif_probability": round(float(row["avg_sif_probability"]), 4),
            })

        return results

    def dimension_monthly_frequency(self, col: str, explode: bool = False) -> Dict[str, List[Dict]]:
        """Monthly frequency for a dimension (LSR, hazard, barrier, etc.)."""
        if not self._has_dates or col not in self.df.columns:
            return {}

        valid = self.df[self.df["_date"].notna()].copy()

        if explode:
            valid[col] = valid[col].apply(safe_parse_json_list)
            valid = valid.explode(col)
            valid = valid[valid[col].notna() & (valid[col] != "")]
            valid = valid.drop_duplicates(subset=["report_id", col, "_month"])

        result = {}
        for dim_val, group in valid.groupby(col):
            monthly = group.groupby("_month").agg(
                count=("report_id", "count"),
                sif_count=("sif_prediction", "sum"),
            ).reset_index().sort_values("_month")

            periods = []
            for _, row in monthly.iterrows():
                periods.append({
                    "period": str(row["_month"]),
                    "count": int(row["count"]),
                    "sif_count": int(row["sif_count"]),
                })
            result[str(dim_val)] = periods

        return result

    def calculate_trends(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add trend indicators comparing each period to the previous one.
        Works on monthly, quarterly, or yearly summaries.
        """
        if len(summaries) < 2:
            return summaries

        enriched = []
        for i, current in enumerate(summaries):
            trend = {}
            if i > 0:
                prev = summaries[i - 1]
                for metric in ["report_volume", "sif_count", "sif_density", "avg_sif_probability"]:
                    curr_val = current.get(metric, 0)
                    prev_val = prev.get(metric, 0)
                    trend[metric] = {
                        "previous_value": prev_val,
                        "current_value": curr_val,
                        "absolute_change": round(curr_val - prev_val, 4),
                        "percentage_change": _safe_pct_change(curr_val, prev_val),
                    }
            enriched.append({**current, "trends": trend if trend else None})

        return enriched

    def full_temporal_analytics(self) -> Dict[str, Any]:
        """Run all temporal analytics."""
        monthly = self.monthly_summary()
        quarterly = self.quarterly_summary()
        yearly = self.yearly_summary()

        warnings = []
        if not self._has_dates:
            warnings.append("MISSING_DATE_DATA")

        return {
            "monthly": self.calculate_trends(monthly),
            "quarterly": self.calculate_trends(quarterly),
            "yearly": self.calculate_trends(yearly),
            "monthly_lsr_frequency": self.dimension_monthly_frequency(
                "predicted_lsr", explode=True
            ),
            "monthly_hazard_frequency": self.dimension_monthly_frequency(
                "predicted_hazards", explode=True
            ),
            "monthly_barrier_frequency": self.dimension_monthly_frequency(
                "predicted_barrier_failures", explode=True
            ),
            "data_quality_warnings": warnings,
        }
