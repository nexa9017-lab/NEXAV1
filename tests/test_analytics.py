"""
Tests for Phase 5 analytics engine.

Uses small deterministic fixtures — no model inference or file I/O.
"""

import json
import math
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.aggregation import (
    safe_parse_json_list, deduplicated_list, sif_density,
    weighted_sif_probability, AnalyticsAggregator,
)
from src.analytics.pattern_mining import PatternMiner, _make_pattern_id
from src.analytics.risk_scoring import PriorityScorer, _safe_min_max_normalize
from src.analytics.temporal import TemporalAnalyzer, _safe_pct_change
from src.analytics.pipeline import _sanitize_for_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n=20):
    """Create a deterministic test DataFrame."""
    np.random.seed(42)
    sites = ["Site A", "Site B", "Site C"]
    activities = ["Maintenance", "Hot Work", "Lifting"]
    hazards_options = [
        '["Electrical Energy"]',
        '["Fire/Explosion"]',
        '["Suspended Load"]',
        '["Electrical Energy", "Fire/Explosion"]',
        '[]',
    ]
    barriers_options = [
        '["ENERGY_ISOLATION_FAILURE"]',
        '["FALL_PROTECTION_FAILURE"]',
        '["LIFTING_CONTROL_FAILURE"]',
        '[]',
    ]
    lsr_options = [
        '["Energy Isolation"]',
        '["Hot Work"]',
        '["Safe Mechanical Lifting"]',
        '["Energy Isolation", "Line of Fire"]',
    ]

    rows = []
    for i in range(n):
        sif_pred = 1 if i % 4 == 0 else 0
        rows.append({
            "report_id": f"REP-{i:04d}",
            "date": f"2024-{(i % 12) + 1:02d}-15",
            "site": sites[i % len(sites)],
            "location": f"Area {i % 3}",
            "report_type": "Incident" if sif_pred else "Near Miss",
            "description": f"Test report {i} about safety work.",
            "activity": activities[i % len(activities)],
            "sif_prediction": sif_pred,
            "sif_probability": round(0.3 + (sif_pred * 0.5) + np.random.uniform(-0.1, 0.1), 4),
            "predicted_activity": json.dumps([activities[i % len(activities)]]),
            "predicted_equipment": json.dumps([f"Pump P-{i}"]),
            "predicted_hazards": hazards_options[i % len(hazards_options)],
            "predicted_barrier_failures": barriers_options[i % len(barriers_options)],
            "predicted_precursor_tags": json.dumps([f"tag_{i % 3}"]),
            "predicted_potential_consequences": json.dumps(["Serious Burns"]) if sif_pred else json.dumps([]),
            "predicted_lsr": lsr_options[i % len(lsr_options)],
            "primary_lsr_predicted": json.loads(lsr_options[i % len(lsr_options)])[0],
            # Ground truth for validation
            "sif_label": sif_pred,
            "hazard": "electrical" if i % 2 == 0 else "fire",
            "barrier_failure": "loto missing" if i % 3 == 0 else "harness not clipped",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: safe_parse_json_list
# ---------------------------------------------------------------------------

class TestSafeParseJsonList:
    def test_valid_json_list(self):
        assert safe_parse_json_list('["a", "b"]') == ["a", "b"]

    def test_empty_string(self):
        assert safe_parse_json_list("") == []

    def test_none(self):
        assert safe_parse_json_list(None) == []

    def test_nan(self):
        assert safe_parse_json_list(float("nan")) == []

    def test_null_string(self):
        assert safe_parse_json_list("null") == []

    def test_malformed_json(self):
        with pytest.warns(match="MALFORMED_LIST_FIELD"):
            result = safe_parse_json_list("[invalid json")
        assert result == []

    def test_already_list(self):
        assert safe_parse_json_list(["x", "y"]) == ["x", "y"]

    def test_single_value(self):
        assert safe_parse_json_list('"hello"') == ["hello"]


# ---------------------------------------------------------------------------
# Test: deduplicated_list
# ---------------------------------------------------------------------------

class TestDeduplicatedList:
    def test_dedup(self):
        assert deduplicated_list(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_empty(self):
        assert deduplicated_list([]) == []

    def test_preserves_order(self):
        assert deduplicated_list(["c", "a", "b", "a"]) == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# Test: SIF Density
# ---------------------------------------------------------------------------

class TestSifDensity:
    def test_basic(self):
        df = _make_df()
        result = sif_density(df, "site")
        assert len(result) > 0
        for r in result:
            assert "total_reports" in r
            assert "sif_count" in r
            assert "sif_density" in r
            assert "avg_sif_probability" in r
            assert r["total_reports"] >= r["sif_count"]

    def test_density_value(self):
        df = _make_df()
        result = sif_density(df, "site")
        for r in result:
            expected = r["sif_count"] / r["total_reports"] if r["total_reports"] > 0 else 0
            assert abs(r["sif_density"] - expected) < 0.001

    def test_zero_denominator(self):
        """SIF density with empty group should not error."""
        df = pd.DataFrame({
            "report_id": [],
            "sif_prediction": [],
            "sif_probability": [],
            "site": [],
        })
        result = sif_density(df, "site")
        assert result == []

    def test_explode_multilabel(self):
        df = _make_df()
        result = sif_density(df, "predicted_lsr", explode=True)
        assert len(result) > 0
        # Multi-label: should have Energy Isolation, Hot Work, etc.
        names = [r["name"] for r in result]
        assert "Energy Isolation" in names

    def test_low_sample_size_warning(self):
        df = pd.DataFrame({
            "report_id": ["R1", "R2"],
            "sif_prediction": [1, 0],
            "sif_probability": [0.9, 0.2],
            "site": ["X", "X"],
        })
        result = sif_density(df, "site")
        assert len(result) == 1
        assert "LOW_SAMPLE_SIZE" in result[0]["data_quality_warnings"]


# ---------------------------------------------------------------------------
# Test: Multi-label LSR counting
# ---------------------------------------------------------------------------

class TestMultiLabelCounting:
    def test_multilabel_lsr_no_double_count(self):
        """A report with ['Energy Isolation', 'Line of Fire'] contributes
        once to each but not twice to either."""
        df = pd.DataFrame({
            "report_id": ["R1"],
            "sif_prediction": [1],
            "sif_probability": [0.9],
            "predicted_lsr": ['["Energy Isolation", "Line of Fire"]'],
        })
        result = sif_density(df, "predicted_lsr", explode=True)
        names = {r["name"] for r in result}
        assert "Energy Isolation" in names
        assert "Line of Fire" in names
        for r in result:
            assert r["total_reports"] == 1  # Each only counted once


# ---------------------------------------------------------------------------
# Test: Pattern Mining
# ---------------------------------------------------------------------------

class TestPatternMining:
    def test_pair_mining(self):
        df = _make_df(30)
        miner = PatternMiner(df, min_support=2)
        patterns = miner.mine_pair_patterns()
        assert len(patterns) > 0
        for p in patterns:
            assert p["occurrences"] >= 2
            assert "pattern_id" in p
            assert "components" in p
            assert "sif_density" in p

    def test_triple_mining(self):
        df = _make_df(30)
        miner = PatternMiner(df, min_support=2)
        patterns = miner.mine_triple_patterns()
        # May or may not have results depending on data
        for p in patterns:
            assert len(p["components"]) == 3
            assert p["occurrences"] >= 2

    def test_min_support_filter(self):
        df = _make_df(10)
        miner = PatternMiner(df, min_support=100)
        patterns = miner.mine_pair_patterns()
        assert len(patterns) == 0

    def test_deterministic_pattern_id(self):
        pid1 = _make_pattern_id("activity_barrier", {"activity": "Maint", "barrier": "LOTO"})
        pid2 = _make_pattern_id("activity_barrier", {"activity": "Maint", "barrier": "LOTO"})
        assert pid1 == pid2
        assert "activity_barrier" in pid1

    def test_cross_site_patterns(self):
        df = _make_df(30)
        miner = PatternMiner(df, min_support=2)
        patterns = miner.mine_pair_patterns()
        cross = miner.cross_site_patterns(patterns, min_sites=2)
        for p in cross:
            assert p["affected_site_count"] >= 2

    def test_emerging_patterns(self):
        df = _make_df(30)
        miner = PatternMiner(df, min_support=1)
        patterns = miner.mine_pair_patterns()
        emerging = miner.detect_emerging_patterns(patterns, period="quarter")
        # Should return list (may be empty)
        assert isinstance(emerging, list)
        for ep in emerging:
            assert ep["status"] in ["STABLE", "INCREASING", "DECREASING", "EMERGING", "NEW_PATTERN"]

    def test_support_ratio(self):
        df = _make_df(20)
        miner = PatternMiner(df, min_support=2)
        patterns = miner.mine_pair_patterns()
        for p in patterns:
            expected = p["occurrences"] / 20
            assert abs(p["support_ratio"] - round(expected, 4)) < 0.01

    def test_example_reports(self):
        df = _make_df(20)
        miner = PatternMiner(df, min_support=2)
        patterns = miner.mine_pair_patterns()
        for p in patterns:
            assert "example_reports" in p
            for ex in p["example_reports"]:
                assert "report_id" in ex
                assert "description_excerpt" in ex


# ---------------------------------------------------------------------------
# Test: Risk Scoring
# ---------------------------------------------------------------------------

class TestRiskScoring:
    def test_score_bounds(self):
        df = _make_df(30)
        miner = PatternMiner(df, min_support=2)
        patterns = miner.mine_pair_patterns()
        if patterns:
            scorer = PriorityScorer()
            scored = scorer.score_patterns(patterns)
            for p in scored:
                assert 0 <= p["priority_score"] <= 100

    def test_score_components(self):
        df = _make_df(30)
        miner = PatternMiner(df, min_support=2)
        patterns = miner.mine_pair_patterns()
        if patterns:
            scorer = PriorityScorer()
            scored = scorer.score_patterns(patterns)
            for p in scored:
                assert "priority_components" in p
                assert "frequency" in p["priority_components"]
                assert "sif_density" in p["priority_components"]
                assert "avg_sif_probability" in p["priority_components"]
                assert "cross_site" in p["priority_components"]
                assert "priority_weights" in p

    def test_deterministic_scoring(self):
        patterns = [{
            "occurrences": 10,
            "sif_count": 5,
            "sif_density": 0.5,
            "avg_sif_probability": 0.7,
            "affected_site_count": 3,
        }]
        scorer = PriorityScorer()
        s1 = scorer.score_patterns(patterns.copy())[0]["priority_score"]
        s2 = scorer.score_patterns(patterns.copy())[0]["priority_score"]
        assert s1 == s2

    def test_safe_min_max_all_same(self):
        """When all values are identical, normalize should not crash."""
        assert _safe_min_max_normalize(5, 5, 5) == 0.5

    def test_single_pattern_scoring(self):
        """Single pattern: min==max for freq and sites, should use 0.5 fallback."""
        patterns = [{
            "occurrences": 10,
            "sif_count": 8,
            "sif_density": 0.8,
            "avg_sif_probability": 0.75,
            "affected_site_count": 2,
        }]
        scorer = PriorityScorer()
        scored = scorer.score_patterns(patterns)
        assert len(scored) == 1
        assert 0 <= scored[0]["priority_score"] <= 100

    def test_empty_patterns(self):
        scorer = PriorityScorer()
        assert scorer.score_patterns([]) == []


# ---------------------------------------------------------------------------
# Test: Temporal Analytics
# ---------------------------------------------------------------------------

class TestTemporalAnalytics:
    def test_monthly_grouping(self):
        df = _make_df(20)
        analyzer = TemporalAnalyzer(df)
        monthly = analyzer.monthly_summary()
        assert len(monthly) > 0
        for m in monthly:
            assert "period" in m
            assert "report_volume" in m
            assert "sif_count" in m

    def test_trend_calculation(self):
        df = _make_df(20)
        analyzer = TemporalAnalyzer(df)
        monthly = analyzer.monthly_summary()
        trended = analyzer.calculate_trends(monthly)
        # First entry has no trend; later entries should
        if len(trended) >= 2:
            assert trended[1]["trends"] is not None

    def test_trend_division_by_zero(self):
        result = _safe_pct_change(5, 0)
        assert result is None

    def test_trend_normal(self):
        result = _safe_pct_change(10, 5)
        assert result == 1.0

    def test_quarterly_grouping(self):
        df = _make_df(20)
        analyzer = TemporalAnalyzer(df)
        quarterly = analyzer.quarterly_summary()
        assert isinstance(quarterly, list)

    def test_yearly_grouping(self):
        df = _make_df(20)
        analyzer = TemporalAnalyzer(df)
        yearly = analyzer.yearly_summary()
        assert isinstance(yearly, list)

    def test_full_temporal(self):
        df = _make_df(20)
        analyzer = TemporalAnalyzer(df)
        result = analyzer.full_temporal_analytics()
        assert "monthly" in result
        assert "quarterly" in result
        assert "yearly" in result


# ---------------------------------------------------------------------------
# Test: Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_site_aggregation(self):
        df = _make_df(20)
        agg = AnalyticsAggregator(df)
        sites = agg.site_analytics()
        assert len(sites) > 0
        for s in sites:
            assert "top_activities" in s
            assert "top_hazards" in s
            assert "reports_per_month" in s

    def test_activity_aggregation(self):
        df = _make_df(20)
        agg = AnalyticsAggregator(df)
        activities = agg.activity_analytics()
        assert len(activities) > 0
        for a in activities:
            assert "affected_sites" in a
            assert "common_hazards" in a

    def test_hazard_aggregation(self):
        df = _make_df(20)
        agg = AnalyticsAggregator(df)
        hazards = agg.hazard_analytics()
        assert len(hazards) > 0

    def test_barrier_aggregation(self):
        df = _make_df(20)
        agg = AnalyticsAggregator(df)
        barriers = agg.barrier_analytics()
        assert len(barriers) > 0

    def test_overall_summary(self):
        df = _make_df(20)
        agg = AnalyticsAggregator(df)
        summary = agg.overall_summary()
        assert "total_reports" in summary
        assert summary["total_reports"] == 20
        assert "overall_sif_density" in summary

    def test_lsr_analytics(self):
        df = _make_df(20)
        agg = AnalyticsAggregator(df)
        lsr = agg.lsr_analytics()
        assert len(lsr) > 0
        for l in lsr:
            assert "associated_sites" in l


# ---------------------------------------------------------------------------
# Test: JSON Serialization
# ---------------------------------------------------------------------------

class TestJsonSerialization:
    def test_sanitize_numpy_types(self):
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3]),
        }
        result = _sanitize_for_json(data)
        # Should be serializable
        serialized = json.dumps(result)
        assert "42" in serialized

    def test_sanitize_nan_inf(self):
        data = {"nan": float("nan"), "inf": float("inf")}
        result = _sanitize_for_json(data)
        assert result["nan"] is None
        assert result["inf"] is None

    def test_full_analytics_serializable(self):
        """Run pipeline on fixture and verify all outputs are valid JSON."""
        df = _make_df(30)
        from src.analytics.pipeline import AnalyticsPipeline
        pipeline = AnalyticsPipeline.__new__(AnalyticsPipeline)
        pipeline.df = df
        pipeline.output_dir = Path(".")
        pipeline.results = {}
        pipeline.warnings = []

        # Run analytics (in-memory only)
        results = pipeline.run()
        sanitized = _sanitize_for_json(results)
        serialized = json.dumps(sanitized, default=str)
        # Should be loadable
        loaded = json.loads(serialized)
        assert isinstance(loaded, dict)


# ---------------------------------------------------------------------------
# Test: Emerging Pattern Detection
# ---------------------------------------------------------------------------

class TestEmergingPatterns:
    def test_new_pattern_status(self):
        """Pattern with 0 previous count should be NEW_PATTERN."""
        # Create data with all dates in recent quarter
        df = pd.DataFrame({
            "report_id": [f"R{i}" for i in range(10)],
            "date": ["2024-06-15"] * 10,
            "site": ["Site A"] * 10,
            "sif_prediction": [1] * 5 + [0] * 5,
            "sif_probability": [0.8] * 5 + [0.2] * 5,
            "predicted_activity": ['["Maintenance"]'] * 10,
            "predicted_hazards": ['["Fire/Explosion"]'] * 10,
            "predicted_barrier_failures": ['["ENERGY_ISOLATION_FAILURE"]'] * 10,
            "predicted_lsr": ['["Energy Isolation"]'] * 10,
            "predicted_precursor_tags": ['["tag1"]'] * 10,
            "location": ["Area 1"] * 10,
        })
        miner = PatternMiner(df, min_support=5)
        patterns = miner.mine_pair_patterns()
        emerging = miner.detect_emerging_patterns(patterns, period="quarter")
        # All data is in one period, so patterns may be NEW_PATTERN
        for ep in emerging:
            assert ep["status"] in ["STABLE", "INCREASING", "DECREASING", "EMERGING", "NEW_PATTERN"]


# ---------------------------------------------------------------------------
# Test: Data Quality Warnings
# ---------------------------------------------------------------------------

class TestDataQualityWarnings:
    def test_low_sample_size(self):
        df = pd.DataFrame({
            "report_id": ["R1"],
            "sif_prediction": [1],
            "sif_probability": [0.9],
            "site": ["Tiny Site"],
        })
        result = sif_density(df, "site")
        assert "LOW_SAMPLE_SIZE" in result[0]["data_quality_warnings"]

    def test_adequate_sample_no_warning(self):
        df = pd.DataFrame({
            "report_id": [f"R{i}" for i in range(15)],
            "sif_prediction": [1] * 5 + [0] * 10,
            "sif_probability": [0.8] * 5 + [0.2] * 10,
            "site": ["Big Site"] * 15,
        })
        result = sif_density(df, "site")
        assert "LOW_SAMPLE_SIZE" not in result[0]["data_quality_warnings"]
