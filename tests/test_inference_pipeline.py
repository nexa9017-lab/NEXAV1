"""
Tests for the Safety Inference Pipeline.
"""

import pytest
from src.pipeline.inference import SafetyInferencePipeline

@pytest.fixture(scope="module")
def pipeline():
    # Because loading takes time, we load it once for all pipeline tests
    p = SafetyInferencePipeline()
    p.load_models()
    return p

def test_pipeline_loading(pipeline):
    assert pipeline.loaded
    assert pipeline.embedder is not None
    assert pipeline.sif_clf is not None
    assert pipeline.lsr_clf is not None
    assert pipeline.extractor is not None

def test_predict_single_valid(pipeline):
    report = {"description": "Worker was grinding near a confined space without a valid permit."}
    result = pipeline.predict_single(report)
    
    assert result["normalized_text"] == report["description"]
    assert "sif" in result
    assert "probability" in result["sif"]
    assert "life_saving_rules" in result
    assert len(result["life_saving_rules"]) > 0
    assert "primary_lsr" in result
    assert "activities" in result
    assert "inference_metadata" in result
    
    # Check probability bounds
    assert 0.0 <= result["sif"]["probability"] <= 1.0

def test_predict_single_empty(pipeline):
    with pytest.raises(ValueError, match="Empty or whitespace-only description"):
        pipeline.predict_single({"description": "   "})
        
def test_predict_batch_order_preservation(pipeline):
    reports = [
        {"description": "First report text."},
        {"description": "Second report text with empty next."},
        {"description": ""}, # Invalid
        {"description": "Fourth report text."}
    ]
    
    results = pipeline.predict_batch(reports)
    
    assert len(results) == 4
    assert results[0]["normalized_text"] == "First report text."
    assert results[1]["normalized_text"] == "Second report text with empty next."
    assert "error" in results[2]
    assert results[3]["normalized_text"] == "Fourth report text."
    
def test_deterministic_output(pipeline):
    # Ensure repeated calls yield the same results
    desc = "Technician opened flange under pressure."
    res1 = pipeline.predict_single({"description": desc})
    res2 = pipeline.predict_single({"description": desc})
    
    assert res1["sif"]["probability"] == res2["sif"]["probability"]
    assert res1["primary_lsr"] == res2["primary_lsr"]
    assert res1["activities"] == res2["activities"]
    assert res1["precursor_tags"] == res2["precursor_tags"]

def test_metadata_format(pipeline):
    res = pipeline.predict_single({"description": "Test"})
    meta = res["inference_metadata"]
    
    assert meta["feature_mode"] == "embedding"
    assert "sif_threshold" in meta["thresholds"]
    assert "lsr_threshold" in meta["thresholds"]
    assert "prototype_notice" in meta
