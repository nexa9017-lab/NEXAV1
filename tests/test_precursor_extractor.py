import pytest
from src.extraction.precursor_extractor import SafetyPrecursorExtractor

@pytest.fixture(scope="module")
def extractor():
    return SafetyPrecursorExtractor()

def _has_value(hazards_list, val):
    return any(h['value'].lower() == val.lower() for h in hazards_list)

def test_negation_not_depressurized(extractor):
    text = "The line was not depressurized before maintenance."
    result = extractor.extract(text)
    assert _has_value(result['hazards'], "Pressurized equipment") or _has_value(result['hazards'], "Stored energy")

def test_negation_not_pressurized(extractor):
    text = "The line was not pressurized before maintenance."
    result = extractor.extract(text)
    assert not _has_value(result['hazards'], "Pressurized equipment")

def test_negation_no_h2s(extractor):
    text = "Gas test confirmed no h2s detected in the vessel."
    result = extractor.extract(text)
    assert not _has_value(result['hazards'], "H2S exposure")

def test_h2s_detected(extractor):
    text = "Gas test confirmed h2s detected in the vessel."
    result = extractor.extract(text)
    assert _has_value(result['hazards'], "H2S exposure")

def test_no_suspended_load(extractor):
    text = "Worker confirmed no suspended load was overhead."
    result = extractor.extract(text)
    assert not _has_value(result['hazards'], "Suspended load")

def test_worker_stood_under_load(extractor):
    text = "The worker stood under the suspended load."
    result = extractor.extract(text)
    assert _has_value(result['unsafe_actions'], "Entering danger zone")
    assert _has_value(result['hazards'], "Suspended load")

def test_loss_of_containment(extractor):
    text = "fluid started weeping from the flange."
    result = extractor.extract(text)
    assert _has_value(result['hazards'], "Loss of containment")

def test_hydrocarbon_release(extractor):
    text = "oil started weeping from the flange."
    result = extractor.extract(text)
    assert _has_value(result['hazards'], "Hydrocarbon release") or _has_value(result['hazards'], "Loss of containment")

def test_fire_explosion_guardrail(extractor):
    text = "worker was out of the line of fire."
    result = extractor.extract(text)
    assert not _has_value(result['hazards'], "Fire/explosion")

def test_equipment_extraction(extractor):
    text = "Pump P-204 was being repaired."
    result = extractor.extract(text)
    assert any(e['value'] == "Pump P-204" and e['type'] == "Pump" for e in result['equipment'])

# --- Phase 4.6 Regression & Contrast Pairs ---

def test_regression_fire_false_positive(extractor):
    text = "Hydraulic torque wrench pressure hose snapped. Worker was out of the line of fire so nobody was struck."
    result = extractor.extract(text)
    assert _has_value(result['hazards'], "Pressurized equipment")
    assert _has_value(result['hazards'], "Loss of containment") or _has_value(result['hazards'], "Hydrocarbon release")
    assert not _has_value(result['hazards'], "Fire/explosion")
    assert not _has_value(result['hazards'], "Line-of-fire exposure")

def test_contrast_pressure_1(extractor):
    text = "The line was pressurized."
    res = extractor.extract(text)
    assert _has_value(res['hazards'], "Pressurized equipment")

def test_contrast_pressure_2(extractor):
    text = "The line was not pressurized."
    res = extractor.extract(text)
    assert not _has_value(res['hazards'], "Pressurized equipment")

def test_contrast_suspended_load_1(extractor):
    text = "The worker stood beneath the suspended load."
    res = extractor.extract(text)
    assert _has_value(res['hazards'], "Suspended load")
    assert _has_value(res['hazards'], "Line-of-fire exposure") or _has_value(res['unsafe_actions'], "Entering danger zone")

def test_contrast_suspended_load_2(extractor):
    text = "The worker stayed clear of the suspended load."
    res = extractor.extract(text)
    # Load may exist, but no exposure
    assert not _has_value(res['hazards'], "Line-of-fire exposure")

def test_contrast_hydrocarbon_1(extractor):
    text = "Hydrocarbon leaked from the flange."
    res = extractor.extract(text)
    assert _has_value(res['hazards'], "Hydrocarbon release")

def test_contrast_hydrocarbon_2(extractor):
    text = "No hydrocarbon leak was found."
    res = extractor.extract(text)
    assert not _has_value(res['hazards'], "Hydrocarbon release")

def test_contrast_energy_1(extractor):
    text = "The line remained live during maintenance."
    res = extractor.extract(text)
    assert _has_value(res['hazards'], "Stored energy")

def test_contrast_energy_2(extractor):
    text = "verified zero energy before maintenance."
    res = extractor.extract(text)
    assert not _has_value(res['hazards'], "Stored energy")

def test_contrast_h2s_1(extractor):
    text = "H2S detected near the sump."
    res = extractor.extract(text)
    assert _has_value(res['hazards'], "H2S exposure")

def test_contrast_h2s_2(extractor):
    text = "No H2S detected near the sump."
    res = extractor.extract(text)
    assert not _has_value(res['hazards'], "H2S exposure")
