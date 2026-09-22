import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.lsr_classifier import LSRClassifier
from src.preprocessing.safety_taxonomy import LIFE_SAVING_RULES

def test_lsr_classifier(tmp_path):
    # 1. Multi-label Parsing
    assert LSRClassifier.safely_parse_lsr_tags("Energy Isolation|Line of Fire") == ["Energy Isolation", "Line of Fire"]
    assert LSRClassifier.safely_parse_lsr_tags("['Hot Work', 'Invalid Rule']") == ["Hot Work"]
    assert LSRClassifier.safely_parse_lsr_tags("None") == []
    
    X_train = np.random.rand(5, 10)
    y_strings = ["Energy Isolation", "Line of Fire|Hot Work", "Driving", "None", "Confined Space"]
    
    X_test = np.random.rand(2, 10)
    
    model = LSRClassifier()
    model.fit(X_train, y_strings)
    
    # 2. Model fitting
    assert model.is_fitted
    assert len(model.mlb.classes_) == len(LIFE_SAVING_RULES)
    
    # 3. Probability matrix shape
    probs = model.predict_proba(X_test)
    assert probs.shape == (2, len(LIFE_SAVING_RULES))
    
    # 4. Primary rule selection
    results = model.predict_with_scores(X_test, threshold=0.01) # Force hits
    assert len(results) == 2
    assert "primary_rule" in results[0]
    
    # 5. Fallback behaviour
    results_empty = model.predict_with_scores(X_test, threshold=0.999) 
    # Usually 0.999 will yield no rules passing, so fallback kicks in
    assert results_empty[0]["primary_rule"] != ""
    assert len(results_empty[0]["rules"]) > 0
    
    # 6. Save/Load
    save_path = tmp_path / "lsr.joblib"
    model.save(save_path)
    
    new_model = LSRClassifier().load(save_path)
    assert new_model.is_fitted
    assert new_model.global_threshold == model.global_threshold
    assert (new_model.predict_proba(X_test) == probs).all()

def test_lsr_embedding_mode(tmp_path):
    X_train = np.random.rand(5, 384)
    y_strings = ["Energy Isolation", "Line of Fire", "Driving", "None", "Confined Space"]
    
    model = LSRClassifier(feature_mode="embedding")
    model.fit(X_train, y_strings)
    
    assert model.feature_mode == "embedding"
    
    save_path = tmp_path / "lsr_emb.joblib"
    model.save(save_path)
    
    new_model = LSRClassifier().load(save_path)
    assert new_model.feature_mode == "embedding"
