import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.sif_classifier import SIFClassifier
from src.preprocessing.feature_engineering import FORBIDDEN_FEATURE_COLUMNS

def test_sif_classifier(tmp_path):
    X_train = np.random.rand(10, 5)
    y_train = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    
    X_test = np.random.rand(3, 5)
    y_test = np.array([0, 1, 0])
    
    feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]
    
    model = SIFClassifier()
    model.fit(X_train, y_train, feature_names=feature_names)
    
    # 1. Model fitting & properties
    assert model.is_fitted
    
    # 2. Probability shape and range
    probs = model.predict_proba(X_test)
    assert probs.shape == (3,)
    assert ((probs >= 0) & (probs <= 1)).all()
    
    # 3. Prediction shape and threshold behavior
    preds_05 = model.predict(X_test, threshold=0.5)
    preds_00 = model.predict(X_test, threshold=0.0)
    assert preds_05.shape == (3,)
    assert (preds_00 == 1).all()
    
    # 4. Evaluation output
    metrics = model.evaluate(X_test, y_test)
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "false_negative_rate" in metrics
    
    # 5. Save/Load
    save_path = tmp_path / "model.joblib"
    model.save(save_path)
    
    new_model = SIFClassifier().load(save_path)
    assert new_model.is_fitted
    assert new_model.threshold == model.threshold
    assert (new_model.predict_proba(X_test) == probs).all()
    
    # 6. Explanation Output
    explanation = model.explain_prediction(X_test[0], top_k=2)
    assert len(explanation) <= 2
    
    # 7. Forbidden Feature Protection
    # Tested dynamically in train_sif.py, but we assert the const is present.
    assert "sif_label" in FORBIDDEN_FEATURE_COLUMNS

def test_sif_embedding_mode(tmp_path):
    X_train = np.random.rand(10, 384)
    y_train = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    
    model = SIFClassifier(feature_mode="embedding")
    model.fit(X_train, y_train)
    
    assert model.feature_mode == "embedding"
    
    save_path = tmp_path / "model_emb.joblib"
    model.save(save_path)
    
    new_model = SIFClassifier().load(save_path)
    assert new_model.feature_mode == "embedding"
