import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_PROCESSED_FILE, VAL_PROCESSED_FILE, TEST_PROCESSED_FILE, CHALLENGE_FILE,
    TFIDF_VECTORIZER_PATH, LSR_MODEL_PATH, LSR_METRICS_PATH, LSR_ENCODER_PATH,
    LSR_LR_C, LSR_LR_MAX_ITER, LSR_LR_CLASS_WEIGHT, LSR_DEFAULT_GLOBAL_THRESHOLD
)
from src.preprocessing.feature_engineering import TfidfFeatureExtractor
from src.models.lsr_classifier import LSRClassifier

def train_lsr():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
    val_df = pd.read_csv(VAL_PROCESSED_FILE)
    test_df = pd.read_csv(TEST_PROCESSED_FILE)
    challenge_df = pd.read_csv(CHALLENGE_FILE)
    
    print("Loading feature extractor...")
    extractor = TfidfFeatureExtractor().load(TFIDF_VECTORIZER_PATH)
    
    X_train = extractor.transform(train_df['description_clean'])
    y_train = train_df['lsr_tags'].values
    
    X_val = extractor.transform(val_df['description_clean'])
    y_val = val_df['lsr_tags'].values
    
    X_test = extractor.transform(test_df['description_clean'])
    y_test = test_df['lsr_tags'].values
    
    X_chal = extractor.transform(challenge_df['description_clean'])
    y_chal = challenge_df['lsr_tags'].values
    
    print("Training LSR Classifier...")
    model = LSRClassifier(C=LSR_LR_C, max_iter=LSR_LR_MAX_ITER, class_weight=LSR_LR_CLASS_WEIGHT)
    model.fit(X_train, y_train)
    
    print(f"Using Global Threshold: {LSR_DEFAULT_GLOBAL_THRESHOLD}")
    model.global_threshold = LSR_DEFAULT_GLOBAL_THRESHOLD
    
    print("\nEvaluating on Test Set...")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Subset Accuracy: {test_metrics['exact_match_ratio']:.4f}")
    
    print("\nPer-Rule Metrics (Test):")
    for cls, cm in test_metrics['per_class'].items():
        print(f"  {cls:30s} | F1: {cm['f1']:.4f} | P: {cm['precision']:.4f} | R: {cm['recall']:.4f} | Support: {cm['support']}")
        
    print("\nEvaluating on Challenge Set...")
    chal_metrics = model.evaluate(X_chal, y_chal)
    print(f"  Challenge Micro F1: {chal_metrics['micro_f1']:.4f}")
    print(f"  Challenge Macro F1: {chal_metrics['macro_f1']:.4f}")
    
    print("\nSaving Model and Metrics...")
    model.save(LSR_MODEL_PATH)
    
    all_metrics = {
        "config": {
            "C": LSR_LR_C,
            "class_weight": LSR_LR_CLASS_WEIGHT,
            "global_threshold": LSR_DEFAULT_GLOBAL_THRESHOLD
        },
        "test": test_metrics,
        "challenge": chal_metrics
    }
    with open(LSR_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    print("\nExample Prediction (Challenge):")
    example_text = challenge_df['description'].iloc[0]
    print(f"Text: {example_text}")
    print(f"True Labels: {y_chal[0]}")
    preds = model.predict_with_scores(X_chal[0])
    print(f"Prediction: {json.dumps(preds[0], indent=2)}")

if __name__ == "__main__":
    train_lsr()
