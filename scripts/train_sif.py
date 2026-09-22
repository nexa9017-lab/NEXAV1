import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_PROCESSED_FILE, VAL_PROCESSED_FILE, TEST_PROCESSED_FILE, CHALLENGE_FILE,
    TFIDF_VECTORIZER_PATH, SIF_MODEL_PATH, SIF_METRICS_PATH, FIGURES_DIR,
    SIF_LR_C, SIF_LR_MAX_ITER, SIF_LR_CLASS_WEIGHT, SIF_MIN_RECALL, SIF_THRESHOLD_SEARCH_RANGE
)
from src.preprocessing.feature_engineering import TfidfFeatureExtractor, FORBIDDEN_FEATURE_COLUMNS
from src.models.sif_classifier import SIFClassifier

def train_sif():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
    val_df = pd.read_csv(VAL_PROCESSED_FILE)
    test_df = pd.read_csv(TEST_PROCESSED_FILE)
    challenge_df = pd.read_csv(CHALLENGE_FILE)
    
    print("Loading feature extractor...")
    extractor = TfidfFeatureExtractor().load(TFIDF_VECTORIZER_PATH)
    vocab = extractor.vectorizer.get_feature_names_out()
    
    # Feature extraction (ensuring no leakage)
    for col in FORBIDDEN_FEATURE_COLUMNS:
        assert col not in train_df[['description_clean']].columns, f"Leakage detected: {col}"
        
    X_train = extractor.transform(train_df['description_clean'])
    y_train = train_df['sif_label'].values
    
    X_val = extractor.transform(val_df['description_clean'])
    y_val = val_df['sif_label'].values
    
    X_test = extractor.transform(test_df['description_clean'])
    y_test = test_df['sif_label'].values
    
    X_chal = extractor.transform(challenge_df['description_clean'])
    y_chal = challenge_df['sif_label'].values
    
    print("Training SIF Classifier...")
    model = SIFClassifier(C=SIF_LR_C, max_iter=SIF_LR_MAX_ITER, class_weight=SIF_LR_CLASS_WEIGHT)
    model.fit(X_train, y_train, feature_names=vocab)
    
    print("Selecting Threshold on Validation Data...")
    best_thresh = 0.5
    best_f1 = -1
    for t in SIF_THRESHOLD_SEARCH_RANGE:
        metrics = model.evaluate(X_val, y_val, threshold=t)
        if metrics['recall'] >= SIF_MIN_RECALL and metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_thresh = t
    
    model.threshold = best_thresh
    print(f"Selected Threshold: {best_thresh:.2f} (Val F1: {best_f1:.4f})")
    
    print("\nEvaluating on Test Set...")
    test_metrics = model.evaluate(X_test, y_test)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
            
    print("\nEvaluating on Challenge Set...")
    chal_metrics = model.evaluate(X_chal, y_chal)
    print(f"  Challenge F1: {chal_metrics['f1']:.4f}")
    print(f"  Challenge Recall: {chal_metrics['recall']:.4f}")
    
    print("\nSaving Model and Metrics...")
    model.save(SIF_MODEL_PATH)
    
    all_metrics = {
        "config": {
            "C": SIF_LR_C,
            "class_weight": SIF_LR_CLASS_WEIGHT,
            "threshold": best_thresh
        },
        "test": test_metrics,
        "challenge": chal_metrics
    }
    with open(SIF_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    print("\nGenerating Figures...")
    # ROC Curve
    y_test_prob = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {test_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('SIF ROC Curve')
    plt.legend()
    plt.savefig(FIGURES_DIR / "sif_roc_curve.png")
    
    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC = {test_metrics['pr_auc']:.3f}")
    plt.title('SIF Precision-Recall Curve')
    plt.legend()
    plt.savefig(FIGURES_DIR / "sif_pr_curve.png")
    
    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-SIF", "SIF"])
    disp.plot()
    plt.title('SIF Confusion Matrix')
    plt.savefig(FIGURES_DIR / "sif_confusion_matrix.png")
    
    print("Figures saved.")
    
    print("\nError Analysis (False Negatives on Test):")
    # Find FNs
    fns = np.where((y_test == 1) & (y_pred == 0))[0]
    for idx in fns[:3]:
        print(f"\n--- FN Example (Index {idx}) ---")
        print(f"Text: {test_df['description'].iloc[idx]}")
        print(f"Probability: {y_test_prob[idx]:.4f}")
        try:
            explanation = model.explain_prediction(X_test[idx])
            print("Influential terms:")
            for term, score in explanation.items():
                print(f"  {term}: {score:.4f}")
        except Exception as e:
            print(f"Explanation failed: {e}")

if __name__ == "__main__":
    train_sif()
