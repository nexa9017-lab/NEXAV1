import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE, 
    TRAIN_PROCESSED_FILE, VAL_PROCESSED_FILE, TEST_PROCESSED_FILE,
    TFIDF_VECTORIZER_PATH, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE, TFIDF_LOWERCASE
)
from src.preprocessing.text_preprocessor import SafetyTextPreprocessor
from src.preprocessing.feature_engineering import TfidfFeatureExtractor, FORBIDDEN_FEATURE_COLUMNS

def preprocess_and_extract_features():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}\n")
    
    # Target Leakage Check
    print("Checking for target leakage in input texts...")
    for col in FORBIDDEN_FEATURE_COLUMNS:
        if col in train_df.columns:
            pass # these columns can be present in the DataFrame for labels, but shouldn't be used as text features.
    
    print("Preprocessing text...")
    preprocessor = SafetyTextPreprocessor()
    
    train_df['description_clean'] = preprocessor.transform_series(train_df['description'])
    val_df['description_clean'] = preprocessor.transform_series(val_df['description'])
    test_df['description_clean'] = preprocessor.transform_series(test_df['description'])
    
    print("Saving processed datasets...")
    train_df.to_csv(TRAIN_PROCESSED_FILE, index=False)
    val_df.to_csv(VAL_PROCESSED_FILE, index=False)
    test_df.to_csv(TEST_PROCESSED_FILE, index=False)
    
    print("Processed datasets generated.")
    
    print("\n--- Example (Train) ---")
    print(f"Raw: {train_df['description'].iloc[0]}")
    print(f"Clean: {train_df['description_clean'].iloc[0]}\n")
    
    print("Fitting TF-IDF on Training Data ONLY...")
    tfidf = TfidfFeatureExtractor(
        max_features=TFIDF_MAX_FEATURES, 
        min_df=TFIDF_MIN_DF, 
        max_df=TFIDF_MAX_DF, 
        ngram_range=TFIDF_NGRAM_RANGE, 
        lowercase=TFIDF_LOWERCASE
    )
    
    # Prevent target leakage: Use ONLY `description_clean` for features.
    X_train = tfidf.fit_transform(train_df['description_clean'])
    
    print("Transforming Validation and Test Data...")
    X_val = tfidf.transform(val_df['description_clean'])
    X_test = tfidf.transform(test_df['description_clean'])
    
    tfidf.save(TFIDF_VECTORIZER_PATH)
    
    print("\nTF-IDF:")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Vocabulary Size: {len(tfidf.vectorizer.vocabulary_)}")

if __name__ == "__main__":
    preprocess_and_extract_features()
