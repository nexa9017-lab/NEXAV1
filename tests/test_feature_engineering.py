import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.preprocessing.feature_engineering import TfidfFeatureExtractor, FORBIDDEN_FEATURE_COLUMNS

def test_tfidf_feature_extractor(tmp_path):
    train_texts = ["worker missed loto", "pump leaked oil", "permit not found"]
    val_texts = ["loto missing", "unknown word"]
    
    extractor = TfidfFeatureExtractor(max_features=10, min_df=1, ngram_range=(1, 2))
    
    # 1. TF-IDF fitting works
    X_train = extractor.fit_transform(train_texts)
    assert extractor.is_fitted
    assert X_train.shape[0] == 3
    
    # 2 & 3. Validation and test transform works
    X_val = extractor.transform(val_texts)
    assert X_val.shape[0] == 2
    
    # 4. Feature dimensions match
    assert X_train.shape[1] == X_val.shape[1]
    
    # 6. Bigrams are supported
    vocab = extractor.vectorizer.vocabulary_
    has_bigrams = any(" " in k for k in vocab.keys())
    assert has_bigrams, "No bigrams found in vocabulary"
    
    # 7. Empty input handling
    X_empty = extractor.transform(["", "  "])
    assert X_empty.shape[0] == 2
    
    # 8. Save/load works
    model_path = tmp_path / "tfidf.joblib"
    extractor.save(model_path)
    
    loaded_extractor = TfidfFeatureExtractor()
    loaded_extractor.load(model_path)
    X_val_loaded = loaded_extractor.transform(val_texts)
    
    assert loaded_extractor.is_fitted
    assert (X_val.toarray() == X_val_loaded.toarray()).all()
    
    # 9. Forbidden target columns check
    assert "sif_label" in FORBIDDEN_FEATURE_COLUMNS
    assert "hazard" in FORBIDDEN_FEATURE_COLUMNS

def test_processed_datasets():
    from src.config import PROCESSED_DATA_DIR, TRAIN_PROCESSED_FILE, VAL_PROCESSED_FILE, TEST_PROCESSED_FILE
    import pandas as pd
    
    # Processed directory can be created
    assert PROCESSED_DATA_DIR.exists()
    
    # Processed CSVs exist
    assert TRAIN_PROCESSED_FILE.exists()
    assert VAL_PROCESSED_FILE.exists()
    assert TEST_PROCESSED_FILE.exists()
    
    train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
    
    # description_clean exists and original description is preserved
    assert 'description_clean' in train_df.columns
    assert 'description' in train_df.columns
    
    # row counts are unchanged (Train size from Phase 1 was 1750)
    assert len(train_df) == 1750

def test_sentence_embedding_features():
    from src.preprocessing.feature_engineering import SentenceEmbeddingFeatures
    import numpy as np
    
    texts = ["this is a test", "another safety report"]
    embedder = SentenceEmbeddingFeatures(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 1. Output is generated
    embeddings = embedder.transform(texts)
    
    # 2. Dimensions are correct (MiniLM-L6-v2 outputs 384 dimensions)
    assert embeddings.shape == (2, 384)
    assert isinstance(embeddings, np.ndarray)

