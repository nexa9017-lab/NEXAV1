import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Data File Paths
ALL_REPORTS_FILE = DATA_DIR / "all_reports.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"
DB_FILE = DATA_DIR / "sif_database.sqlite"

# Processed Data Paths
PROCESSED_DATA_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
VAL_PROCESSED_FILE = PROCESSED_DATA_DIR / "validation_processed.csv"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

# Models and Artifacts Paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"

MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CHALLENGE_FILE = DATA_DIR / "challenge" / "challenge_reports.csv"
os.makedirs(DATA_DIR / "challenge", exist_ok=True)

# Phase 3 and 3.5 Configuration

# SIF Classifier
SIF_MODEL_TFIDF_PATH = MODELS_DIR / "sif_tfidf.joblib"
SIF_MODEL_EMBEDDING_PATH = MODELS_DIR / "sif_embedding.joblib"
SIF_METRICS_PATH = REPORTS_DIR / "sif_metrics.json"
SIF_LR_C = 1.0
SIF_LR_MAX_ITER = 1000
SIF_LR_CLASS_WEIGHT = "balanced"
SIF_MIN_RECALL = 0.85 # The minimum recall we want to target
SIF_THRESHOLD_SEARCH_RANGE = [x / 100.0 for x in range(10, 91, 5)]

# LSR Classifier
LSR_MODEL_TFIDF_PATH = MODELS_DIR / "lsr_tfidf.joblib"
LSR_MODEL_EMBEDDING_PATH = MODELS_DIR / "lsr_embedding.joblib"
LSR_ENCODER_PATH = MODELS_DIR / "lsr_label_encoder.joblib"
LSR_METRICS_PATH = REPORTS_DIR / "lsr_metrics.json"
LSR_LR_C = 1.0
LSR_LR_MAX_ITER = 1000
LSR_LR_CLASS_WEIGHT = "balanced"
LSR_DEFAULT_GLOBAL_THRESHOLD = 0.35
LSR_USE_PER_CLASS_THRESHOLDS = True

# Feature Engineering Settings
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_LOWERCASE = True

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = ARTIFACTS_DIR / "embeddings_cache"
SEMANTIC_SIMILARITY_THRESHOLD = 0.65
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"

# Embeddings Settings
EMBEDDINGS_ENABLED = True

# Generator Settings
NUM_REPORTS = 2500
RANDOM_SEED = 42

# SIF Distribution Target (~20-30%)
SIF_PROBABILITY = 0.25

# Phase 5 — Analytics Configuration
ANALYTICS_DATA_DIR = DATA_DIR / "analytics"
os.makedirs(ANALYTICS_DATA_DIR, exist_ok=True)
ENRICHED_REPORTS_PATH = ANALYTICS_DATA_DIR / "enriched_reports.csv"

ANALYTICS_OUTPUT_DIR = ARTIFACTS_DIR / "analytics"
os.makedirs(ANALYTICS_OUTPUT_DIR, exist_ok=True)

# Pattern Mining
MIN_PATTERN_SUPPORT = 5
LOW_SAMPLE_SIZE_THRESHOLD = 10

# Emerging Pattern Detection
EMERGING_PATTERN_THRESHOLD = 1.0  # 100% increase
EMERGING_PERIOD = "quarter"       # "quarter" or "month"

# Prototype Prioritization Score Weights (must sum to 1.0)
PRIORITY_WEIGHT_FREQUENCY = 0.20
PRIORITY_WEIGHT_SIF_DENSITY = 0.35
PRIORITY_WEIGHT_AVG_PROBABILITY = 0.25
PRIORITY_WEIGHT_CROSS_SITE = 0.20

# Analytics Version
ANALYTICS_VERSION = "0.5.0"
ANALYTICS_PROTOTYPE_NOTICE = (
    "Analytics are generated from synthetic data and model-predicted "
    "fields for demonstration purposes."
)

# ---------------------------------------------------------------------------
# RAG MVP Configuration (Phase R1 & R4)
# ---------------------------------------------------------------------------
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_DIR = ARTIFACTS_DIR / "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
VECTOR_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
VECTOR_METADATA_PATH = VECTOR_STORE_DIR / "index_metadata.json"
DEFAULT_TOP_K = 5

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


