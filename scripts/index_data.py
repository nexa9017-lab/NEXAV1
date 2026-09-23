"""
RAG Historical Safety Incident Indexing Script.
Loads historical safety reports, preprocesses descriptions, generates embeddings,
and builds the persistent FAISS vector store and metadata catalog.
"""
import sys
import time
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_REPORTS_FILE, VECTOR_INDEX_PATH, VECTOR_METADATA_PATH, RAG_EMBEDDING_MODEL
from src.preprocessing.text_preprocessor import SafetyTextPreprocessor
from src.rag.embeddings import IncidentEmbedder
from src.rag.vector_store import VectorStore

ALLOWED_METADATA_COLS = ["report_id", "date", "site", "location", "report_type"]

def run_indexing(
    csv_path: Path = ALL_REPORTS_FILE,
    index_path: Path = VECTOR_INDEX_PATH,
    metadata_path: Path = VECTOR_METADATA_PATH,
    batch_size: int = 64
) -> bool:
    print("RAG Historical Index Build")
    print("--------------------------")
    print(f"Source: {csv_path.as_posix() if hasattr(csv_path, 'as_posix') else csv_path}")

    if not Path(csv_path).exists():
        print(f"Error: Source CSV not found at {csv_path}")
        return False

    start_time = time.time()
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Rows loaded: {total_rows}")

    if "description" not in df.columns:
        print("Error: Required column 'description' not found in CSV.")
        return False

    # Filter out empty or whitespace-only descriptions
    valid_mask = df["description"].fillna("").astype(str).str.strip() != ""
    skipped_count = int((~valid_mask).sum())
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    valid_count = len(valid_df)

    print(f"Valid descriptions: {valid_count}")
    print(f"Skipped descriptions: {skipped_count}")

    if valid_count == 0:
        print("Error: No valid descriptions found to index.")
        return False

    # Initialize existing SafetyTextPreprocessor
    preprocessor = SafetyTextPreprocessor()

    records = []
    cleaned_texts = []

    for _, row in valid_df.iterrows():
        raw_desc = str(row["description"]).strip()
        cleaned_desc = preprocessor.preprocess(raw_desc)
        cleaned_texts.append(cleaned_desc)

        record = {
            "description": raw_desc,
            "clean_description": cleaned_desc
        }
        for col in ALLOWED_METADATA_COLS:
            if col in row and pd.notna(row[col]):
                record[col] = str(row[col])
            else:
                record[col] = None
        records.append(record)

    # Initialize Embedder
    print(f"\nEmbedding model:\n{RAG_EMBEDDING_MODEL}")
    embedder = IncidentEmbedder(model_name=RAG_EMBEDDING_MODEL)
    print(f"\nEmbedding dimension: {embedder.dimension}")

    # Generate embeddings
    embed_start = time.time()
    embeddings = embedder.embed_batch(cleaned_texts, batch_size=batch_size)
    embed_time = round(time.time() - embed_start, 2)
    print(f"Embedding generated in {embed_time}s")

    # Build and Save FAISS Index
    print(f"\nBuilding FAISS IndexFlatIP...")
    store = VectorStore(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=RAG_EMBEDDING_MODEL
    )
    store.build_index(records=records, embeddings=embeddings)
    print(f"Records indexed: {len(records)}")

    store.save_index()
    rel_index_path = Path(index_path).as_posix()
    rel_meta_path = Path(metadata_path).as_posix()
    print(f"\nIndex saved:\n{rel_index_path}")
    print(f"\nMetadata saved:\n{rel_meta_path}")

    # Verify reload
    verify_store = VectorStore(index_path=index_path, metadata_path=metadata_path)
    loaded = verify_store.load_index()
    if not loaded or verify_store.index.ntotal != valid_count:
        print("Error: Verification reload failed.")
        return False

    elapsed = round(time.time() - start_time, 2)
    print(f"\nTotal indexing time: {elapsed}s")
    print("RAG_INDEX_BUILD_SUCCESS = true")
    return True

if __name__ == "__main__":
    success = run_indexing()
    if not success:
        sys.exit(1)
