"""
CLI script to query the RAG vector store for similar historical safety incidents.
Loads existing FAISS index (does not rebuild index, does not call any LLM).
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import VECTOR_INDEX_PATH, VECTOR_METADATA_PATH, RAG_EMBEDDING_MODEL, DEFAULT_TOP_K
from src.rag.embeddings import IncidentEmbedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import IncidentRetriever

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        try:
            query = input("Enter incident description: ").strip()
        except EOFError:
            print("No query entered.")
            return

    if not query:
        print("Error: Empty query provided.")
        sys.exit(1)

    # 1. Load vector store
    t_load_start = time.time()
    store = VectorStore(index_path=VECTOR_INDEX_PATH, metadata_path=VECTOR_METADATA_PATH)
    loaded = store.load_index()
    if not loaded:
        print(f"Error: Vector store artifacts not found at {VECTOR_INDEX_PATH}. Run scripts/index_data.py first.")
        sys.exit(1)
    t_load = round((time.time() - t_load_start) * 1000, 2)

    # 2. Initialize Embedder & Retriever
    embedder = IncidentEmbedder(model_name=RAG_EMBEDDING_MODEL)
    retriever = IncidentRetriever(embedder=embedder, vector_store=store)

    # 3. Retrieve
    t_search_start = time.time()
    results = retriever.retrieve_similar_incidents(query=query, top_k=DEFAULT_TOP_K)
    t_search = round((time.time() - t_search_start) * 1000, 2)

    print(f"\nTop {len(results)} Similar Historical Incidents")
    print("=" * 60)
    for r in results:
        print(f"\n{r['rank']}. {r['incident_id']} (Similarity: {r['similarity_score']:.4f})")
        print(f"   Site: {r.get('site', 'Unknown')} | Location: {r.get('location', 'Unknown')} | Date: {r.get('date', 'Unknown')}")
        print(f"   Type: {r.get('report_type', 'Unknown')}")
        print(f"   Description: {r['description']}")

    print("\n" + "=" * 60)
    print(f"Performance: Index load: {t_load}ms | Retrieval: {t_search}ms")

if __name__ == "__main__":
    main()
