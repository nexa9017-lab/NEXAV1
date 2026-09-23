"""
Retrieval quality evaluation and latency measurement script for Phase R3.
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.vector_store import VectorStore
from src.rag.embeddings import IncidentEmbedder
from src.rag.retriever import IncidentRetriever
from src.config import VECTOR_INDEX_PATH, VECTOR_METADATA_PATH

def evaluate():
    # 1. Measure index load time
    t0 = time.perf_counter()
    store = VectorStore(index_path=VECTOR_INDEX_PATH, metadata_path=VECTOR_METADATA_PATH)
    assert store.load_index() is True
    t_load = (time.perf_counter() - t0) * 1000

    # 2. Measure embedder model load time
    t0 = time.perf_counter()
    embedder = IncidentEmbedder()
    t_model_load = (time.perf_counter() - t0) * 1000

    retriever = IncidentRetriever(embedder=embedder, vector_store=store)

    queries = [
        ("Query A — Energy Isolation", "worker opened a flange before isolation was verified and residual pressure remained"),
        ("Query B — Work at Height", "technician working at height with harness disconnected"),
        ("Query C — Lifting / Line of Fire", "worker entered crane swing radius while suspended load was moving")
    ]

    print("=" * 80)
    print("RETRIEVAL PERFORMANCE OBSERVATIONS")
    print("=" * 80)
    print(f"Index Load Time:    {t_load:.2f} ms")
    print(f"Embedder Load Time: {t_model_load:.2f} ms\n")

    for label, q in queries:
        print("=" * 80)
        print(f"{label}")
        print(f"Query: \"{q}\"")
        print("-" * 80)

        # Measure query preprocess + embedding time
        t0 = time.perf_counter()
        clean_q = retriever.preprocessor.preprocess(q)
        q_vec = embedder.embed_text(clean_q)
        t_embed = (time.perf_counter() - t0) * 1000

        # Measure FAISS search time
        t0 = time.perf_counter()
        scores, indices = store.search_vector(q_vec, top_k=5)
        t_faiss = (time.perf_counter() - t0) * 1000

        # Measure total end-to-end retrieval method time
        t0 = time.perf_counter()
        results = retriever.retrieve_similar_incidents(q, top_k=5)
        t_total = (time.perf_counter() - t0) * 1000

        print(f"Latency Breakdown: Preprocess+Embed: {t_embed:.2f} ms | FAISS Search: {t_faiss:.2f} ms | Total Retrieve: {t_total:.2f} ms")
        print(f"\nTop {len(results)} Retrieved Results:")
        for r in results:
            print(f"  Rank {r['rank']}: [{r['incident_id']}] (Similarity Score: {r['similarity_score']:.4f})")
            print(f"    Site: {r.get('site')} | Location: {r.get('location')} | Type: {r.get('report_type')}")
            print(f"    Description: {r['description']}")
            print()

if __name__ == "__main__":
    evaluate()
