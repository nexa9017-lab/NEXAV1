"""
Integration smoke test script for the Unified Safety RAG Pipeline (Phase R5).
Initializes the real legacy inference pipeline, FAISS vector store retriever,
and Groq LLM client (openai/gpt-oss-120b), executes an end-to-end analysis,
and prints all 4 subsystems separately.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GROQ_MODEL,
    VECTOR_INDEX_PATH,
    VECTOR_METADATA_PATH,
    DEFAULT_TOP_K,
)
from src.pipeline.inference import SafetyInferencePipeline
from src.rag.embeddings import IncidentEmbedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import IncidentRetriever
from src.rag.prompts import SafetyPromptBuilder
from src.rag.llm_client import GroqLLMClient, MockLLMClient
from src.rag.pipeline import SafetyRAGPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_environment():
    """Load GROQ_API_KEY from environment or local .env file if present."""
    if not os.getenv("GROQ_API_KEY"):
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and not val.startswith("your_"):
                            os.environ["GROQ_API_KEY"] = val
                    elif line.startswith("GROQ_MODEL="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ["GROQ_MODEL"] = val


def main():
    load_environment()
    api_key = os.getenv("GROQ_API_KEY")

    print("=" * 80)
    print("PHASE R5: UNIFIED SAFETY RAG PIPELINE INTEGRATION TEST")
    print("=" * 80)

    # 1. Initialize Legacy Pipeline
    print("\n[1/4] Loading Legacy Safety Inference Pipeline...")
    t0 = time.time()
    legacy_pipeline = SafetyInferencePipeline()
    legacy_pipeline.load_models()
    print(f"      Legacy models loaded in {round((time.time() - t0) * 1000, 2)} ms.")

    # 2. Initialize Retriever
    print("\n[2/4] Loading FAISS Vector Store & Semantic Retriever...")
    t0 = time.time()
    embedder = IncidentEmbedder()
    vector_store = VectorStore()
    if not vector_store.load_index():
        print(f"ERROR: Could not load vector store from {VECTOR_INDEX_PATH}. Run scripts/index_data.py first.")
        return
    retriever = IncidentRetriever(embedder=embedder, vector_store=vector_store)
    print(f"      Retriever loaded {len(vector_store.records)} historical records in {round((time.time() - t0) * 1000, 2)} ms.")

    # 3. Initialize Prompt Builder & LLM Client
    print("\n[3/4] Initializing LLM Client...")
    prompt_builder = SafetyPromptBuilder()
    if api_key:
        print(f"      Using live Groq client with model: {GROQ_MODEL}")
        llm_client = GroqLLMClient(api_key=api_key, model=GROQ_MODEL)
    else:
        print("      No GROQ_API_KEY detected. Using MockLLMClient for offline smoke test.")
        llm_client = MockLLMClient()

    # 4. Construct Unified Pipeline
    pipeline = SafetyRAGPipeline(
        legacy_pipeline=legacy_pipeline,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    test_query = (
        "Technician began opening the pump discharge flange before isolation was verified. "
        "Residual pressure remained in the line."
    )

    print("\n[4/4] Executing Unified Analysis on Sample Incident:")
    print(f"      \"{test_query}\"")
    print("-" * 80)

    assessment = pipeline.analyze(test_query, top_k=DEFAULT_TOP_K)

    # -------------------------------------------------------------
    # SECTION 1: LEGACY MODEL ASSESSMENT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. LEGACY MODEL ASSESSMENT")
    print("=" * 80)
    if assessment.model_assessment:
        leg = assessment.model_assessment
        print(f"SIF Prediction:       {leg.sif_prediction} (Probability: {leg.sif_probability})")
        print(f"Primary LSR:          {leg.primary_lsr}")
        print(f"LSR Predictions:      {leg.lsr_predictions}")
        print(f"Activities:           {leg.activities}")
        print(f"Equipment:            {leg.equipment}")
        print(f"Hazards:              {leg.hazards}")
        print(f"Barrier Failures:     {leg.barrier_failures}")
        print(f"Unsafe Actions:       {leg.unsafe_actions}")
        print(f"Unsafe Conditions:    {leg.unsafe_conditions}")
        print(f"Potential Cons.:      {leg.potential_consequences}")
        print(f"Precursor Tags:       {leg.precursor_tags}")
    else:
        print("Legacy Model Assessment Unavailable.")

    # -------------------------------------------------------------
    # SECTION 2: SIMILAR HISTORICAL INCIDENTS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"2. SIMILAR HISTORICAL INCIDENTS (Retrieved {len(assessment.similar_incidents)} cases from FAISS)")
    print("=" * 80)
    for inc in assessment.similar_incidents:
        print(f"Rank {inc.rank} | ID: {inc.incident_id} | Similarity: {inc.similarity_score:.4f} | Site: {inc.site}")
        print(f"  Description: {inc.description[:110]}...")

    # -------------------------------------------------------------
    # SECTION 3: RAG ASSESSMENT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. RAG ASSESSMENT (LLM Reasoning & Precursors)")
    print("=" * 80)
    if assessment.rag_assessment:
        rag = assessment.rag_assessment
        print(f"SIF Potential:        {rag.sif_potential}")
        print(f"Assessment Confidence: {rag.assessment_confidence}")
        print(f"SIF Explanation:      {rag.sif_explanation}")
        print(f"Life-Saving Rules:    {rag.life_saving_rules}")
        print(f"Activity:             {rag.activity}")
        print(f"Equipment:            {rag.equipment}")
        print(f"Hazards:              {rag.hazards}")
        print(f"Barrier Failures:     {rag.barrier_failures}")
        print(f"Unsafe Actions:       {rag.unsafe_actions}")
        print(f"Unsafe Conditions:    {rag.unsafe_conditions}")
        print(f"Potential Cons.:      {rag.potential_consequences}")
        print(f"Recurring Precursor:  {rag.recurring_precursor_pattern}")
        print(f"Incident Evidence:    {rag.incident_evidence}")
        print(f"Historical Patterns:  {len(rag.historical_pattern_observations)} observation(s)")
        for obs in rag.historical_pattern_observations:
            print(f"  - Obs: {obs.observation}")
            print(f"    Supporting IDs: {obs.supporting_incident_ids}")
        print(f"Uncertainty Notes:    {rag.uncertainty_notes}")
    else:
        print("RAG LLM Assessment Unavailable.")

    # -------------------------------------------------------------
    # SECTION 4: CONSISTENCY / DISAGREEMENTS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. CONSISTENCY / DISAGREEMENTS")
    print("=" * 80)
    if assessment.consistency:
        c = assessment.consistency
        print(f"SIF Agreement:        {c.sif_agreement}")
        print(f"LSR Overlap:          {c.lsr_overlap}")
        if c.conflicts:
            print("Detected Conflicts:")
            for conf in c.conflicts:
                print(f"  - [CONFLICT] {conf}")
        else:
            print("Subsystems are consistent; no conflicts detected.")
    else:
        print("Consistency check unavailable (one or both subsystems missing).")

    # -------------------------------------------------------------
    # WARNINGS & METADATA
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. EXECUTION METADATA & WARNINGS")
    print("=" * 80)
    if assessment.warnings:
        print("System Warnings:")
        for w in assessment.warnings:
            print(f"  - [WARNING] {w}")
    else:
        print("Warnings: None")

    print(f"Latencies: {assessment.metadata.get('latencies')}")
    print(f"LLM Model: {assessment.metadata.get('llm_model')}")
    print("=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
