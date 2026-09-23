"""
Direct smoke test script for Groq LLM reasoning (Phase R4).
Verifies GROQ_API_KEY, constructs a prompt, invokes GroqLLMClient, parses response into Pydantic schema, and measures latency.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GROQ_MODEL
from src.rag.prompts import SafetyPromptBuilder
from src.rag.llm_client import GroqLLMClient, parse_assessment_response

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
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

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:

        print("=" * 70)
        print("SKIPPING LIVE GROQ TEST: GROQ_API_KEY environment variable is not set.")
        print("To run live testing:")
        print("  $env:GROQ_API_KEY = 'your-groq-api-key'")
        print("  python scripts/test_groq_llm.py")
        print("=" * 70)
        return

    print("=" * 70)
    print("LIVE GROQ LLM SMOKE TEST (Phase R4)")
    print("=" * 70)
    print(f"Target Model: {GROQ_MODEL}")
    print("API Key: [PRESENT]")

    # 1. Initialize prompt builder and Groq client
    prompt_builder = SafetyPromptBuilder()
    try:
        llm_client = GroqLLMClient(model=GROQ_MODEL)
    except Exception as e:
        print(f"Error initializing GroqLLMClient: {e}")
        return

    # 2. Sample incident and hardcoded supporting context for direct testing
    sample_incident = "Technician began opening the pump discharge flange before isolation was verified. Residual pressure remained in the line."
    sample_cases = [
        {
            "rank": 1,
            "incident_id": "REP-338811A7",
            "similarity_score": 0.6375,
            "site": "Platform Charlie",
            "location": "Boiler Area",
            "date": "2023-04-18",
            "report_type": "Near Miss",
            "description": "During Maintenance of Compressor C-10, the technician started loosening the flange while the line was still pressurized. Isolation valve was closed but Padlock missing. Stored energy discharge occurred."
        },
        {
            "rank": 2,
            "incident_id": "REP-DCCA20C7",
            "similarity_score": 0.6226,
            "site": "Refinery Delta",
            "location": "Boiler Area",
            "date": "2023-05-02",
            "report_type": "Incident",
            "description": "During Maintenance of Compressor C-10, the technician started loosening the flange while the line was still pressurized. Isolation valve was closed but Isolation valve leaking. Steam leak occurred."
        }
    ]

    # 3. Build prompts
    t0 = time.perf_counter()
    system_prompt = prompt_builder.build_system_prompt()
    analysis_prompt = prompt_builder.build_analysis_prompt(sample_incident, sample_cases)
    t_prompt = (time.perf_counter() - t0) * 1000

    # 4. Invoke Groq LLM
    print(f"\nCalling Groq API ({GROQ_MODEL})...")
    t0 = time.perf_counter()
    try:
        raw_output = llm_client.generate(prompt=analysis_prompt, system_prompt=system_prompt)
        t_llm = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return

    # 5. Parse and validate response
    t0 = time.perf_counter()
    try:
        assessment = parse_assessment_response(raw_output)
        t_parse = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"Response parsing failed: {e}")
        print(f"Raw output was:\n{raw_output}")
        return

    print("\n" + "=" * 70)
    print("STRUCTURED SAFETY ASSESSMENT RESULT")
    print("=" * 70)
    print(f"SIF Potential:           {assessment.sif_potential}")
    print(f"Assessment Confidence:   {assessment.assessment_confidence}")
    print(f"SIF Explanation:         {assessment.sif_explanation}")
    print(f"Life-Saving Rules:       {assessment.life_saving_rules}")
    print(f"Activity:                {assessment.activity}")
    print(f"Equipment:               {assessment.equipment}")
    print(f"Hazards:                 {assessment.hazards}")
    print(f"Barrier Failures:        {assessment.barrier_failures}")
    print(f"Unsafe Actions:          {assessment.unsafe_actions}")
    print(f"Unsafe Conditions:       {assessment.unsafe_conditions}")
    print(f"Potential Consequences:  {assessment.potential_consequences}")
    print(f"Recurring Pattern:       {assessment.recurring_precursor_pattern}")
    print(f"Incident Evidence:       {assessment.incident_evidence}")
    print(f"Historical Observations: {len(assessment.historical_pattern_observations)} patterns found")
    for obs in assessment.historical_pattern_observations:
        print(f"  - {obs.observation} (Supporting IDs: {obs.supporting_incident_ids})")
    print(f"Uncertainty Notes:       {assessment.uncertainty_notes}")

    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN:")
    print(f"Prompt Construction: {t_prompt:.2f} ms")
    print(f"Groq LLM Call:       {t_llm:.2f} ms")
    print(f"Response Parsing:    {t_parse:.2f} ms")
    print(f"Total Time:          {(t_prompt + t_llm + t_parse):.2f} ms")
    print("=" * 70)

if __name__ == "__main__":
    main()
