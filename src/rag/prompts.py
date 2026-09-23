"""
Prompt construction and safety taxonomy formatting for the RAG safety MVP.
Formats industrial safety prompts enforcing anti-hallucination and evidence grounding rules.
"""
from typing import Any, Dict, List, Optional
from src.preprocessing.safety_taxonomy import LIFE_SAVING_RULES

class SafetyPromptBuilder:
    """
    Constructs system and analysis prompts for LLM safety intelligence assessment.
    """
    def __init__(self, taxonomy: Optional[Dict[str, Any]] = None):
        self.taxonomy = taxonomy or LIFE_SAVING_RULES
        self.canonical_rules = list(self.taxonomy.keys())

    def build_system_prompt(self) -> str:
        """
        Builds the system instruction prompt establishing safety rules and grounding boundaries.
        """
        rules_list_str = "\n".join(f"- {rule}" for rule in sorted(self.canonical_rules))

        return f"""You are an expert industrial safety intelligence and decision-support assistant.
Your role is to analyze frontline safety observations, near misses, and incident reports to identify Serious Injury and Fatality (SIF) precursors.

CRITICAL REASONING & GROUNDING RULES:
1. PRIMARY SOURCE OF TRUTH: The NEW INCIDENT is the primary source of truth. Base current-incident facts strictly on the text of the new incident.
2. SUPPORTING CONTEXT ONLY: Retrieved historical incidents are supporting context only.
3. PROHIBITION ON FACT COPYING: Never copy facts, equipment tags, site names, barrier failures, or specifics from historical incidents into the new incident analysis unless those facts are explicitly present in the new incident itself. Similar incidents may help identify potential patterns, but they must never overwrite or fabricate facts about the current incident.
4. EVIDENCE GROUNDING: Statements in 'incident_evidence' must be short factual sentences directly supported by the new incident text. Do not invent missing details.
5. UNCERTAINTY & MISSING INFORMATION: If information is ambiguous or absent (e.g. pressure levels, fluid types, exact height), clearly note this in 'uncertainty_notes'.
6. LIFE-SAVING RULE TAXONOMY: You must ONLY select relevant Life-Saving Rules from the following approved canonical list:
{rules_list_str}
Do not invent or use alternative rule names. If no rule applies, return an empty list.
7. HISTORICAL PATTERNS: In 'historical_pattern_observations', summarize multi-factor risk patterns observed across the retrieved cases. You must ONLY reference supporting incident IDs that are explicitly provided in the retrieved cases list. Never invent incident IDs.
8. CONFIDENCE: Set 'assessment_confidence' to "high", "medium", or "low" reflecting qualitative certainty based on description clarity and completeness. This is NOT a calibrated probability.
9. OUTPUT FORMAT: Output valid, parseable JSON ONLY. Do not enclose in markdown blocks. Do not add conversational text or explanation before or after the JSON.
"""

    def build_analysis_prompt(
        self,
        new_incident: str,
        retrieved_incidents: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Constructs the user analysis prompt combining the new incident and supporting historical context.
        """
        if not new_incident or not new_incident.strip():
            raise ValueError("new_incident cannot be empty or whitespace-only.")

        incident_text = new_incident.strip()
        retrieved_cases = retrieved_incidents or []

        # Format retrieved incidents section
        if not retrieved_cases:
            historical_section = "No historical incident records provided.\n"
        else:
            case_blocks = []
            for i, case in enumerate(retrieved_cases, 1):
                inc_id = case.get("incident_id") or case.get("report_id") or f"CASE-{i}"
                score = case.get("similarity_score", 0.0)
                site = case.get("site", "Unknown")
                location = case.get("location", "Unknown")
                date = case.get("date", "Unknown")
                rep_type = case.get("report_type", "Unknown")
                desc = case.get("description", "").strip()

                block = (
                    f"Case {i}:\n"
                    f"Incident ID: {inc_id}\n"
                    f"Similarity Score: {score}\n"
                    f"Site: {site} | Location: {location} | Date: {date}\n"
                    f"Report Type: {rep_type}\n"
                    f"Description: {desc}"
                )
                case_blocks.append(block)
            historical_section = "\n\n".join(case_blocks)

        prompt = f"""NEW INCIDENT
------------
{incident_text}

RETRIEVED HISTORICAL INCIDENTS
------------------------------
{historical_section}

TASK
----
Analyze ONLY the new incident using the retrieved historical incidents strictly as supporting context.
Return a valid JSON object matching the following structure exactly:

{{
  "sif_potential": true/false,
  "assessment_confidence": "high" | "medium" | "low",
  "sif_explanation": "Causal reasoning for the SIF determination based on energy and barriers.",
  "life_saving_rules": ["Approved Rule Name 1", ...],
  "activity": "Operational activity underway or null",
  "equipment": ["Equipment name 1", ...],
  "hazards": ["Hazard description 1", ...],
  "barrier_failures": ["Barrier failure 1", ...],
  "unsafe_actions": ["Unsafe action 1", ...],
  "unsafe_conditions": ["Unsafe condition 1", ...],
  "potential_consequences": ["Plausible worst-case consequence 1", ...],
  "recurring_precursor_pattern": "Textual synthesis of the multi-factor risk pattern or null",
  "incident_evidence": ["Direct factual statement from the new incident 1", ...],
  "historical_pattern_observations": [
    {{
      "observation": "Synthesis of pattern observed in historical context",
      "supporting_incident_ids": ["EXACT_RETRIEVED_ID_1", ...]
    }}
  ],
  "uncertainty_notes": ["Caveat or missing detail 1", ...]
}}
"""
        return prompt
