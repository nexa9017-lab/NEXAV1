"""
Unified Safety RAG Pipeline.
Combines legacy deterministic ML/rules pipeline, FAISS semantic retrieval,
prompt construction, and Groq LLM intelligence into a unified safety assessment.
"""
import time
from typing import Any, Dict, List, Optional, Set

from src.config import DEFAULT_TOP_K
from src.api.schemas import (
    LegacyModelAssessment,
    RetrievedIncident,
    AssessmentConsistency,
    UnifiedSafetyAssessment,
    LLMIncidentAssessment,
)
from src.rag.llm_client import parse_assessment_response, LLMResponseError


class SafetyRAGPipeline:
    """
    Unified Safety Intelligence Pipeline.

    Orchestrates:
    1. Legacy classification & precursor extraction (deterministic baseline)
    2. FAISS dense semantic retrieval (historical context)
    3. LLM structured reasoning with anti-hallucination and evidence grounding
    4. Historical ID validation against retrieved set
    5. Subsystem consistency and conflict detection
    """

    def __init__(
        self,
        legacy_pipeline: Any = None,
        retriever: Any = None,
        prompt_builder: Any = None,
        llm_client: Any = None,
    ):
        """
        Dependency-injected constructor. Does NOT automatically instantiate expensive
        production dependencies so that tests and callers retain full modular control.
        """
        self.legacy_pipeline = legacy_pipeline
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def _extract_string_list(self, items: Any) -> List[str]:
        """Helper to extract clean strings from a list of extraction dicts or strings."""
        if not items or not isinstance(items, list):
            return []
        result = []
        for item in items:
            if isinstance(item, dict):
                val = item.get("value")
                if val is not None and str(val).strip():
                    result.append(str(val).strip())
            elif isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif item is not None:
                result.append(str(item).strip())
        return result

    def _map_legacy_output(self, raw_output: Dict[str, Any]) -> LegacyModelAssessment:
        """
        Maps the actual dictionary returned by SafetyInferencePipeline.predict_single()
        into the strongly typed LegacyModelAssessment schema.
        """
        # SIF mapping
        sif_dict = raw_output.get("sif") if isinstance(raw_output.get("sif"), dict) else {}
        sif_label = sif_dict.get("label")
        if sif_label is not None:
            sif_prediction = (sif_label == "SIF-Potential")
        else:
            sif_prediction = bool(raw_output.get("sif_prediction", False))

        sif_probability = sif_dict.get("probability")
        if sif_probability is None and "sif_probability" in raw_output:
            sif_probability = raw_output.get("sif_probability")
        if sif_probability is not None:
            sif_probability = float(sif_probability)

        # Primary LSR
        primary_lsr = raw_output.get("primary_lsr")

        # LSR predictions list
        lsr_predictions: List[str] = []
        raw_lsr = raw_output.get("life_saving_rules")
        if isinstance(raw_lsr, list):
            for r in raw_lsr:
                if isinstance(r, dict) and "rule" in r:
                    lsr_predictions.append(str(r["rule"]))
                elif isinstance(r, str):
                    lsr_predictions.append(r)
        elif isinstance(raw_output.get("lsr_predictions"), list):
            lsr_predictions = [str(r) for r in raw_output["lsr_predictions"]]

        return LegacyModelAssessment(
            sif_prediction=sif_prediction,
            sif_probability=sif_probability,
            primary_lsr=str(primary_lsr) if primary_lsr else None,
            lsr_predictions=lsr_predictions,
            activities=self._extract_string_list(raw_output.get("activities")),
            equipment=self._extract_string_list(raw_output.get("equipment")),
            hazards=self._extract_string_list(raw_output.get("hazards")),
            barrier_failures=self._extract_string_list(raw_output.get("barrier_failures")),
            unsafe_actions=self._extract_string_list(raw_output.get("unsafe_actions")),
            unsafe_conditions=self._extract_string_list(raw_output.get("unsafe_conditions")),
            potential_consequences=self._extract_string_list(raw_output.get("potential_consequences")),
            precursor_tags=self._extract_string_list(raw_output.get("precursor_tags")),
        )

    def _validate_historical_ids(
        self, rag_assessment: LLMIncidentAssessment, allowed_ids: Set[str]
    ) -> List[str]:
        """
        Verifies that every incident ID referenced in historical_pattern_observations
        actually belongs to the retrieved historical incidents set.
        Strips invalid IDs and generates warning strings.
        """
        warnings: List[str] = []
        for obs in rag_assessment.historical_pattern_observations:
            valid_ids = []
            for inc_id in obs.supporting_incident_ids:
                if inc_id in allowed_ids:
                    valid_ids.append(inc_id)
                else:
                    warnings.append(
                        f"LLM referenced historical incident {inc_id}, "
                        f"which was not present in retrieved context."
                    )
            obs.supporting_incident_ids = valid_ids
        return warnings

    def _evaluate_consistency(
        self,
        model_assessment: Optional[LegacyModelAssessment],
        rag_assessment: Optional[LLMIncidentAssessment],
    ) -> Optional[AssessmentConsistency]:
        """
        Evaluates agreement between legacy classifier and RAG LLM on SIF potential
        and computes Life-Saving Rule overlap and explicit conflict descriptions.
        """
        if model_assessment is None or rag_assessment is None:
            return None

        # SIF Agreement
        legacy_sif = bool(model_assessment.sif_prediction)
        rag_sif = bool(rag_assessment.sif_potential)
        sif_agreement = (legacy_sif == rag_sif)

        # LSR Overlap
        legacy_lsrs = set(model_assessment.lsr_predictions)
        if model_assessment.primary_lsr:
            legacy_lsrs.add(model_assessment.primary_lsr)
        rag_lsrs = set(rag_assessment.life_saving_rules)

        lsr_overlap = sorted(list(legacy_lsrs.intersection(rag_lsrs)))

        conflicts: List[str] = []
        if not sif_agreement:
            legacy_str = "SIF-Potential" if legacy_sif else "Non-SIF"
            rag_str = "SIF-Potential" if rag_sif else "Non-SIF"
            conflicts.append(
                f"Legacy classifier predicted {legacy_str} while the RAG assessment identified {rag_str}."
            )

        if legacy_lsrs and rag_lsrs and not lsr_overlap:
            conflicts.append(
                f"LSR mismatch: Legacy model identified {sorted(list(legacy_lsrs))} "
                f"whereas RAG assessment identified {sorted(list(rag_lsrs))}."
            )

        return AssessmentConsistency(
            sif_agreement=sif_agreement,
            lsr_overlap=lsr_overlap,
            conflicts=conflicts,
        )

    def analyze(
        self, incident_text: str, top_k: int = DEFAULT_TOP_K
    ) -> UnifiedSafetyAssessment:
        """
        Executes the complete unified safety analysis pipeline.

        Execution sequence:
        1. Validate incident text
        2. Run legacy classification & extraction
        3. Retrieve top-k historical incidents from FAISS
        4. If retrieval succeeds, build prompt and invoke LLM
        5. Parse LLM response and validate referenced historical IDs
        6. Compute consistency and conflicts
        7. Return UnifiedSafetyAssessment preserving subsystem separation
        """
        # 1. Validate incident text
        if not isinstance(incident_text, str) or not incident_text.strip():
            raise ValueError("Incident text cannot be empty or whitespace-only.")

        clean_incident_text = incident_text.strip()
        warnings: List[str] = []
        latencies: Dict[str, float] = {}
        t_total_start = time.time()

        # 2. Legacy Pipeline
        model_assessment: Optional[LegacyModelAssessment] = None
        if self.legacy_pipeline is not None:
            try:
                t0 = time.time()
                raw_legacy = self.legacy_pipeline.predict_single({"description": clean_incident_text})
                latencies["legacy_pipeline_ms"] = round((time.time() - t0) * 1000, 2)
                model_assessment = self._map_legacy_output(raw_legacy)
            except Exception as e:
                warnings.append(f"Legacy model inference failed: {str(e)}")
        else:
            warnings.append("Legacy pipeline not configured.")

        # 3. Retrieval
        retrieved_cases: List[Dict[str, Any]] = []
        similar_incidents: List[RetrievedIncident] = []
        retrieval_ok = False

        if self.retriever is not None:
            try:
                t0 = time.time()
                retrieved_cases = self.retriever.retrieve_similar_incidents(
                    clean_incident_text, top_k=top_k
                )
                latencies["retrieval_ms"] = round((time.time() - t0) * 1000, 2)

                if retrieved_cases:
                    retrieval_ok = True
                    for item in retrieved_cases:
                        similar_incidents.append(
                            RetrievedIncident(
                                rank=item.get("rank", 1),
                                vector_id=item.get("vector_id", 0),
                                incident_id=str(item.get("incident_id") or "UNKNOWN"),
                                similarity_score=float(item.get("similarity_score", 0.0)),
                                description=str(item.get("description") or ""),
                                site=item.get("site"),
                                location=item.get("location"),
                                date=item.get("date"),
                                report_type=item.get("report_type"),
                            )
                        )
                else:
                    warnings.append(
                        "No historical incidents retrieved. Skipping LLM historical reasoning."
                    )
            except Exception as e:
                warnings.append(
                    f"Retrieval failed: {str(e)}. Skipping LLM historical reasoning."
                )
        else:
            warnings.append("Retriever not configured. Skipping LLM historical reasoning.")

        # 4. LLM Reasoning (invoked only if retrieval returned supporting context)
        rag_assessment: Optional[LLMIncidentAssessment] = None
        if retrieval_ok and self.prompt_builder is not None and self.llm_client is not None:
            try:
                t0 = time.time()
                system_prompt = self.prompt_builder.build_system_prompt()
                analysis_prompt = self.prompt_builder.build_analysis_prompt(
                    clean_incident_text, retrieved_cases
                )
                latencies["prompt_build_ms"] = round((time.time() - t0) * 1000, 2)

                t0 = time.time()
                raw_llm_response = self.llm_client.generate(
                    analysis_prompt, system_prompt=system_prompt
                )
                latencies["llm_generation_ms"] = round((time.time() - t0) * 1000, 2)

                t0 = time.time()
                rag_assessment = parse_assessment_response(raw_llm_response)
                latencies["response_parse_ms"] = round((time.time() - t0) * 1000, 2)

                # Validate and filter referenced historical IDs
                allowed_ids = {r.incident_id for r in similar_incidents}
                id_warnings = self._validate_historical_ids(rag_assessment, allowed_ids)
                warnings.extend(id_warnings)

            except Exception as e:
                warnings.append(f"RAG LLM assessment failed: {str(e)}")
        elif retrieval_ok:
            if self.prompt_builder is None:
                warnings.append("Prompt builder not configured.")
            if self.llm_client is None:
                warnings.append("LLM client not configured.")

        # 5. Subsystem Consistency
        consistency = self._evaluate_consistency(model_assessment, rag_assessment)

        latencies["total_pipeline_ms"] = round((time.time() - t_total_start) * 1000, 2)

        metadata: Dict[str, Any] = {
            "top_k": top_k,
            "retrieved_count": len(similar_incidents),
            "latencies": latencies,
            "llm_model": getattr(self.llm_client, "model", None) if self.llm_client else None,
        }

        return UnifiedSafetyAssessment(
            incident_text=clean_incident_text,
            model_assessment=model_assessment,
            rag_assessment=rag_assessment,
            similar_incidents=similar_incidents,
            consistency=consistency,
            warnings=warnings,
            metadata=metadata,
        )

