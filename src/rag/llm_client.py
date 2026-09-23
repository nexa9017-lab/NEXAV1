"""
LLM client interface, provider implementations, and response parser for the RAG safety MVP.
Supports Groq API integration and deterministic MockLLMClient for testing.
"""
from abc import ABC, abstractmethod
import json
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from src.config import GROQ_MODEL
from src.api.schemas import LLMIncidentAssessment
from src.preprocessing.safety_taxonomy import LIFE_SAVING_RULES

class LLMResponseError(RuntimeError):
    """Raised when an LLM response is malformed, invalid JSON, or violates schema/taxonomy constraints."""
    pass

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM inference providers.
    """
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Submits prompt to language model and returns raw text response.
        """
        pass

class GroqLLMClient(BaseLLMClient):
    """
    Groq LLM Client calling the specified model (e.g. openai/gpt-oss-120b).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
        temperature: float = 0.1
    ):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Groq API key not found. Set the GROQ_API_KEY environment variable "
                "or pass api_key explicitly."
            )

        from groq import Groq
        self.client = Groq(api_key=key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Attempt with json_object response format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # Fallback if specific model or API version rejects response_format
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                return response.choices[0].message.content or ""
            except Exception as inner_e:
                raise LLMResponseError(f"Groq API call failed: {inner_e}")

class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock LLM client for unit tests without network calls or API keys.
    """
    def __init__(self, response: Optional[str] = None):
        self.response = response or self._default_response()
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

    def _default_response(self) -> str:
        return json.dumps({
            "sif_potential": True,
            "assessment_confidence": "high",
            "sif_explanation": "Opening process equipment before verified isolation with residual pressure creates high-energy hazard.",
            "life_saving_rules": ["Energy Isolation"],
            "activity": "Maintenance",
            "equipment": ["pump discharge flange"],
            "hazards": ["stored energy", "pressurized fluid"],
            "barrier_failures": ["energy isolation not verified"],
            "unsafe_actions": ["opened flange before zero-energy verification"],
            "unsafe_conditions": ["residual pressure in line"],
            "potential_consequences": ["high-pressure fluid release", "serious bodily injury"],
            "recurring_precursor_pattern": "Maintenance on pressurized equipment without verified isolation",
            "incident_evidence": ["Technician began opening pump discharge flange before isolation was verified."],
            "historical_pattern_observations": [
                {
                    "observation": "Historical cases consistently exhibit flange unbolting prior to verified lockout.",
                    "supporting_incident_ids": ["REP-338811A7"]
                }
            ],
            "uncertainty_notes": ["Fluid type and line operating pressure not specified."]
        })

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self.response

def parse_assessment_response(raw_response: str) -> LLMIncidentAssessment:
    """
    Parses and validates raw LLM output into an LLMIncidentAssessment Pydantic model.
    Tolerates JSON wrapped in markdown blocks (```json ... ```).
    Enforces strict Life-Saving Rule validation against canonical taxonomy.
    """
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise LLMResponseError("Received empty response from LLM.")

    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as jde:
        raise LLMResponseError(f"Failed to parse LLM response as JSON: {jde}")

    if not isinstance(data, dict):
        raise LLMResponseError(f"Expected JSON object in LLM response, got {type(data).__name__}.")

    # Validate against Pydantic schema
    try:
        assessment = LLMIncidentAssessment.model_validate(data)
    except ValidationError as ve:
        raise LLMResponseError(f"LLM response failed schema validation: {ve}")

    # Validate Life-Saving Rules against canonical taxonomy
    valid_rules = set(LIFE_SAVING_RULES.keys())
    for rule in assessment.life_saving_rules:
        if rule not in valid_rules:
            raise LLMResponseError(
                f"Unsupported Life-Saving Rule '{rule}' returned by LLM. "
                f"Must be selected from: {sorted(valid_rules)}"
            )

    return assessment
