"""
LLM fallback provider interface.
Supports: Ollama (local), OpenAI-compatible, Anthropic.
"""
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

from app.settings import get_settings
from app.logging_config import get_logger

logger = get_logger("llm_fallback")

EXTRACTION_PROMPT_TEMPLATE = """You are a document data extraction assistant.
Extract the value of "{field_name}" from the following document snippet.

Field description: {field_description}
Field type: {field_type}

Document snippet:
---
{snippet}
---

Rules:
1. Extract ONLY if clearly present in the text. Otherwise return null.
2. Return ONLY a JSON object with no other text: {{"value": <extracted_value_or_null>, "evidence": "<exact_text_span_from_snippet>"}}
3. Do not add explanations or markdown.
"""


class LLMProvider(ABC):
    @abstractmethod
    def extract(self, prompt: str) -> Optional[dict]:
        """Return parsed JSON dict with 'value' and 'evidence' keys."""
        pass


class OllamaProvider(LLMProvider):
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = settings.llm_model

    def extract(self, prompt: str) -> Optional[dict]:
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0, "seed": 42},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "").strip()
                return _parse_llm_json(raw)
        except Exception as e:
            logger.error("ollama_error", error=str(e))
            return None


class CloudProvider(LLMProvider):
    """Generic cloud provider (OpenAI-compatible API)."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.cloud_llm_api_key
        self.provider = settings.cloud_llm_provider
        if self.provider == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-4o-mini"
        elif self.provider == "anthropic":
            self.base_url = "https://api.anthropic.com/v1"
            self.model = "claude-haiku-4-5-20251001"
        else:
            self.base_url = ""
            self.model = ""

    def extract(self, prompt: str) -> Optional[dict]:
        if not self.api_key or not self.base_url:
            logger.warning("cloud_llm_not_configured")
            return None
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            if self.provider == "anthropic":
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": prompt}],
                }
                endpoint = f"{self.base_url}/messages"
            else:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 256,
                }
                endpoint = f"{self.base_url}/chat/completions"

            with httpx.Client(timeout=30) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            if self.provider == "anthropic":
                raw = data["content"][0]["text"].strip()
            else:
                raw = data["choices"][0]["message"]["content"].strip()

            return _parse_llm_json(raw)
        except Exception as e:
            logger.error("cloud_llm_error", provider=self.provider, error=str(e))
            return None


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Extract JSON from LLM response (may have extra text)."""
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract JSON object
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def get_llm_provider() -> Optional[LLMProvider]:
    settings = get_settings()
    if not settings.llm_fallback_enabled:
        return None
    if settings.cloud_llm_provider and settings.cloud_llm_api_key:
        return CloudProvider()
    return OllamaProvider()
