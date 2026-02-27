"""
LLM fallback gate: decides when to trigger LLM and processes results.
"""
from typing import Optional

from app.config.loader import FieldConfig, DocumentConfig
from app.fallback.llm_provider import (
    get_llm_provider, EXTRACTION_PROMPT_TEMPLATE, LLMProvider
)
from app.models import (
    FieldResult, FieldCandidate, ExtractionMethod, ExtractionStatus,
    SourceLocation, Evidence, NormalizedDocument,
)
from app.settings import get_settings
from app.logging_config import get_logger

logger = get_logger("llm_fallback")


def should_use_llm(field_result: FieldResult, field_config: FieldConfig) -> bool:
    """
    Gate: trigger LLM only when:
    - fallback is allowed for this field
    - field is required and missing, OR
    - confidence < threshold
    """
    settings = get_settings()
    if not settings.llm_fallback_enabled:
        return False
    if not field_config.fallback_allowed:
        return False
    if field_result.status == ExtractionStatus.MISSING and field_config.required:
        return True
    if field_result.confidence < settings.llm_confidence_threshold:
        return True
    return False


def _build_snippet(doc: NormalizedDocument, max_chars: int = 2000) -> str:
    """
    Build a representative snippet from the document.
    Prefer first 2 pages as they usually contain header fields.
    """
    snippets = []
    for page in doc.pages[:2]:
        snippets.append(page.full_text[:max_chars // 2])
    return "\n\n".join(snippets)[:max_chars]


def llm_extract_field(
    field_result: FieldResult,
    field_config: FieldConfig,
    doc: NormalizedDocument,
    provider: Optional[LLMProvider] = None,
) -> FieldResult:
    """
    Attempt LLM extraction for a field.
    Returns updated FieldResult (may still be MISSING if LLM returns null).
    """
    if provider is None:
        provider = get_llm_provider()
    if provider is None:
        return field_result

    snippet = _build_snippet(doc)
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        field_name=field_config.name.replace("_", " ").title(),
        field_description=f"A {field_config.type} field",
        field_type=field_config.type,
        snippet=snippet,
    )

    logger.info("llm_fallback_triggered", field=field_config.name)
    result = provider.extract(prompt)

    if not result or result.get("value") is None:
        logger.info("llm_returned_null", field=field_config.name)
        return field_result

    llm_value = result["value"]
    llm_evidence = result.get("evidence", str(llm_value))

    logger.info("llm_extracted", field=field_config.name, value=llm_value)
    return FieldResult(
        field_name=field_config.name,
        value=llm_value,
        raw_value=str(llm_value),
        confidence=0.65,  # LLM results get moderate confidence
        status=ExtractionStatus.NEEDS_REVIEW,
        method=ExtractionMethod.LLM,
        source=None,
        evidence=Evidence(snippet=llm_evidence),
        alternatives=field_result.alternatives,
        validation_errors=field_result.validation_errors,
    )
