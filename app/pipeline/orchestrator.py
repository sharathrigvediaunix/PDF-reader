"""
Main extraction pipeline orchestrator.
Ties together: normalization -> deterministic extraction -> validation -> LLM fallback -> persistence.

IMPORTANT:
- This orchestrator is designed to be called from the Celery worker.
- Therefore, ALL DB writes (artifacts + field_results) use SYNC SQLAlchemy (queries_sync.py)
  to avoid asyncpg/concurrency issues inside Celery (prefork).
"""
from __future__ import annotations

from typing import Optional

from app.config import load_config
from app.extractors.deterministic import extract_field
from app.fallback.gate import should_use_llm, llm_extract_field
from app.fallback.llm_provider import get_llm_provider
from app.models import ExtractionResult, FieldResult
from app.normalization import normalize_document
from app.settings import get_settings
from app.storage import upload_bytes, upload_json
from app.storage.queries_sync import save_artifact_sync, save_field_results_sync
from app.validators import normalize_value, validate_value, compute_validation_summary
from app.logging_config import get_logger

logger = get_logger("pipeline")


def run_extraction_pipeline(
    file_bytes: bytes,
    document_id: str,
    job_id: str,
    document_type: str,
    supplier_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> ExtractionResult:
    """
    Full end-to-end extraction pipeline.
    1) Upload original file to MinIO
    2) Normalize document (PDF text or OCR)
    3) Extract fields deterministically
    4) Validate + score
    5) LLM fallback for low-confidence/missing required fields
    6) Persist results (SYNC DB writes)
    7) Return ExtractionResult
    """
    settings = get_settings()
    log = logger.bind(job_id=job_id, document_id=document_id, document_type=document_type)

    # Stored only if settings.debug is True
    debug_trace: dict[str, dict] = {}

    try:
        # ── 1. Upload original file ────────────────────────────────────────────
        obj_key = f"{document_id}/original/{filename or 'document'}"
        url = upload_bytes(obj_key, file_bytes)
        save_artifact_sync(document_id, "original", url, {"filename": filename})
        log.info("original_uploaded", url=url)

        # ── 2. Normalize document ──────────────────────────────────────────────
        log.info("normalization_start")
        norm_doc = normalize_document(file_bytes, document_id, filename)
        log.info("normalization_done", pages=len(norm_doc.pages))

        # Upload OCR outputs (only when OCR used)
        for page in norm_doc.pages:
            if page.ocr_used:
                ocr_data = {
                    "page_no": page.page_no,
                    "tokens": [
                        {"text": t.text, "bbox": t.bbox.to_dict(), "conf": t.conf}
                        for t in page.tokens
                    ],
                    "full_text": page.full_text,
                }
                ocr_key = f"{document_id}/ocr/page_{page.page_no}.json"
                ocr_url = upload_json(ocr_key, ocr_data)
                save_artifact_sync(document_id, "ocr_json", ocr_url, {"page": page.page_no})

        # ── 3. Load field config ───────────────────────────────────────────────
        doc_config = load_config(document_type)

        # ── 4. Deterministic extraction ────────────────────────────────────────
        log.info("extraction_start", fields=list(doc_config.fields.keys()))
        extracted_fields: dict[str, FieldResult] = {}

        for fname, fconfig in doc_config.fields.items():
            log.debug("extracting_field", field=fname)

            fr = extract_field(
                norm_doc,
                fconfig,
                normalizer_fn=normalize_value,
                validator_fn=validate_value,
            )

            # Validate final value again (kept from your original code)
            if fr.value is not None:
                errors = validate_value(fr.value, fconfig)
                fr.validation_errors = errors
                if errors:
                    log.debug("field_validation_errors", field=fname, errors=errors)

            extracted_fields[fname] = fr

            # Lightweight per-field trace (only uploaded if settings.debug)
            debug_trace[fname] = {
                "method": fr.method.value if fr.method else None,
                "confidence": fr.confidence,
                "status": fr.status.value,
                "candidates": [
                    {
                        "value": c.value,
                        "confidence": c.confidence,
                        "method": c.method.value,
                    }
                    for c in (fr.alternatives or [])[:3]
                ],
            }

        # ── 5. LLM fallback ────────────────────────────────────────────────────
        if settings.llm_fallback_enabled:
            llm_provider = get_llm_provider()
            for fname, fconfig in doc_config.fields.items():
                fr = extracted_fields[fname]
                if should_use_llm(fr, fconfig):
                    log.info("llm_fallback", field=fname)
                    fr2 = llm_extract_field(fr, fconfig, norm_doc, provider=llm_provider)
                    extracted_fields[fname] = fr2
                    # keep trace consistent
                    if fname in debug_trace:
                        debug_trace[fname]["llm_used"] = True

        # ── 6. Compute validation summary ──────────────────────────────────────
        summary = compute_validation_summary(extracted_fields, doc_config)
        log.info(
            "validation_done",
            required_present=summary.required_present,
            overall_confidence=summary.overall_confidence,
        )

        # ── 7. Build result ────────────────────────────────────────────────────
        result = ExtractionResult(
            document_id=document_id,
            job_id=job_id,
            supplier_id=supplier_id,
            document_type=document_type,
            fields=extracted_fields,
            validation_summary=summary,
        )

        # Upload debug trace (optional)
        if settings.debug:
            debug_key = f"{document_id}/debug/debug_trace_{job_id}.json"
            debug_url = upload_json(debug_key, debug_trace)
            save_artifact_sync(document_id, "debug_trace", debug_url, {"job_id": job_id})
            result.debug_trace = debug_trace

        # ── 8. Persist field results (SYNC, Celery-safe) ───────────────────────
        save_field_results_sync(job_id, result)

        log.info("pipeline_complete", fields_extracted=len(extracted_fields))
        return result

    except Exception as e:
        log.error("pipeline_error", error=str(e), exc_info=True)
        raise