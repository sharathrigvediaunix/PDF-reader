"""
Main extraction pipeline orchestrator.
Ties together: normalization -> deterministic extraction -> validation -> LLM fallback -> persistence.

IMPORTANT:
- This orchestrator is designed to be called from the Celery worker.
- Therefore, ALL DB writes (artifacts + field_results) use SYNC SQLAlchemy (queries_sync.py)
  to avoid asyncpg/concurrency issues inside Celery (prefork).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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


class PipelineFileLogger:
    """Appends all pipeline runs to a single pipeline_results.log, one JSON line per stage."""

    def __init__(self, job_id: str, document_id: str, log_dir: str):
        self.job_id = job_id
        self.document_id = document_id
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._path = path / "pipeline_results.log"
        self._file = self._path.open("a", encoding="utf-8")

    def log_stage(self, stage: str, data: dict):
        entry = {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def finalize(self):
        self._file.close()
        try:
            log_bytes = self._path.read_bytes()
            upload_bytes("logs/pipeline_results.log", log_bytes, content_type="application/x-ndjson")
        except Exception:
            pass


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
    flog = PipelineFileLogger(job_id, document_id, settings.log_dir)

    # Stored only if settings.debug is True
    debug_trace: dict[str, dict] = {}

    try:
        # ── 1. Upload original file ────────────────────────────────────────────
        obj_key = f"{document_id}/original/{filename or 'document'}"
        url = upload_bytes(obj_key, file_bytes)
        save_artifact_sync(document_id, "original", url, {"filename": filename})
        log.info("original_uploaded", url=url)
        flog.log_stage("upload", {"url": url, "file_size_bytes": len(file_bytes), "filename": filename})

        # ── 2. Normalize document ──────────────────────────────────────────────
        log.info("normalization_start")
        norm_doc = normalize_document(file_bytes, document_id, filename)
        log.info("normalization_done", pages=len(norm_doc.pages))
        flog.log_stage("normalization", {
            "pages": len(norm_doc.pages),
            "total_tokens": sum(len(p.tokens) for p in norm_doc.pages),
            "tokens_by_page": [
                {
                    "page": p.page_no,
                    "ocr_used": p.ocr_used,
                    "token_count": len(p.tokens),
                    "tokens": [
                        {"text": t.text, "bbox": t.bbox.to_dict(), "line_id": t.line_id, "conf": t.conf}
                        for t in p.tokens
                    ],
                }
                for p in norm_doc.pages
            ],
        })

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
        flog.log_stage("config", {"document_type": document_type, "field_names": list(doc_config.fields.keys())})

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

            if fr.value is not None:
                errors = validate_value(fr.value, fconfig)
                fr.validation_errors = errors
                if errors:
                    log.debug("field_validation_errors", field=fname, errors=errors)

            extracted_fields[fname] = fr
            flog.log_stage("extraction", {
                "field": fname,
                "value": fr.value,
                "confidence": fr.confidence,
                "status": fr.status.value,
                "method": fr.method.value if fr.method else None,
                "validation_errors": fr.validation_errors,
                "candidates": [
                    {"value": c.value, "confidence": c.confidence, "method": c.method.value}
                    for c in (fr.alternatives or [])[:3]
                ],
            })

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
                    if fname in debug_trace:
                        debug_trace[fname]["llm_used"] = True
                    flog.log_stage("llm_fallback", {
                        "field": fname,
                        "value": fr2.value,
                        "confidence": fr2.confidence,
                        "status": fr2.status.value,
                    })

        # ── 6. Compute validation summary ──────────────────────────────────────
        summary = compute_validation_summary(extracted_fields, doc_config)
        log.info(
            "validation_done",
            required_present=summary.required_present,
            overall_confidence=summary.overall_confidence,
        )
        flog.log_stage("validation", {
            "required_present": summary.required_present,
            "required_missing": summary.required_missing,
            "overall_confidence": summary.overall_confidence,
            "cross_field_errors": summary.cross_field_errors,
        })

        # ── 7. Build result ────────────────────────────────────────────────────
        result = ExtractionResult(
            document_id=document_id,
            job_id=job_id,
            supplier_id=supplier_id,
            document_type=document_type,
            fields=extracted_fields,
            validation_summary=summary,
        )
        flog.log_stage("result_build", {
            "field_count": len(extracted_fields),
            "fields": {
                name: {"value": fr.value, "status": fr.status.value, "confidence": fr.confidence}
                for name, fr in extracted_fields.items()
            },
        })

        # Upload debug trace (optional)
        if settings.debug:
            debug_key = f"{document_id}/debug/debug_trace_{job_id}.json"
            debug_url = upload_json(debug_key, debug_trace)
            save_artifact_sync(document_id, "debug_trace", debug_url, {"job_id": job_id})
            result.debug_trace = debug_trace

        # ── 8. Persist field results (SYNC, Celery-safe) ───────────────────────
        try:
            save_field_results_sync(job_id, result)
            flog.log_stage("persistence", {"success": True})
        except Exception as persist_err:
            flog.log_stage("persistence", {"success": False, "error": str(persist_err)})
            raise

        log.info("pipeline_complete", fields_extracted=len(extracted_fields))
        flog.finalize()
        return result

    except Exception as e:
        log.error("pipeline_error", error=str(e), exc_info=True)
        flog.finalize()
        raise