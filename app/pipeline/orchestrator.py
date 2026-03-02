"""
Main extraction pipeline orchestrator.
Upgraded with:
- Document type detection
- Party/supplier identification
- (doc_type × party) template resolution
- LayoutLM fallback (Option A) after deterministic extraction
- Drift detection — flags job when anchors fail or confidence drops
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.config.loader import load_config, identify_party, identify_document_type
from app.extractors.deterministic import extract_field
from app.extractors.layoutlm import layoutlm_extract_field, should_use_layoutlm
from app.fallback.gate import should_use_llm, llm_extract_field
from app.fallback.llm_provider import get_llm_provider
from app.models import (
    ExtractionResult,
    FieldResult,
    JobStatus,
    DriftReport,
    ExtractionStatus,
)
from app.normalization import normalize_document
from app.settings import get_settings
from app.storage import upload_bytes, upload_json
from app.storage.queries_sync import save_artifact_sync, save_field_results_sync
from app.validators import normalize_value, validate_value, compute_validation_summary
from app.logging_config import get_logger

logger = get_logger("pipeline")

# If overall confidence drops below this, flag as DRIFT
DRIFT_CONFIDENCE_THRESHOLD = 0.40

# Minimum fields that must be found to avoid drift flag
MIN_FIELDS_FOUND_RATIO = 0.50


def run_extraction_pipeline(
    file_bytes: bytes,
    document_id: str,
    job_id: str,
    document_type: Optional[str] = None,
    supplier_id: Optional[str] = None,
    filename: Optional[str] = None,
    requested_fields: Optional[list[str]] = None,  # user-selected fields to extract
) -> ExtractionResult:
    """
    Full end-to-end extraction pipeline.

    1) Upload original file to MinIO
    2) Normalize document (PDF text or PaddleOCR) + layout normalization
    3) Auto-detect document type (if not provided)
    4) Auto-detect party/supplier (if not provided)
    5) Load (doc_type × party) template config
    6) Deterministic spatial extraction
    7) LayoutLM fallback for low-confidence fields
    8) LLM fallback (if enabled) for still-missing required fields
    9) Validate + compute summary
    10) Drift detection
    11) Persist results
    """
    settings = get_settings()
    log = logger.bind(job_id=job_id, document_id=document_id)
    debug_trace: dict = {}

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

        # Upload OCR outputs for scanned pages
        for page in norm_doc.pages:
            if page.ocr_used:
                ocr_data = {
                    "page_no": page.page_no,
                    "tokens": [
                        {"text": t.text, "bbox": t.bbox.to_dict(), "conf": t.conf}
                        for t in page.tokens
                    ],
                    "full_text": page.full_text,
                    "layout_lines": len(page.layout_lines),
                    "layout_columns": len(page.layout_columns),
                }
                ocr_key = f"{document_id}/ocr/page_{page.page_no}.json"
                ocr_url = upload_json(ocr_key, ocr_data)
                save_artifact_sync(
                    document_id, "ocr_json", ocr_url, {"page": page.page_no}
                )

        # ── 3. Auto-detect document type ───────────────────────────────────────
        first_page_text = norm_doc.pages[0].full_text if norm_doc.pages else ""

        if not document_type:
            document_type, type_conf = identify_document_type(first_page_text)
            log.info(
                "document_type_detected", doc_type=document_type, confidence=type_conf
            )
        else:
            type_conf = 1.0

        norm_doc.doc_type = document_type

        # ── 4. Auto-detect party/supplier ─────────────────────────────────────
        if not supplier_id:
            party_id, party_conf = identify_party(first_page_text)
            if party_id:
                supplier_id = party_id
                log.info("party_detected", party_id=party_id, confidence=party_conf)
            else:
                log.info("party_not_detected_using_default")

        norm_doc.party_id = supplier_id

        # ── 5. Load (doc_type × party) template ───────────────────────────────
        doc_config = load_config(document_type, supplier_id)
        log.info("template_loaded", doc_type=document_type, party=supplier_id)

        # ── 6. Deterministic spatial extraction ───────────────────────────────
        # Only extract the fields the user requested; if none specified, extract all
        fields_to_extract = doc_config.fields_for_request(requested_fields)
        log.info("extraction_start", fields=list(fields_to_extract.keys()))
        extracted_fields: dict[str, FieldResult] = {}

        for fname, fconfig in fields_to_extract.items():
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
            extracted_fields[fname] = fr

            debug_trace[fname] = {
                "method": fr.method.value if fr.method else None,
                "confidence": fr.confidence,
                "status": fr.status.value,
            }

        # ── 7. LayoutLM fallback ───────────────────────────────────────────────
        layoutlm_used = False
        for fname, fconfig in fields_to_extract.items():
            fr = extracted_fields[fname]
            if should_use_layoutlm(fr, fconfig):
                log.info("layoutlm_fallback", field=fname, current_conf=fr.confidence)
                fr2 = layoutlm_extract_field(norm_doc, fconfig, fr)
                if fr2.confidence > fr.confidence:
                    extracted_fields[fname] = fr2
                    debug_trace[fname]["layoutlm_used"] = True
                    layoutlm_used = True

        if layoutlm_used:
            log.info("layoutlm_fallback_complete")

        # ── 8. LLM fallback (if enabled) ──────────────────────────────────────
        if settings.llm_fallback_enabled:
            llm_provider = get_llm_provider()
            for fname, fconfig in fields_to_extract.items():
                fr = extracted_fields[fname]
                if should_use_llm(fr, fconfig):
                    log.info("llm_fallback", field=fname)
                    fr2 = llm_extract_field(
                        fr, fconfig, norm_doc, provider=llm_provider
                    )
                    extracted_fields[fname] = fr2
                    debug_trace[fname]["llm_used"] = True

        # ── 9. Validation summary ──────────────────────────────────────────────
        summary = compute_validation_summary(extracted_fields, doc_config)
        log.info(
            "validation_done",
            required_present=summary.required_present,
            overall_confidence=summary.overall_confidence,
        )

        # ── 10. Drift detection ────────────────────────────────────────────────
        drift_report = _detect_drift(
            extracted_fields, doc_config, document_type, supplier_id, summary
        )
        if drift_report:
            log.warning(
                "drift_detected",
                missing=drift_report.missing_anchors,
                low_conf=drift_report.low_confidence_fields,
            )

        # ── 11. Build and persist result ───────────────────────────────────────
        result = ExtractionResult(
            document_id=document_id,
            job_id=job_id,
            supplier_id=supplier_id,
            document_type=document_type,
            fields=extracted_fields,
            validation_summary=summary,
            drift_report=drift_report,
        )

        if settings.debug:
            debug_key = f"{document_id}/debug/debug_trace_{job_id}.json"
            debug_url = upload_json(debug_key, debug_trace)
            save_artifact_sync(
                document_id, "debug_trace", debug_url, {"job_id": job_id}
            )
            result.debug_trace = debug_trace

        save_field_results_sync(job_id, result)
        log.info("pipeline_complete", fields_extracted=len(extracted_fields))
        return result

    except Exception as e:
        log.error("pipeline_error", error=str(e), exc_info=True)
        raise


def _detect_drift(
    extracted_fields: dict[str, FieldResult],
    doc_config,
    document_type: str,
    supplier_id: Optional[str],
    summary,
) -> Optional[DriftReport]:
    """
    Check if extraction results suggest template drift.
    Drift is flagged when:
    - Too many required fields are missing
    - Overall confidence drops below threshold
    - A significant portion of fields failed extraction
    """
    missing_anchors = [
        fname
        for fname, fr in extracted_fields.items()
        if fr.status == ExtractionStatus.MISSING and doc_config.fields[fname].required
    ]

    low_conf_fields = [
        fname
        for fname, fr in extracted_fields.items()
        if fr.confidence < DRIFT_CONFIDENCE_THRESHOLD
        and fr.status != ExtractionStatus.MISSING
    ]

    total_fields = len(extracted_fields)
    found_fields = sum(
        1 for fr in extracted_fields.values() if fr.status != ExtractionStatus.MISSING
    )
    found_ratio = found_fields / total_fields if total_fields > 0 else 0.0

    is_drift = (
        len(missing_anchors) > 0
        or summary.overall_confidence < DRIFT_CONFIDENCE_THRESHOLD
        or found_ratio < MIN_FIELDS_FOUND_RATIO
    )

    if not is_drift:
        return None

    return DriftReport(
        supplier_id=supplier_id,
        document_type=document_type,
        missing_anchors=missing_anchors,
        low_confidence_fields=low_conf_fields,
        confidence_scores={
            fname: fr.confidence for fname, fr in extracted_fields.items()
        },
        flagged_at=datetime.now(timezone.utc).isoformat(),
    )
