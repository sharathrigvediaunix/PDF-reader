"""
FastAPI API routes:
- GET  /v1/suppliers/{party_id}/check    — does a template exist for this supplier?
- POST /v1/suppliers/{party_id}/template — upload + save a YAML template
- GET  /v1/suppliers/{party_id}/fields   — list available fields from template
- POST /v1/extract                       — upload doc + specify fields to extract
- GET  /v1/jobs/{job_id}
- GET  /v1/documents/{document_id}/result
- GET  /v1/documents/{document_id}/artifacts
"""

import base64
import json
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models import JobResponse, UploadResponse, JobStatus
from app.logging_config import get_logger
from app.storage.queries import (
    create_document,
    create_job,
    get_job,
    get_artifacts,
    get_latest_result,
)
from app.config.loader import (
    load_config,
    get_available_fields,
    identify_party,
    identify_document_type,
    SUPPLIERS_DIR,
)

logger = get_logger("api")
router = APIRouter()


# ── Supplier / template endpoints ─────────────────────────────────────────────


@router.get("/v1/suppliers/{party_id}/check")
async def check_supplier_template(party_id: str):
    """Return whether a template YAML exists for this supplier."""
    path = os.path.join(SUPPLIERS_DIR, f"{party_id}.yaml")
    exists = os.path.exists(path)
    return {"party_id": party_id, "has_template": exists}


@router.post("/v1/suppliers/{party_id}/template")
async def upload_supplier_template(party_id: str, file: UploadFile = File(...)):
    """
    Save a supplier YAML template.
    After saving, the loader cache is cleared so it picks up the new file
    immediately on the next extraction.
    """
    if not file.filename.endswith(".yaml") and not file.filename.endswith(".yml"):
        raise HTTPException(status_code=400, detail="File must be a .yaml file")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Basic YAML validity check
    import yaml

    try:
        parsed = yaml.safe_load(content)
        if not isinstance(parsed, dict):
            raise ValueError("Not a valid mapping")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    os.makedirs(SUPPLIERS_DIR, exist_ok=True)
    path = os.path.join(SUPPLIERS_DIR, f"{party_id}.yaml")
    with open(path, "wb") as f:
        f.write(content)

    # Clear lru_cache so the new template is picked up immediately
    load_config.cache_clear()

    fields = get_available_fields(party_id)
    logger.info("template_saved", party_id=party_id, fields=fields)
    return {
        "party_id": party_id,
        "saved": True,
        "available_fields": fields,
    }


@router.get("/v1/suppliers/{party_id}/fields")
async def get_supplier_fields(party_id: str, document_type: str = "invoice"):
    """List all extractable fields defined in a supplier's template."""
    fields = get_available_fields(party_id, document_type)
    if not fields:
        raise HTTPException(
            status_code=404, detail=f"No template found for supplier '{party_id}'"
        )
    return {"party_id": party_id, "fields": fields}


@router.post("/v1/suppliers/detect")
async def detect_supplier(file: UploadFile = File(...)):
    """
    Run supplier + doc-type detection on an uploaded file without starting
    extraction. Used by Streamlit to check template existence before asking
    the user for fields.
    """
    from app.normalization import normalize_document

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    norm_doc = normalize_document(file_bytes, "detect", file.filename)
    first_page_text = norm_doc.pages[0].full_text if norm_doc.pages else ""

    party_id, party_conf = identify_party(first_page_text)
    doc_type, type_conf = identify_document_type(first_page_text)

    has_template = False
    available_fields = []
    if party_id:
        path = os.path.join(SUPPLIERS_DIR, f"{party_id}.yaml")
        has_template = os.path.exists(path)
        if has_template:
            available_fields = get_available_fields(party_id, doc_type)

    return {
        "party_id": party_id,
        "party_confidence": party_conf,
        "document_type": doc_type,
        "type_confidence": type_conf,
        "has_template": has_template,
        "available_fields": available_fields,
    }


# ── Extraction endpoints ──────────────────────────────────────────────────────


@router.post("/v1/extract", response_model=UploadResponse)
async def extract_document(
    file: UploadFile = File(...),
    document_type: str = Form(default="invoice"),
    supplier_id: Optional[str] = Form(default=None),
    requested_fields: Optional[str] = Form(default=None),  # JSON array string
):
    """
    Upload a document and start async extraction.
    requested_fields is a JSON array of field names e.g. '["invoice_number","total_amount_usd"]'
    If omitted, all fields in the supplier template are extracted.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    fields_list: Optional[list[str]] = None
    if requested_fields:
        try:
            fields_list = json.loads(requested_fields)
        except Exception:
            raise HTTPException(
                status_code=400, detail="requested_fields must be a JSON array"
            )

    document_id = await create_document(
        document_type=document_type,
        original_filename=file.filename,
        supplier_id=supplier_id,
    )
    job_id = await create_job(document_id)

    logger.info(
        "extraction_submitted",
        document_id=document_id,
        job_id=job_id,
        document_type=document_type,
        filename=file.filename,
        requested_fields=fields_list,
    )

    from app.workers.celery_app import extract_task

    extract_task.delay(
        file_b64=base64.b64encode(file_bytes).decode(),
        document_id=document_id,
        job_id=job_id,
        document_type=document_type,
        supplier_id=supplier_id,
        filename=file.filename,
        requested_fields=fields_list,
    )

    return UploadResponse(
        job_id=job_id,
        document_id=document_id,
        message="Extraction job submitted",
    )


@router.get("/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = JobStatus(job["status"])
    response = JobResponse(
        job_id=job_id,
        document_id=job.get("document_id"),
        status=status,
    )
    if status == JobStatus.FAILED:
        response.error = job.get("error")
    if status == JobStatus.COMPLETED and job.get("document_id"):
        latest = await get_latest_result(job["document_id"])
        if latest:
            response.result = latest  # type: ignore
    return response


@router.get("/v1/documents/{document_id}/result")
async def get_document_result(document_id: str):
    result = await get_latest_result(document_id)
    if not result:
        raise HTTPException(
            status_code=404, detail=f"No result for document {document_id}"
        )
    return result


@router.get("/v1/documents/{document_id}/artifacts")
async def get_document_artifacts(document_id: str):
    artifacts = await get_artifacts(document_id)
    return {"document_id": document_id, "artifacts": artifacts}
