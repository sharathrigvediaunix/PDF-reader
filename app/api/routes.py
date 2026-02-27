"""
FastAPI API routes:
- POST /v1/extract
- GET  /v1/jobs/{job_id}
- GET  /v1/documents/{document_id}/result
- GET  /v1/documents/{document_id}/artifacts
"""
import base64
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models import JobResponse, UploadResponse, JobStatus
from app.logging_config import get_logger
from app.storage.queries import (
    create_document, create_job, get_job, get_artifacts, get_latest_result
)

logger = get_logger("api")
router = APIRouter()

SUPPORTED_DOC_TYPES = {"invoice", "receipt", "purchase_order"}


@router.post("/v1/extract", response_model=UploadResponse)
async def extract_document(
    file: UploadFile = File(...),
    document_type: str = Form(default="invoice"),
    supplier_id: Optional[str] = Form(default=None),
):
    """
    Upload a document and start async extraction.
    Returns job_id and document_id immediately.
    """
    if document_type not in SUPPORTED_DOC_TYPES:
        # Still allow it — config system will error if no yaml found
        logger.warning("unknown_document_type", document_type=document_type)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Create DB records
    document_id = await create_document(
        document_type=document_type,
        original_filename=file.filename,
        supplier_id=supplier_id,
    )
    job_id = await create_job(document_id)

    logger.info("extraction_submitted",
                document_id=document_id, job_id=job_id,
                document_type=document_type, filename=file.filename)

    # Dispatch Celery task
    from app.workers.celery_app import extract_task
    extract_task.delay(
        file_b64=base64.b64encode(file_bytes).decode(),
        document_id=document_id,
        job_id=job_id,
        document_type=document_type,
        supplier_id=supplier_id,
        filename=file.filename,
    )

    return UploadResponse(
        job_id=job_id,
        document_id=document_id,
        message="Extraction job submitted",
    )


@router.get("/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get job status and result if completed."""
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
    """Get latest extraction result for a document."""
    result = await get_latest_result(document_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No completed extraction found for document {document_id}"
        )
    return result


@router.get("/v1/documents/{document_id}/artifacts")
async def get_document_artifacts(document_id: str):
    """List artifacts for a document."""
    artifacts = await get_artifacts(document_id)
    return {"document_id": document_id, "artifacts": artifacts}
