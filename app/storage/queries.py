"""
Async DB queries for documents, jobs, field results, artifacts.
"""
import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.db import Document, Job, FieldResultDB, Artifact, get_session_factory
from app.models import ExtractionResult, JobStatus


async def create_document(
    document_type: str,
    original_filename: Optional[str] = None,
    supplier_id: Optional[str] = None,
) -> str:
    doc_id = str(uuid.uuid4())
    factory = get_session_factory()
    async with factory() as session:
        doc = Document(
            id=doc_id,
            supplier_id=supplier_id,
            document_type=document_type,
            original_filename=original_filename,
        )
        session.add(doc)
        await session.commit()
    return doc_id


async def create_job(document_id: str) -> str:
    job_id = str(uuid.uuid4())
    factory = get_session_factory()
    async with factory() as session:
        job = Job(id=job_id, document_id=document_id, status="pending")
        session.add(job)
        await session.commit()
    return job_id


async def update_job_status(job_id: str, status: str, error: Optional[str] = None):
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=status, error=error, updated_at=datetime.utcnow())
        )
        await session.commit()


async def get_job(job_id: str) -> Optional[dict]:
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return None
        return {
            "id": job.id,
            "document_id": job.document_id,
            "status": job.status,
            "error": job.error,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        }


async def save_field_results(job_id: str, result: ExtractionResult):
    factory = get_session_factory()
    async with factory() as session:
        for fname, fr in result.fields.items():
            db_fr = FieldResultDB(
                job_id=job_id,
                field_name=fname,
                value=str(fr.value) if fr.value is not None else None,
                confidence=fr.confidence,
                status=fr.status.value if fr.status else "missing",
                method=fr.method.value if fr.method else None,
                page=str(fr.source.page) if fr.source else None,
                bbox_json=fr.source.bbox if fr.source else None,
                evidence=fr.evidence.snippet if fr.evidence else None,
            )
            session.add(db_fr)
        await session.commit()


async def save_artifact(document_id: str, kind: str, url: str, meta: Optional[dict] = None):
    factory = get_session_factory()
    async with factory() as session:
        artifact = Artifact(
            document_id=document_id,
            kind=kind,
            url=url,
            meta_json=meta,
        )
        session.add(artifact)
        await session.commit()


async def get_artifacts(document_id: str) -> list[dict]:
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Artifact).where(Artifact.document_id == document_id)
        )
        artifacts = result.scalars().all()
        return [
            {"kind": a.kind, "url": a.url, "meta": a.meta_json}
            for a in artifacts
        ]


async def get_latest_result(document_id: str) -> Optional[dict]:
    """Get most recent completed job's field results for a document."""
    factory = get_session_factory()
    async with factory() as session:
        # Get latest completed job
        result = await session.execute(
            select(Job)
            .where(Job.document_id == document_id, Job.status == "completed")
            .order_by(Job.updated_at.desc())
            .limit(1)
        )
        job = result.scalar_one_or_none()
        if not job:
            return None

        # Get field results
        fr_result = await session.execute(
            select(FieldResultDB).where(FieldResultDB.job_id == job.id)
        )
        frs = fr_result.scalars().all()
        fields = {}
        for fr in frs:
            fields[fr.field_name] = {
                "value": fr.value,
                "confidence": fr.confidence,
                "status": fr.status,
                "method": fr.method,
                "page": fr.page,
                "bbox": fr.bbox_json,
                "evidence": fr.evidence,
            }
        return {"job_id": job.id, "document_id": document_id, "fields": fields}
