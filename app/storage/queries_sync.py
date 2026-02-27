from datetime import datetime
from typing import Optional

from sqlalchemy import update
from app.storage.db import Job, FieldResultDB, Artifact, get_sync_session_factory

def update_job_status_sync(job_id: str, status: str, error: Optional[str] = None) -> None:
    factory = get_sync_session_factory()
    with factory() as session:
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=status, error=error, updated_at=datetime.utcnow())
        )
        session.commit()

def save_field_results_sync(job_id: str, result) -> None:
    factory = get_sync_session_factory()
    with factory() as session:
        rows = []
        for fname, fr in result.fields.items():
            rows.append(FieldResultDB(
                job_id=job_id,
                field_name=fname,
                value=str(fr.value) if fr.value is not None else None,
                confidence=fr.confidence,
                status=fr.status.value if fr.status else "missing",
                method=fr.method.value if fr.method else None,
                page=str(fr.source.page) if fr.source else None,
                bbox_json=fr.source.bbox if fr.source else None,
                evidence=fr.evidence.snippet if fr.evidence else None,
            ))
        session.add_all(rows)
        session.commit()

def save_artifact_sync(document_id: str, kind: str, url: str, meta: Optional[dict] = None) -> None:
    factory = get_sync_session_factory()
    with factory() as session:
        session.add(Artifact(document_id=document_id, kind=kind, url=url, meta_json=meta))
        session.commit()