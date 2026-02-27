"""
Celery worker tasks for document extraction (long-term stable).

- Celery tasks remain synchronous.
- DB updates use sync SQLAlchemy (psycopg/psycopg2) to avoid asyncpg issues.
"""
from celery import Celery

from app.settings import get_settings
from app.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("worker")

settings = get_settings()
celery_app = Celery(
    "docextract",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@celery_app.task(
    name="docextract.extract",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
)
def extract_task(
    self,
    file_b64: str,
    document_id: str,
    job_id: str,
    document_type: str,
    supplier_id: str = None,
    filename: str = None,
):
    import base64

    log = logger.bind(job_id=job_id, document_id=document_id)

    from app.pipeline.orchestrator import run_extraction_pipeline
    from app.storage.queries_sync import update_job_status_sync

    try:
        log.info("task_started")
        update_job_status_sync(job_id, "processing")

        file_bytes = base64.b64decode(file_b64)

        # Pipeline is sync -> keep it sync in Celery
        result = run_extraction_pipeline(
            file_bytes=file_bytes,
            document_id=document_id,
            job_id=job_id,
            document_type=document_type,
            supplier_id=supplier_id,
            filename=filename,
        )

        update_job_status_sync(job_id, "completed")
        log.info("task_completed")
        return result.model_dump(mode="json")

    except Exception as exc:
        log.error("task_failed", error=str(exc), exc_info=True)

        # Mark failed only after retries exhausted; otherwise keep it retrying.
        if self.request.retries >= self.max_retries:
            update_job_status_sync(job_id, "failed", error=str(exc))

        raise self.retry(exc=exc)