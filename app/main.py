"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.logging_config import setup_logging, get_logger
from app.storage.db import init_db
from app.storage.minio_client import ensure_bucket

setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize DB and MinIO bucket."""
    logger.info("startup_init")
    await init_db()
    logger.info("db_initialized")
    try:
        ensure_bucket()
        logger.info("minio_ready")
    except Exception as e:
        logger.warning("minio_init_warning", error=str(e))
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Document Extraction Framework",
    description="Multi-source document extraction with deterministic-first approach",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/status")
async def system_status():
    """
    Check health of all pipeline services:
    Redis, Celery workers, PostgreSQL, MinIO.
    """
    import asyncio
    from app.settings import get_settings

    settings = get_settings()

    results = {}

    # ── Redis ──────────────────────────────────────────────────────────────────
    try:
        import redis

        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        results["redis"] = {"status": "ok"}
    except Exception as e:
        results["redis"] = {"status": "error", "detail": str(e)}

    # ── Celery workers ─────────────────────────────────────────────────────────
    try:
        from app.workers.celery_app import celery_app

        inspect = celery_app.control.inspect(timeout=3)
        active = inspect.active() or {}
        stats = inspect.stats() or {}
        workers = []
        for w_name, w_stats in stats.items():
            active_tasks = len(active.get(w_name, []))
            workers.append(
                {
                    "name": w_name,
                    "active_tasks": active_tasks,
                    "pool": w_stats.get("pool", {}).get("implementation", "unknown"),
                    "concurrency": w_stats.get("pool", {}).get("max-concurrency", "?"),
                }
            )
        results["celery"] = {
            "status": "ok" if workers else "no_workers",
            "workers": workers,
        }
    except Exception as e:
        results["celery"] = {"status": "error", "detail": str(e)}

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    try:
        from app.storage.db import engine

        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        results["postgres"] = {"status": "ok"}
    except Exception as e:
        results["postgres"] = {"status": "error", "detail": str(e)}

    # ── MinIO ──────────────────────────────────────────────────────────────────
    try:
        from app.storage.minio_client import get_client
        from app.settings import get_settings as _s

        s = _s()
        client = get_client()
        client.bucket_exists(s.minio_bucket)
        results["minio"] = {"status": "ok"}
    except Exception as e:
        results["minio"] = {"status": "error", "detail": str(e)}

    overall = "ok" if all(v["status"] == "ok" for v in results.values()) else "degraded"
    results["overall"] = overall
    return results
