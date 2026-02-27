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
