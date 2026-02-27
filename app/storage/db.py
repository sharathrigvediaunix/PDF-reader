"""
SQLAlchemy ORM models + async/sync engine setup.

- FastAPI uses async engine/session (asyncpg).
- Celery worker should use sync engine/session (psycopg / psycopg2) for reliability.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.settings import get_settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    supplier_id = Column(String, nullable=True)
    document_type = Column(String, nullable=False)
    original_filename = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    status = Column(String, default="pending")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FieldResultDB(Base):
    __tablename__ = "field_results"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    field_name = Column(String, nullable=False)
    value = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    status = Column(String, default="missing")
    method = Column(String, nullable=True)
    page = Column(String, nullable=True)
    bbox_json = Column(JSON, nullable=True)
    evidence = Column(Text, nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    kind = Column(String, nullable=False)
    url = Column(Text, nullable=False)
    meta_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─── Engines + session factories ──────────────────────────────────────────────

_async_engine = None
_async_session_factory = None

_sync_engine = None
_sync_session_factory = None


def _to_sync_db_url(async_url: str) -> str:
    """
    Convert postgresql+asyncpg://... -> postgresql+psycopg://...
    If your project uses psycopg2 instead, swap to postgresql+psycopg2://
    """
    if async_url.startswith("postgresql+asyncpg://"):
        return async_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    # If user provided a non-asyncpg URL already, return as-is
    return async_url


def get_async_engine():
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(settings.database_url, echo=False, future=True)
    return _async_engine


def get_async_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = sessionmaker(
            get_async_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


def get_sync_engine():
    """
    Celery-safe sync engine. Uses psycopg driver by default.
    """
    global _sync_engine
    if _sync_engine is None:
        settings = get_settings()
        sync_url = _to_sync_db_url(settings.database_url)
        _sync_engine = create_engine(sync_url, echo=False, future=True, pool_pre_ping=True)
    return _sync_engine


def get_sync_session_factory():
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_sync_engine(), class_=Session, expire_on_commit=False
        )
    return _sync_session_factory


# Backward-compatible names (your existing API code imports these)
def get_engine():
    return get_async_engine()


def get_session_factory():
    return get_async_session_factory()


async def init_db():
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)