"""
SQLAlchemy ORM models + async engine setup.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, DateTime, Text, ForeignKey, JSON, func
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

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
    kind = Column(String, nullable=False)  # "original", "page_image", "ocr_json", "debug_trace"
    url = Column(Text, nullable=False)
    meta_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─── Engine + session ─────────────────────────────────────────────────────────

_engine = None
_async_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False, future=True)
    return _engine


def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


async def init_db():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
