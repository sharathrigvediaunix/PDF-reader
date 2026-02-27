"""
Canonical data models for the document extraction framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────

class ExtractionStatus(str, Enum):
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    MISSING = "missing"


class ExtractionMethod(str, Enum):
    ANCHOR = "anchor"
    REGEX = "regex"
    LAYOUT = "layout"
    LLM = "llm"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── Token / Page models ─────────────────────────────────────────────────────

@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "page": self.page}

    @classmethod
    def from_dict(cls, d: dict) -> "BBox":
        return cls(**d)


@dataclass
class Token:
    text: str
    bbox: BBox
    line_id: int
    conf: float = 1.0  # OCR confidence if applicable


@dataclass
class NormalizedPage:
    page_no: int
    tokens: list[Token]
    full_text: str
    ocr_used: bool
    image_path: Optional[str] = None  # MinIO path


@dataclass
class NormalizedDocument:
    document_id: str
    pages: list[NormalizedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.full_text for p in self.pages)

    @property
    def all_tokens(self) -> list[Token]:
        return [t for p in self.pages for t in p.tokens]


# ─── Extraction result models ─────────────────────────────────────────────────

class SourceLocation(BaseModel):
    page: int
    bbox: Optional[dict] = None  # serialized BBox


class Evidence(BaseModel):
    snippet: str
    context: Optional[str] = None


class FieldCandidate(BaseModel):
    value: Any
    raw_value: str
    confidence: float
    method: ExtractionMethod
    source: Optional[SourceLocation] = None
    evidence: Optional[Evidence] = None


class FieldResult(BaseModel):
    field_name: str
    value: Any = None
    raw_value: Optional[str] = None
    confidence: float = 0.0
    status: ExtractionStatus = ExtractionStatus.MISSING
    method: Optional[ExtractionMethod] = None
    source: Optional[SourceLocation] = None
    evidence: Optional[Evidence] = None
    alternatives: list[FieldCandidate] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class ValidationSummary(BaseModel):
    required_present: bool
    required_missing: list[str] = Field(default_factory=list)
    cross_field_errors: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0


class ExtractionResult(BaseModel):
    document_id: str
    job_id: str
    supplier_id: Optional[str] = None
    document_type: str
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    validation_summary: Optional[ValidationSummary] = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    debug_trace: Optional[dict] = None


# ─── API response models ──────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    document_id: Optional[str] = None
    status: JobStatus
    result: Optional[ExtractionResult] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    job_id: str
    document_id: str
    message: str = "Extraction job submitted"
