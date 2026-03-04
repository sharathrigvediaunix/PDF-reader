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
    LAYOUTLM = "layoutlm"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DRIFT = "drift"  # anchor/template mismatch detected


# ─── Layout structures ───────────────────────────────────────────────────────


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0

    def to_dict(self) -> dict:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BBox":
        return cls(**d)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Token:
    text: str
    bbox: BBox
    line_id: int
    conf: float = 1.0  # OCR confidence if applicable


@dataclass
class LayoutLine:
    """A group of tokens on the same visual line (by Y-proximity)."""

    line_id: int
    tokens: list[Token]
    y_center: float  # representative Y center of the line

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def x_start(self) -> float:
        return min(t.bbox.x0 for t in self.tokens) if self.tokens else 0.0


@dataclass
class LayoutColumn:
    """A detected column boundary (by X-alignment of tokens)."""

    col_id: int
    x_center: float
    x_min: float
    x_max: float
    header_text: Optional[str] = None  # filled during table detection


@dataclass
class NormalizedPage:
    page_no: int
    tokens: list[Token]
    full_text: str
    ocr_used: bool
    image_path: Optional[str] = None  # MinIO path
    pil_image: Optional[Any] = None  # PIL.Image for LayoutLM (in-memory, not persisted)
    layout_lines: list[LayoutLine] = field(default_factory=list)
    layout_columns: list[LayoutColumn] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    document_id: str
    pages: list[NormalizedPage]
    doc_type: Optional[str] = None  # detected document type
    party_id: Optional[str] = None  # detected supplier/party

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.full_text for p in self.pages)

    @property
    def all_tokens(self) -> list[Token]:
        return [t for p in self.pages for t in p.tokens]


# ─── Extraction result models ─────────────────────────────────────────────────


class SourceLocation(BaseModel):
    page: int
    bbox: Optional[dict] = None


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


class DriftReport(BaseModel):
    """Generated when template anchors fail to match."""

    supplier_id: Optional[str] = None
    document_type: str
    missing_anchors: list[str] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    flagged_at: Optional[str] = None


class ExtractionResult(BaseModel):
    document_id: str
    job_id: str
    supplier_id: Optional[str] = None
    document_type: str
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    validation_summary: Optional[ValidationSummary] = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    debug_trace: Optional[dict] = None
    drift_report: Optional[DriftReport] = None


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
