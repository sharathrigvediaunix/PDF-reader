"""
File type detection and routing to correct extraction path.
After extraction, all pages pass through layout normalization
to produce LayoutLines and LayoutColumns for spatial extraction.
"""

import io
from typing import Optional

import magic
from pdf2image import convert_from_bytes
from PIL import Image

from app.models import NormalizedDocument, NormalizedPage
from app.normalization.pdf_extractor import has_text_layer, extract_pdf_text
from app.normalization.ocr_pipeline import ocr_image
from app.normalization.layout_normalizer import normalize_layout
from app.logging_config import get_logger

logger = get_logger("file_router")

SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}


def detect_file_type(data: bytes) -> str:
    """Use python-magic to detect MIME type from file bytes."""
    return magic.from_buffer(data[:2048], mime=True)


def normalize_document(
    file_bytes: bytes,
    document_id: str,
    filename: Optional[str] = None,
) -> NormalizedDocument:
    """
    Route file to correct extraction pipeline:
    - PDF with text layer  -> pdfminer extraction
    - PDF without text layer (scanned) -> rasterize + PaddleOCR
    - Image -> PaddleOCR directly

    All paths then run through layout normalization to produce
    LayoutLines and LayoutColumns on each page.
    """
    mime = detect_file_type(file_bytes)
    logger.info(
        "detected_file_type", document_id=document_id, mime=mime, filename=filename
    )

    if mime == "application/pdf":
        doc = _process_pdf(file_bytes, document_id)
    elif mime in SUPPORTED_IMAGE_TYPES:
        doc = _process_image(file_bytes, document_id)
    else:
        logger.warning("unknown_mime_type", mime=mime, document_id=document_id)
        try:
            doc = _process_pdf(file_bytes, document_id)
        except Exception:
            doc = _process_image(file_bytes, document_id)

    # ── Layout normalization (runs on ALL pages regardless of source) ─────────
    for page in doc.pages:
        normalize_layout(page)

    logger.info("layout_normalized", document_id=document_id, pages=len(doc.pages))
    return doc


def _process_pdf(pdf_bytes: bytes, document_id: str) -> NormalizedDocument:
    """Process PDF: extract text if available, else PaddleOCR."""
    if has_text_layer(pdf_bytes):
        logger.info("pdf_has_text_layer", document_id=document_id)
        doc = extract_pdf_text(pdf_bytes, document_id)
        if doc.pages and any(p.tokens for p in doc.pages):
            return doc
        logger.warning("pdf_text_empty_fallback_ocr", document_id=document_id)

    # Rasterize PDF pages and run PaddleOCR
    logger.info("rasterizing_pdf_for_ocr", document_id=document_id)
    pages = convert_from_bytes(pdf_bytes, dpi=300)  # 300 DPI per design doc spec
    normalized_pages = []
    for page_no, pil_page in enumerate(pages):
        npage = ocr_image(pil_page.convert("RGB"), page_no=page_no)
        normalized_pages.append(npage)

    return NormalizedDocument(document_id=document_id, pages=normalized_pages)


def _process_image(image_bytes: bytes, document_id: str) -> NormalizedDocument:
    """Process a single image file via PaddleOCR."""
    pil = Image.open(io.BytesIO(image_bytes))
    npage = ocr_image(pil, page_no=0)
    return NormalizedDocument(document_id=document_id, pages=[npage])
