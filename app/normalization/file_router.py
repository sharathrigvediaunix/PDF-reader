"""
File type detection and routing to correct extraction path.
"""
import io
import os
import tempfile
from typing import Optional

import magic
from pdf2image import convert_from_bytes
from PIL import Image

from app.models import NormalizedDocument, NormalizedPage
from app.normalization.pdf_extractor import has_text_layer, extract_pdf_text
from app.normalization.ocr_pipeline import ocr_image
from app.logging_config import get_logger

logger = get_logger("file_router")

SUPPORTED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp"
}


def detect_file_type(data: bytes) -> str:
    """Use python-magic to detect MIME type."""
    return magic.from_buffer(data[:2048], mime=True)


def normalize_document(
    file_bytes: bytes,
    document_id: str,
    filename: Optional[str] = None,
) -> NormalizedDocument:
    """
    Route file to correct extraction pipeline:
    - PDF with text layer -> pdfminer extraction
    - PDF without text layer (scanned) -> rasterize + OCR
    - Image -> OCR directly
    """
    mime = detect_file_type(file_bytes)
    logger.info("detected_file_type", document_id=document_id, mime=mime, filename=filename)

    if mime == "application/pdf":
        return _process_pdf(file_bytes, document_id)
    elif mime in SUPPORTED_IMAGE_TYPES:
        return _process_image(file_bytes, document_id)
    else:
        # Try as PDF first, then image
        logger.warning("unknown_mime_type", mime=mime, document_id=document_id)
        try:
            return _process_pdf(file_bytes, document_id)
        except Exception:
            return _process_image(file_bytes, document_id)


def _process_pdf(pdf_bytes: bytes, document_id: str) -> NormalizedDocument:
    """Process PDF: extract text if available, else OCR."""
    if has_text_layer(pdf_bytes):
        logger.info("pdf_has_text_layer", document_id=document_id)
        doc = extract_pdf_text(pdf_bytes, document_id)
        if doc.pages and any(p.tokens for p in doc.pages):
            return doc
        logger.warning("pdf_text_empty_fallback_ocr", document_id=document_id)

    # Rasterize PDF pages and OCR
    logger.info("rasterizing_pdf_for_ocr", document_id=document_id)
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    normalized_pages = []
    for page_no, pil_page in enumerate(pages):
        npage = ocr_image(pil_page, page_no=page_no)
        normalized_pages.append(npage)

    return NormalizedDocument(document_id=document_id, pages=normalized_pages)


def _process_image(image_bytes: bytes, document_id: str) -> NormalizedDocument:
    """Process a single image file via OCR."""
    pil = Image.open(io.BytesIO(image_bytes))
    npage = ocr_image(pil, page_no=0)
    return NormalizedDocument(document_id=document_id, pages=[npage])
