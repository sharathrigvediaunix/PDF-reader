"""
Unit tests for file type detection and routing.
"""
import os
import pytest
from app.normalization.file_router import detect_file_type


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_detect_pdf(invoice_pdf_1):
    mime = detect_file_type(invoice_pdf_1)
    assert mime == "application/pdf"


def test_detect_pdf_fixture_2(invoice_pdf_2):
    mime = detect_file_type(invoice_pdf_2)
    assert mime == "application/pdf"


def test_pdf_has_text_layer(invoice_pdf_1):
    """Digitally generated PDFs should have a text layer."""
    from app.normalization.pdf_extractor import has_text_layer
    assert has_text_layer(invoice_pdf_1) is True


def test_pdf_text_extraction(invoice_pdf_1):
    """PDF text extraction should return tokens for digital PDF."""
    from app.normalization.pdf_extractor import extract_pdf_text
    doc = extract_pdf_text(invoice_pdf_1, "test-doc")
    assert len(doc.pages) > 0
    # Should have found some tokens
    total_tokens = sum(len(p.tokens) for p in doc.pages)
    assert total_tokens > 5


def test_pdf_text_contains_invoice_keywords(invoice_pdf_1):
    """Extracted text should contain known invoice keywords."""
    from app.normalization.pdf_extractor import extract_pdf_text
    doc = extract_pdf_text(invoice_pdf_1, "test-doc")
    full_text = doc.full_text.lower()
    assert "invoice" in full_text


def test_normalize_document_pdf(invoice_pdf_1):
    """normalize_document should return NormalizedDocument for PDF input."""
    from app.normalization import normalize_document
    doc = normalize_document(invoice_pdf_1, "test-normalize")
    assert doc.document_id == "test-normalize"
    assert len(doc.pages) >= 1


def test_detect_png():
    """1x1 PNG should be detected as image/png."""
    # Minimal valid 1x1 PNG
    import base64
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    png_bytes = base64.b64decode(png_b64)
    mime = detect_file_type(png_bytes)
    assert "image" in mime
