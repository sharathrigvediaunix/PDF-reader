"""
Unit tests for anchor-based extraction.
"""
import pytest
from app.extractors.deterministic import anchor_extract
from app.config import load_config


def test_anchor_finds_invoice_number(simple_norm_doc, invoice_config):
    """Anchor 'Invoice Number' should find INV-2024-001."""
    fc = invoice_config.fields["invoice_number"]
    candidates = anchor_extract(simple_norm_doc, fc)
    assert len(candidates) > 0
    values = [c.raw_value.upper() for c in candidates]
    assert any("INV-2024-001" in v for v in values), f"No invoice number found. Got: {values}"


def test_anchor_finds_total(simple_norm_doc, invoice_config):
    """Anchor 'Total Amount' should find $4,895.00."""
    fc = invoice_config.fields["total_amount"]
    candidates = anchor_extract(simple_norm_doc, fc)
    assert len(candidates) > 0
    values = [c.raw_value for c in candidates]
    assert any("4,895" in v or "4895" in v for v in values), f"Total not found. Got: {values}"


def test_anchor_fuzzy_match(invoice_config):
    """Fuzzy anchor matching should handle 'Inv #' variant."""
    from app.models import NormalizedDocument, NormalizedPage, Token, BBox

    text_lines = [
        ["Inv", "#", "XYZ-999"],
    ]
    tokens = []
    for li, words in enumerate(text_lines):
        for i, w in enumerate(words):
            tokens.append(Token(
                text=w,
                bbox=BBox(x0=i * 60, y0=li * 14, x1=i * 60 + 55, y1=li * 14 + 12, page=0),
                line_id=li,
            ))
    page = NormalizedPage(page_no=0, tokens=tokens, full_text="Inv # XYZ-999", ocr_used=False)
    doc = NormalizedDocument(document_id="test-fuzzy", pages=[page])

    fc = invoice_config.fields["invoice_number"]
    candidates = anchor_extract(doc, fc)
    values = [c.raw_value for c in candidates]
    assert any("XYZ-999" in v for v in values), f"Fuzzy match failed. Got: {values}"


def test_anchor_multiword(invoice_config):
    """'Invoice Number' (two-word anchor) should be detected."""
    from app.models import NormalizedDocument, NormalizedPage, Token, BBox

    tokens = []
    words = ["Invoice", "Number:", "ABC-123"]
    for i, w in enumerate(words):
        tokens.append(Token(
            text=w,
            bbox=BBox(x0=i * 80, y0=0, x1=i * 80 + 75, y1=12, page=0),
            line_id=0,
        ))
    page = NormalizedPage(page_no=0, tokens=tokens, full_text="Invoice Number: ABC-123", ocr_used=False)
    doc = NormalizedDocument(document_id="test-multiword", pages=[page])

    fc = invoice_config.fields["invoice_number"]
    candidates = anchor_extract(doc, fc)
    values = [c.raw_value for c in candidates]
    assert any("ABC-123" in v for v in values), f"Multi-word anchor failed. Got: {values}"


def test_anchor_returns_method(simple_norm_doc, invoice_config):
    """Anchor results should have method=anchor."""
    from app.models import ExtractionMethod
    fc = invoice_config.fields["invoice_number"]
    candidates = anchor_extract(simple_norm_doc, fc)
    if candidates:
        assert candidates[0].method == ExtractionMethod.ANCHOR


def test_anchor_no_false_positives(invoice_config):
    """Invoice number anchor should not fire on unrelated text."""
    from app.models import NormalizedDocument, NormalizedPage, Token, BBox

    tokens = []
    for i, w in enumerate(["Hello", "World", "Nothing", "Here"]):
        tokens.append(Token(
            text=w,
            bbox=BBox(x0=i * 60, y0=0, x1=i * 60 + 55, y1=12, page=0),
            line_id=0,
        ))
    page = NormalizedPage(page_no=0, tokens=tokens, full_text="Hello World Nothing Here", ocr_used=False)
    doc = NormalizedDocument(document_id="test-no-false", pages=[page])

    fc = invoice_config.fields["invoice_number"]
    candidates = anchor_extract(doc, fc)
    assert len(candidates) == 0
