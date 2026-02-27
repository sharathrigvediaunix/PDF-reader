"""
Unit tests for regex-based extraction.
"""
import pytest
from app.extractors.deterministic import regex_extract
from app.config import load_config


def make_doc(text: str, doc_id: str = "test"):
    from app.models import NormalizedDocument, NormalizedPage, Token, BBox
    tokens = []
    for li, line in enumerate(text.split("\n")):
        for i, word in enumerate(line.strip().split()):
            tokens.append(Token(
                text=word,
                bbox=BBox(x0=i * 60, y0=li * 14, x1=i * 60 + 55, y1=li * 14 + 12, page=0),
                line_id=li,
            ))
    page = NormalizedPage(page_no=0, tokens=tokens, full_text=text, ocr_used=False)
    return NormalizedDocument(document_id=doc_id, pages=[page])


def test_regex_invoice_number_standard(invoice_config):
    doc = make_doc("Invoice Number: INV-2024-001\nTotal: $100.00")
    fc = invoice_config.fields["invoice_number"]
    candidates = regex_extract(doc, fc)
    values = [c.raw_value for c in candidates]
    assert any("INV-2024-001" in v for v in values), f"Got: {values}"


def test_regex_invoice_date_slash(invoice_config):
    doc = make_doc("Invoice Date: 01/15/2024")
    fc = invoice_config.fields["invoice_date"]
    candidates = regex_extract(doc, fc)
    values = [c.raw_value for c in candidates]
    assert any("01/15/2024" in v for v in values), f"Got: {values}"


def test_regex_invoice_date_text(invoice_config):
    doc = make_doc("Date: February 3, 2024")
    fc = invoice_config.fields["invoice_date"]
    candidates = regex_extract(doc, fc)
    assert len(candidates) > 0


def test_regex_total_amount(invoice_config):
    doc = make_doc("Total Amount: $4,895.00")
    fc = invoice_config.fields["total_amount"]
    candidates = regex_extract(doc, fc)
    values = [c.raw_value for c in candidates]
    assert any("4,895" in v or "4895" in v for v in values), f"Got: {values}"


def test_regex_grand_total(invoice_config):
    doc = make_doc("Grand Total: $8,598.63")
    fc = invoice_config.fields["total_amount"]
    candidates = regex_extract(doc, fc)
    values = [c.raw_value for c in candidates]
    assert any("8,598" in v or "8598" in v for v in values), f"Got: {values}"


def test_regex_tax_amount(invoice_config):
    doc = make_doc("Tax (10%): $445.00")
    fc = invoice_config.fields["tax_amount"]
    candidates = regex_extract(doc, fc)
    values = [c.raw_value for c in candidates]
    assert any("445" in v for v in values), f"Got: {values}"


def test_regex_method_set(invoice_config):
    from app.models import ExtractionMethod
    doc = make_doc("Invoice Number: XYZ-001")
    fc = invoice_config.fields["invoice_number"]
    candidates = regex_extract(doc, fc)
    if candidates:
        assert candidates[0].method == ExtractionMethod.REGEX


def test_regex_empty_doc(invoice_config):
    doc = make_doc("")
    fc = invoice_config.fields["invoice_number"]
    candidates = regex_extract(doc, fc)
    assert candidates == []
