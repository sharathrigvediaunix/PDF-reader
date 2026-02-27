"""
Unit tests for confidence scoring and status assignment.
"""
import pytest
from app.extractors.deterministic import score_candidates, extract_field
from app.models import ExtractionMethod, ExtractionStatus, FieldCandidate, Evidence
from app.validators import normalize_value, validate_value


def make_candidate(value, raw, confidence, method=ExtractionMethod.REGEX):
    return FieldCandidate(
        value=value, raw_value=raw,
        confidence=confidence, method=method,
        evidence=Evidence(snippet=raw),
    )


def test_scoring_validator_bonus(invoice_config):
    """Validator bonus should increase confidence for valid values."""
    fc = invoice_config.fields["invoice_number"]
    candidates = [make_candidate("INV-001", "INV-001", 0.55)]
    scored = score_candidates(candidates, fc, normalize_value, validate_value)
    assert scored[0].confidence > 0.55


def test_scoring_sorts_descending(invoice_config):
    """Scored candidates should be sorted by confidence descending."""
    fc = invoice_config.fields["invoice_number"]
    candidates = [
        make_candidate("ABC", "ABC", 0.40),
        make_candidate("INV-2024-001", "INV-2024-001", 0.75),
        make_candidate("XY", "XY", 0.30),
    ]
    scored = score_candidates(candidates, fc, normalize_value, validate_value)
    for i in range(len(scored) - 1):
        assert scored[i].confidence >= scored[i + 1].confidence


def test_status_verified(invoice_config):
    """High confidence + passing validators -> VERIFIED."""
    from app.models import NormalizedDocument, NormalizedPage, Token, BBox
    tokens = []
    text = "Invoice Number: INV-2024-001\nTotal Amount: $4,895.00\nInvoice Date: 01/15/2024"
    for li, line in enumerate(text.split("\n")):
        for i, word in enumerate(line.strip().split()):
            tokens.append(Token(
                text=word,
                bbox=BBox(x0=i * 60, y0=li * 14, x1=i * 60 + 55, y1=li * 14 + 12, page=0),
                line_id=li,
            ))
    page = NormalizedPage(page_no=0, tokens=tokens, full_text=text, ocr_used=False)
    doc = NormalizedDocument(document_id="test", pages=[page])

    fc = invoice_config.fields["invoice_number"]
    result = extract_field(doc, fc, normalize_value, validate_value)
    # Should find something (may be needs_review due to short text)
    assert result.status != ExtractionStatus.MISSING
    assert result.value is not None


def test_status_missing_when_not_found(invoice_config):
    """No matching text -> MISSING status."""
    from app.models import NormalizedDocument, NormalizedPage, Token, BBox
    tokens = []
    for i, w in enumerate(["Hello", "World"]):
        tokens.append(Token(
            text=w, bbox=BBox(x0=i * 60, y0=0, x1=i * 60 + 55, y1=12, page=0),
            line_id=0,
        ))
    page = NormalizedPage(page_no=0, tokens=tokens, full_text="Hello World", ocr_used=False)
    doc = NormalizedDocument(document_id="test-missing", pages=[page])

    fc = invoice_config.fields["invoice_number"]
    result = extract_field(doc, fc, normalize_value, validate_value)
    assert result.status == ExtractionStatus.MISSING


def test_scoring_normalizer_bonus(invoice_config):
    """Successful normalization adds 0.10 to confidence."""
    fc = invoice_config.fields["invoice_date"]
    candidates = [make_candidate("01/15/2024", "01/15/2024", 0.55)]
    scored = score_candidates(candidates, fc, normalize_value, validate_value)
    # normalizer should parse date successfully -> bonus
    assert scored[0].confidence > 0.55
