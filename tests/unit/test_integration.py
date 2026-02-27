"""
Integration test: run full extraction pipeline on fixture PDFs.
Bypasses Celery/Postgres/MinIO.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _make_doc_from_pdf(pdf_bytes, doc_id="test"):
    from app.normalization import normalize_document
    return normalize_document(pdf_bytes, doc_id)


def _run_deterministic(norm_doc, document_type="invoice"):
    from app.config import load_config
    from app.extractors.deterministic import extract_field
    from app.validators import normalize_value, validate_value

    doc_config = load_config(document_type)
    results = {}
    for fname, fconfig in doc_config.fields.items():
        fr = extract_field(norm_doc, fconfig, normalize_value, validate_value)
        results[fname] = fr
    return results


def test_invoice_1_extraction(invoice_pdf_1):
    """Invoice 1: should find invoice_number, total_amount, invoice_date."""
    doc = _make_doc_from_pdf(invoice_pdf_1, "inv1")
    results = _run_deterministic(doc)

    # At minimum these should be found
    assert results["invoice_number"].value is not None, "invoice_number should be found"
    assert results["total_amount"].value is not None, "total_amount should be found"
    assert results["invoice_date"].value is not None, "invoice_date should be found"

    # Verify values
    inv_no = str(results["invoice_number"].value).upper()
    assert "INV-2024-001" in inv_no or "2024" in inv_no, f"Invoice number: {inv_no}"


def test_invoice_2_extraction(invoice_pdf_2):
    """Invoice 2: should find invoice_number, grand total."""
    doc = _make_doc_from_pdf(invoice_pdf_2, "inv2")
    results = _run_deterministic(doc)

    assert results["total_amount"].value is not None, "total_amount should be found"
    assert results["invoice_number"].value is not None, "invoice_number should be found"


def test_invoice_simple_extraction(invoice_pdf_simple):
    """Simple invoice: minimal fields should be found."""
    doc = _make_doc_from_pdf(invoice_pdf_simple, "inv-simple")
    results = _run_deterministic(doc)

    assert results["total_amount"].value is not None, "total_amount should be found"


def test_all_results_have_method(invoice_pdf_1):
    """All non-missing results should have a method set."""
    from app.models import ExtractionStatus
    doc = _make_doc_from_pdf(invoice_pdf_1, "inv1-method")
    results = _run_deterministic(doc)
    for fname, fr in results.items():
        if fr.status != ExtractionStatus.MISSING:
            assert fr.method is not None, f"Field {fname} has no method"


def test_all_results_have_confidence(invoice_pdf_1):
    """All non-missing results should have confidence > 0."""
    from app.models import ExtractionStatus
    doc = _make_doc_from_pdf(invoice_pdf_1, "inv1-conf")
    results = _run_deterministic(doc)
    for fname, fr in results.items():
        if fr.status != ExtractionStatus.MISSING:
            assert fr.confidence > 0, f"Field {fname} has zero confidence"


def test_validation_summary(invoice_pdf_1):
    """Validation summary should list missing required fields."""
    from app.config import load_config
    from app.validators.rules import compute_validation_summary

    doc = _make_doc_from_pdf(invoice_pdf_1, "inv1-summary")
    results = _run_deterministic(doc)
    doc_config = load_config("invoice")
    summary = compute_validation_summary(results, doc_config)

    assert isinstance(summary.required_missing, list)
    assert isinstance(summary.overall_confidence, float)
    assert 0.0 <= summary.overall_confidence <= 1.0
