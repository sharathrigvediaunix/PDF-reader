"""
pytest conftest: shared fixtures for unit tests.
"""
import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def invoice_pdf_1():
    path = os.path.join(FIXTURES_DIR, "invoice_digital_1.pdf")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def invoice_pdf_2():
    path = os.path.join(FIXTURES_DIR, "invoice_digital_2.pdf")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def invoice_pdf_simple():
    path = os.path.join(FIXTURES_DIR, "invoice_simple.pdf")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def invoice_text_1():
    return """
INVOICE

Vendor: Acme Supplies Inc.
Invoice Number: INV-2024-001
Invoice Date: 01/15/2024
Due Date: 02/15/2024

Subtotal: $4,450.00
Tax (10%): $445.00
Total Amount: $4,895.00
"""


@pytest.fixture
def invoice_text_2():
    return """
Inv # GTS-20240203
Date: February 3, 2024
Grand Total: $8,598.63
Sub Total: $7,925.00
Sales Tax (8.5%): $673.63
"""


@pytest.fixture
def invoice_config():
    from app.config import load_config
    return load_config("invoice")


@pytest.fixture
def simple_norm_doc(invoice_text_1):
    """Create a minimal NormalizedDocument from text."""
    from app.models import NormalizedDocument, NormalizedPage, Token, BBox

    tokens = []
    line_id = 0
    for line in invoice_text_1.strip().split("\n"):
        words = line.strip().split()
        for i, word in enumerate(words):
            tokens.append(Token(
                text=word,
                bbox=BBox(x0=i * 60.0, y0=line_id * 14.0,
                           x1=i * 60.0 + 55.0, y1=line_id * 14.0 + 12.0,
                           page=0),
                line_id=line_id,
            ))
        line_id += 1

    page = NormalizedPage(
        page_no=0,
        tokens=tokens,
        full_text=invoice_text_1,
        ocr_used=False,
    )
    return NormalizedDocument(document_id="test-doc-001", pages=[page])
