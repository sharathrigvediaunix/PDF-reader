#!/usr/bin/env python3
"""
Generate synthetic test fixture PDFs for the test suite.
Run: python scripts/generate_fixtures.py
"""
import os
import sys

# Try to use reportlab if available, else fpdf2
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    USE_REPORTLAB = True
except ImportError:
    USE_REPORTLAB = False

try:
    from fpdf import FPDF
    USE_FPDF = True
except ImportError:
    USE_FPDF = False

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


INVOICE_1_TEXT = """
INVOICE

Vendor: Acme Supplies Inc.
123 Commerce Street, Suite 400
New York, NY 10001

Bill To:
TechCorp Ltd.
456 Innovation Avenue
San Francisco, CA 94107

Invoice Number: INV-2024-001
Invoice Date: 01/15/2024
Due Date: 02/15/2024
Payment Terms: Net 30

Description                          Qty    Unit Price    Amount
-------------------------------------------------------------------
Software Licenses (Annual)            5      $500.00     $2,500.00
Support Services                      1      $750.00       $750.00
Implementation Consulting             8      $150.00     $1,200.00

                                               Subtotal:  $4,450.00
                                               Tax (10%):   $445.00
                                         Total Amount:  $4,895.00

Payment due by: February 15, 2024
Please make checks payable to: Acme Supplies Inc.
"""

INVOICE_2_TEXT = """
TAX INVOICE

FROM:
Global Tech Solutions
789 Enterprise Blvd
Austin, TX 78701
Phone: (512) 555-0100

TO:
Midwest Distribution Co.
321 Warehouse Road
Chicago, IL 60601

Inv # GTS-20240203
Date: February 3, 2024
Payment Due: March 4, 2024

ITEMS:
Hardware Components          $8,200.00
Shipping & Handling            $125.00
Discount                      -$400.00

Sub Total: $7,925.00
Sales Tax (8.5%): $673.63
Grand Total: $8,598.63

Currency: USD
"""

INVOICE_3_TEXT = """
Simple Invoice

invoice number: SI-003
invoice date: 2024-03-20

vendor: Quick Services LLC
customer: Beta Company

total due: $500.00
"""


def write_text_as_pdf_reportlab(text: str, output_path: str):
    c = canvas.Canvas(output_path, pagesize=letter)
    y = 750
    for line in text.strip().split("\n"):
        c.drawString(50, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            y = 750
    c.save()


def write_text_as_pdf_fpdf(text: str, output_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in text.strip().split("\n"):
        pdf.cell(0, 5, line, ln=True)
    pdf.output(output_path)


def write_text_file(text: str, output_path: str):
    """Fallback: write plain text file (not PDF)."""
    with open(output_path, "w") as f:
        f.write(text)


def generate_fixture(name: str, text: str):
    pdf_path = os.path.join(FIXTURES_DIR, f"{name}.pdf")
    txt_path = os.path.join(FIXTURES_DIR, f"{name}.txt")

    # Always write text version for tests that don't need PDF
    write_text_file(text, txt_path)

    if USE_REPORTLAB:
        write_text_as_pdf_reportlab(text, pdf_path)
        print(f"Generated (reportlab): {pdf_path}")
    elif USE_FPDF:
        write_text_as_pdf_fpdf(text, pdf_path)
        print(f"Generated (fpdf2): {pdf_path}")
    else:
        print(f"No PDF library available. Text fixture written: {txt_path}")
        print("  Install reportlab or fpdf2 for PDF fixtures:")
        print("  pip install reportlab  OR  pip install fpdf2")


if __name__ == "__main__":
    generate_fixture("invoice_digital_1", INVOICE_1_TEXT)
    generate_fixture("invoice_digital_2", INVOICE_2_TEXT)
    generate_fixture("invoice_simple", INVOICE_3_TEXT)
    print(f"\nFixtures generated in: {FIXTURES_DIR}")
