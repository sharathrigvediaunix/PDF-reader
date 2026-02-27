"""
PDF text + bbox extraction using pdfminer.six.
Returns NormalizedDocument with token-level data.
"""
import io
from typing import Optional

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage, LTTextBox, LTTextLine, LTChar, LTAnno
from pdfminer.pdfpage import PDFPage

from app.models import NormalizedDocument, NormalizedPage, Token, BBox
from app.logging_config import get_logger

logger = get_logger("pdf_extractor")

MIN_TEXT_CHARS = 20  # threshold to decide if PDF has usable text layer


def has_text_layer(pdf_bytes: bytes) -> bool:
    """Detect if PDF has extractable text."""
    total_chars = 0
    try:
        for page_layout in extract_pages(io.BytesIO(pdf_bytes)):
            for element in page_layout:
                if isinstance(element, LTTextBox):
                    total_chars += len(element.get_text().strip())
            if total_chars >= MIN_TEXT_CHARS:
                return True
    except Exception:
        pass
    return total_chars >= MIN_TEXT_CHARS


def extract_pdf_text(pdf_bytes: bytes, document_id: str) -> NormalizedDocument:
    """
    Extract text + bboxes from digital PDF using pdfminer.six.
    Line IDs are assigned per page.
    """
    pages = []
    line_counter = 0

    try:
        for page_no, page_layout in enumerate(extract_pages(io.BytesIO(pdf_bytes))):
            tokens = []
            page_height = page_layout.height

            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue
                for line in element:
                    if not isinstance(line, LTTextLine):
                        continue
                    line_text = line.get_text().strip()
                    if not line_text:
                        continue

                    # Collect characters into word tokens
                    word_chars = []
                    word_bbox = None

                    def flush_word():
                        nonlocal word_chars, word_bbox
                        if not word_chars:
                            return
                        word_text = "".join(word_chars).strip()
                        if word_text and word_bbox:
                            tokens.append(Token(
                                text=word_text,
                                bbox=word_bbox,
                                line_id=line_counter,
                            ))
                        word_chars = []
                        word_bbox = None

                    for char in line:
                        if isinstance(char, LTChar):
                            if char.get_text() == " " or char.get_text() == "\t":
                                flush_word()
                            else:
                                word_chars.append(char.get_text())
                                char_bbox = BBox(
                                    x0=char.x0, y0=page_height - char.y1,
                                    x1=char.x1, y1=page_height - char.y0,
                                    page=page_no,
                                )
                                if word_bbox is None:
                                    word_bbox = char_bbox
                                else:
                                    word_bbox = BBox(
                                        x0=min(word_bbox.x0, char_bbox.x0),
                                        y0=min(word_bbox.y0, char_bbox.y0),
                                        x1=max(word_bbox.x1, char_bbox.x1),
                                        y1=max(word_bbox.y1, char_bbox.y1),
                                        page=page_no,
                                    )
                    flush_word()
                    line_counter += 1

            # Build full_text from tokens grouped by line
            lines_text: dict[int, list[str]] = {}
            for tok in tokens:
                lines_text.setdefault(tok.line_id, []).append(tok.text)
            full_text = "\n".join(" ".join(words) for words in lines_text.values())

            pages.append(NormalizedPage(
                page_no=page_no,
                tokens=tokens,
                full_text=full_text,
                ocr_used=False,
            ))

        logger.info("pdf_text_extracted", document_id=document_id, pages=len(pages))
    except Exception as e:
        logger.error("pdf_extraction_error", document_id=document_id, error=str(e))
        pages = pages or []

    return NormalizedDocument(document_id=document_id, pages=pages)
