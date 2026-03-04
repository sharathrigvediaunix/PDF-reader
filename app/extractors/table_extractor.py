"""
Dynamic table extractor.
Detects table structure from header keyword X-positions,
extracts rows dynamically until stop condition.
Handles multi-page tables.

No hardcoded column indexes — column boundaries come from
the actual positions of header tokens in the document.
"""
from typing import Optional, Any
import re

from app.models import (
    NormalizedDocument, NormalizedPage, Token, BBox,
    LayoutLine, LayoutColumn,
)
from app.logging_config import get_logger

logger = get_logger("table_extractor")

# How far (pixels) a token can be from a column center and still belong to it
COLUMN_ASSIGNMENT_TOLERANCE = 40.0

# Minimum number of header keywords that must match to confirm table start
MIN_HEADER_KEYWORD_MATCHES = 2




def _fuzzy_contains(text: str, keyword: str, threshold: float = 0.75) -> bool:
    """Check if text loosely contains the keyword."""
    from difflib import SequenceMatcher
    text_lower = text.lower().strip()
    kw_lower = keyword.lower().strip()
    if kw_lower in text_lower:
        return True
    score = SequenceMatcher(None, text_lower, kw_lower).ratio()
    return score >= threshold


def _find_table_header_line(
    layout_lines: list[LayoutLine],
    header_keywords: list[str],
) -> Optional[tuple[int, LayoutLine]]:
    """
    Scan layout lines to find the one that contains the table header.
    Returns (line_index, header_line) or None if not found.
    A line is considered a header if it matches >= MIN_HEADER_KEYWORD_MATCHES keywords.
    """
    for idx, line in enumerate(layout_lines):
        line_text = line.text.lower()
        matches = sum(
            1 for kw in header_keywords
            if kw.lower() in line_text
        )
        if matches >= MIN_HEADER_KEYWORD_MATCHES:
            logger.debug("table_header_found", line_idx=idx, line_text=line.text)
            return idx, line
    return None


def _build_column_map(
    header_line: LayoutLine,
    column_config: list[dict],
) -> dict[str, tuple[float, float]]:
    """
    Build a mapping of {field_name -> (x_min, x_max)} from the header line.
    Uses the X positions of matched header tokens to define column boundaries.

    column_config entries:
        {"field": "quantity", "keywords": ["Qty", "QTY", "Quantity"]}
    """
    col_map: dict[str, tuple[float, float]] = {}

    for col_cfg in column_config:
        field_name = col_cfg["field"]
        keywords = col_cfg.get("keywords", [])

        for tok in header_line.tokens:
            if any(_fuzzy_contains(tok.text, kw) for kw in keywords):
                # Column spans from token center ± tolerance
                half_w = max(tok.bbox.width, COLUMN_ASSIGNMENT_TOLERANCE)
                col_map[field_name] = (
                    tok.bbox.center_x - half_w,
                    tok.bbox.center_x + half_w,
                )
                break

    logger.debug("column_map_built", columns=list(col_map.keys()))
    return col_map


def _assign_token_to_column(
    tok: Token,
    col_map: dict[str, tuple[float, float]],
) -> Optional[str]:
    """Return the field name whose column range contains this token's X center."""
    for field_name, (x_min, x_max) in col_map.items():
        if x_min <= tok.bbox.center_x <= x_max:
            return field_name
    return None


def _is_stop_row(line: LayoutLine, stop_keywords: list[str]) -> bool:
    """Return True if this line signals the end of the table."""
    line_text = line.text.lower()
    return any(kw.lower() in line_text for kw in stop_keywords)


def _parse_cell(text: str, field_type: str) -> Any:
    """Parse a cell value to its configured type."""
    text = text.strip().lstrip(":").strip()
    if not text or text in ("-", "—", "N/A", "n/a"):
        return None
    if field_type == "number":
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        try:
            return float(cleaned) if "." in cleaned else int(cleaned)
        except ValueError:
            return None
    elif field_type == "string":
        return text
    return text


def extract_table(
    doc: NormalizedDocument,
    table_config: dict,
) -> list[dict]:
    """
    Extract all table rows from the document using dynamic column detection.

    table_config schema:
    {
        "header_keywords": ["Item", "Description", "Qty", "Unit Price", "Amount"],
        "stop_keywords": ["Subtotal", "Total", "Sub Total"],
        "columns": [
            {"field": "description", "keywords": ["Description", "Item Description"], "type": "string"},
            {"field": "quantity",    "keywords": ["Qty", "QTY", "Quantity"],          "type": "number"},
            {"field": "unit_price",  "keywords": ["Unit Price", "Price"],             "type": "number"},
            {"field": "amount",      "keywords": ["Amount", "Total"],                 "type": "number"},
        ]
    }

    Returns list of row dicts.
    """
    header_keywords = table_config.get("header_keywords", [])
    stop_keywords = table_config.get("stop_keywords", ["Subtotal", "Total", "Sub Total"])
    column_config = table_config.get("columns", [])

    if not header_keywords or not column_config:
        logger.warning("table_config_incomplete")
        return []

    rows = []
    col_map: Optional[dict] = None
    in_table = False

    for page in doc.pages:
        if not page.layout_lines:
            continue

        for line_idx, line in enumerate(page.layout_lines):
            # ── Find table header ─────────────────────────────────────────────
            if not in_table:
                matches = sum(
                    1 for kw in header_keywords
                    if kw.lower() in line.text.lower()
                )
                if matches >= MIN_HEADER_KEYWORD_MATCHES:
                    col_map = _build_column_map(line, column_config)
                    in_table = True
                    logger.info("table_started", page=page.page_no, line=line.text)
                continue

            # ── Stop condition ────────────────────────────────────────────────
            if _is_stop_row(line, stop_keywords):
                logger.info("table_ended", page=page.page_no, line=line.text)
                # Don't break — might resume on next page
                in_table = False
                col_map = None
                continue

            # ── Skip empty lines ──────────────────────────────────────────────
            if not line.tokens:
                continue

            # ── Extract data row ──────────────────────────────────────────────
            if col_map:
                row: dict[str, Any] = {}
                for tok in line.tokens:
                    field_name = _assign_token_to_column(tok, col_map)
                    if field_name:
                        existing = row.get(field_name, "")
                        row[field_name] = (existing + " " + tok.text).strip()

                # Parse cell values to correct types
                parsed_row: dict[str, Any] = {}
                for col_cfg in column_config:
                    fname = col_cfg["field"]
                    ftype = col_cfg.get("type", "string")
                    raw = row.get(fname, "")
                    parsed_row[fname] = _parse_cell(raw, ftype)

                # Only keep rows that have at least one non-null value
                if any(v is not None for v in parsed_row.values()):
                    rows.append(parsed_row)

    logger.info("table_extracted", total_rows=len(rows))
    return rows
