"""
Layout normalization layer.
Takes raw tokens from PDF or OCR and produces:
  - LayoutLines: tokens grouped by Y-proximity (visual rows)
  - LayoutColumns: X-aligned cluster boundaries (visual columns)

This runs after both pdfminer and PaddleOCR extraction, giving the
spatial extraction layer a coordinate-aware structure to work with.
"""
from typing import Optional
import numpy as np

from app.models import Token, BBox, LayoutLine, LayoutColumn, NormalizedPage
from app.logging_config import get_logger

logger = get_logger("layout_normalizer")

# Tokens within this many pixels vertically are on the same line
LINE_Y_THRESHOLD = 8.0

# Minimum tokens to form a column cluster
MIN_COLUMN_TOKENS = 3

# Column merge tolerance — clusters within this many pixels are merged
COLUMN_MERGE_THRESHOLD = 20.0


def build_layout_lines(tokens: list[Token]) -> list[LayoutLine]:
    """
    Group tokens into visual lines by Y-center proximity.
    Tokens whose Y-centers are within LINE_Y_THRESHOLD pixels
    of each other are placed on the same line.
    Lines are sorted top-to-bottom, tokens within each line left-to-right.
    """
    if not tokens:
        return []

    # Sort tokens top-to-bottom, left-to-right
    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.center_y, t.bbox.x0))

    lines: list[list[Token]] = []
    current_line: list[Token] = [sorted_tokens[0]]
    current_y = sorted_tokens[0].bbox.center_y

    for tok in sorted_tokens[1:]:
        if abs(tok.bbox.center_y - current_y) <= LINE_Y_THRESHOLD:
            current_line.append(tok)
        else:
            lines.append(current_line)
            current_line = [tok]
            current_y = tok.bbox.center_y

    if current_line:
        lines.append(current_line)

    # Build LayoutLine objects
    layout_lines = []
    for idx, line_tokens in enumerate(lines):
        # Sort tokens within line left-to-right
        line_tokens.sort(key=lambda t: t.bbox.x0)
        y_center = sum(t.bbox.center_y for t in line_tokens) / len(line_tokens)
        layout_lines.append(LayoutLine(
            line_id=idx,
            tokens=line_tokens,
            y_center=y_center,
        ))

    logger.debug("layout_lines_built", count=len(layout_lines))
    return layout_lines


def build_layout_columns(tokens: list[Token]) -> list[LayoutColumn]:
    """
    Detect column boundaries by clustering token X-centers.
    Uses simple histogram-based clustering — no ML needed.
    Columns are useful for table structure detection.
    """
    if len(tokens) < MIN_COLUMN_TOKENS:
        return []

    x_centers = np.array([t.bbox.center_x for t in tokens])

    # Build histogram of X positions with 10-pixel bins
    page_width = max(t.bbox.x1 for t in tokens)
    bins = max(10, int(page_width / 10))
    hist, edges = np.histogram(x_centers, bins=bins)

    # Find peaks (high-density X regions = column centers)
    # A peak is a bin with more tokens than its neighbors
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] >= MIN_COLUMN_TOKENS:
            peak_x = (edges[i] + edges[i+1]) / 2
            peaks.append(peak_x)

    if not peaks:
        return []

    # Merge peaks that are too close together
    merged = [peaks[0]]
    for px in peaks[1:]:
        if px - merged[-1] > COLUMN_MERGE_THRESHOLD:
            merged.append(px)
        else:
            # Average the two close peaks
            merged[-1] = (merged[-1] + px) / 2

    # Build LayoutColumn objects
    columns = []
    for col_id, x_center in enumerate(merged):
        # Find tokens near this column center
        nearby = [t for t in tokens if abs(t.bbox.center_x - x_center) <= COLUMN_MERGE_THRESHOLD]
        if not nearby:
            continue
        x_min = min(t.bbox.x0 for t in nearby)
        x_max = max(t.bbox.x1 for t in nearby)
        columns.append(LayoutColumn(
            col_id=col_id,
            x_center=x_center,
            x_min=x_min,
            x_max=x_max,
        ))

    logger.debug("layout_columns_built", count=len(columns))
    return columns


def normalize_layout(page: NormalizedPage) -> NormalizedPage:
    """
    Enrich a NormalizedPage with layout_lines and layout_columns.
    Non-destructive — tokens are unchanged, layout is added.
    """
    if not page.tokens:
        return page

    page.layout_lines = build_layout_lines(page.tokens)
    page.layout_columns = build_layout_columns(page.tokens)

    # Re-assign line_ids on tokens to match layout line grouping
    for layout_line in page.layout_lines:
        for tok in layout_line.tokens:
            tok.line_id = layout_line.line_id

    return page
