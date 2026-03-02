"""
Spatial token search.
Given an anchor token's bounding box, searches for value tokens
in all directions: right, below, below-left, nearby radius.

This replaces the line_id-based search in the original deterministic extractor.
Works for any document layout regardless of where the value appears
relative to the anchor label.
"""
from typing import Optional
import math

from app.models import Token, BBox, LayoutLine
from app.logging_config import get_logger

logger = get_logger("spatial")

# Max pixels to search right of anchor on the same line
SAME_LINE_Y_TOLERANCE = 10.0

# Max pixels to search below an anchor
MAX_BELOW_DISTANCE = 120.0

# Max pixels to search in any direction (radius search)
NEARBY_RADIUS = 150.0

# A token is considered a "label" (not a value) if it's short and ends with ':'
LABEL_PATTERN_MAX_LEN = 30


def _distance(tok_a: Token, tok_b: Token) -> float:
    """Euclidean distance between two token centers."""
    dx = tok_a.bbox.center_x - tok_b.bbox.center_x
    dy = tok_a.bbox.center_y - tok_b.bbox.center_y
    return math.sqrt(dx * dx + dy * dy)


def _is_label_token(tok: Token) -> bool:
    """
    Heuristic: skip tokens that look like field labels rather than values.
    A label is typically short and ends with ':' or is all caps with no digits.
    """
    text = tok.text.strip()
    if text.endswith(":") and len(text) <= LABEL_PATTERN_MAX_LEN:
        return True
    return False


def search_right(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    max_tokens: int = 6,
) -> list[Token]:
    """
    Find tokens to the right of the anchor on the same visual line.
    Y-centers must be within SAME_LINE_Y_TOLERANCE pixels.
    Returns tokens sorted left-to-right.
    """
    results = []
    for tok in tokens:
        if tok is anchor:
            continue
        if tok.bbox.page != page_no:
            continue
        # Same line: Y-centers close
        if abs(tok.bbox.center_y - anchor.bbox.center_y) > SAME_LINE_Y_TOLERANCE:
            continue
        # Must be to the right
        if tok.bbox.x0 <= anchor.bbox.x1:
            continue
        if _is_label_token(tok):
            continue
        results.append(tok)

    results.sort(key=lambda t: t.bbox.x0)
    return results[:max_tokens]


def search_below(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    max_tokens: int = 6,
    x_tolerance: float = 60.0,
) -> list[Token]:
    """
    Find tokens below the anchor, roughly X-aligned with it.
    Useful for cases like:
        Invoice No:
        GP-204/2025        <- value is directly below
    """
    results = []
    anchor_x_center = anchor.bbox.center_x

    for tok in tokens:
        if tok is anchor:
            continue
        if tok.bbox.page != page_no:
            continue
        # Must be below
        if tok.bbox.y0 <= anchor.bbox.y1:
            continue
        # Not too far below
        if tok.bbox.y0 - anchor.bbox.y1 > MAX_BELOW_DISTANCE:
            continue
        # X-aligned with anchor
        if abs(tok.bbox.center_x - anchor_x_center) > x_tolerance:
            continue
        if _is_label_token(tok):
            continue
        results.append(tok)

    results.sort(key=lambda t: (t.bbox.y0, t.bbox.x0))
    return results[:max_tokens]


def search_nearby(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    max_tokens: int = 4,
) -> list[Token]:
    """
    Find the closest tokens to the anchor within NEARBY_RADIUS pixels.
    Direction-agnostic fallback when right and below searches fail.
    """
    candidates = []
    for tok in tokens:
        if tok is anchor:
            continue
        if tok.bbox.page != page_no:
            continue
        # Must not be to the left (values are rarely to the left of labels)
        if tok.bbox.x1 < anchor.bbox.x0 - 20:
            continue
        dist = _distance(anchor, tok)
        if dist <= NEARBY_RADIUS:
            if not _is_label_token(tok):
                candidates.append((dist, tok))

    candidates.sort(key=lambda x: x[0])
    return [tok for _, tok in candidates[:max_tokens]]


def spatial_search(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    strategy: str = "right_then_below",
) -> list[Token]:
    """
    Main spatial search entry point.

    Strategies:
      right_then_below  - try right first, fall back to below (default)
      below_then_right  - try below first, fall back to right
      right_only        - same line only
      below_only        - below only
      any               - nearest token in any direction
    """
    if strategy == "right_only":
        return search_right(anchor, tokens, page_no)

    elif strategy == "below_only":
        return search_below(anchor, tokens, page_no)

    elif strategy == "below_then_right":
        result = search_below(anchor, tokens, page_no)
        if not result:
            result = search_right(anchor, tokens, page_no)
        if not result:
            result = search_nearby(anchor, tokens, page_no)
        return result

    elif strategy == "any":
        return search_nearby(anchor, tokens, page_no)

    else:  # right_then_below (default)
        result = search_right(anchor, tokens, page_no)
        if not result:
            result = search_below(anchor, tokens, page_no)
        if not result:
            result = search_nearby(anchor, tokens, page_no)
        return result
