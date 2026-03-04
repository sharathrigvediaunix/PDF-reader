"""
Spatial token search.
Given an anchor token's bounding box, searches for value tokens
in configured directions within a pixel window.

Key improvements over v1:
- Pattern-aware: if field patterns are passed, prefer tokens that match them
- Stronger label filtering: skips all-alpha tokens with no digits (labels)
- x-distance cap on same-line search so it doesn't grab distant column headers
- Wider below search x_tolerance
- Falls back through right → below → wide-right → nearby, trying pattern match at each stage
"""

import math
import re
from typing import Optional

from app.models import Token, BBox
from app.logging_config import get_logger

logger = get_logger("spatial")

# Max Y delta to consider tokens on the same visual line
SAME_LINE_Y_TOLERANCE = 12.0

# Max pixels below anchor to search
MAX_BELOW_DISTANCE = 150.0

# Max pixels to the right on same line (default, overridden by field window)
DEFAULT_MAX_RIGHT_DISTANCE = 500.0

# X alignment tolerance for "below" search
DEFAULT_X_TOLERANCE = 120.0

# Fallback radius for nearby search
NEARBY_RADIUS = 200.0


# ── Label detection ───────────────────────────────────────────────────────────


def _is_label_token(tok: Token) -> bool:
    """
    Return True if this token looks like a field label rather than a value.
    Labels are:
      - Pure alphabetic words with no digits (SHIPPING, TAX, DATE, ORDER)
      - Short tokens ending with ':' or '#'
      - Separator lines (_____, -----)
      - Single-char punctuation
    Values always contain at least one digit OR are short alpha codes.
    """
    text = tok.text.strip()
    if not text:
        return True

    # Separator lines
    if all(c in "_-=" for c in text):
        return True

    # Single punctuation
    if len(text) == 1 and not text.isalnum():
        return True

    # Ends with label punctuation and no digits
    if text.endswith((":", "#")) and not any(c.isdigit() for c in text):
        return True

    # Pure alphabetic (no digits) and longer than 2 chars → label word
    # Allow short alpha codes like "FOB", "USD" but reject "SHIPPING", "TAX" etc.
    alpha_only = re.sub(r"[^a-zA-Z]", "", text)
    if alpha_only == text and len(text) > 3:
        return True

    return False


def _matches_any_pattern(tok: Token, patterns: list[re.Pattern]) -> bool:
    """Return True if token text matches any of the compiled field patterns."""
    if not patterns:
        return False
    for pat in patterns:
        if pat.search(tok.text):
            return True
    return False


def _score_token(
    tok: Token,
    patterns: list[re.Pattern],
    distance: float,
) -> float:
    """
    Score a candidate value token.
    Higher = better.
    Pattern match gives a big bonus so matching tokens always beat non-matching ones.
    """
    score = 1000.0 / (distance + 1.0)  # closer = higher base score
    if _is_label_token(tok):
        score -= 500.0  # heavy penalty for labels
    if patterns and _matches_any_pattern(tok, patterns):
        score += 1000.0  # pattern match wins
    return score


# ── Directional searches ──────────────────────────────────────────────────────


def search_right(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    max_x_distance: float = DEFAULT_MAX_RIGHT_DISTANCE,
    patterns: Optional[list[re.Pattern]] = None,
    max_tokens: int = 8,
) -> list[Token]:
    """
    Tokens to the right of anchor on the same visual line.
    Capped at max_x_distance so we don't grab tokens in distant columns.
    If patterns provided, returns only matching tokens first; falls back to all.
    """
    candidates = []
    for tok in tokens:
        if tok is anchor:
            continue
        if tok.bbox.page != page_no:
            continue
        if abs(tok.bbox.center_y - anchor.bbox.center_y) > SAME_LINE_Y_TOLERANCE:
            continue
        if tok.bbox.x0 <= anchor.bbox.x1:
            continue
        x_dist = tok.bbox.x0 - anchor.bbox.x1
        if x_dist > max_x_distance:
            continue
        candidates.append(tok)

    candidates.sort(key=lambda t: t.bbox.x0)

    if patterns:
        pattern_matches = [t for t in candidates if _matches_any_pattern(t, patterns)]
        if pattern_matches:
            return pattern_matches[:max_tokens]

    # Filter labels only if no pattern matches found
    non_labels = [t for t in candidates if not _is_label_token(t)]
    return (non_labels or candidates)[:max_tokens]


def search_below(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    max_y_distance: float = MAX_BELOW_DISTANCE,
    x_tolerance: float = DEFAULT_X_TOLERANCE,
    patterns: Optional[list[re.Pattern]] = None,
    max_tokens: int = 6,
) -> list[Token]:
    """
    Tokens below anchor, X-aligned within x_tolerance of anchor center.
    """
    candidates = []
    anchor_x = anchor.bbox.center_x

    for tok in tokens:
        if tok is anchor:
            continue
        if tok.bbox.page != page_no:
            continue
        if tok.bbox.y0 <= anchor.bbox.y1:
            continue
        y_dist = tok.bbox.y0 - anchor.bbox.y1
        if y_dist > max_y_distance:
            continue
        if abs(tok.bbox.center_x - anchor_x) > x_tolerance:
            continue
        candidates.append(tok)

    candidates.sort(key=lambda t: (t.bbox.y0, t.bbox.x0))

    if patterns:
        pattern_matches = [t for t in candidates if _matches_any_pattern(t, patterns)]
        if pattern_matches:
            return pattern_matches[:max_tokens]

    non_labels = [t for t in candidates if not _is_label_token(t)]
    return (non_labels or candidates)[:max_tokens]


def search_below_wide(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    patterns: Optional[list[re.Pattern]] = None,
) -> list[Token]:
    """
    Wide below search — scans up to 3 lines below with relaxed X tolerance.
    Used as a fallback when tight below fails.
    If patterns given, returns only pattern-matching tokens.
    """
    return search_below(
        anchor,
        tokens,
        page_no,
        max_y_distance=MAX_BELOW_DISTANCE * 2,
        x_tolerance=DEFAULT_X_TOLERANCE * 2,
        patterns=patterns,
    )


def search_right_wide(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    patterns: Optional[list[re.Pattern]] = None,
) -> list[Token]:
    """
    Wide right search — no x-distance cap, scans entire same line.
    If patterns given, returns only pattern-matching tokens.
    """
    return search_right(
        anchor,
        tokens,
        page_no,
        max_x_distance=9999.0,
        patterns=patterns,
    )


def search_nearby(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    patterns: Optional[list[re.Pattern]] = None,
    max_tokens: int = 4,
) -> list[Token]:
    """
    Nearest tokens in any direction within NEARBY_RADIUS.
    Pattern-matching tokens scored highest.
    """
    candidates = []
    for tok in tokens:
        if tok is anchor:
            continue
        if tok.bbox.page != page_no:
            continue
        dist = math.sqrt(
            (tok.bbox.center_x - anchor.bbox.center_x) ** 2
            + (tok.bbox.center_y - anchor.bbox.center_y) ** 2
        )
        if dist <= NEARBY_RADIUS:
            score = _score_token(tok, patterns or [], dist)
            candidates.append((score, dist, tok))

    candidates.sort(key=lambda x: -x[0])  # highest score first
    return [tok for _, _, tok in candidates[:max_tokens]]


# ── Main entry point ──────────────────────────────────────────────────────────


def spatial_search(
    anchor: Token,
    tokens: list[Token],
    page_no: int,
    strategy: str = "right_then_below",
    window: Optional[dict] = None,
    patterns: Optional[list[re.Pattern]] = None,
) -> list[Token]:
    """
    Search for value tokens near the anchor using the configured strategy.

    Strategies:
      right_only        - same line right only
      same_line_right   - same as right_only
      same_line_or_next - right first, then 1 line below (tight)
      right_then_below  - right → below → wide fallbacks
      below_then_right  - below → right → wide fallbacks
      next_3_lines      - below search with relaxed distance
      any               - pattern-aware nearest-token search

    window: {x: int, y: int} pixel search box from YAML field config.
            x caps the right-search distance, y caps the below distance.
    patterns: compiled regex patterns from field config.
              When provided, tokens matching a pattern are strongly preferred.
    """
    max_x = float((window or {}).get("x", DEFAULT_MAX_RIGHT_DISTANCE))
    max_y = float((window or {}).get("y", MAX_BELOW_DISTANCE))

    def _right():
        return search_right(
            anchor, tokens, page_no, max_x_distance=max_x, patterns=patterns
        )

    def _below():
        return search_below(
            anchor, tokens, page_no, max_y_distance=max_y, patterns=patterns
        )

    def _right_wide():
        return search_right_wide(anchor, tokens, page_no, patterns=patterns)

    def _below_wide():
        return search_below_wide(anchor, tokens, page_no, patterns=patterns)

    def _nearby():
        return search_nearby(anchor, tokens, page_no, patterns=patterns)

    def _first_non_empty(*fns):
        for fn in fns:
            result = fn()
            if result:
                return result
        return []

    if strategy in ("right_only", "same_line_right"):
        return _first_non_empty(_right, _right_wide)

    elif strategy == "same_line_or_next":
        # Try right on same line, then directly below, then wider fallbacks
        return _first_non_empty(_right, _below, _right_wide, _below_wide, _nearby)

    elif strategy == "below_then_right":
        return _first_non_empty(_below, _right, _below_wide, _right_wide, _nearby)

    elif strategy == "next_3_lines":
        return _first_non_empty(_below, _below_wide, _right, _nearby)

    elif strategy == "any":
        return _first_non_empty(_nearby, _right, _below)

    else:  # right_then_below (default)
        return _first_non_empty(_right, _below, _right_wide, _below_wide, _nearby)
