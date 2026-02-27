"""
Deterministic extraction engine.
1. Anchor-based key-value extraction (fuzzy anchor matching)
2. Regex extraction over full text
3. Candidate scoring: method_weight + parse_success + validator_bonus - conflict_penalty
"""
import re
from difflib import SequenceMatcher
from typing import Any, Optional

from app.config.loader import FieldConfig, DocumentConfig
from app.models import (
    NormalizedDocument, Token, BBox,
    FieldResult, FieldCandidate, ExtractionMethod,
    SourceLocation, Evidence, ExtractionStatus,
)
from app.logging_config import get_logger

logger = get_logger("deterministic_extractor")

# Method base weights for scoring
METHOD_WEIGHTS = {
    ExtractionMethod.ANCHOR: 0.75,
    ExtractionMethod.REGEX: 0.55,
    ExtractionMethod.LAYOUT: 0.50,
}

# Fuzzy match threshold for anchor detection
ANCHOR_FUZZY_THRESHOLD = 0.80


def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _token_text_near(
    tokens: list[Token],
    anchor_idx: int,
    search_window: str,
    page_no: int,
) -> list[tuple[str, Optional[BBox]]]:
    """
    Extract text near an anchor token based on search_window strategy.
    Returns list of (text, bbox) tuples.
    """
    anchor_tok = tokens[anchor_idx]
    anchor_line = anchor_tok.line_id
    results = []

    if search_window in ("same_line_or_next", "same_line_right"):
        # same line after anchor
        for tok in tokens[anchor_idx + 1:]:
            if tok.line_id == anchor_line and tok.bbox.page == page_no:
                results.append((tok.text, tok.bbox))
            elif tok.line_id > anchor_line:
                break

        # If nothing on same line and window allows next line
        if not results and search_window == "same_line_or_next":
            next_line_id = None
            for tok in tokens[anchor_idx + 1:]:
                if tok.line_id != anchor_line:
                    next_line_id = tok.line_id
                    break
            if next_line_id is not None:
                for tok in tokens:
                    if tok.line_id == next_line_id and tok.bbox.page == page_no:
                        results.append((tok.text, tok.bbox))

    elif search_window == "next_3_lines":
        lines_found = set()
        for tok in tokens[anchor_idx + 1:]:
            if tok.bbox.page == page_no and tok.line_id > anchor_line:
                lines_found.add(tok.line_id)
                results.append((tok.text, tok.bbox))
                if len(lines_found) >= 3:
                    break

    return results


def _best_bbox(bboxes: list[BBox]) -> Optional[BBox]:
    if not bboxes:
        return None
    return BBox(
        x0=min(b.x0 for b in bboxes),
        y0=min(b.y0 for b in bboxes),
        x1=max(b.x1 for b in bboxes),
        y1=max(b.y1 for b in bboxes),
        page=bboxes[0].page,
    )


def anchor_extract(
    doc: NormalizedDocument,
    field_config: FieldConfig,
) -> list[FieldCandidate]:
    """
    Scan tokens for anchor phrases, extract value from surrounding tokens.
    Supports fuzzy matching for anchor variants.
    """
    candidates = []

    for page in doc.pages:
        tokens = page.tokens
        n = len(tokens)

        for i, tok in enumerate(tokens):
            tok_lower = tok.text.lower().strip(":#")

            for anchor in field_config.anchors:
                anchor_words = anchor.lower().split()
                # Try multi-word anchor match
                match_score = 0.0
                matched_span = 1

                if len(anchor_words) == 1:
                    match_score = _fuzzy_score(tok_lower, anchor_words[0])
                    matched_span = 1
                else:
                    # Multi-word: check consecutive tokens
                    if i + len(anchor_words) <= n:
                        span_text = " ".join(
                            tokens[i + k].text.lower().strip(":#")
                            for k in range(len(anchor_words))
                        )
                        match_score = _fuzzy_score(span_text, anchor)
                        matched_span = len(anchor_words)

                if match_score >= ANCHOR_FUZZY_THRESHOLD:
                    # Extract value tokens after anchor
                    value_tokens = _token_text_near(
                        tokens,
                        i + matched_span - 1,
                        field_config.search_window,
                        page.page_no,
                    )

                    if not value_tokens:
                        continue

                    # Strip colon from first token
                    cleaned = []
                    for txt, bbox in value_tokens:
                        txt = txt.strip().lstrip(":").strip()
                        if txt:
                            cleaned.append((txt, bbox))

                    if not cleaned:
                        continue

                    raw_value = " ".join(t for t, _ in cleaned)
                    bboxes = [b for _, b in cleaned if b is not None]
                    best_bb = _best_bbox(bboxes)

                    # Context snippet
                    ctx_start = max(0, i - 2)
                    ctx_tokens = tokens[ctx_start: i + matched_span + len(cleaned)]
                    snippet = " ".join(t.text for t in ctx_tokens)

                    candidates.append(FieldCandidate(
                        value=raw_value,
                        raw_value=raw_value,
                        confidence=METHOD_WEIGHTS[ExtractionMethod.ANCHOR] * match_score,
                        method=ExtractionMethod.ANCHOR,
                        source=SourceLocation(
                            page=page.page_no,
                            bbox=best_bb.to_dict() if best_bb else None,
                        ),
                        evidence=Evidence(snippet=snippet),
                    ))

    logger.debug("anchor_candidates", field=field_config.name, count=len(candidates))
    return candidates


def regex_extract(
    doc: NormalizedDocument,
    field_config: FieldConfig,
) -> list[FieldCandidate]:
    """
    Apply regex patterns over full text, map matches back to token bboxes.
    """
    candidates = []
    full_text = doc.full_text

    for pattern in field_config.compiled_patterns:
        for match in pattern.finditer(full_text):
            # Try to get the capture group (group 1), else full match
            try:
                raw_value = match.group(1).strip()
            except IndexError:
                raw_value = match.group(0).strip()

            if not raw_value:
                continue

            # Find approximate bbox by scanning tokens
            bbox, page_no = _find_token_bbox(doc, raw_value)

            snippet = full_text[max(0, match.start() - 40): match.end() + 40]

            candidates.append(FieldCandidate(
                value=raw_value,
                raw_value=raw_value,
                confidence=METHOD_WEIGHTS[ExtractionMethod.REGEX],
                method=ExtractionMethod.REGEX,
                source=SourceLocation(
                    page=page_no,
                    bbox=bbox.to_dict() if bbox else None,
                ),
                evidence=Evidence(snippet=snippet.strip()),
            ))

    logger.debug("regex_candidates", field=field_config.name, count=len(candidates))
    return candidates


def _find_token_bbox(doc: NormalizedDocument, text: str) -> tuple[Optional[BBox], int]:
    """Approximate token bbox lookup for a matched text value."""
    words = text.lower().split()
    if not words:
        return None, 0

    for page in doc.pages:
        for i, tok in enumerate(page.tokens):
            if tok.text.lower().strip("$,.:") == words[0].strip("$,."):
                # Check if subsequent tokens match
                match_bboxes = [tok.bbox]
                all_match = True
                for j, w in enumerate(words[1:], 1):
                    if i + j < len(page.tokens):
                        if page.tokens[i + j].text.lower().strip("$,.") == w.strip("$,."):
                            match_bboxes.append(page.tokens[i + j].bbox)
                        else:
                            all_match = False
                            break
                if all_match:
                    bbox = _best_bbox(match_bboxes)
                    return bbox, page.page_no

    return None, 0


def score_candidates(
    candidates: list[FieldCandidate],
    field_config: FieldConfig,
    normalizer_fn,
    validator_fn,
) -> list[FieldCandidate]:
    """
    Score and rank candidates:
    - method_weight (base)
    - +0.1 if normalization succeeds
    - +0.15 if validation passes
    - -0.2 for conflicts (duplicate values disagreeing)
    """
    scored = []
    for c in candidates:
        score = c.confidence
        # Try normalization
        try:
            norm_val = normalizer_fn(c.raw_value, field_config)
            score += 0.10
            c = c.model_copy(update={"value": norm_val})
        except Exception:
            norm_val = c.raw_value

        # Try validation
        try:
            errors = validator_fn(norm_val, field_config)
            if not errors:
                score += 0.15
            else:
                score -= 0.05 * len(errors)
        except Exception:
            pass

        scored.append(c.model_copy(update={"confidence": min(score, 1.0)}))

    # Dedup: check for conflicting values among top candidates
    if len(scored) > 1:
        top_val = scored[0].value if scored else None
        for c in scored[1:]:
            if c.value != top_val and c.confidence > 0.4:
                # Conflict: slightly penalize all candidates
                scored = [
                    x.model_copy(update={"confidence": x.confidence - 0.05})
                    for x in scored
                ]
                break

    scored.sort(key=lambda x: x.confidence, reverse=True)
    return scored


def extract_field(
    doc: NormalizedDocument,
    field_config: FieldConfig,
    normalizer_fn,
    validator_fn,
) -> FieldResult:
    """
    Run deterministic extraction for a single field.
    Returns best FieldResult with alternatives in debug.
    """
    all_candidates: list[FieldCandidate] = []

    # 1. Anchor-based
    all_candidates.extend(anchor_extract(doc, field_config))

    # 2. Regex-based
    all_candidates.extend(regex_extract(doc, field_config))

    if not all_candidates:
        return FieldResult(
            field_name=field_config.name,
            status=ExtractionStatus.MISSING,
        )

    # Score & rank
    scored = score_candidates(all_candidates, field_config, normalizer_fn, validator_fn)
    best = scored[0]

    # Determine status based on confidence
    if best.confidence >= 0.85:
        status = ExtractionStatus.VERIFIED
    elif best.confidence >= 0.50:
        status = ExtractionStatus.NEEDS_REVIEW
    else:
        status = ExtractionStatus.MISSING

    return FieldResult(
        field_name=field_config.name,
        value=best.value,
        raw_value=best.raw_value,
        confidence=best.confidence,
        status=status,
        method=best.method,
        source=best.source,
        evidence=best.evidence,
        alternatives=scored[1:5],  # keep top 4 alternatives
    )
