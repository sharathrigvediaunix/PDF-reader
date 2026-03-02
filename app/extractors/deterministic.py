"""
Deterministic extraction engine.
1. Spatial anchor-based extraction (bbox search in all directions)
2. Regex extraction over full text (fallback)
3. Candidate scoring: method_weight + ocr_conf + parse_success + validator_bonus
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
from app.extractors.spatial import spatial_search
from app.logging_config import get_logger

logger = get_logger("deterministic_extractor")

# Method base weights for confidence scoring
METHOD_WEIGHTS = {
    ExtractionMethod.ANCHOR: 0.75,
    ExtractionMethod.REGEX: 0.55,
    ExtractionMethod.LAYOUT: 0.50,
}

# Fuzzy match threshold for anchor detection
ANCHOR_FUZZY_THRESHOLD = 0.80

# Map search_window config values to spatial strategy names
SEARCH_WINDOW_MAP = {
    "same_line_right":   "right_only",
    "same_line_or_next": "right_then_below",
    "next_3_lines":      "below_then_right",
    "full_text":         "any",
    "any":               "any",
}


def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


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


def _avg_conf(tokens: list[Token]) -> float:
    """Average OCR confidence of a set of tokens."""
    if not tokens:
        return 1.0
    return sum(t.conf for t in tokens) / len(tokens)


def anchor_extract(
    doc: NormalizedDocument,
    field_config: FieldConfig,
) -> list[FieldCandidate]:
    """
    Spatially-aware anchor extraction.
    For each page, scan tokens for anchor phrases using fuzzy matching.
    Once anchor is found, use spatial search (right/below/nearby)
    to find the value — regardless of layout.
    """
    candidates = []
    spatial_strategy = SEARCH_WINDOW_MAP.get(
        field_config.search_window, "right_then_below"
    )

    for page in doc.pages:
        tokens = page.tokens
        n = len(tokens)

        for i, tok in enumerate(tokens):
            tok_lower = tok.text.lower().strip(":#")

            for anchor in field_config.anchors:
                anchor_words = anchor.lower().split()
                match_score = 0.0
                matched_span = 1

                if len(anchor_words) == 1:
                    match_score = _fuzzy_score(tok_lower, anchor_words[0])
                    matched_span = 1
                else:
                    if i + len(anchor_words) <= n:
                        span_text = " ".join(
                            tokens[i + k].text.lower().strip(":#")
                            for k in range(len(anchor_words))
                        )
                        match_score = _fuzzy_score(span_text, anchor)
                        matched_span = len(anchor_words)

                if match_score < ANCHOR_FUZZY_THRESHOLD:
                    continue

                # ── Spatial search for value tokens ───────────────────────────
                anchor_tok = tokens[i + matched_span - 1]
                value_tokens = spatial_search(
                    anchor=anchor_tok,
                    tokens=tokens,
                    page_no=page.page_no,
                    strategy=spatial_strategy,
                )

                if not value_tokens:
                    continue

                # Strip leading colons from first token
                cleaned_tokens = []
                for vt in value_tokens:
                    text = vt.text.strip().lstrip(":").strip()
                    if text:
                        cleaned_tokens.append((text, vt.bbox, vt.conf))

                if not cleaned_tokens:
                    continue

                raw_value = " ".join(t for t, _, _ in cleaned_tokens)
                bboxes = [b for _, b, _ in cleaned_tokens]
                best_bb = _best_bbox(bboxes)

                # OCR confidence from value tokens
                ocr_conf = _avg_conf(value_tokens)

                # Context snippet
                ctx_start = max(0, i - 2)
                ctx_tokens = tokens[ctx_start: i + matched_span + len(cleaned_tokens)]
                snippet = " ".join(t.text for t in ctx_tokens)

                # Base confidence: method weight × anchor match × OCR confidence
                base_conf = METHOD_WEIGHTS[ExtractionMethod.ANCHOR] * match_score * ocr_conf

                candidates.append(FieldCandidate(
                    value=raw_value,
                    raw_value=raw_value,
                    confidence=base_conf,
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
    Remains as a fallback when spatial anchor search fails.
    """
    candidates = []
    full_text = doc.full_text

    for pattern in field_config.compiled_patterns:
        for match in pattern.finditer(full_text):
            try:
                raw_value = match.group(1).strip()
            except IndexError:
                raw_value = match.group(0).strip()

            if not raw_value:
                continue

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
                    return _best_bbox(match_bboxes), page.page_no

    return None, 0


def score_candidates(
    candidates: list[FieldCandidate],
    field_config: FieldConfig,
    normalizer_fn,
    validator_fn,
) -> list[FieldCandidate]:
    """
    Score and rank candidates.
    score = base_confidence (anchor×match×ocr OR regex_weight)
          + 0.10 if normalization succeeds
          + 0.15 if all validators pass
          - 0.05 per validation error
          - 0.05 conflict penalty if top candidates disagree
    """
    scored = []
    for c in candidates:
        score = c.confidence

        try:
            norm_val = normalizer_fn(c.raw_value, field_config)
            score += 0.10
            c = c.model_copy(update={"value": norm_val})
        except Exception:
            norm_val = c.raw_value

        try:
            errors = validator_fn(norm_val, field_config)
            if not errors:
                score += 0.15
            else:
                score -= 0.05 * len(errors)
        except Exception:
            pass

        scored.append(c.model_copy(update={"confidence": min(score, 1.0)}))

    # Conflict penalty
    if len(scored) > 1:
        top_val = scored[0].value if scored else None
        for c in scored[1:]:
            if c.value != top_val and c.confidence > 0.4:
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
    1. Spatial anchor search
    2. Regex fallback
    3. Score and pick best candidate
    """
    all_candidates: list[FieldCandidate] = []

    all_candidates.extend(anchor_extract(doc, field_config))
    all_candidates.extend(regex_extract(doc, field_config))

    if not all_candidates:
        return FieldResult(
            field_name=field_config.name,
            status=ExtractionStatus.MISSING,
        )

    scored = score_candidates(all_candidates, field_config, normalizer_fn, validator_fn)
    best = scored[0]

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
        alternatives=scored[1:5],
    )
