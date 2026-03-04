"""
LayoutLM v3 fallback extractor (Option A — Phase 1).

Uses microsoft/layoutlmv3-base pretrained checkpoint.
Called ONLY when deterministic extraction confidence is below threshold.
Does NOT require fine-tuning to provide value — the pretrained model
understands document layout, key-value relationships, and table structure.

Fine-tuning on your own invoice data (Phase 2) will significantly
improve accuracy beyond what the pretrained model provides.

Architecture:
  - Input: page image + token bboxes (normalized 0-1000)
  - Model: LayoutLMv3ForTokenClassification
  - Output: per-token field classification

For Phase 1, we use it in a zero-shot / feature-extraction mode:
  - Get token embeddings from the pretrained model
  - Find tokens most semantically similar to the field label
  - Return the token with highest similarity as the candidate value
"""

from __future__ import annotations

from typing import Optional, Any

from PIL import Image, ImageDraw

from app.models import (
    NormalizedDocument,
    NormalizedPage,
    Token,
    BBox,
    FieldResult,
    FieldCandidate,
    ExtractionMethod,
    SourceLocation,
    Evidence,
    ExtractionStatus,
)
from app.config.loader import FieldConfig
from app.logging_config import get_logger

logger = get_logger("layoutlm")

MODEL_NAME = "microsoft/layoutlmv3-base"
LAYOUTLM_CONFIDENCE_BASE = 0.65  # base confidence for LayoutLM results
LAYOUTLM_FALLBACK_THRESHOLD = 0.50  # only trigger if deterministic conf < this

# Lazy-loaded model and processor
_model = None
_processor = None


def _load_model():
    """Lazy-load LayoutLMv3 model and processor."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    try:
        from transformers import (
            LayoutLMv3Processor,
            LayoutLMv3Model,
        )
        import torch

        logger.info("layoutlm_loading", model=MODEL_NAME)
        _processor = LayoutLMv3Processor.from_pretrained(
            MODEL_NAME,
            apply_ocr=False,  # we supply our own OCR tokens + bboxes
        )
        _model = LayoutLMv3Model.from_pretrained(MODEL_NAME)
        _model.eval()
        logger.info("layoutlm_loaded")
        return _model, _processor

    except ImportError:
        logger.error("transformers_not_installed")
        raise RuntimeError(
            "transformers and torch are required for LayoutLM. "
            "Run: pip install transformers torch"
        )


def _normalize_bbox(bbox: BBox, page_width: float, page_height: float) -> list[int]:
    """
    Normalize bbox to 0-1000 range as required by LayoutLMv3.
    LayoutLMv3 expects [x0, y0, x1, y1] normalized to 0-1000.
    """
    return [
        int(1000 * bbox.x0 / page_width),
        int(1000 * bbox.y0 / page_height),
        int(1000 * bbox.x1 / page_width),
        int(1000 * bbox.y1 / page_height),
    ]


def _page_dimensions(page: NormalizedPage) -> tuple[float, float]:
    """Estimate page dimensions from token bboxes."""
    if not page.tokens:
        return 800.0, 1100.0
    max_x = max(t.bbox.x1 for t in page.tokens)
    max_y = max(t.bbox.y1 for t in page.tokens)
    return max(max_x, 1.0), max(max_y, 1.0)


def _render_page_image(page: NormalizedPage) -> Image.Image:
    """
    Return a PIL Image for the page.
    - Scanned pages: use the stored pil_image captured during OCR.
    - Digital PDF pages: render a synthetic image by drawing token bboxes as
      filled rectangles on a white canvas.  This gives LayoutLMv3 real spatial
      structure to work with instead of a completely blank image.
    """
    page_w, page_h = _page_dimensions(page)

    if page.pil_image is not None:
        # Best case — real rasterised image from OCR pipeline
        return page.pil_image.convert("RGB")

    # Fallback: synthesise a layout image from token bounding boxes
    img = Image.new("RGB", (int(page_w), int(page_h)), color=255)
    draw = ImageDraw.Draw(img)
    for tok in page.tokens:
        b = tok.bbox
        # Draw a light-grey filled rect per token to represent text blocks
        draw.rectangle([b.x0, b.y0, b.x1, b.y1], fill=180)
    return img


def layoutlm_extract_field(
    doc: NormalizedDocument,
    field_config: FieldConfig,
    current_result: FieldResult,
) -> FieldResult:
    """
    Use LayoutLMv3 to extract a field when deterministic confidence is low.

    Strategy (Phase 1 — pretrained model, no fine-tuning):
    1. For each page, get contextual embeddings from LayoutLMv3
    2. Find the anchor token using field anchors
    3. Get embedding of anchor token
    4. Find the token with highest cosine similarity to the anchor
       that is spatially near the anchor (right or below)
    5. Return that token as the field value candidate

    This leverages LayoutLMv3's pretrained understanding of document
    structure without any task-specific fine-tuning.
    """
    try:
        import torch
        import torch.nn.functional as F

        model, processor = _load_model()
    except Exception as e:
        logger.error("layoutlm_load_failed", error=str(e))
        return current_result

    best_candidate: Optional[FieldCandidate] = None

    for page in doc.pages:
        if not page.tokens:
            continue

        page_w, page_h = _page_dimensions(page)
        words = [t.text for t in page.tokens]
        boxes = [_normalize_bbox(t.bbox, page_w, page_h) for t in page.tokens]

        # Build a proper page image — real for scanned pages, synthetic for digital PDFs
        image = _render_page_image(page)

        try:
            encoding = processor(
                image,
                words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
        except Exception as e:
            logger.warning("layoutlm_encoding_failed", error=str(e))
            continue

        with torch.no_grad():
            outputs = model(**encoding)

        # Token embeddings: shape [1, seq_len, hidden_size]
        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

        # Find anchor token indices
        anchor_indices = []
        for anchor in field_config.anchors:
            anchor_words = anchor.lower().split()
            for i, word in enumerate(words):
                if word.lower().strip(":#") == anchor_words[0]:
                    anchor_indices.append(i)

        if not anchor_indices:
            continue

        anchor_idx = anchor_indices[0]

        # Clamp to processor token count (words may be split by tokenizer)
        input_ids = encoding["input_ids"][0]
        if anchor_idx >= len(input_ids):
            continue

        anchor_embedding = token_embeddings[anchor_idx]  # [hidden_size]

        # Find the most similar nearby token
        anchor_tok = page.tokens[anchor_idx]
        best_sim = -1.0
        best_tok = None

        for j, tok in enumerate(page.tokens):
            if j == anchor_idx:
                continue
            if j >= token_embeddings.shape[0]:
                break

            # Only consider tokens to the right or below anchor
            is_right = (
                abs(tok.bbox.center_y - anchor_tok.bbox.center_y) <= 15
                and tok.bbox.x0 > anchor_tok.bbox.x1
            )
            is_below = (
                tok.bbox.y0 > anchor_tok.bbox.y1
                and tok.bbox.y0 - anchor_tok.bbox.y1 <= 100
                and abs(tok.bbox.center_x - anchor_tok.bbox.center_x) <= 80
            )

            if not (is_right or is_below):
                continue

            # Skip obvious label tokens
            if tok.text.strip().endswith(":") and len(tok.text) < 30:
                continue

            tok_embedding = token_embeddings[j]
            sim = F.cosine_similarity(
                anchor_embedding.unsqueeze(0),
                tok_embedding.unsqueeze(0),
            ).item()

            if sim > best_sim:
                best_sim = sim
                best_tok = tok

        if best_tok is None:
            continue

        # Compute confidence: base + cosine similarity contribution
        confidence = min(
            LAYOUTLM_CONFIDENCE_BASE + (best_sim - 0.5) * 0.3,
            0.80,  # cap at 0.80 for pretrained (fine-tuned can go higher)
        )

        candidate = FieldCandidate(
            value=best_tok.text,
            raw_value=best_tok.text,
            confidence=confidence,
            method=ExtractionMethod.LAYOUTLM,
            source=SourceLocation(
                page=page.page_no,
                bbox=best_tok.bbox.to_dict(),
            ),
            evidence=Evidence(
                snippet=f"LayoutLMv3 similarity={best_sim:.3f} anchor='{field_config.anchors[0] if field_config.anchors else ''}'",
            ),
        )

        if best_candidate is None or confidence > best_candidate.confidence:
            best_candidate = candidate

    if best_candidate is None:
        logger.debug("layoutlm_no_candidate", field=field_config.name)
        return current_result

    # Only upgrade the result if LayoutLM is more confident
    if best_candidate.confidence <= current_result.confidence:
        return current_result

    logger.info(
        "layoutlm_upgraded_field",
        field=field_config.name,
        old_conf=current_result.confidence,
        new_conf=best_candidate.confidence,
    )

    status = (
        ExtractionStatus.NEEDS_REVIEW
        if best_candidate.confidence < 0.75
        else ExtractionStatus.VERIFIED
    )

    return FieldResult(
        field_name=field_config.name,
        value=best_candidate.value,
        raw_value=best_candidate.raw_value,
        confidence=best_candidate.confidence,
        status=status,
        method=ExtractionMethod.LAYOUTLM,
        source=best_candidate.source,
        evidence=best_candidate.evidence,
        alternatives=current_result.alternatives,
        validation_errors=current_result.validation_errors,
    )


def should_use_layoutlm(result: FieldResult, field_config: FieldConfig) -> bool:
    """
    Return True if LayoutLM should be tried for this field.
    Triggered when:
    - Field is required AND missing/low-confidence
    - Confidence is below LAYOUTLM_FALLBACK_THRESHOLD
    - field_config.fallback_allowed is True
    """
    if not field_config.fallback_allowed:
        return False
    if result.status == ExtractionStatus.MISSING:
        return True
    if result.confidence < LAYOUTLM_FALLBACK_THRESHOLD:
        return True
    return False
