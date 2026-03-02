"""
OCR pipeline: preprocess image -> PaddleOCR -> NormalizedPage with token bboxes.
PaddleOCR replaces Tesseract for better accuracy on structured documents,
tables, and mixed-language content (e.g. Chinese + English invoices).
"""

import io
import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.models import NormalizedPage, Token, BBox
from app.logging_config import get_logger

logger = get_logger("ocr")

# PaddleOCR is imported lazily to avoid slow startup when not needed
_paddle_ocr = None


def _get_paddle_ocr():
    """Lazy-load PaddleOCR engine (expensive to initialize)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR

            # use_angle_cls: auto-rotate detected text blocks
            # lang: en covers most invoice content; add 'ch' for Chinese
            # use_gpu: set True if GPU available
            _paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
            logger.info("paddleocr_initialized")
        except ImportError:
            logger.error("paddleocr_not_installed")
            raise RuntimeError(
                "PaddleOCR is not installed. Run: pip install paddlepaddle paddleocr"
            )
    return _paddle_ocr


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Deterministic image preprocessing pipeline:
    1. Grayscale
    2. Denoise
    3. Adaptive threshold (binarize)
    4. Deskew
    """
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # 3. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=5,
    )

    # 4. Deskew
    thresh = _deskew(thresh)

    return thresh


def _deskew(img: np.ndarray) -> np.ndarray:
    """Correct skew using Hough line transform."""
    try:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return img
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception:
        return img


def _paddle_result_to_tokens(
    result: list,
    page_no: int,
) -> list[Token]:
    """
    Convert PaddleOCR result format to Token list.

    PaddleOCR returns:
    [
      [
        [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  # quad bbox
        ("text", confidence)
      ],
      ...
    ]
    We convert the quad bbox to axis-aligned BBox (x0,y0,x1,y1).
    Line IDs are assigned by Y-clustering (done later in layout_normalizer).
    """
    tokens = []
    if not result or not result[0]:
        return tokens

    for item in result[0]:
        if not item or len(item) < 2:
            continue

        quad = item[0]  # 4 corner points
        text_conf = item[1]  # ("text", confidence)

        if not quad or not text_conf:
            continue

        text = str(text_conf[0]).strip()
        conf = float(text_conf[1]) if text_conf[1] is not None else 1.0

        if not text:
            continue

        # Convert quad to axis-aligned bbox
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        bbox = BBox(
            x0=min(xs),
            y0=min(ys),
            x1=max(xs),
            y1=max(ys),
            page=page_no,
        )

        # PaddleOCR returns whole text blocks — split into word tokens
        words = text.split()
        if not words:
            continue

        word_width = bbox.width / len(words)
        for w_idx, word in enumerate(words):
            word_bbox = BBox(
                x0=bbox.x0 + w_idx * word_width,
                y0=bbox.y0,
                x1=bbox.x0 + (w_idx + 1) * word_width,
                y1=bbox.y1,
                page=page_no,
            )
            tokens.append(
                Token(
                    text=word,
                    bbox=word_bbox,
                    line_id=0,  # reassigned by layout_normalizer
                    conf=conf,
                )
            )

    return tokens


def ocr_image(
    pil_image: Image.Image,
    page_no: int,
    lang: str = "en",
) -> NormalizedPage:
    """
    Run PaddleOCR on a PIL image.
    Returns NormalizedPage with word-level tokens and bboxes.
    """
    # Convert to numpy array for preprocessing
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Preprocess (deskew, denoise, binarize)
    processed = preprocess_image(img_cv)

    # Convert back to RGB PIL for PaddleOCR
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    # Run PaddleOCR
    ocr = _get_paddle_ocr()
    result = ocr.ocr(processed_rgb, cls=True)

    # Convert to tokens
    tokens = _paddle_result_to_tokens(result, page_no)

    # Build full_text from tokens (rough ordering by Y then X)
    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y0, t.bbox.x0))
    full_text = " ".join(t.text for t in sorted_tokens)

    logger.info("ocr_complete", page_no=page_no, token_count=len(tokens))

    return NormalizedPage(
        page_no=page_no,
        tokens=tokens,
        full_text=full_text,
        ocr_used=True,
        pil_image=pil_image,  # store original (pre-preprocess) image for LayoutLM
    )


def ocr_bytes(image_bytes: bytes, page_no: int = 0) -> NormalizedPage:
    """OCR from raw image bytes."""
    pil = Image.open(io.BytesIO(image_bytes))
    return ocr_image(pil, page_no=page_no)
