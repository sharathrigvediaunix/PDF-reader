"""
OCR pipeline: preprocess image -> run Tesseract -> return NormalizedPage with token bboxes.
"""
import io
import math
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.models import NormalizedPage, Token, BBox
from app.logging_config import get_logger

logger = get_logger("ocr")


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Deterministic image preprocessing:
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
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=5
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
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img


def ocr_image(
    pil_image: Image.Image,
    page_no: int,
    lang: str = "eng",
) -> NormalizedPage:
    """
    Run Tesseract OCR on a PIL image, return NormalizedPage with word bboxes.
    Uses TSV output for word-level bboxes.
    """
    # Convert to OpenCV format
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Preprocess
    processed = preprocess_image(img_cv)

    # Convert back to PIL for pytesseract
    processed_pil = Image.fromarray(processed)

    # Run Tesseract with TSV output (word-level bboxes)
    tsv_data = pytesseract.image_to_data(
        processed_pil,
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",  # LSTM OCR, assume uniform block
    )

    tokens = []
    lines_text: dict[int, list[str]] = {}
    n_boxes = len(tsv_data["text"])

    for i in range(n_boxes):
        word = tsv_data["text"][i].strip()
        if not word:
            continue
        conf = float(tsv_data["conf"][i])
        if conf < 0:  # -1 means non-word
            continue
        x = tsv_data["left"][i]
        y = tsv_data["top"][i]
        w = tsv_data["width"][i]
        h = tsv_data["height"][i]
        line_num = tsv_data["line_num"][i]
        par_num = tsv_data["par_num"][i]
        block_num = tsv_data["block_num"][i]
        line_id = block_num * 1000 + par_num * 100 + line_num

        bbox = BBox(x0=x, y0=y, x1=x + w, y1=y + h, page=page_no)
        tokens.append(Token(
            text=word,
            bbox=bbox,
            line_id=line_id,
            conf=conf / 100.0,  # normalize to 0-1
        ))
        lines_text.setdefault(line_id, []).append(word)

    full_text = "\n".join(" ".join(map(str, words)) for _, words in sorted(lines_text.items()))

    logger.info("ocr_complete", page_no=page_no, token_count=len(tokens))
    return NormalizedPage(
        page_no=page_no,
        tokens=tokens,
        full_text=full_text,
        ocr_used=True,
    )


def ocr_bytes(image_bytes: bytes, page_no: int = 0) -> NormalizedPage:
    """OCR from raw image bytes."""
    pil = Image.open(io.BytesIO(image_bytes))
    return ocr_image(pil, page_no=page_no)
