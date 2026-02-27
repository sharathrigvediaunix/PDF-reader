"""
Field value normalizers.
Normalizer functions take a raw string and FieldConfig, return normalized value.
"""
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from dateutil import parser as dateutil_parser

from app.config.loader import FieldConfig


class NormalizationError(ValueError):
    pass


def normalize_value(raw: str, field_config: FieldConfig) -> Any:
    """
    Apply normalizers sequentially as listed in field config.
    Returns normalized value.
    """
    value = raw
    for norm in field_config.normalizers:
        if norm == "strip":
            value = str(value).strip()
        elif norm == "upper":
            value = str(value).upper()
        elif norm == "lower":
            value = str(value).lower()
        elif norm == "title_case":
            value = str(value).title()
        elif norm == "parse_date":
            value = _parse_date(str(value))
        elif norm == "parse_money":
            value = _parse_money(str(value))
        elif norm == "remove_currency_symbol":
            value = re.sub(r'[$€£¥₹]', '', str(value)).strip()
        # unknown normalizers are silently ignored

    return value


def _parse_date(raw: str) -> str:
    """Parse various date formats to ISO 8601 (YYYY-MM-DD)."""
    raw = raw.strip()
    if not raw:
        raise NormalizationError("empty date string")
    try:
        dt = dateutil_parser.parse(raw, dayfirst=False)
        return dt.date().isoformat()
    except (ValueError, OverflowError) as e:
        raise NormalizationError(f"Cannot parse date: {raw}") from e


def _parse_money(raw: str) -> float:
    """
    Parse money strings like "$1,234.56" or "1.234,56" (European style).
    Returns float.
    """
    raw = raw.strip()
    # Remove currency symbols and spaces
    cleaned = re.sub(r'[€£¥₹$\s]', '', raw)
    # Handle European number format (1.234,56 -> 1234.56)
    if re.match(r'^\d{1,3}(\.\d{3})*(,\d{2})?$', cleaned):
        cleaned = cleaned.replace('.', '').replace(',', '.')
    else:
        # Standard: remove commas
        cleaned = cleaned.replace(',', '')
    try:
        return float(Decimal(cleaned))
    except (InvalidOperation, ValueError) as e:
        raise NormalizationError(f"Cannot parse money: {raw}") from e
