"""
Field validators and cross-field validation.
Validators take normalized value and return list of error strings (empty = valid).
"""
import re
from datetime import date, datetime
from typing import Any

from app.config.loader import FieldConfig, DocumentConfig
from app.models import FieldResult, ExtractionStatus, ValidationSummary
from app.logging_config import get_logger

logger = get_logger("validators")


def validate_value(value: Any, field_config: FieldConfig) -> list[str]:
    """
    Apply all validators from field_config.validators.
    Returns list of validation error messages.
    """
    errors = []
    if value is None:
        if field_config.required:
            errors.append(f"{field_config.name} is required but missing")
        return errors

    for validator in field_config.validators:
        if isinstance(validator, str):
            vname = validator
            vargs = None
        elif isinstance(validator, dict):
            vname, vargs = next(iter(validator.items()))
        else:
            # Could be "min_length:3" format
            parts = str(validator).split(":")
            vname = parts[0]
            vargs = parts[1] if len(parts) > 1 else None

        err = _apply_validator(vname, vargs, value, field_config)
        if err:
            errors.append(err)

    return errors


def _apply_validator(vname: str, vargs: Any, value: Any, field_config: FieldConfig) -> str:
    """Return error string or empty string if valid."""
    try:
        if ":" in vname:
            parts = vname.split(":", 1)
            vname = parts[0]
            vargs = parts[1]

        if vname == "min_length":
            min_len = int(vargs) if vargs else 1
            if len(str(value)) < min_len:
                return f"{field_config.name} too short (min {min_len} chars)"

        elif vname == "max_length":
            max_len = int(vargs) if vargs else 255
            if len(str(value)) > max_len:
                return f"{field_config.name} too long (max {max_len} chars)"

        elif vname == "valid_date":
            # Accept ISO date strings
            if isinstance(value, str):
                datetime.strptime(value, "%Y-%m-%d")
            elif not isinstance(value, (date, datetime)):
                return f"{field_config.name} is not a valid date"

        elif vname == "not_future":
            today = date.today()
            if isinstance(value, str):
                d = datetime.strptime(value, "%Y-%m-%d").date()
            elif isinstance(value, datetime):
                d = value.date()
            elif isinstance(value, date):
                d = value
            else:
                return ""
            if d > today:
                return f"{field_config.name} is in the future"

        elif vname == "positive_number":
            if float(value) < 0:
                return f"{field_config.name} must be positive"

        elif vname == "non_empty":
            if not str(value).strip():
                return f"{field_config.name} is empty"

    except (ValueError, TypeError) as e:
        return f"{field_config.name} validation error: {str(e)}"

    return ""


def run_cross_field_validation(
    fields: dict[str, FieldResult],
    doc_config: DocumentConfig,
) -> list[str]:
    """
    Run cross-field validations defined in the document config.
    Returns list of error messages.
    """
    errors = []

    for cv in doc_config.cross_field_validations:
        # Get values for involved fields
        vals = {}
        all_present = True
        for fname in cv.fields:
            fr = fields.get(fname)
            if fr and fr.value is not None:
                try:
                    vals[fname] = float(fr.value) if isinstance(fr.value, (int, float, str)) else fr.value
                except (ValueError, TypeError):
                    vals[fname] = fr.value
            else:
                all_present = False
                break

        if not all_present:
            continue  # skip if any field missing

        try:
            # Simple rule evaluation (safe subset)
            result = _eval_rule(cv.rule, vals)
            if not result:
                errors.append(cv.description)
        except Exception as e:
            logger.warning("cross_field_eval_error", rule=cv.rule, error=str(e))

    return errors


def _eval_rule(rule: str, values: dict) -> bool:
    """
    Evaluate simple comparison rules like "total_amount >= tax_amount".
    Only supports: >=, <=, >, <, ==, !=
    """
    # Replace field names with values
    expr = rule
    for fname, val in sorted(values.items(), key=lambda x: -len(x[0])):
        if isinstance(val, (int, float)):
            expr = expr.replace(fname, str(val))
        elif isinstance(val, str):
            expr = expr.replace(fname, f'"{val}"')

    # Only allow safe comparison operators
    if not re.match(r'^[\d\.\s"<>=!]+$', expr):
        return True  # can't evaluate, skip

    return bool(eval(expr))  # noqa: S307  # limited to numeric comparisons


def compute_validation_summary(
    fields: dict[str, FieldResult],
    doc_config: DocumentConfig,
) -> ValidationSummary:
    """Compute overall validation summary."""
    required_missing = []
    for fname, fc in doc_config.fields.items():
        if fc.required:
            fr = fields.get(fname)
            if not fr or fr.status == ExtractionStatus.MISSING:
                required_missing.append(fname)

    cross_errors = run_cross_field_validation(fields, doc_config)

    verified_fields = [f for f in fields.values() if f.status == ExtractionStatus.VERIFIED]
    all_fields = [f for f in fields.values() if f.status != ExtractionStatus.MISSING]

    overall_conf = (
        sum(f.confidence for f in verified_fields) / max(len(all_fields), 1)
        if all_fields else 0.0
    )

    return ValidationSummary(
        required_present=len(required_missing) == 0,
        required_missing=required_missing,
        cross_field_errors=cross_errors,
        overall_confidence=round(overall_conf, 3),
    )
