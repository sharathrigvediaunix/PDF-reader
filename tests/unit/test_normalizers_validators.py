"""
Unit tests for normalizers and validators.
"""
import pytest
from app.validators.normalizers import normalize_value, NormalizationError, _parse_date, _parse_money
from app.validators.rules import validate_value


def make_fc(field_name, normalizers, validators, required=False, field_type="string"):
    from app.config.loader import FieldConfig
    return FieldConfig(
        name=field_name,
        type=field_type,
        required=required,
        anchors=[],
        patterns=[],
        search_window="same_line_or_next",
        normalizers=normalizers,
        validators=validators,
        fallback_allowed=True,
    )


# ── Normalizers ────────────────────────────────────────────────────────────────

def test_strip_normalizer():
    fc = make_fc("test", ["strip"], [])
    assert normalize_value("  hello  ", fc) == "hello"


def test_upper_normalizer():
    fc = make_fc("test", ["upper"], [])
    assert normalize_value("abc", fc) == "ABC"


def test_parse_date_slash():
    assert _parse_date("01/15/2024") == "2024-01-15"


def test_parse_date_iso():
    assert _parse_date("2024-03-20") == "2024-03-20"


def test_parse_date_text():
    result = _parse_date("February 3, 2024")
    assert result == "2024-02-03"


def test_parse_date_invalid():
    with pytest.raises(NormalizationError):
        _parse_date("not a date")


def test_parse_money_standard():
    assert _parse_money("$4,895.00") == pytest.approx(4895.00)


def test_parse_money_plain():
    assert _parse_money("1234.56") == pytest.approx(1234.56)


def test_parse_money_commas():
    assert _parse_money("8,598.63") == pytest.approx(8598.63)


def test_parse_money_invalid():
    with pytest.raises(NormalizationError):
        _parse_money("not-a-number")


# ── Validators ─────────────────────────────────────────────────────────────────

def test_validate_min_length_pass():
    fc = make_fc("test", [], ["min_length:3"])
    errors = validate_value("hello", fc)
    assert errors == []


def test_validate_min_length_fail():
    fc = make_fc("test", [], ["min_length:5"])
    errors = validate_value("hi", fc)
    assert len(errors) > 0


def test_validate_positive_number_pass():
    fc = make_fc("test", [], ["positive_number"], field_type="money")
    errors = validate_value(100.0, fc)
    assert errors == []


def test_validate_positive_number_fail():
    fc = make_fc("test", [], ["positive_number"], field_type="money")
    errors = validate_value(-5.0, fc)
    assert len(errors) > 0


def test_validate_valid_date_pass():
    fc = make_fc("test", [], ["valid_date"], field_type="date")
    errors = validate_value("2024-01-15", fc)
    assert errors == []


def test_validate_valid_date_fail():
    fc = make_fc("test", [], ["valid_date"], field_type="date")
    errors = validate_value("not-a-date", fc)
    assert len(errors) > 0


def test_validate_not_future_pass():
    fc = make_fc("test", [], ["not_future"], field_type="date")
    errors = validate_value("2020-01-01", fc)
    assert errors == []


def test_validate_none_required():
    fc = make_fc("test", [], [], required=True)
    errors = validate_value(None, fc)
    assert len(errors) > 0


def test_validate_none_not_required():
    fc = make_fc("test", [], [], required=False)
    errors = validate_value(None, fc)
    assert errors == []


# ── Cross-field validation ─────────────────────────────────────────────────────

def test_cross_field_total_gte_tax():
    from app.config import load_config
    from app.models import FieldResult, ExtractionStatus, ExtractionMethod
    from app.validators.rules import run_cross_field_validation

    doc_config = load_config("invoice")
    fields = {
        "total_amount": FieldResult(
            field_name="total_amount", value=1000.0, confidence=0.9,
            status=ExtractionStatus.VERIFIED, method=ExtractionMethod.REGEX,
        ),
        "tax_amount": FieldResult(
            field_name="tax_amount", value=200.0, confidence=0.9,
            status=ExtractionStatus.VERIFIED, method=ExtractionMethod.REGEX,
        ),
    }
    errors = run_cross_field_validation(fields, doc_config)
    assert errors == []


def test_cross_field_total_lt_tax_fails():
    from app.config import load_config
    from app.models import FieldResult, ExtractionStatus, ExtractionMethod
    from app.validators.rules import run_cross_field_validation

    doc_config = load_config("invoice")
    fields = {
        "total_amount": FieldResult(
            field_name="total_amount", value=100.0, confidence=0.9,
            status=ExtractionStatus.VERIFIED, method=ExtractionMethod.REGEX,
        ),
        "tax_amount": FieldResult(
            field_name="tax_amount", value=500.0, confidence=0.9,
            status=ExtractionStatus.VERIFIED, method=ExtractionMethod.REGEX,
        ),
    }
    errors = run_cross_field_validation(fields, doc_config)
    assert len(errors) > 0
