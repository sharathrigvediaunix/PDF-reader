"""
YAML field config loader. One YAML per document_type.
"""
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__))


@dataclass
class FieldConfig:
    name: str
    type: str
    required: bool
    anchors: list[str]
    patterns: list[str]
    search_window: str
    normalizers: list[str]
    validators: list[Any]
    fallback_allowed: bool

    @property
    def compiled_patterns(self) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.patterns]


@dataclass
class CrossFieldValidation:
    name: str
    description: str
    fields: list[str]
    rule: str


@dataclass
class DocumentConfig:
    document_type: str
    fields: dict[str, FieldConfig]
    cross_field_validations: list[CrossFieldValidation]


@lru_cache(maxsize=32)
def load_config(document_type: str) -> DocumentConfig:
    config_path = os.path.join(CONFIG_DIR, f"{document_type}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found for document_type={document_type}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    fields = {}
    for fname, fdata in raw.get("fields", {}).items():
        fields[fname] = FieldConfig(
            name=fname,
            type=fdata.get("type", "string"),
            required=fdata.get("required", False),
            anchors=fdata.get("anchors", []),
            patterns=fdata.get("patterns", []),
            search_window=fdata.get("search_window", "same_line_or_next"),
            normalizers=fdata.get("normalizers", []),
            validators=fdata.get("validators", []),
            fallback_allowed=fdata.get("fallback_allowed", True),
        )

    cross_vals = []
    for cv in raw.get("cross_field_validations", []):
        cross_vals.append(CrossFieldValidation(
            name=cv["name"],
            description=cv["description"],
            fields=cv["fields"],
            rule=cv["rule"],
        ))

    return DocumentConfig(
        document_type=raw.get("document_type", document_type),
        fields=fields,
        cross_field_validations=cross_vals,
    )
