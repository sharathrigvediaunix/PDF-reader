"""
Config loader for the supplier-template extraction system.

Each supplier YAML in config/suppliers/*.yaml defines the structural blueprint
for that supplier's documents — anchors, patterns, field positions, table
layouts, repeating record blocks. This is the "learning" about how that
supplier formats their documents.

At extraction time, the caller passes `requested_fields` — only those fields
are extracted from the blueprint. If no fields are requested, all fields in
the blueprint are extracted.

Resolution order:
  1. suppliers/{party_id}.yaml   — rich supplier profile (preferred)
  2. {doc_type}__{party_id}.yaml — legacy exact-match template
  3. {doc_type}__default.yaml    — doc_type fallback
  4. invoice__default.yaml       — global fallback
"""

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import yaml

CONFIG_DIR = os.path.dirname(__file__)
SUPPLIERS_DIR = os.path.join(CONFIG_DIR, "suppliers")


# ── Field config ──────────────────────────────────────────────────────────────


@dataclass
class FieldConfig:
    name: str
    type: str  # string | int | money | date
    required: bool
    anchors: list[str]
    patterns: list[str]
    search_window: str  # maps to spatial extractor strategy
    direction: list[str]  # right | below | left | above
    window: dict  # {x: int, y: int} pixel search window
    normalizers: list[str]
    validators: list[Any]
    fallback_allowed: bool
    multi: bool = False  # True for fields_multi entries
    split_on: list[str] = field(default_factory=list)
    dedupe: bool = False
    mode: str = "single"  # single | extract_all

    @property
    def compiled_patterns(self) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.patterns]


# ── Table / record config ─────────────────────────────────────────────────────


@dataclass
class TableColumnConfig:
    name: str
    type: str
    patterns: list[str] = field(default_factory=list)


@dataclass
class TableConfig:
    name: str
    start_anchors: list[str]
    end_anchors: list[str]
    row_mode: str
    columns: dict[str, TableColumnConfig]
    header_keywords: list[str] = field(default_factory=list)
    stop_keywords: list[str] = field(default_factory=list)


@dataclass
class RecordConfig:
    """
    A repeating block within a document (e.g. one PO block per invoice in a
    Grainger summary bill). Each block has its own fields_single / fields_multi
    / tables.
    """

    name: str
    start_anchors: list[str]
    end_anchors: list[str]
    record_id_anchors: list[str]
    record_id_patterns: list[str]
    fields: dict[str, FieldConfig]
    tables: dict[str, TableConfig] = field(default_factory=dict)


# ── Document config ───────────────────────────────────────────────────────────


@dataclass
class DocumentConfig:
    document_type: str
    party_id: Optional[str]
    profile_id: Optional[str]
    fields: dict[str, FieldConfig]
    tables: dict[str, TableConfig]
    records: dict[str, RecordConfig]
    normalization: dict
    global_rules: dict
    cross_field_validations: list = field(default_factory=list)

    def fields_for_request(
        self, requested_fields: Optional[list[str]]
    ) -> dict[str, FieldConfig]:
        """
        Return only the FieldConfig entries the user asked for.
        If requested_fields is None or empty, return all blueprint fields.
        Unknown names are silently ignored.
        """
        if not requested_fields:
            return self.fields
        return {
            fname: fcfg
            for fname, fcfg in self.fields.items()
            if fname in requested_fields
        }

    def available_fields(self) -> list[str]:
        """All field names defined in this supplier's blueprint."""
        return list(self.fields.keys())


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _direction_to_search_window(direction: list[str]) -> str:
    d = set(d.lower() for d in direction)
    if d == {"right"}:
        return "same_line_right"
    if "right" in d and "below" in d and len(d) == 2:
        return "same_line_or_next"
    if "below" in d:
        return "next_3_lines"
    return "any"


def _parse_field(fname: str, fdata: dict, multi: bool = False) -> FieldConfig:
    direction = fdata.get("direction", ["right", "below"])
    if isinstance(direction, str):
        direction = [direction]
    normalizers = fdata.get("normalize", fdata.get("normalizers", []))
    if isinstance(normalizers, str):
        normalizers = [normalizers]
    return FieldConfig(
        name=fname,
        type=fdata.get("type", "string"),
        required=fdata.get("required", False),
        anchors=fdata.get("anchors", []),
        patterns=fdata.get("patterns", []),
        search_window=_direction_to_search_window(direction),
        direction=direction,
        window=fdata.get("window", {"x": 300, "y": 80}),
        normalizers=normalizers,
        validators=fdata.get("validators", []),
        fallback_allowed=fdata.get("fallback_allowed", True),
        multi=multi,
        split_on=fdata.get("split_on", []),
        dedupe=fdata.get("dedupe", False),
        mode=fdata.get("mode", "extract_all" if multi else "single"),
    )


def _parse_table(tname: str, tdata: dict) -> TableConfig:
    columns = {}
    for cname, cdata in (tdata.get("columns") or {}).items():
        if cdata is None:
            cdata = {}
        columns[cname] = TableColumnConfig(
            name=cname,
            type=cdata.get("type", "string"),
            patterns=cdata.get("patterns", []),
        )
    return TableConfig(
        name=tname,
        start_anchors=tdata.get("start_anchors", tdata.get("header_keywords", [])),
        end_anchors=tdata.get("end_anchors", tdata.get("stop_keywords", [])),
        row_mode=tdata.get("row_mode", "by_line_y_cluster"),
        columns=columns,
        header_keywords=tdata.get("header_keywords", []),
        stop_keywords=tdata.get("stop_keywords", []),
    )


def _parse_record(rname: str, rdata: dict) -> RecordConfig:
    fields = {}
    for fname, fdata in (rdata.get("fields_single") or {}).items():
        fields[fname] = _parse_field(fname, fdata, multi=False)
    for fname, fdata in (rdata.get("fields_multi") or {}).items():
        fields[fname] = _parse_field(fname, fdata, multi=True)
    tables = {}
    for tname, tdata in (rdata.get("tables") or {}).items():
        tables[tname] = _parse_table(tname, tdata)
    rid = rdata.get("record_id") or {}
    return RecordConfig(
        name=rname,
        start_anchors=rdata.get("start_anchors", []),
        end_anchors=rdata.get("end_anchors", []),
        record_id_anchors=rid.get("anchors", []),
        record_id_patterns=rid.get("patterns", []),
        fields=fields,
        tables=tables,
    )


def _parse_supplier_profile(
    raw: dict, document_type: str, party_id: Optional[str]
) -> DocumentConfig:
    fields: dict[str, FieldConfig] = {}
    for fname, fdata in (raw.get("fields_single") or {}).items():
        fields[fname] = _parse_field(fname, fdata, multi=False)
    for fname, fdata in (raw.get("fields_multi") or {}).items():
        fields[fname] = _parse_field(fname, fdata, multi=True)

    tables: dict[str, TableConfig] = {}
    for tname, tdata in (raw.get("tables") or {}).items():
        tables[tname] = _parse_table(tname, tdata)

    records: dict[str, RecordConfig] = {}
    for rname, rdata in (raw.get("records") or {}).items():
        rec = _parse_record(rname, rdata)
        records[rname] = rec
        # Surface record fields at top level so standard extractor can reach them
        for fname, fcfg in rec.fields.items():
            if fname not in fields:
                fields[fname] = fcfg

    return DocumentConfig(
        document_type=document_type,
        party_id=party_id,
        profile_id=raw.get("profile_id"),
        fields=fields,
        tables=tables,
        records=records,
        normalization=raw.get("normalization") or {},
        global_rules=raw.get("global_rules") or {},
    )


def _parse_legacy_config(
    raw: dict, document_type: str, party_id: Optional[str]
) -> DocumentConfig:
    fields: dict[str, FieldConfig] = {}
    for fname, fdata in (raw.get("fields") or {}).items():
        direction = fdata.get("direction", ["right", "below"])
        if isinstance(direction, str):
            direction = [direction]
        fields[fname] = FieldConfig(
            name=fname,
            type=fdata.get("type", "string"),
            required=fdata.get("required", False),
            anchors=fdata.get("anchors", []),
            patterns=fdata.get("patterns", []),
            search_window=fdata.get("search_window", "same_line_or_next"),
            direction=direction,
            window=fdata.get("window", {"x": 300, "y": 80}),
            normalizers=fdata.get("normalizers", []),
            validators=fdata.get("validators", []),
            fallback_allowed=fdata.get("fallback_allowed", True),
        )
    table_config = None
    if "table" in raw:
        table_config = _parse_table("default", raw["table"])
    return DocumentConfig(
        document_type=raw.get("document_type", document_type),
        party_id=party_id,
        profile_id=None,
        fields=fields,
        tables={"default": table_config} if table_config else {},
        records={},
        normalization={},
        global_rules={},
    )


# ── Template loader ───────────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def load_config(document_type: str, party_id: Optional[str] = None) -> DocumentConfig:
    """Load the structural blueprint for a (document_type, party_id) combination."""
    if party_id:
        supplier_path = os.path.join(SUPPLIERS_DIR, f"{party_id}.yaml")
        if os.path.exists(supplier_path):
            with open(supplier_path) as f:
                raw = yaml.safe_load(f)
            return _parse_supplier_profile(raw, document_type, party_id)

    candidates = []
    if party_id:
        candidates.append(f"{document_type}__{party_id}.yaml")
    candidates.append(f"{document_type}__default.yaml")
    if "invoice__default.yaml" not in candidates:
        candidates.append("invoice__default.yaml")

    for filename in candidates:
        path = os.path.join(CONFIG_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                raw = yaml.safe_load(f)
            return _parse_legacy_config(raw, document_type, party_id)

    raise FileNotFoundError(
        f"No config found for document_type={document_type}, party_id={party_id}."
    )


def get_available_fields(party_id: str, document_type: str = "invoice") -> list[str]:
    """Return all field names defined in a supplier's blueprint."""
    try:
        return load_config(document_type, party_id).available_fields()
    except FileNotFoundError:
        return []


# ── Supplier identification ───────────────────────────────────────────────────


@lru_cache(maxsize=32)
def _load_supplier_profiles() -> list[dict]:
    if not os.path.exists(SUPPLIERS_DIR):
        return []
    profiles = []
    for fname in sorted(os.listdir(SUPPLIERS_DIR)):
        if not fname.endswith(".yaml"):
            continue
        with open(os.path.join(SUPPLIERS_DIR, fname)) as f:
            cfg = yaml.safe_load(f)
        if cfg:
            cfg["_file_id"] = fname[:-5]  # filename stem = party_id
            profiles.append(cfg)
    return profiles


def identify_party(text: str) -> tuple[Optional[str], float]:
    """
    Match document text against all supplier profiles.
    Uses vendor.name_contains (new schema) or fingerprints (legacy).
    Returns (party_id, confidence).
    """
    text_upper = text.upper()
    best_id: Optional[str] = None
    best_score = 0.0

    for profile in _load_supplier_profiles():
        keywords = (profile.get("vendor") or {}).get("name_contains", []) + profile.get(
            "fingerprints", []
        )
        if not keywords:
            continue
        matches = sum(1 for kw in keywords if str(kw).upper() in text_upper)
        score = matches / len(keywords)
        if score > best_score:
            best_score = score
            best_id = profile["_file_id"]

    return best_id, round(best_score, 2)


def identify_document_type(text: str) -> tuple[str, float]:
    text_upper = text.upper()
    type_markers = {
        "summary_bill": ["SUMMARY BILL INVOICE", "SUMMARY BILL NUMBER"],
        "commercial_invoice": ["COMMERCIAL INVOICE"],
        "packing_list": ["PACKING LIST", "PACKING SLIP"],
        "invoice": ["INVOICE", "TAX INVOICE"],
        "purchase_order": ["PURCHASE ORDER", "P.O.", "PO NUMBER"],
        "sales_order": ["SALES ORDER", "SO NUMBER"],
        "delivery_note": ["DELIVERY NOTE", "DELIVERY ORDER"],
        "credit_memo": ["CREDIT MEMO", "CREDIT NOTE"],
    }
    for doc_type, markers in type_markers.items():
        for marker in markers:
            if marker in text_upper:
                return doc_type, 0.95
    return "invoice", 0.50
