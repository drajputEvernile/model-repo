"""Candidate log: one feature row per candidate each field's rules produced on a page.

Pages where a field produced nothing get one placeholder row, so recall can be measured
from the log alone. Rows are keyed by a candidate_id that is stable across reruns of the
same OCR, which is what labels are joined on.

Heading fields (one per layout detector) use the same row: the heading text is the value,
region is the kind (title / section / page_header), rule_source the detector class,
rule_score the detector score and detail the level; the heading_* columns hold the layout
features (height against body text, overlap with KV keys and values, whole line or not).

rule_* is always what the rules decided (a feature for training). A run with a trained
version also fills model_score / model_accepted / model_selected / model_level
(training/model.py). What the run extracted (every key-value pair it found, which accuracy
counts) is model_accepted when filled, else rule_accepted (run_accepted()); what it selected
for the record output is model_selected, else rule_selected (run_selected()).
"""

from __future__ import annotations

import csv
import hashlib
import re
from functools import lru_cache

from ..heading.extract import DETECTORS
from ..util import config
from ..util.dates import find_dates
from ..util.geometry import Box, Word, median_height, union_boxes
from ..util.keys import KV_ROOT, KeyHit
from ..util.mentions import locate_value

from .normalize import normalize

COLUMNS = [
    # identity
    "run_id", "model_version", "catalog_hash", "ner_model", "record_id", "source_system", "file_name",
    "page_number", "page_count", "ocr_sha1", "field", "candidate_id", "is_placeholder",
    # rule output
    "key", "region", "rule_source", "rule_score", "rule_accepted", "rule_selected", "rule_note",
    "value", "value_norm", "detail",
    # key
    "key_text", "key_match", "key_edit_distance", "key_trusted_reason", "key_weak", "key_cluster_size",
    "key_n_words", "key_words", "key_x0", "key_y0", "key_x1", "key_y1",
    # value location and key -> value relation (word indexes are OCR word positions on the page)
    "value_found", "value_words", "value_x0", "value_y0", "value_x1", "value_y1",
    "relation", "word_gap", "line_gap", "dx",
    # value shape
    "value_len", "value_n_words", "digit_ratio", "alpha_ratio", "upper_ratio",
    "has_comma", "has_initial", "value_is_date",
    # page context
    "page_w", "page_h", "page_frac", "n_keys_page", "n_trusted_keys_page", "n_field_keys",
    "n_field_candidates", "n_field_accepted", "page_value_count", "page_distinct_values",
    # record context
    "record_value_pages", "record_value_share", "record_distinct_values",
    # heading layout
    "heading_height_ratio", "heading_key_overlap", "heading_key_field", "heading_value_overlap",
    "heading_whole_line", "heading_line_count", "heading_ends_colon", "heading_position", "heading_tail_words",
    # trained model (blank in a v0 run): probability, what it extracted, what it selected, heading level it gave
    "model_score", "model_accepted", "model_selected", "model_level",
]


@lru_cache(maxsize=1)
def catalog_hash() -> str:
    """Hash of every key catalog, so a label set can be tied to the keys it was made with."""
    digest = hashlib.sha1()
    for path in sorted([*KV_ROOT.glob("*/keys.json"), *KV_ROOT.glob("*/roles.json"), *KV_ROOT.glob("*/key_blocklist.json")]):
        digest.update(path.relative_to(KV_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


@lru_cache(maxsize=1)
def record_sources() -> dict[str, str]:
    """RecordId -> source system / layout, from the optional record_sources.csv."""
    path = config.Training_Data / "record_sources.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("RecordId") or "").strip(): str(row.get("Source") or "").strip()
            for row in csv.DictReader(handle)
            if row.get("RecordId")
        }


def ocr_sha1(words: list[Word]) -> str:
    return hashlib.sha1("\n".join(word.content for word in words).encode("utf-8")).hexdigest()[:16]


def field_value(field: str, row) -> tuple[str, str]:
    """(main value, field-specific detail) of a rule row."""
    if field in DETECTORS:
        return row.text, row.level
    if field == "electronic_signature":
        return row.provider_name, row.signature_date
    if field == "page_no":
        return row.page_no, row.page_total
    if field == "dos":
        return row.value, row.tier
    if field == "provider_name":
        return row.value, row.profile
    return row.value, ""


def run_selected(row) -> bool:
    """What the run selected: the trained model's choice when it made one, else the rules'."""
    model = str(row.get("model_selected", "") or "")
    return (model if model else str(row.get("rule_selected", "") or "")) in {"1", "True", "true"}


def run_accepted(row) -> bool:
    """Whether the run extracted this key-value pair: the trained model's call when it made one, else the rules'."""
    model = str(row.get("model_accepted", "") or "")
    return (model if model else str(row.get("rule_accepted", "") or "")) in {"1", "True", "true"}


def run_level(row) -> str:
    return str(row.get("model_level", "") or "") or str(row.get("detail", "") or "")


def _relation(key_box: Box | None, value_box: Box | None, line_h: float) -> str:
    if key_box is None:
        return "keyless"
    if value_box is None:
        return "unknown"
    if abs(value_box.cy - key_box.cy) <= 0.6 * line_h:
        if value_box.left >= key_box.right - 0.5 * line_h:
            return "right"
        if value_box.right <= key_box.left + 0.5 * line_h:
            return "left"
        return "overlap"
    return "below" if value_box.cy > key_box.cy else "above"


def _word_gap(key_idx: list[int], value_idx: list[int], positions: dict[int, int]) -> int | None:
    """Words between the key and the value in OCR reading order."""
    keys = [positions[i] for i in key_idx if i in positions]
    values = [positions[i] for i in value_idx if i in positions]
    if not keys or not values:
        return None
    if min(values) > max(keys):
        return min(values) - max(keys) - 1
    if max(values) < min(keys):
        return min(keys) - max(values) - 1
    return 0


def _ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _shape(value: str) -> dict:
    letters = sum(ch.isalpha() for ch in value)
    words = value.split()
    return {
        "value_len": len(value),
        "value_n_words": len(words),
        "digit_ratio": _ratio(sum(ch.isdigit() for ch in value), len(value)),
        "alpha_ratio": _ratio(letters, len(value)),
        "upper_ratio": _ratio(sum(ch.isupper() for ch in value), letters),
        "has_comma": int("," in value),
        "has_initial": int(any(re.fullmatch(r"[A-Za-z]\.?", word) for word in words)),
        "value_is_date": int(bool(find_dates(value))),
    }


def _norm_box(box: Box | None, page_w: float, page_h: float, prefix: str) -> dict:
    if box is None:
        return {f"{prefix}_{name}": "" for name in ("x0", "y0", "x1", "y1")}
    return {
        f"{prefix}_x0": round(box.left / page_w, 4),
        f"{prefix}_y0": round(box.top / page_h, 4),
        f"{prefix}_x1": round(box.right / page_w, 4),
        f"{prefix}_y1": round(box.bottom / page_h, 4),
    }


def _candidate_id(parts: list[str], seen: set[str]) -> str:
    base = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    candidate, n = base, 1
    while candidate in seen:
        n += 1
        candidate = f"{base}-{n}"
    seen.add(candidate)
    return candidate


def _heading_rows(
    base: dict,
    record_id: str,
    page_number: str,
    field: str,
    rows: list,
    norms: list[str],
    accepted_norms: list[str],
    page_w: float,
    page_h: float,
    seen: set[str],
) -> list[dict]:
    out: list[dict] = []
    for row, norm in zip(rows, norms):
        indexes = [word.index for word in row.words]
        out.append({
            **base,
            # Detector class and score stay out of the id, so a threshold or model change keeps the labels.
            "candidate_id": _candidate_id(
                [record_id, page_number, field, ",".join(map(str, indexes)), norm], seen
            ),
            "is_placeholder": 0,
            "region": row.kind,
            "rule_source": row.det_class,
            "rule_score": row.score,
            "rule_accepted": int(row.accepted),
            "rule_selected": int(row.selected),
            "rule_note": row.note,
            "value": row.text,
            "value_norm": norm,
            "detail": row.level,
            "key_match": "keyless",
            "key_n_words": 0,
            "value_found": 1,
            "value_words": " ".join(map(str, indexes)),
            **_norm_box(row.box, page_w, page_h, "value"),
            **_shape(row.text),
            "page_value_count": accepted_norms.count(norm) if norm else 0,
            "heading_height_ratio": row.height_ratio,
            "heading_key_overlap": row.key_overlap,
            "heading_key_field": row.key_field,
            "heading_value_overlap": row.value_overlap,
            "heading_whole_line": int(row.whole_line),
            "heading_line_count": row.line_count,
            "heading_ends_colon": int(row.words[-1].content.rstrip().endswith(":")) if row.words else 0,
            "heading_position": row.position,
            "heading_tail_words": row.tail_words,
        })
    return out


def page_rows(record_id: str, page_count: int, result, fields: list[str]) -> list[dict]:
    """Feature rows for one PageResult (record context is filled in by document_rows)."""
    words = result.words
    positions = {word.index: position for position, word in enumerate(words)}
    by_index = {word.index: word for word in words}
    page_w, page_h = result.page_w or 1.0, result.page_h or 1.0
    line_h = median_height(words) or 10.0
    page_number = str(result.page.get("pageNumber") or "")
    page_index = int(page_number) if page_number.isdigit() else 0
    identity = {
        "record_id": record_id,
        "source_system": record_sources().get(record_id, ""),
        "file_name": str(result.page.get("fileName") or ""),
        "page_number": page_number,
        "page_count": page_count,
        "ocr_sha1": ocr_sha1(words),
        "page_w": page_w,
        "page_h": page_h,
        "page_frac": _ratio(page_index, page_count),
        "n_keys_page": len(result.hits),
        "n_trusted_keys_page": sum(hit.trusted for hit in result.hits),
    }
    seen: set[str] = set()
    out: list[dict] = []
    for field in fields:
        rows = result.rows.get(field) or []
        values = [field_value(field, row) for row in rows]
        norms = [normalize(field, value, detail) for value, detail in values]
        accepted_norms = [norm for row, norm in zip(rows, norms) if row.accepted and norm]
        base = {
            **identity,
            "field": field,
            "n_field_keys": sum(hit.field == field for hit in result.hits),
            "n_field_candidates": len(rows),
            "n_field_accepted": sum(bool(row.accepted) for row in rows),
            "page_distinct_values": len(set(accepted_norms)),
        }
        if not rows:
            out.append({
                **base,
                "candidate_id": _candidate_id([record_id, page_number, field, "__none__"], seen),
                "is_placeholder": 1,
            })
            continue
        if field in DETECTORS:
            out.extend(_heading_rows(base, record_id, page_number, field, rows, norms, accepted_norms, page_w, page_h, seen))
            continue
        for row, (value, detail), norm in zip(rows, values, norms):
            hit: KeyHit | None = getattr(row, "key_hit", None)
            key_box = hit.box if hit is not None else None
            if field == "page_no":
                value_box = row.box
                value_idx = [
                    word.index for word in words
                    if value_box is not None
                    and value_box.left <= word.box.cx <= value_box.right
                    and value_box.top <= word.box.cy <= value_box.bottom
                ]
            else:
                run = locate_value(field, row, value, words) if value else []
                value_box = union_boxes([word.box for word in run])
                value_idx = [word.index for word in run]
            key_h = key_box.height() if key_box is not None and key_box.height() > 0 else line_h
            gap = _word_gap(hit.word_indexes, value_idx, positions) if hit is not None else None
            relation = _relation(key_box, value_box, key_h)
            out.append({
                **base,
                "candidate_id": _candidate_id(
                    [
                        record_id, page_number, field, row.key,
                        ",".join(map(str, hit.word_indexes)) if hit is not None
                        else ",".join(str(word.index) for word in getattr(row, "value_at", None) or []),
                        norm, row.source,
                    ],
                    seen,
                ),
                "is_placeholder": 0,
                "key": row.key,
                "region": row.region,
                "rule_source": row.source,
                "rule_score": row.score,
                "rule_accepted": int(bool(row.accepted)),
                "rule_selected": int(bool(row.selected)),
                "value": value,
                "value_norm": norm,
                "detail": detail,
                "key_text": (
                    " ".join(by_index[i].content for i in hit.word_indexes if i in by_index)
                    if hit is not None else ""
                ),
                "key_match": hit.match if hit is not None else "keyless",
                "key_edit_distance": hit.edit_distance if hit is not None else "",
                "key_trusted_reason": hit.trusted_reason if hit is not None else "",
                "key_weak": int(hit.weak) if hit is not None else "",
                "key_cluster_size": hit.cluster_size if hit is not None else "",
                "key_n_words": len(hit.word_indexes) if hit is not None else 0,
                "key_words": " ".join(map(str, hit.word_indexes)) if hit is not None else "",
                **_norm_box(key_box, page_w, page_h, "key"),
                "value_found": int(value_box is not None),
                "value_words": " ".join(map(str, value_idx)),
                **_norm_box(value_box, page_w, page_h, "value"),
                "relation": relation,
                "word_gap": "" if gap is None else gap,
                "line_gap": (
                    round((value_box.cy - key_box.cy) / key_h, 3)
                    if key_box is not None and value_box is not None else ""
                ),
                "dx": (
                    round((value_box.left - key_box.right) / page_w, 4)
                    if key_box is not None and value_box is not None else ""
                ),
                **_shape(value),
                "page_value_count": accepted_norms.count(norm) if norm else 0,
            })
    return out


def document_rows(result, run_id: str, model_version: str, fields: list[str]) -> list[dict]:
    """Candidate log rows for a DocumentResult, with record-level agreement features."""
    page_count = len(result.pages)
    rows: list[dict] = []
    for page in result.pages:
        rows.extend(page_rows(result.record_id, page_count, page, fields))

    pages_with: dict[tuple[str, str], set[str]] = {}
    field_pages: dict[str, set[str]] = {}
    for row in rows:
        if row.get("is_placeholder") or not row.get("rule_accepted") or not row.get("value_norm"):
            continue
        pages_with.setdefault((row["field"], row["value_norm"]), set()).add(row["page_number"])
        field_pages.setdefault(row["field"], set()).add(row["page_number"])
    distinct: dict[str, int] = {}
    for field, _ in pages_with:
        distinct[field] = distinct.get(field, 0) + 1

    stamp = {
        "run_id": run_id,
        "model_version": model_version,
        "catalog_hash": catalog_hash(),
        "ner_model": config.Ner_Model_Path.name,
    }
    out: list[dict] = []
    for row in rows:
        row.update(stamp)
        field = row["field"]
        count = len(pages_with.get((field, row.get("value_norm") or ""), ()))
        row["record_value_pages"] = count
        row["record_value_share"] = _ratio(count, len(field_pages.get(field, ())))
        row["record_distinct_values"] = distinct.get(field, 0)
        # Blank, not NaN, so integer columns stay integers in the CSV.
        out.append({column: row.get(column, "") for column in COLUMNS})
    return out
