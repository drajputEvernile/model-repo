"""Manual review labels: what reviewers say is on each page, per field.

Stored per run in {run}/review/labels.json. One entry per page and field:

    candidates  verdict (correct / incorrect + reason) on each candidate the rules produced;
                "not_a_key" may name the neighbour word that makes the key prose, which is
                added to that field's key_blocklist.json; "wrong_field" names the field the
                value belongs to (belongs_to); "other_selected" (a selected candidate whose
                key lost to another key of the field on the page) names that other key's
                candidate (prefer_candidate, with its key and key words)
    added       values the rules missed, with the OCR words the reviewer picked; a correction
                typed under a "wrong_value" candidate carries its for_candidate and that
                candidate's key
    not_present the field is not on the page
    truth       every true value (correct candidates + added), which is what accuracy,
                ranker training and NER training read; a Member ID truth carries its
                id_type (member_id / mrn / ssn / encounter / other, see Member_ID/id_types.py)

Accuracy counts key-value pairs (score_pairs). A pair the run extracted is right when a true
pair has the same key and value (same candidate, or same normalized value at the same words
with the same key words); an extracted pair with a wrong key or value is wrong; a true pair
the run did not extract is missed. Accuracy = right / (right + wrong + missed). A field not
on the page with nothing extracted is one right; a correction typed under a wrong candidate
is the same error as that candidate, not a second miss. Labels are tied to the OCR text by
ocr_sha1, so a rerun over the same pages is scored with the same labels.

Heading fields (one per layout detector, see Heading/extract.py) are reviewed the same way
with their own reasons: a correct heading and an added one carry a level (Heading /
Subheading), and "wrong_span" opens a correction like "wrong_value". Headings are scored
separately (evaluate_headings: line-level precision / recall, level accuracy and pages
exactly right) and are not part of the KV accuracy.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from collections import Counter

import pandas as pd

from ..heading.extract import DETECTORS, LEVELS
from ..member_id.id_types import ID_TYPES, guess_id_type
from ..util.keys import add_key_blocks

from .features import run_accepted as _run_accepted
from .features import run_level, run_selected as _run_selected
from .normalize import normalize_typed

SCHEMA_VERSION = 2

KV_FIELDS: dict[str, str] = {
    "dob": "Member DOB",
    "member_id": "Member ID",
    "name": "Member Name",
    "provider_name": "Provider Name",
    "electronic_signature": "E-Signature",
    "dos": "Date of Service",
    "page_no": "Page No",
}
HEADING_FIELDS: dict[str, str] = {name: detector.label for name, detector in DETECTORS.items()}
FIELDS: dict[str, str] = {**KV_FIELDS, **HEADING_FIELDS}

REASONS: dict[str, str] = {
    "wrong_value": "Wrong value",
    "wrong_field": "Belongs to another field",
    "other_selected": "Should not be selected (another key was right)",
    "not_a_value": "Not a value (label / noise)",
    "not_a_key": "Not a real key here (prose / heading)",
    "wrong_key_value": "Wrong key & value",
}

HEADING_REASONS: dict[str, str] = {
    "not_heading": "Not a heading (body text)",
    "is_key": "It's a KV key / field label",
    "wrong_span": "Wrong text (cut off / too much)",
    "noise": "Noise (logo, clock, page number)",
}

# Reasons whose candidate gets a correction box (the right value / heading text).
CORRECTION_REASONS = frozenset({"wrong_value", "wrong_span"})

_CANDIDATE_COLUMNS = [
    "record_id", "file_name", "page_number", "ocr_sha1", "field", "candidate_id", "is_placeholder",
    "key", "key_text", "region", "rule_source", "rule_score", "rule_accepted", "rule_selected", "rule_note",
    "model_accepted", "model_selected", "model_level",
    "value", "value_norm", "detail", "key_words", "value_words",
    "key_x0", "key_y0", "key_x1", "key_y1", "value_x0", "value_y0", "value_x1", "value_y1",
]


def page_key(record_id: str, page_number: str | int, file_name: str) -> str:
    return f"{record_id}|{page_number}|{file_name}"


def labels_path(run_dir: Path) -> Path:
    return run_dir / "review" / "labels.json"


def workbook_path(run_dir: Path) -> Path:
    return run_dir / "review" / "manual_review.xlsx"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    os.replace(temp, path)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=64)
def _read_labels(path: str, mtime: float) -> dict[str, Any]:
    del mtime
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        data = {}
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("pages", {})
    return data


def load_labels(run_dir: Path) -> dict[str, Any]:
    path = labels_path(run_dir)
    # Callers mutate the result; never hand out the cached object.
    return json.loads(json.dumps(_read_labels(str(path), _mtime(path))))


def save_labels(run_dir: Path, data: dict[str, Any]) -> None:
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = _now()
    _atomic_write(labels_path(run_dir), json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- candidate logs


def _read_log(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in _CANDIDATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[_CANDIDATE_COLUMNS]


@lru_cache(maxsize=256)
def _log(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return _read_log(Path(path))


def candidate_logs(run_dir: Path) -> list[Path]:
    folder = run_dir / "candidates"
    return sorted(folder.glob("*.csv")) if folder.is_dir() else []


def record_log(run_dir: Path, record_id: str) -> pd.DataFrame:
    path = run_dir / "candidates" / f"{record_id}.csv"
    if not path.is_file():
        return pd.DataFrame(columns=_CANDIDATE_COLUMNS)
    return _log(str(path), _mtime(path))


def run_log(run_dir: Path) -> pd.DataFrame:
    frames = [_log(str(path), _mtime(path)) for path in candidate_logs(run_dir)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_CANDIDATE_COLUMNS)


def _ints(text: Any) -> list[int]:
    return [int(part) for part in str(text or "").split() if part.lstrip("-").isdigit()]


def _box(row: dict[str, Any], prefix: str) -> list[float] | None:
    try:
        return [float(row[f"{prefix}_{name}"]) for name in ("x0", "y0", "x1", "y1")]
    except (KeyError, TypeError, ValueError):
        return None


def page_candidates(run_dir: Path, record_id: str, page_number: str, file_name: str) -> dict[str, dict[str, Any]]:
    """field -> {ocr_sha1, candidates: [...]} for one page, from the run's candidate log."""
    frame = record_log(run_dir, record_id)
    page = frame[(frame["page_number"] == str(page_number)) & (frame["file_name"] == file_name)]
    out: dict[str, dict[str, Any]] = {field: {"ocr_sha1": "", "candidates": []} for field in FIELDS}
    for row in page.to_dict("records"):
        field = row["field"]
        if field not in out:
            continue
        out[field]["ocr_sha1"] = row["ocr_sha1"]
        if row["is_placeholder"] == "1":
            continue
        out[field]["candidates"].append(
            {
                "candidate_id": row["candidate_id"],
                "key": row["key"],
                "key_text": row["key_text"],
                "region": row["region"],
                "source": row["rule_source"],
                "score": row["rule_score"],
                "accepted": _run_accepted(row),
                "selected": _run_selected(row),
                "note": row["rule_note"],
                "value": row["value"],
                "value_norm": row["value_norm"],
                "detail": run_level(row),
                "key_words": _ints(row["key_words"]),
                "value_words": _ints(row["value_words"]),
                "key_box": _box(row, "key"),
                "value_box": _box(row, "value"),
                # The catalog key, not the OCR text: '55N' was matched as 'SSN'.
                "id_type": guess_id_type(row["key"] or row["key_text"]) if field == "member_id" else "",
            }
        )
    # A field the run did not log (headings in a run made before them) still belongs to this OCR.
    page_hash = next((slot["ocr_sha1"] for slot in out.values() if slot["ocr_sha1"]), "")
    for slot in out.values():
        slot["ocr_sha1"] = slot["ocr_sha1"] or page_hash
    return out


# ---------------------------------------------------------------- saving a review


def is_extracted(c: dict[str, Any], heading: bool = False) -> bool:
    """What a review must judge: every key-value pair the run extracted (the pairs accuracy
    counts), or the lines detected as headings."""
    return bool(c["selected"] or (not heading and c["accepted"] and c["value_norm"]))


def _extracted(candidates: list[dict[str, Any]], heading: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": c["candidate_id"], "value": c["value"], "value_norm": c["value_norm"],
            "key_words": c["key_words"], "value_words": c["value_words"],
        }
        for c in candidates
        if is_extracted(c, heading)
    ]


def _norms(items: list[dict[str, Any]]) -> set[str]:
    return {item["value_norm"] for item in items if item.get("value_norm")}


def same_pair(true: dict[str, Any], found: dict[str, Any]) -> bool:
    """Whether an extracted candidate is this true key-value pair."""
    if true.get("candidate_id") and true["candidate_id"] == found.get("candidate_id"):
        return True
    if not true.get("value_norm") or true["value_norm"] != found.get("value_norm"):
        return False
    true_words, found_words = set(true.get("value_words") or []), set(found.get("value_words") or [])
    if true_words and found_words and not true_words & found_words:
        return False
    true_key = set(true.get("key_words") or [])
    return not true_key or bool(true_key & set(found.get("key_words") or []))


def score_pairs(
    found: list[dict[str, Any]], label: dict[str, Any], details: list[tuple[str, dict[str, Any]]] | None = None
) -> Counter:
    """right / wrong / missed key-value pairs of one reviewed page-field.

    found: the pairs the run extracted (candidate_id, value_norm, key_words, value_words).
    details, when given, gets ("right" | "wrong" | "missed", pair) for every pair counted.
    """
    details = details if details is not None else []
    truth = [item for item in label.get("truth") or [] if item.get("value_norm")]
    verdicts = label.get("candidates") or {}
    score: Counter = Counter()
    if not truth and not found:
        score["right"] += int(bool(label.get("not_present")))
        return score
    open_truth = list(range(len(truth)))
    wrong_ids: set[str] = set()
    for c in found:
        verdict = (verdicts.get(c.get("candidate_id", "")) or {}).get("verdict")
        match = None
        if verdict != "incorrect":
            match = next((i for i in open_truth if same_pair(truth[i], c)), None)
        if match is None:
            score["wrong"] += 1
            wrong_ids.add(c.get("candidate_id", ""))
            details.append(("wrong", c))
        else:
            score["right"] += 1
            open_truth.remove(match)
            details.append(("right", c))
    for i in open_truth:
        if truth[i].get("for_candidate") not in wrong_ids - {""}:
            score["missed"] += 1
            details.append(("missed", truth[i]))
    return score


def auto_review(candidates: list[dict[str, Any]], label: dict[str, Any], heading: bool = False) -> dict[str, Any]:
    """Another run's review of this page-field applied to this run's candidates, judged like
    the accuracy: a candidate matching a true pair is correct, any other extracted one is
    incorrect (its old verdict kept when it has one), and true pairs nothing matched are added."""
    if heading:
        return label
    details: list[tuple[str, dict[str, Any]]] = []
    score_pairs(_extracted(candidates), label, details)
    old = label.get("candidates") or {}
    verdicts: dict[str, dict[str, Any]] = {}
    added: list[dict[str, Any]] = []
    for kind, pair in details:
        cid = pair.get("candidate_id", "")
        if kind == "right":
            verdicts[cid] = {"verdict": "correct", "reason": ""}
        elif kind == "wrong":
            verdicts[cid] = old.get(cid) or {"verdict": "incorrect", "reason": "wrong_key_value"}
        else:
            added.append({**pair, "for_candidate": ""})
    corrected = {cid for cid, verdict in verdicts.items() if verdict.get("reason") in CORRECTION_REASONS}
    added += [item for item in label.get("truth") or [] if item.get("for_candidate") in corrected]
    keep = ("value", "value2", "key", "for_candidate", "level", "id_type", "value_words", "key_words")
    return {
        **label,
        "candidates": verdicts,
        "added": [{name: item.get(name, [] if name.endswith("_words") else "") for name in keep} for item in added],
        "auto": True,
    }


def pair_accuracy(score: Counter) -> float | None:
    total = score["right"] + score["wrong"] + score["missed"]
    return round(100 * score["right"] / total, 1) if total else None


def build_label(field: str, found: dict[str, Any], submission: dict[str, Any], reviewer: str) -> dict[str, Any]:
    """Validate one page-field review against the run's candidates and derive the truth."""
    if field not in FIELDS:
        raise ValueError(f"Unknown field: {field}")
    heading = field in HEADING_FIELDS
    reasons = HEADING_REASONS if heading else REASONS
    candidates = {c["candidate_id"]: c for c in found["candidates"]}

    verdicts: dict[str, dict[str, Any]] = {}
    for cid, raw in (submission.get("candidates") or {}).items():
        if cid not in candidates:
            raise ValueError(f"Candidate {cid} is not on this page for {FIELDS[field]}.")
        verdict = str((raw or {}).get("verdict") or "")
        if verdict not in {"correct", "incorrect"}:
            continue
        reason = str((raw or {}).get("reason") or "")
        if verdict == "incorrect" and reason and reason not in reasons:
            raise ValueError(f"Unknown reason: {reason}")
        c = candidates[cid]
        blocks = verdict == "incorrect" and reason == "not_a_key" and bool(c["key_words"])
        belongs_to = str(raw.get("belongs_to") or "") if verdict == "incorrect" and reason == "wrong_field" else ""
        if reason == "wrong_field" and verdict == "incorrect" and (belongs_to not in KV_FIELDS or belongs_to == field):
            raise ValueError(f"Pick the field '{c['value']}' belongs to.")
        prefer = str(raw.get("prefer_candidate") or "") if verdict == "incorrect" and reason == "other_selected" else ""
        if verdict == "incorrect" and reason == "other_selected":
            if not c["selected"]:
                raise ValueError(f"'{c['value']}' was not selected, so it can't be 'Should not be selected'.")
            other = candidates.get(prefer)
            if other is None or not other["key_words"] or other["key_words"] == c["key_words"]:
                raise ValueError(f"Pick the key that should have been used instead of '{c['key_text'] or c['key']}'.")
        level = ""
        if heading and verdict == "correct":
            level = str(raw.get("level") or c["detail"])
            if level not in LEVELS:
                raise ValueError(f"Pick the level of the heading '{c['value']}'.")
        id_type = ""
        if field == "member_id" and verdict == "correct":
            id_type = str(raw.get("id_type") or c.get("id_type") or guess_id_type(c["key"] or c["key_text"]))
            if id_type not in ID_TYPES:
                raise ValueError(f"Pick the ID type of '{c['value']}'.")
        verdicts[cid] = {
            "verdict": verdict,
            "reason": reason if verdict == "incorrect" else "",
            "belongs_to": belongs_to,
            "prefer_candidate": prefer,
            "prefer_key": candidates[prefer]["key"] if prefer else "",
            "prefer_key_text": candidates[prefer]["key_text"] if prefer else "",
            "prefer_key_words": candidates[prefer]["key_words"] if prefer else [],
            "level": level,
            "id_type": id_type,
            "block_before": str(raw.get("block_before") or "").strip() if blocks else "",
            "block_after": str(raw.get("block_after") or "").strip() if blocks else "",
            "key": c["key"],
            "key_text": c["key_text"],
            "region": c["region"],
            "value": c["value"],
            "value_norm": c["value_norm"],
            "detail": c["detail"],
            "selected": c["selected"],
        }

    extracted_ids = {c["candidate_id"] for c in candidates.values() if is_extracted(c, heading)}
    if extracted_ids - set(verdicts):
        raise ValueError(f"Mark every extracted {FIELDS[field]} value correct or incorrect first.")
    unreasoned = [
        v for cid, v in verdicts.items() if v["verdict"] == "incorrect" and cid in extracted_ids and not v["reason"]
    ]
    if unreasoned:
        raise ValueError("Pick a reason for each extracted value marked incorrect.")

    added: list[dict[str, Any]] = []
    for raw in submission.get("added") or []:
        value = str((raw or {}).get("value") or "").strip()
        if not value:
            continue
        norm = normalize_typed(field, value)
        if not norm:
            raise ValueError(f"'{value}' is not a valid {FIELDS[field]}.")
        for_candidate = str(raw.get("for_candidate") or "")
        key = "" if heading else str(raw.get("key") or "").strip()
        key_words = [] if heading else sorted({int(i) for i in raw.get("key_words") or []})
        if for_candidate:
            if verdicts.get(for_candidate, {}).get("reason") not in CORRECTION_REASONS:
                raise ValueError(f"The correction '{value}' belongs to a candidate that is not marked for correction.")
            # The key was right, only the value was wrong: the correction keeps that key.
            key = key or candidates[for_candidate]["key"]
            key_words = key_words or candidates[for_candidate]["key_words"]
        level = str(raw.get("level") or "") if heading else ""
        if heading and level not in LEVELS:
            raise ValueError(f"Pick the level of the heading '{value}'.")
        id_type = str(raw.get("id_type") or guess_id_type(key)) if field == "member_id" else ""
        if field == "member_id" and id_type not in ID_TYPES:
            raise ValueError(f"Pick the ID type of '{value}' (or enter its key).")
        added.append(
            {
                "value": value,
                "value_norm": norm,
                "for_candidate": for_candidate,
                "level": level,
                "id_type": id_type,
                "value2": str(raw.get("value2") or "").strip(),
                "key": key,
                "value_words": sorted({int(i) for i in raw.get("value_words") or []}),
                "key_words": key_words,
            }
        )

    not_present = bool(submission.get("not_present"))
    truth: list[dict[str, Any]] = []
    for cid, v in verdicts.items():
        if v["verdict"] == "correct":
            c = candidates[cid]
            truth.append(
                {
                    "source": "candidate",
                    "candidate_id": cid,
                    "value": c["value"],
                    "value_norm": c["value_norm"],
                    "value2": c["detail"] if field == "electronic_signature" else "",
                    "level": v["level"],
                    "id_type": v["id_type"],
                    "key": c["key"],
                    "value_words": c["value_words"],
                    "key_words": c["key_words"],
                }
            )
    truth.extend({"source": "added", **item} for item in added)

    if not_present and truth:
        if heading:
            raise ValueError("The page is marked as having no headings, but a heading is ticked or added.")
        raise ValueError(f"{FIELDS[field]} is marked not on the page but has correct or added values.")
    if not not_present and not truth:
        if heading:
            raise ValueError("Tick a heading, add a missed one, or mark that the page has no headings.")
        raise ValueError(f"Mark a correct value, add the missed value, or mark {FIELDS[field]} as not on this page.")

    extracted = _extracted(found["candidates"], heading)
    label = {"not_present": not_present, "candidates": verdicts, "truth": truth}
    pairs = score_pairs(extracted, label)
    return {
        **label,
        "added": added,
        "extracted": extracted,
        "pairs": {name: pairs[name] for name in ("right", "wrong", "missed")},
        "correct": _norms(extracted) == _norms(truth) if heading else not (pairs["wrong"] or pairs["missed"]),
        "notes": str(submission.get("notes") or "").strip(),
        "reviewer": reviewer.strip(),
        "reviewed_at": _now(),
    }


def save_field_review(
    run_dir: Path,
    record_id: str,
    page_number: str,
    file_name: str,
    field: str,
    submission: dict[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    found = page_candidates(run_dir, record_id, page_number, file_name)
    if field not in found:
        raise ValueError(f"Unknown field: {field}")
    label = build_label(field, found[field], submission, reviewer)
    data = load_labels(run_dir)
    key = page_key(record_id, page_number, file_name)
    entry = data["pages"].setdefault(
        key,
        {"record_id": record_id, "page_number": str(page_number), "file_name": file_name, "fields": {}},
    )
    entry["ocr_sha1"] = found[field]["ocr_sha1"]
    entry["fields"][field] = label
    save_labels(run_dir, data)
    _refresh_workbook(run_dir, data)
    for verdict in label["candidates"].values():
        if verdict["block_before"] or verdict["block_after"]:
            add_key_blocks(field, verdict["key"], verdict["block_before"], verdict["block_after"])
    return label


def _refresh_workbook(run_dir: Path, data: dict[str, Any]) -> None:
    # labels.json is the record; the workbook is a copy that may be open (locked) in Excel.
    try:
        write_workbook(run_dir, data)
    except OSError:
        pass


def clear_field_review(run_dir: Path, record_id: str, page_number: str, file_name: str, field: str) -> None:
    data = load_labels(run_dir)
    key = page_key(record_id, page_number, file_name)
    entry = data["pages"].get(key)
    if not entry or field not in entry.get("fields", {}):
        return
    del entry["fields"][field]
    if not entry["fields"] and not entry.get("page_info"):
        del data["pages"][key]
    save_labels(run_dir, data)
    _refresh_workbook(run_dir, data)


# ---------------------------------------------------------------- page information (collected only)
# Facts the reviewer records about a page that no extractor detects and no accuracy counts,
# e.g. the page's correct position in the document. Stored per page beside the field reviews.


def save_page_info(
    run_dir: Path, record_id: str, page_number: str, file_name: str, page_sequence: str, reviewer: str
) -> dict[str, Any] | None:
    """Save (or, when page_sequence is blank, remove) the page's correct sequence number."""
    text = str(page_sequence or "").strip()
    if text and not (text.isdigit() and int(text) > 0):
        raise ValueError("The correct page sequence number must be a whole number of 1 or more.")
    data = load_labels(run_dir)
    key = page_key(record_id, page_number, file_name)
    entry = data["pages"].setdefault(
        key,
        {"record_id": record_id, "page_number": str(page_number), "file_name": file_name, "fields": {}},
    )
    if text:
        entry["page_info"] = {"page_sequence": int(text), "reviewer": reviewer.strip(), "saved_at": _now()}
    else:
        entry.pop("page_info", None)
        if not entry.get("fields"):
            del data["pages"][key]
    save_labels(run_dir, data)
    _refresh_workbook(run_dir, data)
    return entry.get("page_info") if text else None


def page_info(run_dir: Path, record_id: str, page_number: str, file_name: str) -> dict[str, Any] | None:
    key = page_key(record_id, page_number, file_name)
    return load_labels(run_dir)["pages"].get(key, {}).get("page_info")


def pooled_page_info(run_dirs: list[Path], key: str) -> dict[str, Any] | None:
    """The latest page information saved for this page in any of the runs (a rerun keeps it)."""
    best: dict[str, Any] | None = None
    for run_dir in run_dirs:
        path = labels_path(run_dir)
        if not path.is_file():
            continue
        info = _read_labels(str(path), _mtime(path))["pages"].get(key, {}).get("page_info")
        if info and (best is None or info.get("saved_at", "") > best.get("saved_at", "")):
            best = {**info, "run": run_dir.name}
    return best


# ---------------------------------------------------------------- pooled labels and accuracy


def pooled_labels(run_dirs: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """(page key, field) -> latest label across runs, with the page's ocr_sha1."""
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in run_dirs:
        path = labels_path(run_dir)
        if not path.is_file():
            continue
        for key, entry in _read_labels(str(path), _mtime(path))["pages"].items():
            for field, label in entry.get("fields", {}).items():
                current = pooled.get((key, field))
                if current is None or label.get("reviewed_at", "") > current.get("reviewed_at", ""):
                    pooled[(key, field)] = {**label, "ocr_sha1": entry.get("ocr_sha1", ""), "run": run_dir.name}
    return pooled


def extracted_pairs(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    """(page key, field) -> {pairs extracted, ocr_sha1} for every page-field of a candidate log."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        key = (page_key(row["record_id"], row["page_number"], row["file_name"]), row["field"])
        slot = out.setdefault(key, {"found": [], "ocr_sha1": row["ocr_sha1"]})
        if row["is_placeholder"] != "1" and row["value_norm"] and (_run_accepted(row) or _run_selected(row)):
            slot["found"].append(
                {
                    "candidate_id": row["candidate_id"],
                    "value": row["value"],
                    "value_norm": row["value_norm"],
                    "key_text": row["key_text"],
                    "region": row["region"],
                    "key_words": _ints(row["key_words"]),
                    "value_words": _ints(row["value_words"]),
                }
            )
    return out


def _same_ocr(label: dict[str, Any], ocr_hash: str) -> bool:
    return not (label.get("ocr_sha1") and ocr_hash and label["ocr_sha1"] != ocr_hash)


def _pair_totals(score: Counter) -> dict[str, Any]:
    return {
        "correct": score["right"],
        "wrong": score["wrong"],
        "missed": score["missed"],
        "total": score["right"] + score["wrong"] + score["missed"],
        "accuracy": pair_accuracy(score),
    }


def evaluate(run_dir: Path, pooled: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """KV key-value pair accuracy of a run over every page-field that has a matching label:
    correct = right pairs, total = right + wrong + missed."""
    return evaluate_log(run_log(run_dir), pooled)


def evaluate_log(frame: pd.DataFrame, pooled: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    per_field: dict[str, Counter] = {field: Counter() for field in KV_FIELDS}
    for key, extracted in extracted_pairs(frame).items():
        label = pooled.get(key)
        field = key[1]
        if label is None or field not in per_field or not _same_ocr(label, extracted["ocr_sha1"]):
            continue
        per_field[field].update(score_pairs(extracted["found"], label))
        per_field[field]["page_fields"] += 1
    overall: Counter = sum(per_field.values(), Counter())
    return {
        **_pair_totals(overall),
        "page_fields": overall["page_fields"],
        "fields": {field: {**_pair_totals(score), "page_fields": score["page_fields"]} for field, score in per_field.items()},
    }


def _pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def evaluate_headings(run_dir: Path, pooled: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per heading detector: line-level precision / recall, level accuracy and exact pages.

    A selected heading is a true positive when a true heading on the page has the same text
    (normalized); its level is right when the reviewed level matches the predicted one.
    """
    frame = run_log(run_dir)
    frame = frame[frame["field"].isin(list(HEADING_FIELDS))]
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        key = (page_key(row["record_id"], row["page_number"], row["file_name"]), row["field"])
        slot = selected.setdefault(key, {"ocr_sha1": row["ocr_sha1"], "norms": Counter(), "levels": {}})
        if _run_selected(row) and row["value_norm"]:
            slot["norms"][row["value_norm"]] += 1
            slot["levels"].setdefault(row["value_norm"], run_level(row))

    out = {field: {"tp": 0, "fp": 0, "fn": 0, "level_ok": 0, "pages": 0, "pages_exact": 0} for field in HEADING_FIELDS}
    for key, slot in selected.items():
        label = pooled.get(key)
        if label is None or not _same_ocr(label, slot["ocr_sha1"]):
            continue
        truth = Counter(item["value_norm"] for item in label.get("truth") or [] if item.get("value_norm"))
        levels: dict[str, str] = {}
        for item in label.get("truth") or []:
            levels.setdefault(item.get("value_norm", ""), item.get("level", ""))
        matched = slot["norms"] & truth
        tp = sum(matched.values())
        score = out[key[1]]
        score["tp"] += tp
        score["fp"] += sum(slot["norms"].values()) - tp
        score["fn"] += sum(truth.values()) - tp
        score["level_ok"] += sum(n for norm, n in matched.items() if slot["levels"].get(norm) == levels.get(norm))
        score["pages"] += 1
        score["pages_exact"] += int(slot["norms"] == truth)
    for score in out.values():
        score["precision"] = _pct(score["tp"], score["tp"] + score["fp"])
        score["recall"] = _pct(score["tp"], score["tp"] + score["fn"])
        score["level_accuracy"] = _pct(score["level_ok"], score["tp"])
        score["page_accuracy"] = _pct(score["pages_exact"], score["pages"])
    return out


def heading_pairs(score: dict[str, Any]) -> Counter:
    """A heading detector's lines as pairs, judged like key-value pairs: a detected heading is
    right when its text and its level are right, wrong when it is no true heading or has the
    wrong level, and a true heading nobody detected is missed."""
    return Counter(
        right=score["level_ok"],
        wrong=score["fp"] + score["tp"] - score["level_ok"],
        missed=score["fn"],
        page_fields=score["pages"],
    )


def evaluate_all(run_dir: Path, pooled: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Every extraction module of a run: the KV fields and the heading detectors, each as
    right / wrong / missed pairs; the overall accuracy adds them all up. 'kv' keeps the
    key-value subtotal, 'headings' the detectors' precision / recall / level details."""
    kv = evaluate(run_dir, pooled)
    headings = evaluate_headings(run_dir, pooled)
    fields = dict(kv["fields"])
    overall = Counter(right=kv["correct"], wrong=kv["wrong"], missed=kv["missed"], page_fields=kv["page_fields"])
    for name, score in headings.items():
        pairs = heading_pairs(score)
        fields[name] = {**_pair_totals(pairs), "page_fields": pairs["page_fields"]}
        overall.update(pairs)
    return {
        **_pair_totals(overall),
        "page_fields": overall["page_fields"],
        "fields": fields,
        "kv": {name: kv[name] for name in ("correct", "wrong", "missed", "total", "accuracy", "page_fields")},
        "headings": headings,
    }


def reviewed_fields(run_dir: Path, pooled: dict[tuple[str, str], dict[str, Any]] | None = None) -> dict[str, int]:
    """page key -> number of fields reviewed in this run, or (with pooled) reviewed in any run
    on the same OCR: a rerun of a reviewed batch is reviewed."""
    path = labels_path(run_dir)
    fields: dict[str, set[str]] = {}
    if path.is_file():
        for key, entry in _read_labels(str(path), _mtime(path))["pages"].items():
            fields[key] = set(entry.get("fields", {}))
    if pooled:
        frame = run_log(run_dir)
        if not frame.empty:
            pages = frame[["record_id", "page_number", "file_name", "field", "ocr_sha1"]].drop_duplicates()
            for row in pages.to_dict("records"):
                key = page_key(row["record_id"], row["page_number"], row["file_name"])
                label = pooled.get((key, row["field"]))
                if label is not None and _same_ocr(label, row["ocr_sha1"]):
                    fields.setdefault(key, set()).add(row["field"])
    return {key: len(names) for key, names in fields.items()}


def labels_signature(run_dirs: list[Path]) -> tuple[float, ...]:
    return tuple(_mtime(labels_path(run_dir)) for run_dir in run_dirs)


# ---------------------------------------------------------------- workbook


def write_workbook(run_dir: Path, data: dict[str, Any] | None = None) -> Path:
    data = data if data is not None else load_labels(run_dir)
    page_fields: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    page_info_rows: list[dict[str, Any]] = []
    for entry in data["pages"].values():
        base = {
            "RecordId": entry["record_id"],
            "FileName": entry["file_name"],
            "PageNumber": entry["page_number"],
        }
        if entry.get("page_info"):
            info = entry["page_info"]
            page_info_rows.append(
                {**base, "CorrectPageSequence": info.get("page_sequence", ""),
                 "Reviewer": info.get("reviewer", ""), "SavedAt": info.get("saved_at", "")}
            )
        for field, label in entry.get("fields", {}).items():
            where = {**base, "Field": FIELDS.get(field, field)}
            kv = field in KV_FIELDS
            pairs = score_pairs(label.get("extracted") or [], label) if kv else Counter()
            page_fields.append(
                {
                    **where,
                    "Extracted": " | ".join(item["value"] for item in label.get("extracted", [])),
                    "Truth": " | ".join(item["value"] for item in label.get("truth", [])),
                    "NotPresent": label.get("not_present", False),
                    "Right": pairs["right"] if kv else "",
                    "Wrong": pairs["wrong"] if kv else "",
                    "Missed": pairs["missed"] if kv else "",
                    "Correct": not (pairs["wrong"] or pairs["missed"]) if kv else label.get("correct", False),
                    "Reviewer": label.get("reviewer", ""),
                    "ReviewedAt": label.get("reviewed_at", ""),
                    "Notes": label.get("notes", ""),
                }
            )
            for cid, verdict in label.get("candidates", {}).items():
                candidates.append(
                    {
                        **where,
                        "CandidateId": cid,
                        "Key": verdict.get("key", ""),
                        "KeyText": verdict.get("key_text", ""),
                        "Region": verdict.get("region", ""),
                        "Value": verdict.get("value", ""),
                        "Selected": verdict.get("selected", False),
                        "Verdict": verdict.get("verdict", ""),
                        "Reason": {**REASONS, **HEADING_REASONS}.get(verdict.get("reason", ""), ""),
                        "BelongsTo": FIELDS.get(verdict.get("belongs_to", ""), ""),
                        "PreferKey": verdict.get("prefer_key_text", "") or verdict.get("prefer_key", ""),
                        "IdType": ID_TYPES.get(verdict.get("id_type", ""), ""),
                        "Level": verdict.get("level", ""),
                        "BlockBefore": verdict.get("block_before", ""),
                        "BlockAfter": verdict.get("block_after", ""),
                    }
                )
            for item in label.get("added", []):
                added.append(
                    {
                        **where,
                        "ForCandidate": item.get("for_candidate", ""),
                        "Key": item.get("key", ""),
                        "Value": item.get("value", ""),
                        "IdType": ID_TYPES.get(item.get("id_type", ""), ""),
                        "Level": item.get("level", ""),
                        "Value2": item.get("value2", ""),
                        "ValueWords": " ".join(map(str, item.get("value_words", []))),
                        "KeyWords": " ".join(map(str, item.get("key_words", []))),
                    }
                )

    pooled = pooled_labels([run_dir])
    score = evaluate_all(run_dir, pooled)
    headings = [
        {
            "Detector": FIELDS[field],
            "PagesReviewed": item["pages"],
            "Precision%": item["precision"],
            "Recall%": item["recall"],
            "LevelAccuracy%": item["level_accuracy"],
            "PagesExact%": item["page_accuracy"],
            "TP": item["tp"],
            "FP": item["fp"],
            "FN": item["fn"],
        }
        for field, item in score["headings"].items()
    ]
    def accuracy_row(name: str, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "Field": name,
            "PageFields": item["page_fields"],
            "Right": item["correct"],
            "Wrong": item["wrong"],
            "Missed": item["missed"],
            "Accuracy%": item["accuracy"],
        }

    accuracy = [accuracy_row(FIELDS[field], item) for field, item in score["fields"].items() if field in KV_FIELDS]
    accuracy.append(accuracy_row("Key-value fields", score["kv"]))
    accuracy += [accuracy_row(FIELDS[field], item) for field, item in score["fields"].items() if field in HEADING_FIELDS]
    accuracy.append(accuracy_row("All modules", score))

    path = workbook_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(accuracy).to_excel(writer, sheet_name="Field_Accuracy", index=False)
        pd.DataFrame(headings).to_excel(writer, sheet_name="Heading_Accuracy", index=False)
        pd.DataFrame(page_fields, columns=[
            "RecordId", "FileName", "PageNumber", "Field", "Extracted", "Truth",
            "NotPresent", "Right", "Wrong", "Missed", "Correct", "Reviewer", "ReviewedAt", "Notes",
        ]).to_excel(writer, sheet_name="Page_Fields", index=False)
        pd.DataFrame(candidates, columns=[
            "RecordId", "FileName", "PageNumber", "Field", "CandidateId", "Key", "KeyText",
            "Region", "Value", "Selected", "Verdict", "Reason", "BelongsTo", "PreferKey", "Level",
            "BlockBefore", "BlockAfter",
        ]).to_excel(writer, sheet_name="Candidates", index=False)
        pd.DataFrame(added, columns=[
            "RecordId", "FileName", "PageNumber", "Field", "ForCandidate", "Key", "Value", "Level", "Value2",
            "ValueWords", "KeyWords",
        ]).to_excel(writer, sheet_name="Added", index=False)
        pd.DataFrame(page_info_rows, columns=[
            "RecordId", "FileName", "PageNumber", "CorrectPageSequence", "Reviewer", "SavedAt",
        ]).to_excel(writer, sheet_name="Page_Info", index=False)
    return path
