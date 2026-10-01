"""Training data for the ranker and the heading classifier: candidate logs joined with reviews.

    python -m stages.lib.extraction.training.dataset [RUN ...]      (default: every KV_Run_* under config.Run_Output)

Every reviewed page-field becomes one group of candidates. A candidate's label is the
reviewer's verdict when it has one, else whether it is one of the true key-value pairs
(labels.same_pair: same value at the same words, under the same key words); so "Wrong key &
value" on a right date stays a negative, and so does the right date read at another key.
The newest run with a candidate log for the same OCR text (ocr_sha1) supplies the candidates,
so a page reviewed once and rerun five times is used once. True pairs no candidate matches
are generator misses (misses.csv): a rules / catalog job, since a ranker only chooses among
candidates.

Output: {config.Training_Data}/datasets/ds_{stamp}/candidates.csv, misses.csv, manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..member_id.id_types import guess_id_type
from .labels import HEADING_FIELDS, KV_FIELDS, page_key, pooled_labels, same_pair
from .ner_export import run_dirs, run_started
from ..util import config

DATASETS = Path(config.Training_Data) / "datasets"
SPLIT_FILE = Path(config.Training_Data) / "splits" / "test_v1.txt"
# Extra columns next to the candidate log's own.
LABEL_COLUMNS = [
    "group_id", "label", "label_source", "true_level", "id_type", "page_not_present", "page_truth_count", "truth_json",
]


def is_test_record(record_id: str) -> bool:
    """About 1 record in 5 is a test record, fixed by its id: the split never changes for a
    record and only grows as records are added, and a record's pages are never split."""
    return int(hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0


def split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of the 'train', 'test' or 'all' split; records the test split to SPLIT_FILE."""
    if name == "all" or frame.empty:
        return frame
    test = frame["record_id"].map(is_test_record)
    records = sorted(set(frame.loc[test, "record_id"]))
    known = set(SPLIT_FILE.read_text(encoding="utf-8").split()) if SPLIT_FILE.is_file() else set()
    if set(records) - known:
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text("\n".join(sorted(known | set(records))) + "\n", encoding="utf-8")
    return frame[test] if name == "test" else frame[~test]


def _read_log(run_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in sorted((run_dir / "candidates").glob("*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _true_level(cid: str, norm: str, label: dict[str, Any]) -> str:
    verdict = (label.get("candidates") or {}).get(cid) or {}
    if verdict.get("verdict") == "correct" and verdict.get("level"):
        return verdict["level"]
    return next((item.get("level", "") for item in label.get("truth") or [] if item.get("value_norm") == norm), "")


def _ints(text: Any) -> list[int]:
    return [int(part) for part in str(text or "").split() if part.lstrip("-").isdigit()]


def _pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "value_norm": row["value_norm"],
        "key_words": _ints(row["key_words"]),
        "value_words": _ints(row["value_words"]),
    }


TRUTH_KEYS = ("candidate_id", "value_norm", "key_words", "value_words", "for_candidate")


def build(names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    runs = run_dirs(names)
    pooled = pooled_labels(runs)
    fields = set(KV_FIELDS) | set(HEADING_FIELDS)
    used: set[tuple[str, str]] = set()
    groups: list[pd.DataFrame] = []
    misses: list[dict[str, Any]] = []
    stats: Counter = Counter()
    sources: Counter = Counter()

    # Newest run first: its candidates reflect the current rules.
    for run_dir in sorted(runs, key=run_started, reverse=True):
        log = _read_log(run_dir)
        if log.empty:
            continue
        log = log[log["field"].isin(fields)]
        for (record_id, page_number, file_name, field), rows in log.groupby(
            ["record_id", "page_number", "file_name", "field"], sort=False
        ):
            key = (page_key(record_id, page_number, file_name), field)
            label = pooled.get(key)
            if label is None or key in used:
                continue
            ocr = rows["ocr_sha1"].iloc[0]
            if label.get("ocr_sha1") and ocr and label["ocr_sha1"] != ocr:
                stats["ocr_changed"] += 1
                continue
            used.add(key)
            sources[run_dir.name] += 1
            truth = [item for item in label.get("truth") or [] if item.get("value_norm")]
            verdicts = label.get("candidates") or {}
            # Placeholder rows stay (label 0) so pages where the rules found nothing are still scored.
            cands = rows.copy()

            positives: list[dict[str, Any]] = []
            if not cands.empty:
                labels: list[int] = []
                label_sources: list[str] = []
                for row in cands.to_dict("records"):
                    verdict = (verdicts.get(row["candidate_id"]) or {}).get("verdict")
                    if row["is_placeholder"] == "1":
                        labels.append(0)
                        label_sources.append("placeholder")
                    elif verdict in {"correct", "incorrect"}:
                        labels.append(int(verdict == "correct"))
                        label_sources.append("verdict")
                    else:
                        labels.append(int(bool(row["value_norm"]) and any(same_pair(item, _pair(row)) for item in truth)))
                        label_sources.append("pair_match")
                    if labels[-1]:
                        positives.append(_pair(row))
                cands["label"] = labels
                cands["label_source"] = label_sources
                cands["group_id"] = f"{key[0]}|{field}"
                cands["true_level"] = [
                    _true_level(cid, norm, label) if field in HEADING_FIELDS and lab else ""
                    for cid, norm, lab in zip(cands["candidate_id"], cands["value_norm"], labels)
                ]
                cands["id_type"] = [
                    guess_id_type(key_ or text) if field == "member_id" else ""
                    for text, key_ in zip(cands["key_text"], cands["key"])
                ]
                cands["page_not_present"] = int(bool(label.get("not_present")))
                cands["page_truth_count"] = len(truth)
                cands["truth_json"] = json.dumps(
                    [{name: item.get(name) for name in TRUTH_KEYS if item.get(name)} for item in truth], ensure_ascii=False
                )
                groups.append(cands)
            for item in truth:
                if any(same_pair(item, found) for found in positives):
                    continue
                misses.append(
                    {
                        "record_id": record_id, "page_number": page_number, "file_name": file_name, "field": field,
                        "value": item.get("value", ""), "value_norm": item["value_norm"], "key": item.get("key", ""),
                        "run": run_dir.name,
                    }
                )
            stats[f"groups:{field}"] += 1

    frame = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame(columns=LABEL_COLUMNS)
    miss_frame = pd.DataFrame(misses, columns=["record_id", "page_number", "file_name", "field", "value", "value_norm", "key", "run"])
    per_field: dict[str, dict[str, int]] = {}
    for field in sorted(fields):
        rows = frame[frame["field"] == field] if not frame.empty else frame
        real = rows[rows["is_placeholder"] != "1"] if not rows.empty else rows
        per_field[field] = {
            "groups": int(stats[f"groups:{field}"]),
            "groups_with_candidates": int(real["group_id"].nunique()) if not real.empty else 0,
            "candidates": int(len(real)),
            "positives": int(rows["label"].sum()) if not rows.empty else 0,
            "not_present_groups": int(rows.loc[rows["page_not_present"] == 1, "group_id"].nunique()) if not rows.empty else 0,
            "generator_misses": int((miss_frame["field"] == field).sum()),
        }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": dict(sources),
        "records": sorted(frame["record_id"].unique().tolist()) if not frame.empty else [],
        "pages": int(frame[["record_id", "page_number", "file_name"]].drop_duplicates().shape[0]) if not frame.empty else 0,
        "fields": per_field,
        "pages_ocr_changed": int(stats["ocr_changed"]),
        "catalog_hashes": sorted(frame["catalog_hash"].unique().tolist()) if not frame.empty else [],
    }
    return frame, miss_frame, manifest


def write(names: list[str] | None = None, out_root: Path | None = None) -> Path:
    frame, misses, manifest = build(names)
    folder = (out_root or DATASETS) / f"ds_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "candidates.csv", index=False)
    misses.to_csv(folder / "misses.csv", index=False)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return folder


def latest() -> Path | None:
    folders = sorted(path for path in DATASETS.glob("ds_*") if (path / "candidates.csv").is_file()) if DATASETS.is_dir() else []
    return folders[-1] if folders else None


def load(folder: Path | str | None = None) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    path = Path(folder) if folder else latest()
    if path is not None and not path.is_absolute() and not path.exists():
        path = DATASETS / path
    if path is None or not (path / "candidates.csv").is_file():
        raise SystemExit("No dataset yet: run `python -m stages.lib.extraction.training.dataset` first.")
    frame = pd.read_csv(path / "candidates.csv", dtype=str, keep_default_na=False)
    for column in ("label", "page_not_present", "page_truth_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    return frame, json.loads((path / "manifest.json").read_text(encoding="utf-8")), path


def main() -> int:
    folder = write(sys.argv[1:] or None)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    print(folder)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
