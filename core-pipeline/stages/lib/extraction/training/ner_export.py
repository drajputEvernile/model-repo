"""GLiNER fine-tuning data from reviewed pages.

    python -m stages.lib.extraction.training.ner_export [RUN ...]      (default: every KV_Run_* under config.Run_Output)

Only pages where every NER field was reviewed are exported, so a word that is not inside a
span is a true negative. Tokens are the page's OCR words in reading order, cut into windows
at line boundaries. Spans come from the OCR words the value was read from (or the reviewer
picked); values without word positions are located by their text. For fields whose meaning
does not depend on the surrounding key (EVERY_MENTION) every other occurrence of a true value
on the page is labelled too, so repeats are not taught as negatives. A span can carry more
than one label (the provider who also signed).

Output: {config.Training_Data}/ner/gliner_train_{timestamp}.json  ([{"tokenized_text", "ner"}])
        plus a .manifest.json with counts and sources.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..member_id.id_types import NER_ID_LABELS, guess_id_type
from .features import ocr_sha1
from .labels import pooled_labels
from .ocr import page_words
from ..util import config
from ..util.geometry import Word, group_lines
from ..util.mentions import find_run, mention_runs, value_parts

# member_id is split by ID type (NER_ID_LABELS); "member id" is only its fallback label.
NER_LABELS: dict[str, str] = {
    "dob": "date of birth",
    "member_id": "member id",
    "name": "patient name",
    "provider_name": "provider name",
    "electronic_signature": "signing provider",
    "dos": "date of service",
}
WINDOW_TOKENS = 256
# A DOS or a signer is only one by its key (a print date, a provider named in the header), so
# only the reviewed occurrence is labelled for those.
EVERY_MENTION = {"dob", "member_id", "name", "provider_name"}


def span_label(field: str, item: dict[str, Any]) -> str:
    if field == "member_id":
        id_type = item.get("id_type") or guess_id_type(item.get("key", "")) or "member_id"
        return NER_ID_LABELS.get(id_type, NER_LABELS[field])
    return NER_LABELS[field]


def all_labels() -> list[str]:
    return list(dict.fromkeys([label for field, label in NER_LABELS.items() if field != "member_id"] + list(NER_ID_LABELS.values())))


def _run_meta(run_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _start_time(run_dir: Path, meta: dict[str, Any]) -> str:
    started = meta.get("start_time") or datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
    return str(started)


def run_started(run_dir: Path) -> tuple[str, int]:
    """Sort key, oldest first: the UTC start in run.json (else the folder time), but never
    before the run it reran. Clocks alone don't order runs: folder names are local times, and
    a machine clock can jump back, so a rerun started 'earlier' than its source still sorts
    after it (the depth breaks the tie)."""
    chain: list[tuple[Path, dict[str, Any]]] = []
    folder, seen = run_dir, set()
    while folder.name not in seen and len(chain) < 50:
        seen.add(folder.name)
        meta = _run_meta(folder)
        chain.append((folder, meta))
        source = meta.get("source_run")
        if not source or not (folder.parent / source).is_dir():
            break
        folder = folder.parent / source
    effective, depth = "", 0
    for depth, (folder, meta) in enumerate(reversed(chain)):
        effective = max(effective, _start_time(folder, meta))
    return effective, depth


def run_dirs(names: list[str] | None = None) -> list[Path]:
    """Runs oldest first."""
    root = Path(config.Run_Output)
    if names:
        return [path if path.is_absolute() else root / path for path in map(Path, names)]
    return sorted((path for path in root.glob("KV_Run_*") if path.is_dir()), key=run_started)


def _span(item: dict[str, Any], position: dict[int, int], words: list[Word]) -> tuple[int, int] | None:
    indexes = [position[i] for i in item.get("value_words") or [] if i in position]
    if not indexes:
        run = find_run(words, value_parts(item.get("value", "")), None)
        indexes = [position[word.index] for word in run if word.index in position]
    if not indexes:
        return None
    return min(indexes), max(indexes)


def _mentions(field: str, item: dict[str, Any], position: dict[int, int], words: list[Word]) -> list[tuple[int, int]]:
    """Every occurrence of the item's value on the page (dates in any format, names in either order)."""
    out: list[tuple[int, int]] = []
    for run in mention_runs(field, str(item.get("value") or ""), words):
        indexes = [position[word.index] for word in run if word.index in position]
        if indexes:
            out.append((min(indexes), max(indexes)))
    return out


def _windows(lines: list[list[Word]]) -> list[tuple[int, int]]:
    """[start, end) token ranges of at most WINDOW_TOKENS, split between lines."""
    out: list[tuple[int, int]] = []
    start = count = 0
    for line in lines:
        if count and count + len(line) > WINDOW_TOKENS:
            out.append((start, start + count))
            start, count = start + count, 0
        count += len(line)
    if count:
        out.append((start, start + count))
    return out


def page_examples(record_id: str, file_name: str, ocr_hash: str, fields: dict[str, dict]) -> tuple[list[dict], Counter]:
    words, _, _ = page_words(record_id, file_name)
    stats: Counter = Counter()
    if not words:
        stats["missing_ocr"] += 1
        return [], stats
    if ocr_hash and ocr_sha1(words) != ocr_hash:
        stats["ocr_changed"] += 1
        return [], stats
    lines = group_lines(words)
    ordered = [word for line in lines for word in line]
    position = {word.index: n for n, word in enumerate(ordered)}

    spans: list[tuple[int, int, str]] = []
    for field in NER_LABELS:
        for item in fields[field].get("truth") or []:
            span = _span(item, position, ordered)
            if span is None:
                stats["unlocated"] += 1
                continue
            label = span_label(field, item)
            spans.append((span[0], span[1], label))
            stats[label] += 1
            if field not in EVERY_MENTION:
                continue
            for left, right in _mentions(field, item, position, ordered):
                if any(left <= r and l <= right for l, r, other in spans if other == label):
                    continue
                spans.append((left, right, label))
                stats[label] += 1
                stats["extra_mentions"] += 1

    examples: list[dict] = []
    for start, end in _windows(lines):
        inside = [
            [left - start, right - start, label]
            for left, right, label in sorted(set(spans))
            if start <= left and right < end
        ]
        stats["cut_spans"] += sum(1 for left, right, _ in spans if left < end <= right or left < start <= right)
        examples.append({"tokenized_text": [word.content for word in ordered[start:end]], "ner": inside})
    return examples, stats


def export(names: list[str] | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    runs = run_dirs(names)
    pooled = pooled_labels(runs)
    pages: dict[str, dict[str, Any]] = {}
    for (key, field), label in pooled.items():
        pages.setdefault(key, {})[field] = label

    examples: list[dict] = []
    stats: Counter = Counter()
    exported = skipped = 0
    for key, fields in sorted(pages.items()):
        if not all(field in fields for field in NER_LABELS):
            skipped += 1
            continue
        record_id, _, file_name = key.split("|", 2)
        hashes = {label.get("ocr_sha1", "") for label in fields.values()} - {""}
        page, page_stats = page_examples(record_id, file_name, next(iter(hashes), ""), fields)
        stats.update(page_stats)
        if page:
            exported += 1
            examples.extend(page)

    out_dir = out_dir or Path(config.Training_Data) / "ner"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = out_dir / f"gliner_train_{stamp}.json"
    data_path.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": config.Ner_Model_Path.name,
        "labels": all_labels(),
        "runs": [path.name for path in runs],
        "pages_exported": exported,
        "pages_partially_reviewed": skipped,
        "examples": len(examples),
        "spans": {label: stats[label] for label in all_labels()},
        "extra_mentions": stats["extra_mentions"],
        "unlocated_values": stats["unlocated"],
        "cut_spans": stats["cut_spans"],
        "pages_ocr_changed": stats["ocr_changed"],
        "pages_missing_ocr": stats["missing_ocr"],
        "data": data_path.name,
    }
    data_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    print(json.dumps(export(sys.argv[1:] or None), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
