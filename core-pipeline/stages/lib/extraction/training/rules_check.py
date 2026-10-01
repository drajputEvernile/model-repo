"""Score the current rules against the reviews without writing a run.

    python -m stages.lib.extraction.training.rules_check [--run RUN] [--show FIELD ...]

Re-extracts the documents of a reviewed run in memory (no overlays, no headings, no model)
and scores every reviewed page-field with the reviews' key-value pairs, exactly like the
UI's accuracy. --show lists the wrong and missed pairs of those fields. For trying rule
changes before a real rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import pipeline
from .labels import (
    KV_FIELDS,
    extracted_pairs,
    evaluate_log,
    pooled_labels,
    run_log,
    score_pairs,
)
from .ner_export import run_dirs
from ..util import config
from ..util.documents import load_local_documents


def current_log(run_dir: Path) -> pd.DataFrame:
    """The candidate log the current rules make for the run's documents."""
    records = set(run_log(run_dir)["record_id"])
    frames = []
    for document in load_local_documents(config.OCR_Input):
        if document.record_id in records:
            result = pipeline.extract_document(document, headings=False)
            frames.append(result.candidates(run_dir.name, "v0"))
    return pd.concat(frames, ignore_index=True).fillna("").astype(str)


def _pair_text(pair: dict) -> str:
    key = pair.get("key_text") or pair.get("key") or ""
    where = f" [{pair['region']}]" if pair.get("region") else ""
    return f"{pair.get('value', pair.get('value_norm', ''))!r} key={key!r}{where} words={pair.get('value_words')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="reviewed run (default: the newest KV_Run_*)")
    parser.add_argument("--show", nargs="*", default=[], help="fields whose wrong / missed pairs to list")
    args = parser.parse_args()
    runs = run_dirs(None)
    run_dir = next(path for path in runs if path.name == args.run) if args.run else runs[-1]
    pooled = pooled_labels(runs)
    log = current_log(run_dir)

    before, after = evaluate_log(run_log(run_dir), pooled), evaluate_log(log, pooled)
    print(f"{'field':22s} {'run':>6s} {'rules now':>10s}   right wrong missed")
    for field in KV_FIELDS:
        old, new = before["fields"][field], after["fields"][field]
        print(f"{field:22s} {str(old['accuracy']):>6s} {str(new['accuracy']):>10s}   {new['correct']:5d} {new['wrong']:5d} {new['missed']:6d}")
    print(f"{'All':22s} {str(before['accuracy']):>6s} {str(after['accuracy']):>10s}   {after['correct']:5d} {after['wrong']:5d} {after['missed']:6d}")

    for (key, field), extracted in sorted(extracted_pairs(log).items()):
        label = pooled.get((key, field))
        if field not in args.show or label is None:
            continue
        details: list = []
        score_pairs(extracted["found"], label, details)
        problems = [(status, pair) for status, pair in details if status != "right"]
        if problems:
            print(f"\n{key} {field}")
            for status, pair in problems:
                print(f"  {status:6s} {_pair_text(pair)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
