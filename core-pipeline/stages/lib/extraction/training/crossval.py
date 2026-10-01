"""Leave-one-record-out cross-validation: how a trained version does on documents it never saw.

    python -m stages.lib.extraction.training.crossval [--dataset ds_x] [--min-groups 10]

For every record in the dataset, each field's model is trained (as train.py does, threshold
and heading vocabulary included) on the other records only and scored on that record's
reviewed pages. The held-out
counts are summed over all records, so every page is scored exactly once by a model that did
not train on its document. v0 (rules) is scored on the same pages for comparison. Nothing is
saved to the registry; the report is written to the dataset folder as crossval.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from .dataset import load
from .evaluate import evaluate_frame
from .labels import FIELDS, pair_accuracy
from .model import CATEGORICAL, NUMERIC, FieldModel, Version
from .train import train_field

KEYS = ("right", "wrong", "missed", "pages")


def _version(models: dict[str, FieldModel]) -> Version:
    """An in-memory version (no registry folder) holding the fold's models."""
    version = object.__new__(Version)
    version.name = "crossval"
    version.numeric = NUMERIC
    version.categorical = CATEGORICAL
    version.fields = models
    return version


def crossval(dataset: str | None, min_groups: int) -> dict[str, Any]:
    frame, _, folder = load(dataset)
    records = sorted(set(frame["record_id"]))
    if len(records) < 2:
        raise SystemExit(f"{folder.name} has {len(records)} record(s); cross-validation needs at least 2.")
    fields: dict[str, dict[str, Counter]] = {}
    folds: list[dict[str, Any]] = []
    for record in records:
        train_rows = frame[frame["record_id"] != record]
        test_rows = frame[frame["record_id"] == record]
        models: dict[str, FieldModel] = {}
        for field in FIELDS:
            rows = train_rows[train_rows["field"] == field]
            if rows.empty:
                continue
            _, model = train_field(rows, field, min_groups)
            if model is not None:
                models[field] = model
        result = evaluate_frame(test_rows, _version(models))
        fold = {"record": record, "trained": sorted(models)}
        for field, entry in result["fields"].items():
            slot = fields.setdefault(field, {"v0": Counter(), "model": Counter()})
            for name in ("v0", "model"):
                slot[name].update({key: entry[name][key] for key in KEYS})
            fold[field] = entry["model"]["accuracy"]
        folds.append(fold)
        print(f"  held out {record}: " + ", ".join(f"{f} {fold.get(f)}" for f in result["fields"]), flush=True)

    def summary(counts: Counter) -> dict[str, Any]:
        return {**{key: counts[key] for key in KEYS}, "accuracy": pair_accuracy(counts)}

    out: dict[str, Any] = {"dataset": folder.name, "records": records, "min_groups": min_groups, "fields": {}, "folds": folds}
    totals = {"v0": Counter(), "model": Counter()}
    kv = {"v0": Counter(), "model": Counter()}
    for field, slot in fields.items():
        out["fields"][field] = {name: summary(counts) for name, counts in slot.items()}
        for name, counts in slot.items():
            totals[name].update(counts)
            if not field.startswith("heading_"):
                kv[name].update(counts)
    out["kv_overall"] = {name: summary(counts) for name, counts in kv.items()}
    out["overall"] = {name: summary(counts) for name, counts in totals.items()}
    (folder / "crossval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def report(result: dict[str, Any]) -> str:
    lines = [f"{'module':22s} {'v0':>6s} {'held-out':>9s}   right wrong missed"]
    rows = [*result["fields"].items(), ("KV overall", result["kv_overall"]), ("All modules", result["overall"])]
    for name, entry in rows:
        model = entry["model"]
        lines.append(
            f"{name:22s} {str(entry['v0']['accuracy']):>6s} {str(model['accuracy']):>9s}   "
            f"{model['right']:5d} {model['wrong']:5d} {model['missed']:6d}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="dataset folder (default: the latest ds_*)")
    parser.add_argument("--min-groups", type=int, default=10, help="reviewed page-fields a field needs for a model")
    args = parser.parse_args()
    result = crossval(args.dataset, args.min_groups)
    print(report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
