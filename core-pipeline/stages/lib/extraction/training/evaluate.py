"""Rules (v0) against a trained version on the same reviewed pages, from a dataset snapshot.

    python -m stages.lib.extraction.training.evaluate [--version v001] [--dataset ds_x] [--split test|train|all]

Both are scored on the same candidates, so the difference is only in the choosing. Per KV
field, over the key-value pairs each extracts (labels.score_pairs): accuracy = right / (right
+ wrong + missed), precision = right / (right + wrong), recall = right / (right + missed),
false positives on pages without the field, and candidate recall (share of true pairs that
are among the candidates: the ceiling any ranker can reach). Headings: line-level
precision / recall, level accuracy, pages exact.
With --version the result is also written to the version folder as metrics_{split}.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import pandas as pd

from .dataset import load, split
from .labels import FIELDS, HEADING_FIELDS, heading_pairs, pair_accuracy, same_pair, score_pairs
from .model import Version, load_version, prepare, version_dir


def _pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).isin({"1", "True", "true"})


def _truth_items(text: str) -> list[dict[str, Any]]:
    # Datasets built before pair scoring stored only the true value_norms.
    return [item if isinstance(item, dict) else {"value_norm": item} for item in json.loads(text or "[]")]


def _ints(text: Any) -> list[int]:
    return [int(part) for part in str(text or "").split() if part.lstrip("-").isdigit()]


def _pairs(rows: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": cid, "value_norm": norm,
            "key_words": _ints(key_words), "value_words": _ints(value_words),
        }
        for cid, norm, key_words, value_words in zip(
            rows["candidate_id"], rows["value_norm"], rows["key_words"], rows["value_words"]
        )
        if norm
    ]


def _metrics(rows: pd.DataFrame, chosen: pd.Series, levels: pd.Series, heading: bool) -> dict[str, Any]:
    n: Counter = Counter()
    for _, group in rows.groupby("group_id", sort=False):
        items = _truth_items(group["truth_json"].iloc[0])
        truth = {item["value_norm"] for item in items if item.get("value_norm")}
        real = group[~_truthy(group["is_placeholder"])]
        picked = real[chosen.loc[real.index].to_numpy()] if not real.empty else real
        selected = set(picked["value_norm"]) - {""}
        n["pages"] += 1
        if heading:
            hits = selected & truth
            n["tp"] += len(hits)
            n["fp"] += len(selected - truth)
            n["fn"] += len(truth - selected)
            n["pages_exact"] += int(selected == truth)
            for norm in hits:
                guess = levels.loc[picked.index[picked["value_norm"] == norm][0]]
                true = next((lvl for lvl in real.loc[real["value_norm"] == norm, "true_level"] if lvl), "")
                n["level_ok"] += int(bool(true) and guess == true)
            continue
        truth_pairs = [item for item in items if item.get("value_norm")]
        # Candidates labelled false are wrong pairs even when their value matches (wrong key).
        label = {
            "truth": truth_pairs,
            "not_present": bool(int(group["page_not_present"].iloc[0])),
            "candidates": {cid: {"verdict": "incorrect"} for cid in real.loc[real["label"] == 0, "candidate_id"]},
        }
        score = score_pairs(_pairs(picked), label)
        n.update({name: score[name] for name in ("right", "wrong", "missed")})
        positives = _pairs(real[real["label"] == 1])
        n["true_pairs"] += len(truth_pairs)
        n["candidate_recall"] += sum(1 for item in truth_pairs if any(same_pair(item, p) for p in positives))
        n["false_positives"] += score["wrong"] if not truth_pairs else 0
    if heading:
        pairs = heading_pairs(n)
        return {
            "pages": n["pages"], "tp": n["tp"], "fp": n["fp"], "fn": n["fn"], "level_ok": n["level_ok"],
            "right": pairs["right"], "wrong": pairs["wrong"], "missed": pairs["missed"],
            "accuracy": pair_accuracy(pairs),
            "precision": _pct(n["tp"], n["tp"] + n["fp"]),
            "recall": _pct(n["tp"], n["tp"] + n["fn"]),
            "level_accuracy": _pct(n["level_ok"], n["tp"]),
            "page_accuracy": _pct(n["pages_exact"], n["pages"]),
        }
    return {
        "pages": n["pages"], "right": n["right"], "wrong": n["wrong"], "missed": n["missed"],
        "accuracy": pair_accuracy(n),
        "precision": _pct(n["right"], n["right"] + n["wrong"]),
        "recall": _pct(n["right"], n["right"] + n["missed"]),
        "false_positives": n["false_positives"],
        "candidate_recall": _pct(n["candidate_recall"], n["true_pairs"]),
    }


def choices(rows: pd.DataFrame, field: str, version: Version | None) -> tuple[pd.Series, pd.Series]:
    """(taken, level) per row of one field: the version's where it has a model, else the rules'.
    KV fields take every pair the run would extract; headings the lines it would select."""
    real = ~_truthy(rows["is_placeholder"])
    chosen = _truthy(rows["rule_selected"]) & real
    if field not in HEADING_FIELDS:
        chosen |= _truthy(rows["rule_accepted"]) & real & (rows["value_norm"].astype(str) != "")
    levels = rows["detail"].astype(str)
    if version is not None and field in version.fields:
        prepared = prepare(rows)
        if not prepared.empty:
            probs, picked, predicted = version.score(prepared, field)
            if field not in HEADING_FIELDS:
                picked = picked | (probs >= version.fields[field].threshold)
            chosen = pd.Series(False, index=rows.index)
            chosen.loc[prepared.index] = picked
            if version.fields[field].level is not None:
                levels = levels.copy()
                levels.loc[prepared.index] = predicted
    return chosen, levels


def evaluate_frame(frame: pd.DataFrame, version: Version | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    totals = {"v0": Counter(), "model": Counter()}
    kv_totals = {"v0": Counter(), "model": Counter()}
    for field in FIELDS:
        rows = frame[frame["field"] == field]
        if rows.empty:
            continue
        heading = field in HEADING_FIELDS
        entry: dict[str, Any] = {"trained": bool(version and field in version.fields)}
        for name, model in (("v0", None), ("model", version)):
            if name == "model" and version is None:
                continue
            chosen, levels = choices(rows, field, model)
            entry[name] = _metrics(rows, chosen, levels, heading)
            counts = {key: entry[name][key] for key in ("right", "wrong", "missed", "pages")}
            totals[name].update(counts)
            if not heading:
                kv_totals[name].update(counts)
        fields[field] = entry

    def summed(groups: dict[str, Counter]) -> dict[str, Any]:
        return {
            name: {**{key: t[key] for key in ("right", "wrong", "missed", "pages")}, "accuracy": pair_accuracy(t)}
            for name, t in groups.items()
            if t["pages"]
        }

    return {"fields": fields, "kv_overall": summed(kv_totals), "overall": summed(totals)}


def report(result: dict[str, Any]) -> str:
    lines = [f"{'field':22s} {'pages':>5s}  {'v0 acc':>7s} {'model':>7s}  {'cand.rec':>8s}  trained  (KV acc = key-value pairs)"]
    for field, entry in result["fields"].items():
        v0, model = entry.get("v0", {}), entry.get("model", {})
        if field in HEADING_FIELDS:
            after = f"  ->  P {model.get('precision')} R {model.get('recall')}" if entry["trained"] else ""
            lines.append(
                f"{field:22s} {v0.get('pages', 0):5d}  {str(v0.get('accuracy')):>7s} {str(model.get('accuracy', '-')):>7s}"
                f"  {'':>8s}  {'yes' if entry['trained'] else 'no'}   P {v0.get('precision')} R {v0.get('recall')}{after}"
            )
            continue
        lines.append(
            f"{field:22s} {v0.get('pages', 0):5d}  {str(v0.get('accuracy')):>7s} {str(model.get('accuracy', '-')):>7s}"
            f"  {str(v0.get('candidate_recall')):>8s}  {'yes' if entry['trained'] else 'no'}"
        )
    for title, what, key in (
        ("KV overall", "key-value pairs", "kv_overall"),
        ("All modules", "key-value pairs and headings", "overall"),
    ):
        for name, item in result.get(key, {}).items():
            lines.append(
                f"{title} {name}: {item['accuracy']}% of {what} "
                f"({item['right']} right, {item['wrong']} wrong, {item['missed']} missed; {item['pages']} page-fields)"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="trained version to compare with v0 (e.g. v001)")
    parser.add_argument("--dataset", help="dataset folder (default: the latest ds_*)")
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = parser.parse_args()
    frame, _, folder = load(args.dataset)
    rows = split(frame, args.split)
    if rows.empty:
        raise SystemExit(f"No reviewed pages in the {args.split} split of {folder.name}.")
    version = load_version(args.version) if args.version else None
    result = {"dataset": folder.name, "split": args.split, "version": args.version, **evaluate_frame(rows, version)}
    print(report(result))
    if args.version:
        path = version_dir(args.version) / f"metrics_{args.split}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
