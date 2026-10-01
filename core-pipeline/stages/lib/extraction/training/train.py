"""Train a new version: one LightGBM model per field (plus a heading level model) from a dataset.

    python -m stages.lib.extraction.training.train [--dataset ds_x] [--min-groups 30] [--description "..."] [--split train|all]

Trains on the train split only (dataset.split: about 1 record in 5 is held out for testing,
by record), or with --split all on every record when the test set is a separate batch.
Heading models also learn the heading vocabulary of the reviews (see training/model.py).
Tunes each field's threshold on out-of-fold scores grouped by record (KV: the
key-value pair accuracy of the pairs at or above it; headings: line F1),
then scores v0 and the new version on the test split (and on the train split, marked as
optimistic). A field with fewer than --min-groups reviewed page-fields, or without both true
and false candidates, gets no model and keeps the rules in this version.

Output: {config.Model_Registry}/vNNN/ (next free number), see training/model.py. Everything
runs on CPU in seconds to minutes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import load, split
from .evaluate import _metrics, evaluate_frame, report
from .labels import FIELDS, HEADING_FIELDS
from .model import (
    CATEGORICAL,
    LEVEL_POSITIVE,
    NUMERIC,
    FieldModel,
    Version,
    add_vocab,
    categories_of,
    heading_vocab,
    matrix,
    prepare,
    select,
    vocab_counts,
)
from ..util import config

PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "min_data_per_group": 5,
    "cat_smooth": 10,
    "verbose": -1,
    "seed": 7,
    "num_threads": 0,
}
ROUNDS = 200
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def _fit(x: pd.DataFrame, y: np.ndarray):
    import lightgbm as lgb

    params = {**PARAMS, "min_data_in_leaf": int(max(2, min(20, len(y) // 10)))}
    return lgb.train(params, lgb.Dataset(x, label=y, categorical_feature=CATEGORICAL, free_raw_data=False), ROUNDS)


def _objective(rows: pd.DataFrame, chosen: np.ndarray, field: str) -> float:
    """What the threshold maximizes: key-value pair accuracy (KV) or line-level F1 (headings)."""
    heading = field in HEADING_FIELDS
    metrics = _metrics(rows, pd.Series(chosen, index=rows.index), rows["detail"].astype(str), heading)
    if heading:
        p, r = (metrics["precision"] or 0.0), (metrics["recall"] or 0.0)
        return 2 * p * r / (p + r) if p + r else 0.0
    return metrics["accuracy"] or 0.0


def _taken(rows: pd.DataFrame, probs: np.ndarray, field: str, threshold: float) -> np.ndarray:
    """What the run keeps at this threshold: every KV pair at or above it (the extracted pairs
    accuracy counts), the selected lines for headings."""
    if field in HEADING_FIELDS:
        return select(rows["group_id"], probs, True, threshold)
    return probs >= threshold


def _fold_matrices(
    rows: pd.DataFrame, fit_idx: np.ndarray, val_idx: np.ndarray, categories: dict, field: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit / validation inputs; a heading vocabulary comes from the fit records only."""
    fit, val = rows.iloc[fit_idx], rows.iloc[val_idx]
    if field in HEADING_FIELDS:
        vocab = heading_vocab(fit)
        fit, val = add_vocab(fit, vocab), add_vocab(val, vocab)
    return matrix(fit, categories), matrix(val, categories)


def _threshold(rows: pd.DataFrame, categories: dict, y: np.ndarray, field: str) -> tuple[float, str]:
    """Best threshold on out-of-fold probabilities (folds by record); 0.5 when there are too few records."""
    from sklearn.model_selection import GroupKFold

    records = rows["record_id"].to_numpy()
    n_folds = min(5, len(set(records)))
    if n_folds < 2:
        return 0.5, "default (one record: no out-of-fold scores)"
    oof = np.zeros(len(rows))
    for fit_idx, val_idx in GroupKFold(n_splits=n_folds).split(rows, y, records):
        if len(set(y[fit_idx])) < 2:
            oof[val_idx] = y[fit_idx].mean()
            continue
        x_fit, x_val = _fold_matrices(rows, fit_idx, val_idx, categories, field)
        oof[val_idx] = _fit(x_fit, y[fit_idx]).predict(x_val)
    scored = [(_objective(rows, _taken(rows, oof, field, t), field), -abs(t - 0.5), t) for t in THRESHOLDS]
    best = max(scored)
    return best[2], f"out-of-fold over {n_folds} record folds (objective {best[0]:.1f})"


def train_field(rows: pd.DataFrame, field: str, min_groups: int) -> tuple[dict[str, Any], FieldModel | None]:
    """(report, model) for one field's training rows; model is None when the field keeps the rules."""
    candidates = prepare(rows)
    groups = rows["group_id"].nunique()
    info: dict[str, Any] = {"groups": int(groups), "candidates": int(len(candidates))}
    y = candidates["label"].to_numpy().astype(int) if not candidates.empty else np.array([])
    if groups < min_groups:
        return {**info, "trained": False, "reason": f"{groups} reviewed page-fields, needs {min_groups}"}, None
    if len(set(y)) < 2:
        return {**info, "trained": False, "reason": "needs both true and false candidates"}, None
    categories = categories_of(candidates)
    threshold, how = _threshold(candidates, categories, y, field)
    vocab = heading_vocab(candidates) if field in HEADING_FIELDS else None
    if vocab is not None:
        candidates = add_vocab(candidates, vocab)
        info["vocabulary"] = {
            "texts": len(vocab),
            "true_heading_texts": sum(1 for true, _ in vocab.values() if true),
        }
    x = matrix(candidates, categories)
    ranker = _fit(x, y)
    gain = ranker.feature_importance(importance_type="gain")
    top = sorted(zip(x.columns, gain), key=lambda item: -item[1])[:10]
    info.update(
        trained=True,
        positives=int(y.sum()),
        threshold=threshold,
        threshold_from=how,
        top_features=[{"feature": name, "gain": round(float(value), 2)} for name, value in top if value > 0],
    )
    level = None
    if field in HEADING_FIELDS:
        positives = candidates[(candidates["label"] == 1) & (candidates["true_level"] != "")]
        levels = (positives["true_level"] == LEVEL_POSITIVE).astype(int).to_numpy()
        if len(positives) >= min_groups and len(set(levels)) == 2:
            level = _fit(matrix(positives, categories), levels)
            info["level_model"] = f"trained on {len(positives)} true headings"
        else:
            info["level_model"] = "not trained (too few true headings with both levels); rules' level kept"
    return info, FieldModel(ranker, float(threshold), categories, level, vocab)


def _next_version() -> str:
    root = Path(config.Model_Registry)
    numbers = [int(path.name[1:]) for path in root.glob("v[0-9][0-9][0-9]") if path.is_dir()] if root.is_dir() else []
    return f"v{max(numbers, default=0) + 1:03d}"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.REPO_ROOT, check=False
        ).stdout.strip()
    except OSError:
        return ""


def train(dataset: str | None, min_groups: int, description: str, split_name: str = "train") -> Path:
    frame, ds_manifest, ds_folder = load(dataset)
    train_rows = split(frame, split_name)
    test_rows = split(frame, "test") if split_name == "train" else frame.iloc[0:0]
    if train_rows.empty:
        raise SystemExit(f"No reviewed pages in the {split_name} split of {ds_folder.name}.")

    version = _next_version()
    folder = Path(config.Model_Registry) / version
    fields: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    categories: dict[str, Any] = {}
    models: dict[str, FieldModel] = {}
    for field in FIELDS:
        rows = train_rows[train_rows["field"] == field]
        if rows.empty:
            fields[field] = {"groups": 0, "trained": False, "reason": "no reviewed pages"}
            continue
        info, model = train_field(rows, field, min_groups)
        fields[field] = info
        if model is not None:
            thresholds[field] = model.threshold
            categories[field] = model.categories
            models[field] = model

    if not models:
        print(json.dumps(fields, indent=2))
        raise SystemExit(f"No field has enough reviewed pages to train (--min-groups {min_groups}); nothing saved.")

    folder.mkdir(parents=True, exist_ok=True)
    for field, model in models.items():
        model.booster.save_model(str(folder / f"ranker_{field}.txt"))
        if model.level is not None:
            model.level.save_model(str(folder / f"level_{field}.txt"))
        if model.vocab is not None:
            (folder / f"vocab_{field}.json").write_text(
                json.dumps(vocab_counts(model.vocab), indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8"
            )
    (folder / "features.json").write_text(
        json.dumps({"numeric": NUMERIC, "categorical": CATEGORICAL, "categories": categories}, indent=2), encoding="utf-8"
    )
    (folder / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    trained = Version(version)
    metrics = {
        "test": evaluate_frame(test_rows, trained) if not test_rows.empty else None,
        "train_optimistic": evaluate_frame(train_rows, trained),
    }
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "version": version,
        "description": description or f"LightGBM ranker for {', '.join(models)}; other fields use the rules.",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": ds_folder.name,
        "dataset_pages": ds_manifest.get("pages"),
        "split": split_name,
        "train_records": sorted(set(train_rows["record_id"])),
        "test_records": sorted(set(test_rows["record_id"])),
        "min_groups": min_groups,
        "params": {**PARAMS, "rounds": ROUNDS},
        "fields_trained": sorted(models),
        "fields": fields,
        "git_commit": _git_commit(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{version}: trained {', '.join(sorted(models))}")
    for field, info in fields.items():
        if not info.get("trained"):
            print(f"  {field:22s} rules kept: {info.get('reason')}")
    if metrics["test"]:
        print("test split:")
        print(report(metrics["test"]))
    elif split_name == "all":
        print("trained on every reviewed record; test it on another batch's dataset:")
        print(f"  Training\\evaluate.py --dataset <ds of the test batch> --split all --version {version}")
    else:
        print("test split: no reviewed test records yet (metrics.json has train-split numbers only, optimistic)")
    print(folder)
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="dataset folder (default: the latest ds_*)")
    parser.add_argument("--min-groups", type=int, default=30, help="reviewed page-fields a field needs for a model")
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "all"],
        help="train: hold out about 1 record in 5 as the test split; all: train on every record "
        "(when the test set is a separate batch, scored with evaluate.py --dataset ... --split all)",
    )
    args = parser.parse_args()
    train(args.dataset, args.min_groups, args.description, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
