"""Trained candidate selection, shared by train.py, evaluate.py and the pipeline.

One LightGBM binary model per field scores each candidate of a (page, field) group with the
probability that it is a true value. A single-value field takes its best candidate when that
probability reaches the field's threshold, else nothing; a multi-value field (Member ID,
headings) takes every candidate at or above it. Headings also get a level model (Heading vs
Subheading). A field without a model keeps the rules' choice. The rules still generate every
candidate: the model never invents a value.

Heading vocabulary: a heading model also sees how the candidate's text was reviewed in other
documents (heading_vocab_*: in how many records it was a true heading, in how many a false
one). The vocabulary is learnt from the training reviews and saved with the version, so it
grows with every reviewed batch; heading_common is the match with common_headings.txt.
A training row's vocabulary counts leave out its own record, so the model learns what the
vocabulary is worth on documents it has not seen.

Version folder {config.Model_Registry}/vNNN/:
    ranker_{field}.txt, level_{field}.txt   LightGBM boosters
    vocab_{field}.json                      heading vocabulary {text: [true records, false records]}
                                            (texts never confirmed as headings only as a hash)
    features.json                           feature columns and per-field category lists
    thresholds.json                         per-field minimum probability to select
    manifest.json, metrics.json             what it was trained on and how it scored
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..heading.extract import DETECTORS, common_heading
from ..member_id.id_types import guess_id_type
from ..util import config

NUMERIC = [
    "rule_score", "rule_accepted", "rule_selected",
    "key_edit_distance", "key_weak", "key_cluster_size", "key_n_words",
    "key_x0", "key_y0", "key_x1", "key_y1",
    "value_found", "value_x0", "value_y0", "value_x1", "value_y1", "word_gap", "line_gap", "dx",
    "value_len", "value_n_words", "digit_ratio", "alpha_ratio", "upper_ratio", "has_comma", "has_initial",
    "value_is_date",
    "page_count", "page_frac", "n_keys_page", "n_trusted_keys_page", "n_field_keys", "n_field_candidates",
    "n_field_accepted", "page_value_count", "page_distinct_values",
    "record_value_pages", "record_value_share", "record_distinct_values",
    "heading_height_ratio", "heading_key_overlap", "heading_value_overlap", "heading_whole_line",
    "heading_line_count", "heading_ends_colon", "heading_tail_words", "heading_common",
    "heading_vocab_true", "heading_vocab_false", "heading_vocab_rate",
    # derived within the (page, field) group
    "score_rank", "score_gap", "same_value_in_group",
]
CATEGORICAL = [
    "key", "key_match", "key_trusted_reason", "region", "relation", "rule_source", "detail_cat",
    "heading_key_field", "heading_position", "id_type",
]
MULTI_VALUE = frozenset({"member_id", *DETECTORS})
# Fields whose detail is a small category (DOS tier, provider profile, heading level), not a value.
_DETAIL_CATEGORY = frozenset({"dos", "provider_name", *DETECTORS})
LEVEL_POSITIVE = "Heading"


def version_dir(version: str) -> Path:
    return Path(config.Model_Registry) / version


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).isin({"1", "True", "true"})


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Candidate rows (placeholders dropped) with group id and derived group features."""
    frame = frame[~_truthy(frame["is_placeholder"])].copy()
    if frame.empty:
        return frame
    frame["group_id"] = (
        frame["record_id"].astype(str) + "|" + frame["page_number"].astype(str) + "|"
        + frame["file_name"].astype(str) + "|" + frame["field"].astype(str)
    )
    score = pd.to_numeric(frame["rule_score"], errors="coerce").fillna(0.0)
    frame["score_rank"] = score.groupby(frame["group_id"]).rank(ascending=False, method="min")
    frame["score_gap"] = score.groupby(frame["group_id"]).transform("max") - score
    norm = frame["value_norm"].astype(str)
    frame["same_value_in_group"] = norm.groupby([frame["group_id"], norm]).transform("size").where(norm != "", 0)
    frame["detail_cat"] = np.where(frame["field"].isin(_DETAIL_CATEGORY), frame["detail"].astype(str), "")
    if "id_type" not in frame.columns:
        frame["id_type"] = ""
    missing = (frame["field"] == "member_id") & (frame["id_type"].astype(str) == "")
    frame.loc[missing, "id_type"] = [
        guess_id_type(str(key) or str(text)) for text, key in zip(frame.loc[missing, "key_text"], frame.loc[missing, "key"])
    ]
    heading = frame["field"].isin(DETECTORS)
    frame["heading_common"] = np.nan
    frame.loc[heading, "heading_common"] = [common_heading(str(text)) for text in frame.loc[heading, "value"]]
    return frame


VOCAB_COLUMNS = ("heading_vocab_true", "heading_vocab_false", "heading_vocab_rate")
# Record sets while training; counts once saved with a version.
Vocab = dict[str, tuple[Any, Any]]


def heading_vocab(frame: pd.DataFrame) -> Vocab:
    """Heading text (value_norm) -> (records where it is a true heading, records where it is a false one)."""
    vocab: dict[str, tuple[set[str], set[str]]] = {}
    real = frame[frame["value_norm"].astype(str) != ""]
    for norm, record, label in zip(real["value_norm"].astype(str), real["record_id"].astype(str), real["label"].astype(int)):
        entry = vocab.setdefault(norm, (set(), set()))
        entry[0 if label else 1].add(record)
    return vocab


def _hidden(norm: str) -> str:
    return "sha1:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def vocab_counts(vocab: Vocab) -> dict[str, list[int]]:
    """What a version saves: counts only, no record ids. A text never confirmed as a heading
    can be page content (a name, a date), so it is saved as a hash, and only when it was
    rejected in at least two records."""
    out: dict[str, list[int]] = {}
    for norm, (true, false) in vocab.items():
        counts = [len(true), len(false)] if isinstance(true, set) else [int(true), int(false)]
        if counts[0]:
            out[norm] = counts
        elif counts[1] >= 2:
            out[_hidden(norm)] = counts
    return out


def add_vocab(frame: pd.DataFrame, vocab: Vocab) -> pd.DataFrame:
    """heading_vocab_* for each row; with record sets the row's own record is not counted."""
    frame = frame.copy()
    true_counts, false_counts = [], []
    for norm, record in zip(frame["value_norm"].astype(str), frame["record_id"].astype(str)):
        true, false = (vocab.get(norm) or vocab.get(_hidden(norm)) or (0, 0)) if norm else (0, 0)
        if isinstance(true, set):
            true, false = len(true - {record}), len(false - {record})
        true_counts.append(int(true))
        false_counts.append(int(false))
    t, f = np.array(true_counts, dtype=float), np.array(false_counts, dtype=float)
    frame["heading_vocab_true"] = t
    frame["heading_vocab_false"] = f
    frame["heading_vocab_rate"] = np.where(t + f > 0, (t + 0.5) / (t + f + 1.0), np.nan)
    return frame


def categories_of(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {column: sorted(set(frame[column].astype(str)) - {""}) for column in CATEGORICAL}


def matrix(
    frame: pd.DataFrame,
    categories: dict[str, list[str]],
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> pd.DataFrame:
    """Model input: numeric columns as floats (blank = missing), categoricals on a fixed vocabulary.
    A trained version passes the columns it was trained with."""
    out = pd.DataFrame(index=frame.index)
    for column in numeric or NUMERIC:
        values = frame[column] if column in frame.columns else pd.Series("", index=frame.index)
        out[column] = pd.to_numeric(values.replace("", np.nan), errors="coerce").astype(float)
    for column in categorical or CATEGORICAL:
        values = frame[column].astype(str) if column in frame.columns else pd.Series("", index=frame.index)
        out[column] = pd.Categorical(values.where(values != "", None), categories=categories.get(column, []))
    return out


def select(groups: pd.Series, probs: np.ndarray, multi: bool, threshold: float) -> np.ndarray:
    """Which candidates a field takes: all above threshold (multi) or the best one if above it."""
    above = probs >= threshold
    if multi:
        return above
    best = pd.Series(probs, index=range(len(probs))).groupby(groups.to_numpy()).idxmax().to_numpy()
    mask = np.zeros(len(probs), dtype=bool)
    mask[best] = True
    return mask & above


@dataclass
class FieldModel:
    booster: Any
    threshold: float
    categories: dict[str, list[str]]
    level: Any = None
    vocab: Vocab | None = None


class Version:
    """A trained version loaded from the registry."""

    def __init__(self, name: str):
        import lightgbm as lgb

        folder = version_dir(name)
        features = json.loads((folder / "features.json").read_text(encoding="utf-8"))
        thresholds = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
        self.name = name
        self.numeric: list[str] = features.get("numeric") or NUMERIC
        self.categorical: list[str] = features.get("categorical") or CATEGORICAL
        self.fields: dict[str, FieldModel] = {}
        for field, threshold in thresholds.items():
            path = folder / f"ranker_{field}.txt"
            if not path.is_file():
                continue
            level = folder / f"level_{field}.txt"
            vocab = folder / f"vocab_{field}.json"
            self.fields[field] = FieldModel(
                booster=lgb.Booster(model_file=str(path)),
                threshold=float(threshold),
                categories=features["categories"][field],
                level=lgb.Booster(model_file=str(level)) if level.is_file() else None,
                vocab={
                    norm: tuple(counts)
                    for norm, counts in json.loads(vocab.read_text(encoding="utf-8")).items()
                } if vocab.is_file() else None,
            )

    def score(self, frame: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """(probabilities, selected, levels) for prepared rows of one field. A candidate is
        extracted (accepted) when its probability reaches the threshold."""
        model = self.fields[field]
        if model.vocab is not None:
            frame = add_vocab(frame, model.vocab)
        x = matrix(frame, model.categories, self.numeric, self.categorical)
        probs = model.booster.predict(x)
        chosen = select(frame["group_id"], probs, field in MULTI_VALUE, model.threshold)
        levels = [""] * len(frame)
        if model.level is not None:
            levels = [LEVEL_POSITIVE if p >= 0.5 else "Subheading" for p in model.level.predict(x)]
        return probs, chosen, levels

    def apply(self, log: pd.DataFrame) -> pd.DataFrame:
        """The candidate log with model_score / model_accepted / model_selected / model_level
        filled for modelled fields."""
        log = log.copy()
        for column in ("model_score", "model_accepted", "model_selected", "model_level"):
            log[column] = ""
        for field, model in self.fields.items():
            rows = prepare(log[log["field"] == field])
            if rows.empty:
                continue
            probs, chosen, levels = self.score(rows, field)
            log.loc[rows.index, "model_score"] = [f"{p:.4f}" for p in probs]
            log.loc[rows.index, "model_accepted"] = ["1" if p >= model.threshold else "0" for p in probs]
            log.loc[rows.index, "model_selected"] = ["1" if c else "0" for c in chosen]
            log.loc[rows.index, "model_level"] = levels
        return log


@lru_cache(maxsize=4)
def load_version(name: str) -> Version:
    return Version(name)
