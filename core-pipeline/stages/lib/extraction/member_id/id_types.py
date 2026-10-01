"""Which kind of ID a Member ID value is, from the key it was read from.

Member ID stays one extraction field, but its true values are labelled by type so training
can tell them apart (NER labels, ranker features). id_types.json lists keys per type; keys
not listed are guessed from their words. Reviewers change the type in the Review UI (not part of this repo).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ID_TYPES: dict[str, str] = {
    "member_id": "Member ID",
    "mrn": "MRN",
    "ssn": "SSN",
    "encounter": "Encounter number",
    "other": "Other ID",
}

# Label each type gets in the GLiNER training data.
NER_ID_LABELS: dict[str, str] = {
    "member_id": "member id",
    "mrn": "medical record number",
    "ssn": "social security number",
    "encounter": "encounter number",
    "other": "other id",
}

_PATH = Path(__file__).with_name("id_types.json")
_WORDS = re.compile(r"[a-z0-9]+")

_GUESSES: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b(ssn?|social)\b")),
    # before mrn: 'Patient Account #' is an account, not a record number
    ("encounter", re.compile(r"\b(encounter|enc|visit|account|acct|acc|fin|financial|csn|har)\b")),
    ("mrn", re.compile(r"\b(mrn|mr|medical|med|chart|empi|unit|patient|pt)\b")),
    ("member_id", re.compile(r"\b(member|subscriber|insurance|insured|ins|policy|pol|id)\b")),
]


def _form(key: str) -> str:
    return " ".join(_WORDS.findall(key.casefold()))


@lru_cache(maxsize=1)
def _catalog() -> dict[str, str]:
    data = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.is_file() else {}
    return {_form(key): id_type for id_type, keys in data.items() if id_type in ID_TYPES for key in keys}


def guess_id_type(key: str) -> str:
    """ID type for a key as printed ('MRN:' -> mrn); '' when there is no key."""
    form = _form(key or "")
    if not form:
        return ""
    known = _catalog().get(form)
    if known:
        return known
    for id_type, pattern in _GUESSES:
        if pattern.search(form):
            return id_type
    return "other"
