"""Ported verbatim from the V1 Member_Verification rule set (name_2_words_rules).

Only the imports changed: V1 reached its siblings through sys.path inserts,
this package uses relative imports. The decision logic is byte-for-byte the
reference's.
"""
from __future__ import annotations

from .name_common import (
    BOTH_FULL,
    INITIAL,
    classify_two_word_name,
    tokenize,
)
from .base_rules import combine_evidences, is_present


def verify_two_word_name(
    found_name: str,
    first_name: str,
    last_name: str,
    dob_ok: bool,
    id_ok: bool,
) -> str:
    if not is_present(found_name):
        return combine_evidences(name_ok=False, dob_ok=dob_ok, id_ok=id_ok)
    kind = classify_two_word_name(tokenize(found_name), first_name, last_name)
    if kind == BOTH_FULL:
        return combine_evidences(name_ok=True, dob_ok=dob_ok, id_ok=id_ok)
    if kind == INITIAL:
        return combine_evidences(
            name_ok=True,
            dob_ok=dob_ok,
            id_ok=id_ok,
            initial_only=True,
        )
    return combine_evidences(name_ok=False, dob_ok=dob_ok, id_ok=id_ok)
