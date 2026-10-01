"""Ported verbatim from the V1 Member_Verification rule set (name_3_words_rules).

Only the imports changed (relative instead of sys.path inserts).
"""
from __future__ import annotations

from .name_common import (
    ALL_FULL,
    TWO_FULL,
    classify_three_word_name,
    tokenize,
)
from .base_rules import combine_evidences, is_present


def verify_three_word_name(
    found_name: str,
    first_name: str,
    middle_name: str,
    last_name: str,
    dob_ok: bool,
    id_ok: bool,
) -> str:
    if not is_present(found_name):
        return combine_evidences(name_ok=False, dob_ok=dob_ok, id_ok=id_ok)
    kind = classify_three_word_name(
        tokenize(found_name),
        first_name,
        middle_name,
        last_name,
    )
    if kind in {ALL_FULL, TWO_FULL}:
        return combine_evidences(name_ok=True, dob_ok=dob_ok, id_ok=id_ok)
    return combine_evidences(name_ok=False, dob_ok=dob_ok, id_ok=id_ok)
