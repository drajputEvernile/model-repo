"""Ported from the V1 Member_Verification rule set (wrong_member_rules).

The decision is unchanged. The page text it was handed (and immediately discarded) is no longer
a parameter: the names come from the extraction, which has already read the page.
"""
from __future__ import annotations

from collections.abc import Sequence

from .name_common import name_matches


def wrong_member_on_page(
    expected: dict[str, str],
    name_mode: str,
    ner_names: Sequence[str] = (),
) -> bool:
    """True when a patient-name sentence on the page names another member.

    ``ner_names`` are the member names the extraction found on this page. The
    page carries a wrong member when there is at least one of them and not one
    verifies as the expected member.
    """
    names = [name for name in ner_names if name and name.strip()]
    if not names:
        return False
    first = expected.get("DummyFirstName", "")
    middle = expected.get("DummyMiddleName", "")
    last = expected.get("DummyLastName", "")
    return not any(
        name_matches(name, first, last, middle, name_mode) for name in names
    )
