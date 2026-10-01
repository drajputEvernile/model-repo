"""Does an extracted DOB / member ID say what the manifest says?

Ported from the V1 rule-based extractors (``extract_dob`` / ``extract_member_id``). Those
searched a page's text for the manifest's value; the extraction now reads the page's
fields itself, so the same comparison runs on each value it extracted instead. The tests are
unchanged: a DOB matches when its day, month and year sit together in any order, a member
ID when the exact value appears with no letter or digit beside it.
"""
from __future__ import annotations

import re

from ...extraction.util.dates import canonical_date

_WORD = re.compile(r"[A-Za-z0-9]+")
_DUMMY_DOB = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def _date_parts(dummy_dob: str) -> tuple[str, str, str] | None:
    """DummyDOB is MM/DD/YYYY."""
    match = _DUMMY_DOB.fullmatch((dummy_dob or "").strip())
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _norm(token: str) -> str:
    if token.isdigit():
        return str(int(token))
    return token


def _window_has_dob(window: list[str], year_index: int, month: str, day: str, year: str) -> bool:
    month_n, day_n, year_n = _norm(month), _norm(day), year
    orders = {
        (day_n, month_n, year_n),
        (month_n, day_n, year_n),
        (year_n, month_n, day_n),
        (year_n, day_n, month_n),
    }
    if year_index >= 2:
        before = tuple(_norm(token) for token in window[year_index - 2 : year_index + 1])
        if before in orders:
            return True
    if year_index + 2 < len(window):
        after = tuple(_norm(token) for token in window[year_index : year_index + 3])
        if after in orders:
            return True
    return False


def date_parts_match(text: str, dummy_dob: str) -> bool:
    parts = _date_parts(dummy_dob)
    if not parts:
        return False
    month, day, year = parts
    words = _WORD.findall(text or "")
    for index, word in enumerate(words):
        if word != year:
            continue
        start = max(0, index - 3)
        stop = min(len(words), index + 4)
        if _window_has_dob(words[start:stop], index - start, month, day, year):
            return True
    return False


def dob_matches(value: str, dummy_dob: str) -> bool:
    """The extracted DOB is the manifest's. A month written as a name ('March 20, 1981') is
    compared as the date it is, which the digit-window test cannot do."""
    if not (value or "").strip() or not (dummy_dob or "").strip():
        return False
    if date_parts_match(value, dummy_dob):
        return True
    found = canonical_date(value)
    return bool(_ISO.fullmatch(found)) and found == canonical_date(dummy_dob)


def member_id_matches(value: str, member_id: str) -> bool:
    """The manifest's member ID appears in the extracted ID exactly (case aside)."""
    expected = (member_id or "").strip()
    if not expected or expected.upper() == "N/A" or not (value or "").strip():
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(expected) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, value, flags=re.IGNORECASE))
