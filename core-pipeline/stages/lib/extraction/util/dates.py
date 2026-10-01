"""Shared date finders for DOB, DOS and electronic-signature extraction.

Supports numeric dates, dates that mix numbers with short/full month names,
and date ranges ("05/10/2024 - 05/10/2024", "1/4/2024 to 1/6/2024").
"""

from __future__ import annotations

import re
from datetime import date, datetime

_CLEAN = re.compile(r"\s+")

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

_DATE_BODY = (
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?"
    r"|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|"
    # OCR form seen in dob.csv: 26.Nov.1953 / 28.Feb.2024
    rf"\d{{1,2}}\.{_MONTH}\.\d{{2,4}}"
    r"|"
    rf"{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}"
    r"|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+\d{{2,4}}"
    r"|"
    rf"\d{{1,2}}[/-]{_MONTH}[/-]\d{{2,4}}"
    r"|"
    rf"{_MONTH}[/-]\d{{1,2}}[/-]\d{{2,4}}"
)

# Capture group 1 is always the full date text.
_DATE = re.compile(rf"\b({_DATE_BODY})\b", flags=re.IGNORECASE)

# Groups 1 and 2 are the two ends of the range.
_RANGE = re.compile(
    rf"\b({_DATE_BODY})\s*(?:-|–|—|to|thru|through)\s*({_DATE_BODY})\b",
    flags=re.IGNORECASE,
)

_YEAR_TAIL = re.compile(r"(\d{2,4})\s*$")
_ISO_YEAR = re.compile(r"^(\d{4})-")


def find_dates(text: str) -> list[re.Match[str]]:
    return list(_DATE.finditer(text or ""))


def find_date_ranges(text: str) -> list[re.Match[str]]:
    return list(_RANGE.finditer(text or ""))


def normalize_date(raw: str) -> str:
    text = _CLEAN.sub(" ", (raw or "").strip()).rstrip(".,;")
    if not text:
        return ""
    iso = re.match(
        r"^(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?$",
        text,
        flags=re.IGNORECASE,
    )
    if iso:
        return iso.group(1)
    return text


_CANON_FORMATS = (
    "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%Y-%m-%d",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%b. %d, %Y",
    "%d %B %Y", "%d %b %Y", "%d.%b.%Y", "%d-%b-%Y", "%d/%b/%Y",
)


def canonical_date(raw: str) -> str:
    """ISO form for comparing dates written differently ('5/3/2024' == '05/03/2024')."""
    text = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", normalize_date(raw), flags=re.IGNORECASE)
    text = re.sub(r"\bSept\b", "Sep", text, flags=re.IGNORECASE)
    for fmt in _CANON_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text.casefold()


def date_year(raw: str) -> int | None:
    """Four-digit year of a matched date; two-digit years map to 20xx unless in the future."""
    text = normalize_date(raw)
    iso = _ISO_YEAR.match(text)
    if iso:
        return int(iso.group(1))
    tail = _YEAR_TAIL.search(text)
    if not tail:
        return None
    year = int(tail.group(1))
    if len(tail.group(1)) == 2:
        pivot = date.today().year % 100 + 1
        year += 2000 if year <= pivot else 1900
    return year
