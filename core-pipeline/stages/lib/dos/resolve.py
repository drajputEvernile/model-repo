"""Date of service: resolve the chart from the dates the extraction found.

The extraction layer (``stages/lib/extraction``) finds each page's date of service — the
labelled date, an admit / discharge pair, its own score — and stages it. What it cannot know
is which *encounter* a page belongs to. That is decided here, in page order, by the page's type:

  A progress note opens a span. A later page whose own date is weak (at or below
  ``span_override_score``), or that has none, takes the progress note's date. A face sheet,
  med list and the like never replace an encounter already found. Demographics and
  injection pages, and pages with no date, keep ``DOS_DEFAULT_DATE``.

Page types, the default date and the override score are in ``dos_canon.json``; reloaded on change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from stages.lib.canon_store import CANON_DIR, CanonFile
from stages.lib.extraction.staging import StagedPage
from stages.lib.extraction.util.dates import canonical_date
from stages.lib.page_classify.codeable_classify import page_type_of

_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class DosProfile:
    default_date: str  # ISO
    span_override_score: float
    non_encounter_page_types: frozenset[str]
    span_page_types: frozenset[str]
    default_page_types: frozenset[str]
    default_page_type_contains: tuple[str, ...]


def _build_profile(data: dict[str, Any]) -> DosProfile:
    datetime.strptime(data["DOS_DEFAULT_DATE"], "%Y-%m-%d")
    return DosProfile(
        default_date=data["DOS_DEFAULT_DATE"],
        span_override_score=float(data["span_override_score"]),
        non_encounter_page_types=frozenset(t.casefold() for t in data["non_encounter_page_types"]),
        span_page_types=frozenset(t.casefold() for t in data["span_page_types"]),
        default_page_types=frozenset(t.casefold() for t in data.get("default_page_types") or []),
        default_page_type_contains=tuple(t.casefold() for t in data.get("default_page_type_contains") or []),
    )


# keyword-canon/dos_canon.json — reloaded when the file changes.
_PROFILE: CanonFile[DosProfile] = CanonFile(CANON_DIR / "dos_canon.json", _build_profile)


def profile() -> DosProfile:
    return _PROFILE.get()


def _uses_default_date(page_type: str, prof: DosProfile) -> bool:
    """Demographics and injection pages keep the default date."""
    if page_type in prof.default_page_types:
        return True
    return any(part in page_type for part in prof.default_page_type_contains)


# --- the extraction's dates --------------------------------------------------


@dataclass
class PageDate:
    dos_from: str  # ISO
    dos_to: str
    confidence: float
    keyword: str
    source: str  # "rules"
    is_pair: bool = False


def iso_to_mdy(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{m}-{d}-{y}"


def iso_date(value: Any) -> str:
    """ISO form of an extracted date, or "" when it is not a real calendar day."""
    iso = canonical_date(str(value or ""))
    if not _ISO.fullmatch(iso):
        return ""
    try:
        datetime.strptime(iso, "%Y-%m-%d")
    except ValueError:
        return ""
    return iso


def page_dates(staged: Optional[StagedPage]) -> list[PageDate]:
    """Every date of service the extraction chose on the page, the primary first.

    An admit date and a discharge date on the same page are one range. Otherwise each chosen
    date is its own.
    """
    if staged is None:
        return []
    rows = staged.selected("dos")
    admit = next((row for row in rows if row.get("tier") == "admit"), None)
    discharge = next((row for row in rows if row.get("tier") == "discharge"), None)
    if admit and discharge:
        spans = [
            (
                iso_date(admit.get("dos_from")),
                iso_date(discharge.get("dos_to")),
                f"{admit.get('key') or ''}+{discharge.get('key') or ''}",
                max(float(admit.get("score") or 0), float(discharge.get("score") or 0)),
                True,
            )
        ]
    else:
        spans = [
            (
                iso_date(row.get("dos_from")),
                iso_date(row.get("dos_to")),
                row.get("key") or "",
                float(row.get("score") or 0),
                False,
            )
            for row in rows
        ]
    return [
        PageDate(
            dos_from=start,
            dos_to=end or start,
            confidence=score,
            keyword=keyword,
            source="rules",
            is_pair=pair,
        )
        for start, end, keyword, score, pair in spans
        if start
    ]


# --- resolve the chart -------------------------------------------------------


def page_type_name(page_text: str, page_number: Any) -> str:
    try:
        number = int(page_number) if page_number is not None else None
    except (TypeError, ValueError):
        number = None
    match = page_type_of(page_text, page_number=number)
    return match.page_type if match is not None else ""


def _row(
    page: dict,
    page_date: Optional[PageDate],
    doc: Optional[PageDate],
    dates: list[PageDate],
    *,
    match_type: str,
    prof: DosProfile,
) -> dict:
    """One output row. MM-DD-YYYY columns plus their ISO twins."""
    is_default = doc is None
    doc_from = doc.dos_from if doc else prof.default_date
    doc_to = doc.dos_to if doc else prof.default_date
    if page_date is not None:
        confidence = page_date.confidence
    elif doc is not None:
        confidence = doc.confidence
    else:
        confidence = 0.0
    return {
        "page_name": page.get("page_name") or str(page.get("page")),
        "page_number": page.get("page"),
        "dos_from": iso_to_mdy(page_date.dos_from) if page_date else "",
        "dos_to": iso_to_mdy(page_date.dos_to) if page_date else "",
        "dos_from_iso": page_date.dos_from if page_date else "",
        "dos_to_iso": page_date.dos_to if page_date else "",
        "doc_dos_from": iso_to_mdy(doc_from),
        "doc_dos_to": iso_to_mdy(doc_to),
        "doc_dos_from_iso": doc_from,
        "doc_dos_to_iso": doc_to,
        "match_type": match_type,
        "keyword": page_date.keyword if page_date else None,
        "confidence": round(confidence, 4),
        "page_source": page_date.source if page_date else "",
        "is_default": is_default,
        # Every date the page carries, for the `dates` array on the DB row.
        "dates": [
            {
                "dos_from": item.dos_from,
                "dos_to": item.dos_to,
                "source_keyword": (item.keyword or "")[:100],
                "confidence": round(item.confidence, 4),
            }
            for item in dates
        ],
    }


def resolve_chart(pages: list[dict]) -> list[dict]:
    """One row per page, in page order.

    Each page is ``{page_name, page, page_text, dates}`` — ``dates`` from :func:`page_dates`.
    Page level (``dos_from`` / ``dos_to``): the date found on that page, or blank. Document
    level (``doc_dos_*``): the encounter the page belongs to.
    """
    prof = profile()
    rows: list[dict] = []
    current: Optional[PageDate] = None
    span_open = False

    for page in pages:
        page_text = page["page_text"]
        page_type = page_type_name(page_text, page.get("page")).casefold() if page_text.strip() else ""
        dates: list[PageDate] = page["dates"]
        found = dates[0] if dates else None

        assigned = found
        if _uses_default_date(page_type, prof):
            # Demographics and injection pages do not take a carried date.
            assigned, doc, match_type = None, None, "default_page"
        elif page_type in prof.span_page_types and found is not None:
            current, span_open = found, True
            assigned, doc, match_type = found, found, "span_start"
        elif page_type in prof.span_page_types:
            # A new progress note with no date of its own ends the previous span.
            current, span_open = None, False
            assigned, doc, match_type = None, None, "no_date_found"
        elif span_open and current is not None and (
            found is None or found.confidence <= prof.span_override_score
        ):
            # A weak date, or none, is replaced by the progress note's date.
            assigned, doc, match_type = current, current, "span"
        elif found is None:
            assigned, doc, match_type = None, None, "no_date_found"
        elif page_type in prof.non_encounter_page_types:
            # A facesheet or med list never replaces an encounter already found.
            assigned = found
            doc = current or found
            match_type = "non_encounter_page"
        elif span_open and found.confidence > prof.span_override_score:
            assigned, doc = found, found
            match_type = "admit_discharge_pair" if found.is_pair else "page_date"
        else:
            current, span_open = found, False
            assigned, doc = found, found
            match_type = "admit_discharge_pair" if found.is_pair else "page_date"

        # The page's own dates; a carried date is the only one a span page has.
        carried = [] if assigned is None else (dates if assigned is found else [assigned])
        rows.append(_row(page, assigned, doc, carried, match_type=match_type, prof=prof))

    return rows
