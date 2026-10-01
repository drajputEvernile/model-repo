"""Explicit page-marker detection."""
from __future__ import annotations

import re
from collections import Counter

from .types import ExplicitMarker, PageFeatures

MARKER_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("page_x_of_y", re.compile(r"page\s+(\d{1,4})\s+of\s+(\d{1,4})\b", re.IGNORECASE)),
    ("page_x_slash_y", re.compile(r"page\s+(\d{1,6})\s*/\s*0*(\d{1,4})(?!\s*/|\d)", re.IGNORECASE)),
    ("pg_x_of_y", re.compile(r"pg\.?\s*(\d{1,4})\s+of\s+(\d{1,4})\b", re.IGNORECASE)),
    ("pg_x", re.compile(r"pg\.?\s+(\d{1,4})\b", re.IGNORECASE)),
    ("p_dot_x_of_y", re.compile(r"\bp\.?\s*(\d{1,4})\s+of\s+(\d{1,4})\b", re.IGNORECASE)),
    ("x_of_y", re.compile(r"\b(\d{1,4})\s+of\s+(\d{1,4})\b", re.IGNORECASE)),
    ("x_slash_y", re.compile(r"(?<![\d/])(\d{1,6})\s*/\s*0*(\d{1,4})(?!\s*/|\d)", re.IGNORECASE)),
    ("page_x", re.compile(r"page\s+(\d{1,4})\b", re.IGNORECASE)),
    ("dash_x_dash", re.compile(r"(?<!\d)-\s*(\d{1,4})\s*-(?!\d)", re.IGNORECASE)),
)

FOOTER_ONLY_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("footer_plain_x", re.compile(r"(?m)^\s*(\d{1,4})\s*$")),
)

_BODY_PATTERNS = frozenset(("page_x_of_y", "page_x_slash_y", "pg_x_of_y", "p_dot_x_of_y"))

# Patterns that literally say "page": trustworthy even as a lone hit. Bare
# number patterns ("1/25", "1 of 3", footer digits) are date/dose noise unless
# they form a consistent series across pages.
_EXPLICIT_PATTERNS = frozenset(
    ("page_x_of_y", "page_x_slash_y", "pg_x_of_y", "p_dot_x_of_y", "pg_x", "page_x")
)


def detect_markers(
    features: list[PageFeatures],
    *,
    doc_page_count: int | None = None,
) -> dict[str, ExplicitMarker]:
    """Pick one marker per page, preferring the document's dominant pagination.

    Markers are grouped into families by their "of N" total. A family covering
    several pages is the record's real pagination — trusted even when N is far
    larger than the upload (a partial upload of a bigger record still orders by
    those numbers). Lone markers are only trusted when their pattern literally
    says "page" and the total is plausible for this upload.
    """
    candidates_by_page: dict[str, list[ExplicitMarker]] = {
        f.page_id: _all_candidates(f) for f in features
    }

    families: dict[int, dict[str, ExplicitMarker]] = {}
    for pid, cands in candidates_by_page.items():
        for m in cands:
            if m.total_pages is None:
                continue
            fam = families.setdefault(m.total_pages, {})
            cur = fam.get(pid)
            if cur is None or _explicitness(m) > _explicitness(cur):
                fam[pid] = m

    best_total: int | None = None
    best_score: tuple = ()
    for total, members in families.items():
        if len(members) < 2:
            continue
        nums = [m.page_num for m in members.values()]
        distinct = len(set(nums))
        if distinct < 2:
            continue
        explicit = sum(1 for m in members.values() if m.pattern in _EXPLICIT_PATTERNS)
        score = (len(members), explicit, distinct)
        if score > best_score:
            best_score = score
            best_total = total

    markers: dict[str, ExplicitMarker] = {}
    if best_total is not None:
        markers = dict(families[best_total])

    # Pages outside the dominant family may still carry a lone explicit
    # "Page x of y" whose total fits this upload.
    n = doc_page_count or 0
    for pid, cands in candidates_by_page.items():
        if pid in markers:
            continue
        for m in cands:
            if m.pattern not in _EXPLICIT_PATTERNS or m.total_pages is None:
                continue
            if best_total is not None and m.total_pages != best_total:
                continue
            if n and m.total_pages > max(n * 2, n + 20):
                continue
            markers[pid] = m
            break
    return markers


def _explicitness(m: ExplicitMarker) -> tuple:
    return (1 if m.pattern in _EXPLICIT_PATTERNS else 0, m.confidence)


def _all_candidates(feature: PageFeatures) -> list[ExplicitMarker]:
    if feature.marker_extracted:
        if feature.extracted_marker is None:
            return []
        feature.extracted_marker.source = "extracted"
        return [feature.extracted_marker]
    candidates: list[ExplicitMarker] = []
    footer_marker = detect_footer_marker(feature.footer_raw)
    if footer_marker:
        footer_marker.source = "footer"
        candidates.append(footer_marker)
    header_marker = detect_marker(feature.header_raw)
    if header_marker:
        header_marker.source = "header"
        candidates.append(header_marker)
    body_marker = _detect_body_marker(feature.full_text)
    if body_marker:
        body_marker.source = "body"
        candidates.append(body_marker)
    return candidates


def _best_marker(feature: PageFeatures, *, doc_page_count: int | None = None) -> ExplicitMarker | None:
    candidates: list[ExplicitMarker] = []

    footer_marker = detect_footer_marker(feature.footer_raw)
    header_marker = detect_marker(feature.header_raw)
    if footer_marker:
        candidates.append(footer_marker)
    if header_marker:
        candidates.append(header_marker)

    body_marker = _detect_body_marker(feature.full_text)
    if body_marker:
        candidates.append(body_marker)

    if doc_page_count:
        plausible = [
            m for m in candidates
            if m.total_pages is None or (
                m.total_pages <= max(doc_page_count * 2, doc_page_count + 20)
                and not (m.total_pages > 80 and doc_page_count < 50)
            )
        ]
        if plausible:
            candidates = plausible

    if not candidates:
        return None

    def _rank(m: ExplicitMarker) -> tuple:
        tp = m.total_pages
        absolute = tp is not None and tp == doc_page_count
        plausible_total = tp is not None and doc_page_count and tp <= doc_page_count + 5
        return (
            0 if absolute else 1 if plausible_total else 2,
            0 if m.total_pages is not None else 1,
            -m.confidence,
        )

    best = min(candidates, key=_rank)
    if footer_marker and best is footer_marker:
        best.source = "footer"
    elif header_marker and best is header_marker:
        best.source = "header"
    else:
        best.source = "body"
    return best


def detect_marker(text: str) -> ExplicitMarker | None:
    for pattern_name, pattern in MARKER_PATTERNS:
        for match in pattern.finditer(text or ""):
            total_pages = int(match.group(2)) if len(match.groups()) > 1 else None
            raw_page_num = int(match.group(1))

            if pattern_name == "x_slash_y" and total_pages is not None and raw_page_num > total_pages:
                if total_pages > 9 and raw_page_num > 50:
                    continue

            page_num = _normalize_page_num(raw_page_num, total_pages)
            if page_num <= 0 or (total_pages is not None and (total_pages <= 0 or page_num > total_pages)):
                continue
            return ExplicitMarker(
                page_num=page_num,
                total_pages=total_pages,
                pattern=pattern_name,
                confidence=_marker_confidence(pattern_name, total_pages),
            )
    return None


def detect_footer_marker(text: str) -> ExplicitMarker | None:
    marker = detect_marker(text)
    if marker:
        return marker
    for pattern_name, pattern in FOOTER_ONLY_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        page_num = int(match.group(1))
        if page_num <= 0:
            continue
        return ExplicitMarker(page_num=page_num, total_pages=None, pattern=pattern_name, confidence=0.78)
    return None


def _detect_body_marker(text: str) -> ExplicitMarker | None:
    for pattern_name, pattern in MARKER_PATTERNS:
        if pattern_name not in _BODY_PATTERNS:
            continue
        for match in pattern.finditer(text or ""):
            total_pages = int(match.group(2)) if len(match.groups()) > 1 else None
            page_num = _normalize_page_num(int(match.group(1)), total_pages)
            if page_num <= 0 or (total_pages is not None and (total_pages <= 0 or page_num > total_pages)):
                continue
            return ExplicitMarker(
                page_num=page_num,
                total_pages=total_pages,
                pattern=pattern_name,
                confidence=_marker_confidence(pattern_name, total_pages),
            )
    return None


def find_marker_conflicts(markers: dict[str, ExplicitMarker]) -> set[str]:
    complete = {pid: m for pid, m in markers.items() if m.total_pages is not None}
    conflicts: set[str] = set()
    if len({m.total_pages for m in complete.values()}) > 1:
        conflicts.update(complete.keys())

    counts = Counter(m.page_num for m in complete.values())
    dup_numbers = {num for num, count in counts.items() if count > 1}
    if dup_numbers:
        conflicts.update(pid for pid, m in complete.items() if m.page_num in dup_numbers)
    return conflicts


def _normalize_page_num(raw_page_num: int, total_pages: int | None) -> int:
    if total_pages is None or raw_page_num <= total_pages:
        return raw_page_num
    total_width = len(str(total_pages))
    suffix = raw_page_num % (10 ** total_width)
    if 1 <= suffix <= total_pages:
        return suffix
    return raw_page_num


def _marker_confidence(pattern_name: str, total_pages: int | None) -> float:
    if total_pages is None:
        return 0.85
    if pattern_name.startswith(("page_", "p_dot_", "pg_")):
        return 1.0
    return 0.92
