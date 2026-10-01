"""Printed page labels ("Page 2 of 5", "P.063/131", "1/2") from header/footer bands.

Patterns from Data/examp/page.csv:
  - Page word + number / "of" total / slash total: Page 6, Page #: 2, PAGE: 002 OF 003, PAGE 03/04
  - OCR spellings of the page word: Pago, Fago, Pg., pq, P. / P (P only right before a number)
  - Bare "N of M" and bare "N/M" (the whole line segment must be just that)
  - Bare page number only when alone on its line inside the core header/footer band

Header (28%) and footer (20%) only, of the image or of the text's extent (geometry.edge_band:
screenshots end the page well above the image bottom); mid-page is never read. A page can carry
several labels (fax header + chart header + footer), so every valid label is
kept and selected. Words are read in raw OCR order: sorting by line position
interleaves footers like "Page 1 of 2" with the patient line below it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..util.geometry import (
    FOOTER_CORE_FRAC,
    HEADER_CORE_FRAC,
    Box,
    Word,
    edge_band,
    median_height,
    union_boxes,
)

_OF = r"(?:of|af|0f)"
_NUM = r"\d{1,4}"
_END = r"(?![\d/])"
_TOTAL = rf"(?:\s*{_OF}\s*|\s*/\s*)(?P<t>{_NUM}){_END}"

# Order matters: labeled forms first so "Page 4 of 5" is not re-read as bare "4 of 5".
_PATTERNS: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "label",
        0.95,
        re.compile(
            rf"(?<![A-Za-z])(?P<label>page|pago|fago|pg|pq)\b[\s.:#]*(?P<n>{_NUM})(?:{_TOTAL}|{_END})",
            re.IGNORECASE,
        ),
    ),
    (
        "p_short",
        0.85,
        re.compile(
            rf"(?<![A-Za-z])(?P<label>p)(?:\.\s*(?P<n>{_NUM})(?:\s*/\s*(?P<t>{_NUM}))?{_END}"
            rf"|\s+(?P<n2>{_NUM})\s*/\s*(?P<t2>{_NUM}){_END})",
            re.IGNORECASE,
        ),
    ),
]
# Bare forms must be the whole line segment ("4 of 7" yes, "2 of 3 criteria" no).
_BARE_OF = re.compile(rf"^\s*(?P<n>{_NUM})\s+{_OF}\s+(?P<t>{_NUM})\s*$", re.IGNORECASE)
_BARE_SLASH = re.compile(r"^\s*(?P<n>\d{1,3})\s*/\s*(?P<t>\d{1,3})\s*$")
_BARE_NUMBER = re.compile(r"^\s*(?P<n>\d{1,3})\s*$")
_EDGE_PUNCT = " :;,.|#"


@dataclass
class PageHit:
    key: str
    region: str
    sentence: str
    value: str = ""
    page_no: str = ""
    page_total: str = ""
    score: float = 0.0
    accepted: bool = False
    selected: bool = False
    source: str = ""
    box: Box | None = None


@dataclass
class _Segment:
    words: list[Word]
    text: str
    spans: list[tuple[int, int]]


def _valid(page_no: str, total: str) -> bool:
    try:
        number = int(page_no)
    except ValueError:
        return False
    if number < 1:
        return False
    if not total:
        return True
    count = int(total)
    return 1 <= count and number <= count


def _band_words(words: list[Word], page_h: float) -> dict[str, list[Word]]:
    bands: dict[str, list[Word]] = {"header": [], "footer": []}
    if page_h <= 0:
        return bands
    text = union_boxes([word.box for word in words])
    for word in sorted(words, key=lambda item: item.index):
        band = edge_band(word.box, page_h, text)
        if band:
            bands[band].append(word)
    return bands


def _segments(words: list[Word], line_h: float) -> list[_Segment]:
    """Runs of words that are consecutive in OCR order, on one line, and close together."""
    runs: list[list[Word]] = []
    for word in words:
        if runs:
            prev = runs[-1][-1]
            same_line = abs(word.box.cy - prev.box.cy) <= 0.6 * line_h
            moving_right = word.box.left >= prev.box.left
            close = word.box.left - prev.box.right <= 4.0 * line_h
            if same_line and moving_right and close:
                runs[-1].append(word)
                continue
        runs.append([word])
    segments: list[_Segment] = []
    for run in runs:
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for word in run:
            if parts:
                cursor += 1
            spans.append((cursor, cursor + len(word.content)))
            parts.append(word.content)
            cursor += len(word.content)
        segments.append(_Segment(words=run, text=" ".join(parts), spans=spans))
    return segments


def _span_box(segment: _Segment, start: int, end: int) -> Box | None:
    boxes = [
        word.box
        for word, (left, right) in zip(segment.words, segment.spans)
        if left < end and right > start
    ]
    return union_boxes(boxes)


def _alone_on_line(word: Word, band: list[Word], line_h: float) -> bool:
    return not any(
        other.index != word.index and abs(other.box.cy - word.box.cy) <= 0.6 * line_h
        for other in band
    )


def _in_core(box: Box, page_h: float, region: str) -> bool:
    # On the image only: a bare number is the weakest label.
    frac = box.cy / page_h
    if region == "header":
        return frac <= HEADER_CORE_FRAC
    return frac >= 1.0 - FOOTER_CORE_FRAC


def _hit(region: str, segment: _Segment, value: str, page_no: str, total: str,
         score: float, source: str, box: Box | None) -> PageHit:
    return PageHit(
        key="No Key Header" if region == "header" else "No Key Footer",
        region=f"keyless_{region}",
        sentence=segment.text,
        value=value.strip(_EDGE_PUNCT),
        page_no=str(int(page_no)),
        page_total=str(int(total)) if total else "",
        score=score,
        accepted=True,
        selected=True,
        source=source,
        box=box,
    )


def _segment_hits(segment: _Segment, region: str) -> list[PageHit]:
    hits: list[PageHit] = []
    taken: list[tuple[int, int]] = []
    for source, score, pattern in _PATTERNS:
        for match in pattern.finditer(segment.text):
            start, end = match.span()
            if any(start < right and end > left for left, right in taken):
                continue
            groups = match.groupdict()
            page_no = groups.get("n") or groups.get("n2") or ""
            total = groups.get("t") or groups.get("t2") or ""
            if not _valid(page_no, total):
                continue
            taken.append((start, end))
            hits.append(
                _hit(region, segment, match.group(0), page_no, total, score, source,
                     _span_box(segment, start, end))
            )
    if hits:
        return hits
    for source, score, pattern in (("bare_of", 0.8, _BARE_OF), ("bare_slash", 0.75, _BARE_SLASH)):
        bare = pattern.match(segment.text.strip(_EDGE_PUNCT))
        if bare and _valid(bare.group("n"), bare.group("t")):
            hits.append(
                _hit(region, segment, segment.text, bare.group("n"), bare.group("t"), score,
                     source, union_boxes([word.box for word in segment.words]))
            )
            break
    return hits


def extract_page(words: list[Word], page_h: float) -> list[PageHit]:
    """Every page label in the header and footer bands (duplicates within a band merged)."""
    line_h = median_height(words)
    rows: list[PageHit] = []
    for region, band in _band_words(words, page_h).items():
        seen: set[str] = set()
        for segment in _segments(band, line_h):
            found = _segment_hits(segment, region)
            if not found and len(segment.words) == 1:
                word = segment.words[0]
                bare = _BARE_NUMBER.match(word.content.strip(_EDGE_PUNCT))
                if (
                    bare
                    and _valid(bare.group("n"), "")
                    and _in_core(word.box, page_h, region)
                    and _alone_on_line(word, band, line_h)
                ):
                    found = [_hit(region, segment, bare.group("n"), bare.group("n"), "", 0.6,
                                  "bare_number", word.box)]
            for hit in found:
                marker = hit.value.casefold()
                if marker in seen:
                    continue
                seen.add(marker)
                rows.append(hit)
    return rows
