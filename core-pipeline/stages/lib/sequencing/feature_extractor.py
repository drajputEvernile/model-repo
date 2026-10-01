"""Build PageFeatures from OCR text (imaging-pipeline adapter)."""
from __future__ import annotations

import re
import string
from typing import Any

from .engine.types import ExplicitMarker, PageFeatures
from .page_heuristics import is_near_blank

EDGE_LINES = 15
_MAX_EDGE_CHARS = 1200


def features_from_text(
    *,
    page_id: str | int,
    page_number: int,
    text: str,
    is_classified: bool = False,
    text_fingerprint: str | None = None,
    extracted_marker: ExplicitMarker | None = None,
    marker_extracted: bool = False,
) -> PageFeatures:
    full_text = text or ""
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    if not lines and full_text.strip():
        lines = [full_text.strip()]

    header_raw = " ".join(lines[:5])
    footer_raw = " ".join(lines[-5:]) if lines else ""
    top_text = _cap_edge_text(lines[:EDGE_LINES])
    bottom_text = _cap_edge_text(lines[-EDGE_LINES:]) if lines else ""
    word_count = len(re.findall(r"\b\w+\b", full_text))
    junk = bool(is_classified)
    if not junk and is_near_blank(full_text, word_count=word_count):
        junk = True

    return PageFeatures(
        page_id=str(page_id),
        original_page_number=int(page_number or 0),
        is_classified=junk,
        header_text=_normalize(header_raw),
        footer_text=_normalize(footer_raw),
        header_raw=header_raw,
        footer_raw=footer_raw,
        top_lines_text=top_text,
        bottom_lines_text=bottom_text,
        full_text=full_text,
        word_count=word_count,
        has_structured=False,
        identity_text="",
        text_fingerprint=text_fingerprint,
        extracted_marker=extracted_marker,
        marker_extracted=marker_extracted,
    )


def features_from_pages(pages: list[dict[str, Any]]) -> list[PageFeatures]:
    """Each dict: page_id, page_number, text, is_classified (optional).

    Optional ``marker_extracted`` / ``page_marker``: the printed page number the extraction
    found (an ExplicitMarker, or None when it found none on a page it read).
    """
    out: list[PageFeatures] = []
    for page in pages:
        out.append(
            features_from_text(
                page_id=page["page_id"],
                page_number=int(page.get("page_number") or 0),
                text=str(page.get("text") or ""),
                is_classified=bool(page.get("is_classified")),
                text_fingerprint=page.get("text_fingerprint"),
                extracted_marker=page.get("page_marker"),
                marker_extracted=bool(page.get("marker_extracted")),
            )
        )
    return out


def _cap_edge_text(lines: list[str]) -> str:
    text = "\n".join(lines)
    if len(text) <= _MAX_EDGE_CHARS:
        return text
    return text[:_MAX_EDGE_CHARS]


def _normalize(value: str) -> str:
    cleaned = "".join(
        ch.lower() if ch not in string.punctuation else " " for ch in (value or "")
    )
    return re.sub(r"\s+", " ", cleaned).strip()
