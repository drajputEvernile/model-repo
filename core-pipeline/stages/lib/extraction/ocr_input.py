"""What the extractors read from a page's OCR.

Every extractor works on word boxes: a key and its value are found by where they sit, not
by the order of the text. Azure Document Intelligence (final2) stores them per page in
``pagesMeta[].words`` as ``{content, polygon}`` with the page's pixel size; the standalone
OCR JSON the module was built on keeps the same words and size at the top of the page.
Both are read here.

Docling / RapidOCR (final1) keeps text only, so a page that went no further than final1
(a high-quality printed page skips Azure) has no boxes to read and is left out — see
``extraction_page``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _usable(word: Any) -> bool:
    """A word with text and a four-corner polygon (the shape util/geometry.py reads)."""
    if not isinstance(word, dict) or not str(word.get("content") or "").strip():
        return False
    polygon = word.get("polygon")
    return isinstance(polygon, (list, tuple)) and len(polygon) >= 8


def _word_source(ocr_page: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The dict holding this page's ``words`` / ``width`` / ``height``, or None."""
    if any(_usable(word) for word in ocr_page.get("words") or []):
        return ocr_page
    for meta in ocr_page.get("pagesMeta") or ocr_page.get("pages_meta") or []:
        if isinstance(meta, dict) and any(_usable(word) for word in meta.get("words") or []):
            return meta
    return None


def has_word_boxes(ocr_page: Optional[dict[str, Any]]) -> bool:
    return isinstance(ocr_page, dict) and _word_source(ocr_page) is not None


def extraction_page(
    ocr_page: Optional[dict[str, Any]],
    *,
    page_name: str,
    page_number: Optional[int],
    image_path: Path,
) -> Optional[dict[str, Any]]:
    """One page in the shape the extractors take, or None when it has no word boxes.

    ``image_path`` is the image the OCR read (the corrected page when rotation changed it):
    the heading detector looks at the same pixels the word boxes were measured on.
    """
    if not isinstance(ocr_page, dict):
        return None
    source = _word_source(ocr_page)
    if source is None:
        return None
    return {
        "pageNumber": page_number,
        "fileName": page_name,
        "width": source.get("width"),
        "height": source.get("height"),
        "words": source["words"],
        "imagePath": str(image_path),
    }
