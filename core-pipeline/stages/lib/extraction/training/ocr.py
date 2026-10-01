"""OCR pages by (RecordId, FileName), with words indexed the way the extractors index them."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..util import config
from ..util.geometry import Word, group_lines, page_size, words_from_page


def _record_json(record_id: str) -> Path | None:
    folder = Path(config.OCR_Input) / record_id
    if not folder.is_dir():
        return None
    jsons = [path for path in folder.glob("*.json") if path.is_file()]
    return max(jsons, key=lambda path: path.stat().st_size) if jsons else None


@lru_cache(maxsize=8)
def _pages(path: str, mtime: float) -> dict[str, dict[str, Any]]:
    del mtime
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, list):
        return {}
    return {
        str(page.get("fileName") or "").casefold(): page
        for page in pages
        if isinstance(page, dict)
    }


def load_page(record_id: str, file_name: str) -> dict[str, Any] | None:
    path = _record_json(record_id)
    if path is None:
        return None
    return _pages(str(path), path.stat().st_mtime).get(file_name.casefold())


def page_words(record_id: str, file_name: str) -> tuple[list[Word], float, float]:
    page = load_page(record_id, file_name)
    if page is None:
        return [], 1.0, 1.0
    words = words_from_page(page)
    page_w, page_h = page_size(page, words)
    return words, page_w, page_h


def page_lines(record_id: str, file_name: str) -> dict[str, Any]:
    """Words grouped into reading-order lines, boxes normalized to 0..1 of the page."""
    page = load_page(record_id, file_name)
    if page is None:
        return {"available": False, "lines": [], "text": ""}
    words = words_from_page(page)
    page_w, page_h = page_size(page, words)
    lines = [
        [
            {
                "i": word.index,
                "t": word.content,
                "b": [
                    round(word.box.left / page_w, 5),
                    round(word.box.top / page_h, 5),
                    round(word.box.right / page_w, 5),
                    round(word.box.bottom / page_h, 5),
                ],
            }
            for word in line
        ]
        for line in group_lines(words)
    ] if words else []
    text = str(page.get("content") or "").strip() or "\n".join(
        " ".join(word["t"] for word in line) for line in lines
    )
    return {"available": True, "lines": lines, "text": text}
