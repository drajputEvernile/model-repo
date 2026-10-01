"""Staging: what the extraction found, kept until the chart is done.

The extraction stage runs once, right after OCR, and finds every field the later stages
need — member name / DOB / ID, DOS, provider, e-signature, printed page number, headings.
Those stages read it from here when their turn comes, so none of them reads a page's OCR
for these fields again. Nothing in staging is a result: member verification, DOS and the
rest write their own rows, and the orchestrator drops the staging when the chart completes.

One JSON file per chart at ``<chart>/staging/extraction.json``::

    {"version": 1, "chart_name": …, "model_version": "v002", "extracted_at": …,
     "pages": {"<page name>": {"page_number": 1, "width": …, "height": …,
                               "fields": {"name": [<hit>, …], "dos": […], …}}}}

A hit is the extractor's own row (key, region, sentence, value, score, accepted, selected,
source, …) with boxes as ``[left, top, right, bottom]``. ``selected`` is the value the
extraction chose; ``accepted`` is every candidate it stands behind.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from config import staging_dir

from .util.geometry import Box

if TYPE_CHECKING:
    from .pipeline import DocumentResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FILE_NAME = "extraction.json"

# Row attributes that are not data: the key a row was read from and the heading's own words.
_NOT_DATA = frozenset({"key_hit", "words"})


def staging_path(chart_name: str) -> Path:
    return staging_dir(chart_name) / FILE_NAME


def _plain(value: Any) -> Any:
    if isinstance(value, Box):
        return [round(value.left, 2), round(value.top, 2), round(value.right, 2), round(value.bottom, 2)]
    if isinstance(value, float):
        return round(value, 4)
    return value


def hit_record(hit: Any) -> dict[str, Any]:
    """One extractor row as plain JSON."""
    return {
        field.name: _plain(getattr(hit, field.name))
        for field in dataclasses.fields(hit)
        if field.name not in _NOT_DATA
    }


def build(chart_name: str, result: "DocumentResult", model_version: str) -> dict[str, Any]:
    pages: dict[str, Any] = {}
    for page in result.pages:
        pages[str(page.page.get("fileName") or "")] = {
            "page_number": page.page.get("pageNumber"),
            "width": page.page_w,
            "height": page.page_h,
            "fields": {
                field_id: [hit_record(hit) for hit in hits]
                for field_id, hits in page.rows.items()
            },
        }
    return {
        "version": SCHEMA_VERSION,
        "chart_name": chart_name,
        "model_version": model_version,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }


def write(chart_name: str, payload: dict[str, Any]) -> Path:
    """Replace the chart's staging (write to a temp name, then move, so a reader never sees half)."""
    path = staging_path(chart_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read(chart_name: str) -> Optional["Staged"]:
    """The chart's staging, or None when the extraction has not run (or was dropped)."""
    path = staging_path(chart_name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable extraction staging %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
        logger.warning("Extraction staging %s is from another version; ignoring it", path)
        return None
    return Staged(data)


def drop(chart_name: str) -> bool:
    """Delete the chart's staging; True when there was any."""
    folder = staging_dir(chart_name)
    if not folder.is_dir():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


class StagedPage:
    """One page's staged rows."""

    def __init__(self, name: str, data: dict[str, Any]):
        self.name = name
        self.page_number = data.get("page_number")
        self.width = data.get("width")
        self.height = data.get("height")
        self.fields: dict[str, list[dict[str, Any]]] = data.get("fields") or {}

    def rows(self, field_id: str) -> list[dict[str, Any]]:
        return self.fields.get(field_id) or []

    def accepted(self, field_id: str) -> list[dict[str, Any]]:
        """Every candidate the extraction stands behind, in page order."""
        return [row for row in self.rows(field_id) if row.get("accepted")]

    def selected(self, field_id: str) -> list[dict[str, Any]]:
        """The candidates it chose: one for a single-value field, several for IDs and headings.

        A heading row has no separate choice (it is a heading or it is not), so every accepted
        heading counts.
        """
        return [row for row in self.accepted(field_id) if row.get("selected", True)]


class Staged:
    """A chart's staged extraction."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.model_version: str = data.get("model_version") or ""
        self._pages = {name: StagedPage(name, page) for name, page in (data.get("pages") or {}).items()}

    def __contains__(self, page_name: str) -> bool:
        return page_name in self._pages

    def __iter__(self) -> Iterator[StagedPage]:
        return iter(self._pages.values())

    def __len__(self) -> int:
        return len(self._pages)

    def page(self, page_name: str) -> Optional[StagedPage]:
        return self._pages.get(page_name)
