"""Stage: key/value extraction (runs right after OCR).

Reads the word boxes Final2 (Azure) stored for each page, runs every extractor over the
chart in one pass (``engine.extract_chart``) and stages what they found
(``staging.py``) for the stages that use it: member verification, date of service and page
sequencing. This replaces the separate member, DOS, page-number and section-header
extraction those stages each did for themselves.

The one thing it writes outside staging is the headings: the pages' ``section_headers``,
in the OCR JSON and ``ocr_results`` rows where the section-header stage used to put them,
so the review UI draws them exactly as before.

A page with no word boxes (Azure did not read it — a high-quality printed page skips the
billed call, so only Final1 text exists) cannot be extracted. It is skipped as
``no_word_boxes`` and left out of staging; the stages that read staging treat it as a page
the extraction found nothing on.

Staging is chart-wide on purpose: a value's features include how often it repeats across the
document, so the pages are extracted together, not one by one.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from config import page_image_path
from db import connect, get_ocr_texts, upsert_ocr_result
from db.paths import ocr_dir, write_final1_json, write_final2_json
from stages._support import (
    mark_completed,
    mark_processing,
    mark_skipped,
    stage_run,
)

from . import engine, staging
from .heading.extract import heading_fields
from .ocr_input import extraction_page

logger = logging.getLogger(__name__)

STAGE = "kv_extract"

# Heading detector field ids → the section_headers level the review UI draws (1 = heading).
HEADING_LEVEL = {"Heading": 1, "Subheading": 2}


def _page_json(raw: Optional[str]) -> Optional[dict[str, Any]]:
    """A stored OCR row as a dict, or None for plain text (no boxes in it)."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _norm_box(box: list[float], page_w: float, page_h: float) -> Optional[dict[str, float]]:
    """CSS fractions (0–1) of a top-left pixel box, the ``norm`` the section_headers items carry."""
    if page_w <= 0 or page_h <= 0:
        return None

    def clip(value: float) -> float:
        return round(max(0.0, min(1.0, float(value))), 5)

    left, top, right, bottom = box
    return {
        "left": clip(left / page_w),
        "top": clip(top / page_h),
        "width": clip((right - left) / page_w),
        "height": clip((bottom - top) / page_h),
    }


def section_headers_of(staged_page: staging.StagedPage) -> list[dict[str, Any]]:
    """The page's headings in the shape the OCR JSON and the review UI read."""
    page_w = float(staged_page.width or 0)
    page_h = float(staged_page.height or 0)
    headers: list[dict[str, Any]] = []
    for field_id in heading_fields():
        for row in staged_page.selected(field_id):
            box = row.get("box")
            headers.append(
                {
                    "text": row.get("text") or "",
                    "level": HEADING_LEVEL.get(row.get("level") or "", 2),
                    "bbox": list(box) if box else [],
                    "page_width": page_w or None,
                    "page_height": page_h or None,
                    "coord_origin": "TOPLEFT" if box else None,
                    "norm": _norm_box(box, page_w, page_h) if box else None,
                }
            )
    return headers


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _rewrite_ocr_json(
    chart_name: str, kind: str, headers_by_page: dict[str, list[dict[str, Any]]]
) -> bool:
    """Put each page's headings into ``ocr/<chart>_final1.json`` or ``_final2.json``."""
    path = ocr_dir(chart_name) / f"{chart_name}_{kind}.json"
    doc = _load_json(path)
    if doc is None:
        return False
    pages = []
    for page in doc.get("pages") or []:
        if isinstance(page, dict) and str(page.get("fileName") or "") in headers_by_page:
            page = {**page, "section_headers": headers_by_page[str(page["fileName"])]}
        pages.append(page)
    if kind == "final1":
        write_final1_json(chart_name, pages, model=str(doc.get("model") or "docling+rapidocr"))
    else:
        write_final2_json(chart_name, pages)
    return True


def _store_headings(
    conn: Any, chart_id: int, page_ids: dict[str, int], headers_by_page: dict[str, list[dict[str, Any]]]
) -> None:
    """Merge the headings into the stored ``ocr_results`` JSON of every engine that read the page."""
    for ocr_type in ("azuredocintel", "docling"):
        stored = get_ocr_texts(conn, chart_id, ocr_type)
        for name, headers in headers_by_page.items():
            page_json = _page_json(stored.get(page_ids[name]))
            if page_json is None:
                continue
            page_json["section_headers"] = headers
            upsert_ocr_result(
                conn,
                chart_id=chart_id,
                page_id=page_ids[name],
                ocr_type=ocr_type,
                raw_text=json.dumps(page_json, default=str),
            )


def run(chart_id: int, *, force: bool = False) -> dict[str, Any]:
    with stage_run(chart_id, STAGE, force=force) as ctx:
        if not ctx.todo:
            return {"chart_id": chart_id, "pages_done": 0, "skipped": ctx.skipped, "reason": "nothing_to_do"}

        with connect() as conn:
            final2 = get_ocr_texts(conn, chart_id, "azuredocintel")

        # The whole chart is extracted together, whichever pages are still to do.
        pages: list[dict[str, Any]] = []
        no_boxes: list[int] = []
        for page in ctx.pages:
            extractable = extraction_page(
                _page_json(final2.get(page["id"])),
                page_name=page["page_name"],
                page_number=page.get("page_number"),
                image_path=page_image_path(ctx.chart_name, page["page_name"]),
            )
            if extractable is None:
                no_boxes.append(page["id"])
            else:
                pages.append(extractable)

        with connect() as conn:
            mark_skipped(conn, ctx, no_boxes, "no_word_boxes")
            for page in ctx.pages_todo:
                mark_processing(conn, ctx, page["id"])
        if no_boxes:
            logger.warning(
                "chart %s: %d of %d page(s) have no word boxes (no Final2 read); "
                "nothing is extracted from them",
                ctx.chart_name, len(no_boxes), len(ctx.pages),
            )

        version = engine.model_version()
        result = engine.extract_chart(ctx.chart_name, pages) if pages else None
        payload = staging.build(ctx.chart_name, result, version) if result else {
            "version": staging.SCHEMA_VERSION,
            "chart_name": ctx.chart_name,
            "model_version": version,
            "pages": {},
        }
        path = staging.write(ctx.chart_name, payload)
        staged = staging.read(ctx.chart_name)

        # Headings go where the section-header stage wrote them.
        page_ids = {page["page_name"]: page["id"] for page in ctx.pages}
        headers_by_page = {
            staged_page.name: section_headers_of(staged_page)
            for staged_page in staged or []
            if staged_page.name in page_ids
        }
        with connect() as conn:
            _store_headings(conn, chart_id, page_ids, headers_by_page)
            for page in ctx.pages_todo:
                mark_completed(conn, ctx, page["id"])
        for kind in ("final1", "final2"):
            _rewrite_ocr_json(ctx.chart_name, kind, headers_by_page)

        headings = sum(len(headers) for headers in headers_by_page.values())
        logger.info(
            "Key/value extraction: %d page(s) staged, %d heading(s) (model %s)",
            len(headers_by_page), headings, version,
        )
        return {
            "chart_id": chart_id,
            "staging": str(path),
            "model_version": version,
            "pages_done": ctx.done,
            "pages_extracted": len(pages),
            "pages_no_word_boxes": len(no_boxes),
            "headings": headings,
            "skipped": ctx.skipped,
        }


def ensure_staging(chart_id: int, chart_name: str) -> staging.Staged:
    """The chart's staging, running the extraction first when there is none.

    Staging is dropped when a chart completes, so re-running one later stage on its own
    (``only=["member_verify"]``) finds nothing. The extraction needs no OCR — it reads what
    OCR stored — so it is simply run again, rather than asking the caller to.
    """
    staged = staging.read(chart_name)
    if staged is None:
        logger.info("chart %s: no extraction staging; running the extraction first", chart_name)
        run(chart_id, force=True)
        staged = staging.read(chart_name)
    if staged is None:
        raise RuntimeError(f"chart {chart_name}: the extraction produced no staging")
    return staged
