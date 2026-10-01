"""Stage: date-of-service from the extraction.

The dates are not found here: the key/value extraction stage (``stages/lib/extraction``)
read every page once, right after OCR, and staged each page's date of service. This stage
resolves them across the chart (``resolve.py``) — which encounter a page belongs to, what
a page with no date of its own inherits — and writes one DB row and one CSV row per page.

Every row records ``extraction_method='rules'``: the dates come from the extraction's rules
and the trained model that picks among them, with no LLM pass.
"""
from __future__ import annotations

import logging
from typing import Any

from db import (
    connect,
    get_blank_junk_flags,
    get_ocr_texts,
    get_quality_map,
    upsert_dos,
)
from db.paths import imaging_csv, write_csv
from stages._support import (
    BJ_EXCLUDE,
    best_page_text,
    mark_completed,
    mark_skipped,
    stage_run,
)
from stages.lib.dos.resolve import page_dates, resolve_chart
from stages.lib.extraction.stage import ensure_staging

logger = logging.getLogger(__name__)

STAGE = "dos_extract"

METHOD = "rules"

DOS_COLS = [
    "chart_name",
    "page_name",
    "page_number",
    "dos_from",
    "dos_to",
    "dos_from_iso",
    "dos_to_iso",
    "doc_dos_from",
    "doc_dos_to",
    "doc_dos_from_iso",
    "doc_dos_to_iso",
    "match_type",
    "keyword",
    "confidence",
    "extraction_method",
]


def _page_texts(
    conn: Any, chart_id: int, pages: list[dict[str, Any]], eligible: set[int]
) -> dict[int, str]:
    """Each eligible page's text, which decides its page type (progress note, face sheet, …).

    Text preference: final2 → final1 → prelim. Prelim is never used for
    handwritten / uncertain / mixed / low-quality pages.
    """
    prelim = get_ocr_texts(conn, chart_id, "tesseract")
    final1 = get_ocr_texts(conn, chart_id, "docling")
    final2 = get_ocr_texts(conn, chart_id, "azuredocintel")
    quality = get_quality_map(conn, chart_id)
    return {
        page["id"]: best_page_text(
            final2=final2.get(page["id"]),
            final1=final1.get(page["id"]),
            prelim=prelim.get(page["id"]),
            quality_row=quality.get(page["id"]),
        )
        for page in pages
        if page["id"] in eligible
    }


def run(chart_id: int, *, force: bool = False) -> dict[str, Any]:
    with stage_run(chart_id, STAGE, force=force) as ctx:
        with connect() as conn:
            bj = get_blank_junk_flags(conn, chart_id, final_only=True)
            drop = [
                pid for pid in ctx.todo
                if bj.get(pid, "not_blank_junk") in BJ_EXCLUDE
            ]
            mark_skipped(conn, ctx, drop, "blank_junk")
            eligible = set(ctx.todo)
            texts = _page_texts(conn, chart_id, ctx.pages, eligible)

        staged = ensure_staging(chart_id, ctx.chart_name)

        # Every page is resolved, in order: a progress note's span runs through the pages
        # after it, whether or not this stage writes them. Pages it does not read carry no
        # date and no text.
        resolved = resolve_chart(
            [
                {
                    "page_name": page["page_name"],
                    "page": page.get("page_number"),
                    "page_text": texts.get(page["id"], ""),
                    "dates": page_dates(staged.page(page["page_name"])) if page["id"] in eligible else [],
                }
                for page in ctx.pages
            ]
        )

        by_name = {p["page_name"]: p for p in ctx.pages}
        csv_rows: list[dict[str, Any]] = []
        written: set[int] = set()

        with connect() as conn:
            for hit in resolved:
                page = by_name.get(str(hit.get("page_name") or ""))
                if page is None or page["id"] not in eligible or page["id"] in written:
                    continue
                page_id = page["id"]
                written.add(page_id)
                upsert_dos(
                    conn,
                    chart_id=chart_id,
                    page_id=page_id,
                    date_of_service_from=hit.get("dos_from_iso") or None,
                    date_of_service_to=hit.get("dos_to_iso") or None,
                    date_of_service_from_doclevel=hit.get("doc_dos_from_iso") or None,
                    date_of_service_to_doclevel=hit.get("doc_dos_to_iso") or None,
                    confidence=hit.get("confidence"),
                    all_dates=hit["dates"],
                    extraction_method=METHOD,
                )
                mark_completed(conn, ctx, page_id)

                csv_rows.append(
                    {
                        "chart_name": ctx.chart_name,
                        "page_name": page["page_name"],
                        "page_number": page.get("page_number"),
                        "dos_from": hit.get("dos_from") or "",
                        "dos_to": hit.get("dos_to") or "",
                        "dos_from_iso": hit.get("dos_from_iso") or "",
                        "dos_to_iso": hit.get("dos_to_iso") or "",
                        "doc_dos_from": hit.get("doc_dos_from") or "",
                        "doc_dos_to": hit.get("doc_dos_to") or "",
                        "doc_dos_from_iso": hit.get("doc_dos_from_iso") or "",
                        "doc_dos_to_iso": hit.get("doc_dos_to_iso") or "",
                        "match_type": hit.get("match_type") or "",
                        "keyword": hit.get("keyword") or "",
                        "confidence": hit.get("confidence"),
                        "extraction_method": METHOD,
                    }
                )

        path = write_csv(imaging_csv(ctx.chart_name, "dos"), DOS_COLS, csv_rows)
        return {
            "chart_id": chart_id,
            "dos_csv": str(path),
            "pages_done": ctx.done,
            "skipped": ctx.skipped,
            "extraction_method": METHOD,
        }
