"""Stage: page sequencing (suggested order; does not reorder files).

Ported from advantmed-document-processing ``adapters/sequencing``:
explicit page markers (the printed page numbers the key/value
extraction staged) → multi-stream split → header groups → optional
cross-encoder continuation → cover pages last.

Writes DB + ``imaging/<chart>_sequencing.csv``.
``currentSequence`` = file/page order; ``actualSequence`` = suggested ``seq``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from config import SEQUENCING_CROSS_ENCODER
from db import (
    connect,
    get_blank_junk_flags,
    get_ocr_texts,
    get_quality_map,
    upsert_sequencing,
)
from db.paths import imaging_csv, write_csv
from stages._support import (
    BJ_EXCLUDE,
    best_page_text,
    mark_completed,
    mark_skipped,
    stage_run,
)
from stages.lib.extraction.stage import ensure_staging
from stages.lib.sequencing import compute_sequence_assignments
from stages.lib.sequencing.engine.types import ExplicitMarker

logger = logging.getLogger(__name__)

STAGE = "page_sequencing"

SEQ_COLS = [
    "chart_name",
    "page_name",
    "page_number",
    "current_sequence",
    "actual_sequence",
    "sequence_method",
    "confidence",
    "review_flag",
    "stream_id",
    "ocr_source",
]


def _ocr_source_label(
    *,
    final2: Optional[str],
    final1: Optional[str],
    prelim: Optional[str],
) -> str:
    if final2 and str(final2).strip():
        return "final2"
    if final1 and str(final1).strip():
        return "final1"
    if prelim and str(prelim).strip():
        return "prelim"
    return ""


def _staged_marker(staged_page: Any) -> Optional[ExplicitMarker]:
    """The printed page number the extraction chose on a page, as a sequencing marker."""
    if staged_page is None:
        return None
    for row in staged_page.selected("page_no"):
        try:
            number = int(row.get("page_no") or 0)
            total = int(row["page_total"]) if row.get("page_total") else None
        except (TypeError, ValueError):
            continue
        if number < 1 or (total is not None and (total < 1 or number > total)):
            continue
        return ExplicitMarker(
            page_num=number,
            total_pages=total,
            pattern="page_x_of_y" if total else "page_x",
            confidence=1.0 if total else 0.85,
        )
    return None


def run(chart_id: int, *, force: bool = False) -> dict[str, Any]:
    with stage_run(chart_id, STAGE, force=force) as ctx:
        with connect() as conn:
            bj = get_blank_junk_flags(conn, chart_id, final_only=True)
            drop = [
                pid
                for pid in ctx.todo
                if bj.get(pid, "not_blank_junk") in BJ_EXCLUDE
            ]
            mark_skipped(conn, ctx, drop, "blank_junk")
            todo = set(ctx.todo)

            prelim = get_ocr_texts(conn, chart_id, "tesseract")
            final1 = get_ocr_texts(conn, chart_id, "docling")
            final2 = get_ocr_texts(conn, chart_id, "azuredocintel")
            quality = get_quality_map(conn, chart_id)

        staged = ensure_staging(chart_id, ctx.chart_name)

        # Sequencing needs the full main-page set (order is chart-global).
        seq_inputs: list[dict[str, Any]] = []
        sources: dict[int, str] = {}
        page_by_id: dict[int, dict[str, Any]] = {}
        for page in ctx.pages:
            page_id = page["id"]
            page_by_id[page_id] = page
            is_bj = bj.get(page_id, "not_blank_junk") in BJ_EXCLUDE
            f2 = final2.get(page_id)
            f1 = final1.get(page_id)
            pr = prelim.get(page_id)
            text = ""
            if not is_bj:
                text = best_page_text(
                    final2=f2,
                    final1=f1,
                    prelim=pr,
                    quality_row=quality.get(page_id),
                )
            sources[page_id] = _ocr_source_label(final2=f2, final1=f1, prelim=pr)
            seq_inputs.append(
                {
                    "page_id": page_id,
                    "page_number": int(page.get("page_number") or 0),
                    "text": text,
                    "is_classified": is_bj,
                    # A page the extraction could not read has no marker to find.
                    "marker_extracted": True,
                    "page_marker": _staged_marker(staged.page(page["page_name"])),
                }
            )

        assignments = compute_sequence_assignments(
            seq_inputs,
            cross_encoder_enabled=bool(SEQUENCING_CROSS_ENCODER),
        )
        by_str_id = {str(a["page_id"]): a for a in assignments}

        csv_rows: list[dict[str, Any]] = []
        with connect() as conn:
            for page in ctx.pages:
                page_id = page["id"]
                a = by_str_id.get(str(page_id)) or {}
                current = int(page.get("page_number") or 0) or None
                actual = a.get("sequence_position")
                conf = a.get("sequence_confidence")
                method = a.get("sequence_method") or ""
                review = bool(a.get("sequence_review_flag"))

                if page_id in todo or force:
                    upsert_sequencing(
                        conn,
                        chart_id=chart_id,
                        page_id=page_id,
                        original_page_number=current,
                        seq=int(actual) if actual is not None else None,
                        confidence=float(conf) if conf is not None else None,
                        sequence_method=method or None,
                        review_flag=review,
                    )
                if page_id in todo:
                    mark_completed(conn, ctx, page_id)

                csv_rows.append(
                    {
                        "chart_name": ctx.chart_name,
                        "page_name": page["page_name"],
                        "page_number": page.get("page_number"),
                        "current_sequence": current or "",
                        "actual_sequence": actual if actual is not None else "",
                        "sequence_method": method,
                        "confidence": conf if conf is not None else "",
                        "review_flag": "y" if review else "n",
                        "stream_id": a.get("stream_id") or "",
                        "ocr_source": sources.get(page_id, ""),
                    }
                )

        path = write_csv(
            imaging_csv(ctx.chart_name, "sequencing"), SEQ_COLS, csv_rows
        )
        logger.info(
            "page_sequencing chart=%s pages=%d → %s (cross_encoder=%s)",
            ctx.chart_name,
            len(csv_rows),
            path,
            SEQUENCING_CROSS_ENCODER,
        )
        return {
            "chart_id": chart_id,
            "sequencing_csv": str(path),
            "pages_done": ctx.done,
            "skipped": ctx.skipped,
            "cross_encoder": bool(SEQUENCING_CROSS_ENCODER),
        }
