"""Paginated / filtered folder listing for the landing page.

Prefer ``chart_list`` summaries when ``DATABASE_URL`` is usable so Local Mode
does not walk every chart's pages/ocr/imaging tree on each request. Disk is
only used for charts missing from Postgres (light page count) or when no DB
is configured (full local scan as before).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Sequence

from app.core.schemas import FolderSummary, OcrRunStatus
from app.services.chart_run_batch import (
    db_lookup_enabled,
    database_url_usable,
    with_run_batch_default,
)
from app.services.ground_truth import charts_with_ground_truth
from app.services.db import connection


logger = logging.getLogger("review_ui.folder_list")

SortKey = Literal["filename", "pages", "updated"]
SortDir = Literal["asc", "desc"]

IMAGE_RE = re.compile(r"\.(jpe?g|png|webp|tif{1,2})$", re.IGNORECASE)

# chart_list.status (+ current_stage) → landing pill. Mirrors postgres adapter.
_CHART_STATUS_TO_OCR: dict[str, str] = {
    "received": "QUEUED",
    "downloading": "IN_PROGRESS",
    "processing": "IN_PROGRESS",
    "completed": "IMAGING_COMPLETED",
    "failed": "FAILED",
    "needs_review": "IMAGING_COMPLETED",
    "rejected": "IMAGING_COMPLETED",
    "ocr_prelim": "IN_PROGRESS",
    "ocr_quality": "IN_PROGRESS",
    "blank_junk": "IMAGING_IN_PROGRESS",
    "ocr_final1": "IN_PROGRESS",
    "ocr_final2": "IN_PROGRESS",
    "section_headers": "IMAGING_IN_PROGRESS",  # charts run before kv_extract replaced it
    "kv_extract": "IMAGING_IN_PROGRESS",
    "member_verify": "IMAGING_IN_PROGRESS",
    "dos_extract": "IMAGING_IN_PROGRESS",
    "page_subtype": "IMAGING_IN_PROGRESS",
    "encounter_type": "IMAGING_IN_PROGRESS",
    "page_sequencing": "IMAGING_IN_PROGRESS",
}
_IMAGING_STAGES = frozenset({"blank_junk", "member_verify", "dos_extract"})


def ocr_status_for(status: str | None, current_stage: str | None) -> OcrRunStatus:
    key = str(status or "").lower()
    mapped = _CHART_STATUS_TO_OCR.get(key, "QUEUED")
    if key == "processing" and str(current_stage or "").lower() in _IMAGING_STAGES:
        return "IMAGING_IN_PROGRESS"  # type: ignore[return-value]
    return mapped  # type: ignore[return-value]


@dataclass
class FolderListParams:
    q: str = ""
    status: list[str] = field(default_factory=list)
    run: list[str] = field(default_factory=list)
    batch: list[str] = field(default_factory=list)
    sort: SortKey = "updated"
    sort_dir: SortDir = "desc"
    limit: Optional[int] = None  # None / <=0 → return all matching
    offset: int = 0


@dataclass
class FolderListResult:
    items: list[FolderSummary]
    total: int
    page_count_sum: int
    ocr_processed_sum: int
    run_options: list[str]
    batch_options: list[str]
    limit: Optional[int]
    offset: int


def quick_page_count(folder_dir: Path) -> int:
    """Count image files under pages/ (or folder root) without OCR/imaging I/O."""
    pages = folder_dir / "pages"
    root = pages if pages.is_dir() else folder_dir
    if not root.is_dir():
        return 0
    n = 0
    try:
        for p in root.iterdir():
            if p.is_file() and IMAGE_RE.search(p.name) and not p.name.startswith("."):
                n += 1
    except OSError:
        return 0
    return n


def fetch_chart_summaries_from_db(
    database_url: str,
    *,
    db_schema: str = "public",
) -> list[FolderSummary]:
    """Landing rows from ``chart_list`` only (no disk walk).

    Excludes ``source='local'`` intake rows — the review UI lists blob (and
    non-empty manifest) charts only.
    """
    if not db_lookup_enabled(database_url, db_schema):
        return []
    try:
        with connection(database_url, db_schema) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chart_name, page_count, status, updated_at,
                           current_stage, run_id, batch_id
                      FROM chart_list
                     WHERE source <> 'local'
                       AND (source <> 'manifest' OR page_count > 0)
                     ORDER BY updated_at DESC NULLS LAST, id DESC
                    """
                )
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("chart_list summary lookup failed: %s", exc)
        return []

    out: list[FolderSummary] = []
    for chart_name, page_count, status, updated_at, current_stage, run_id, batch_id in rows:
        name = str(chart_name or "").strip()
        if not name:
            continue
        page_n = int(page_count or 0)
        status_ui = ocr_status_for(status, current_stage)
        # Approximate progress columns from chart status (no per-page disk scan).
        imaging_n = page_n if status_ui == "IMAGING_COMPLETED" else 0
        ocr_n = (
            page_n
            if status_ui
            in {"COMPLETED", "IMAGING_COMPLETED", "IMAGING_IN_PROGRESS", "IN_PROGRESS"}
            else 0
        )
        stored_run, stored_batch = with_run_batch_default(
            str(run_id) if run_id else None,
            str(batch_id) if batch_id else None,
        )
        out.append(
            FolderSummary(
                id=name,
                name=name,
                page_count=page_n,
                ocr_processed=ocr_n,
                imaging_processed=imaging_n,
                ocr_status=status_ui,
                last_updated_at=updated_at,
                run_id=stored_run,
                batch_id=stored_batch,
            )
        )
    return out


def list_disk_folder_names_light(data_root: Path) -> list[FolderSummary]:
    """Chart dirs on disk with a cheap page count — no OCR/imaging scan."""
    if not data_root.is_dir():
        return []
    out: list[FolderSummary] = []
    try:
        entries = sorted(data_root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        mtime = None
        try:
            mtime = datetime.fromtimestamp(entry.stat().st_mtime).astimezone()
        except OSError:
            pass
        disk_run, disk_batch = with_run_batch_default(None, None)
        out.append(
            FolderSummary(
                id=entry.name,
                name=entry.name,
                page_count=quick_page_count(entry),
                ocr_processed=0,
                imaging_processed=0,
                ocr_status="QUEUED",
                last_updated_at=mtime,
                run_id=disk_run,
                batch_id=disk_batch,
            )
        )
    return out


def merge_db_and_disk(
    db_rows: Sequence[FolderSummary],
    disk_rows: Sequence[FolderSummary],
) -> list[FolderSummary]:
    """DB wins for known charts; disk-only charts appended."""
    by_id: dict[str, FolderSummary] = {f.id: f for f in db_rows}
    for d in disk_rows:
        if d.id in by_id:
            existing = by_id[d.id]
            if existing.page_count <= 0 and d.page_count > 0:
                existing.page_count = d.page_count
            if existing.last_updated_at is None and d.last_updated_at is not None:
                existing.last_updated_at = d.last_updated_at
            continue
        by_id[d.id] = d
    return list(by_id.values())


def _sort_key_value(f: FolderSummary, sort: SortKey):
    if sort == "filename":
        return (f.name or "").casefold()
    if sort == "pages":
        return f.page_count or 0
    # updated
    ts = f.last_updated_at
    if ts is None:
        return datetime.min
    if getattr(ts, "tzinfo", None) is None:
        return ts
    return ts


def filter_sort_page(
    rows: Sequence[FolderSummary],
    params: FolderListParams,
) -> FolderListResult:
    """Apply q/status/run/batch filters, sort, then paginate.

    Run facets come from the full set. Batch facets are empty until a run is
    selected, then only batches that appear on charts in those run(s).
    """
    # Review UI never lists Queued charts — drop them before facets/filters.
    active = [f for f in rows if f.ocr_status != "QUEUED"]

    run_options = sorted(
        {f.run_id for f in active if f.run_id},
        key=lambda s: s.casefold(),
    )

    q = (params.q or "").strip().casefold()
    status_set = {s for s in params.status if s}
    run_set = {s for s in params.run if s}
    batch_set = {s for s in params.batch if s}

    # Batch options are locked to selected run(s).
    if run_set:
        batch_options = sorted(
            {
                f.batch_id
                for f in active
                if f.batch_id and f.run_id in run_set
            },
            key=lambda s: s.casefold(),
        )
    else:
        batch_options = []

    filtered: list[FolderSummary] = []
    for f in active:
        if q and q not in (f.name or "").casefold():
            continue
        if status_set and f.ocr_status not in status_set:
            continue
        if run_set and (f.run_id is None or f.run_id not in run_set):
            continue
        if batch_set and (f.batch_id is None or f.batch_id not in batch_set):
            continue
        filtered.append(f)

    reverse = params.sort_dir == "desc"
    filtered.sort(
        key=lambda f: _sort_key_value(f, params.sort),
        reverse=reverse,
    )

    total = len(filtered)
    page_count_sum = sum(f.page_count or 0 for f in filtered)
    ocr_processed_sum = sum(f.ocr_processed or 0 for f in filtered)

    offset = max(0, int(params.offset or 0))
    limit = params.limit
    if limit is None or int(limit) <= 0:
        items = filtered[offset:] if offset else filtered
        eff_limit: Optional[int] = None
    else:
        lim = int(limit)
        items = filtered[offset : offset + lim]
        eff_limit = lim

    return FolderListResult(
        items=items,
        total=total,
        page_count_sum=page_count_sum,
        ocr_processed_sum=ocr_processed_sum,
        run_options=run_options,
        batch_options=batch_options,
        limit=eff_limit,
        offset=offset,
    )


def build_folder_list(
    *,
    database_url: str | None,
    db_schema: str = "public",
    data_root: Path | None,
    full_local_rows: Sequence[FolderSummary] | None,
    params: FolderListParams,
) -> FolderListResult:
    """Compose landing list: DB summaries preferred; light disk or full local fallback."""
    if database_url and database_url_usable(database_url):
        db_rows = fetch_chart_summaries_from_db(database_url, db_schema=db_schema)
        disk_rows: list[FolderSummary] = []
        if data_root is not None:
            disk_rows = list_disk_folder_names_light(data_root)
        merged = merge_db_and_disk(db_rows, disk_rows)
        have = charts_with_ground_truth(database_url, db_schema)
        if have:
            merged = [
                row.model_copy(update={"ground_truth_available": row.name in have})
                for row in merged
            ]
        return filter_sort_page(merged, params)

    # No DB — use caller-provided full local scan (existing behaviour).
    rows = list(full_local_rows or [])
    if not rows and data_root is not None:
        rows = list_disk_folder_names_light(data_root)
    return filter_sort_page(rows, params)
