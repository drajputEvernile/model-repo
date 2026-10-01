"""Production Mode repository — read pipeline outputs from Postgres.

Page images come from Azure Blob using ``chart_list.blob_container`` +
``blob_path`` (Raw_Input), with Processed ``output_path`` pages/corrected as
fallbacks. OCR text, manifest, quality, blank/junk, member, verification, and
DOS come from schema tables. ``DATA_ROOT`` is optional (used when a local
workspace copy exists).
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.adapters.base import FolderRepository
from app.adapters.local.repository import LocalFolderRepository, _fmt_dos_display, _normalize_kind
from app.core.schemas import (
    FolderDetail,
    FolderSummary,
    ImagingDocumentResponse,
    ImagingManifestDetails,
    ImagingPageResult,
    ImagingSectionsProcessed,
    ImagingVerificationDetails,
    OcrSectionHeader,
    OcrTextResponse,
    PageSummary,
)
from app.services import db
from app.services.chart_run_batch import with_run_batch_default
from app.services.db import psycopg_url as _psycopg_url
from app.services.ground_truth import attach_ground_truth
from app.services.imaging_overlays import display_page_type, empty_imaging_pages

logger = logging.getLogger("review_ui.postgres")

# API kind → ocr_results.ocr_type
KIND_TO_OCR_TYPE: dict[str, str] = {
    "preliminary": "tesseract",
    "final1": "docling",
    "final2": "azuredocintel",
}

# schema v7 lifecycle values.
CHART_STATUS_TO_OCR: dict[str, str] = {
    "received": "QUEUED",
    "downloading": "IN_PROGRESS",
    "processing": "IN_PROGRESS",
    "completed": "IMAGING_COMPLETED",
    "failed": "FAILED",
    "needs_review": "IMAGING_COMPLETED",
    "rejected": "IMAGING_COMPLETED",
    # v6 packed the stage name into status. Kept so a database that has not run
    # migration 002 yet still renders sensibly.
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

# v7: once status is just "processing", which stage it is in comes from
# current_stage. These stages mean the imaging modules are running.
IMAGING_STAGES = frozenset(
    {
        "blank_junk",
        "section_headers",
        "kv_extract",
        "member_verify",
        "dos_extract",
        "page_subtype",
        "encounter_type",
        "page_sequencing",
    }
)

# A chart is Imaging Completed only when every stage in pipeline_stage is done
# for every page — chart_list.status 'completed' / 'needs_review' / 'rejected'.
_IMAGING_DONE_STATUSES = frozenset({"completed", "needs_review", "rejected"})

# How long page → blob locations are cached (Raw_Input listing order = page_number).
_BLOB_LIST_TTL_SEC = 300.0


def _ocr_status_for(status: str | None, current_stage: str | None) -> str:
    """UI status pill from (chart_list.status, chart_list.current_stage)."""
    key = str(status or "").lower()
    mapped = CHART_STATUS_TO_OCR.get(key, "QUEUED")
    if key == "processing" and str(current_stage or "").lower() in IMAGING_STAGES:
        return "IMAGING_IN_PROGRESS"
    return mapped


def _fmt_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return _fmt_dos_display(value)


def _blank_junk_ui(flag: str | None, subtype: str | None) -> tuple[str | None, bool | None, str | None]:
    """Map DB blank_junk_flag → UI blankOrJunk / isDuplicate / pageType."""
    if not flag:
        return None, None, None
    f = flag.strip().lower()
    if f == "blank":
        return "Yes (Blank)", False, "Not Available"
    if f == "duplicate":
        return "No", True, "Not Available"
    if f == "junk":
        return "Yes (Junk)", False, subtype or "Junk"
    if f == "not_blank_junk":
        return "No", False, "Not Available"
    return None, None, None


class PostgresFolderRepository(FolderRepository):
    def __init__(
        self,
        database_url: str,
        data_root: Path | None = None,
        metadata_root: Path | None = None,
        db_schema: str = "public",
    ):
        self.database_url = _psycopg_url(database_url)
        self.db_schema = (db_schema or "public").strip() or "public"
        self._local = (
            LocalFolderRepository(data_root, metadata_root=metadata_root)
            if data_root is not None
            else None
        )
        # Per-chart caches for the page image route (TTL _BLOB_LIST_TTL_SEC):
        # folder_id → sorted (key, etag) under the Raw_Input prefix
        self._raw_blob_lists: dict[str, tuple[float, list[tuple[str, str]]]] = {}
        # folder_id → page_number → page_list/chart_list location row
        self._page_rows_cache: dict[str, tuple[float, dict[int, tuple]]] = {}
        # (folder_id, page_number) → resolved blob location
        self._resolved_blobs: dict[tuple[str, int], tuple[float, dict[str, str]]] = {}

    def _optional_local(self) -> LocalFolderRepository | None:
        return self._local

    def _connect(self):
        return db.connection(self.database_url, self.db_schema)

    def list_folders(self) -> list[FolderSummary]:
        """DB chart_list summaries; light disk merge for local-only folders.

        Avoids a full LocalFolderRepository scan (pages/ocr/imaging per chart).
        """
        from app.services.folder_list import (
            FolderListParams,
            build_folder_list,
        )

        result = build_folder_list(
            database_url=self.database_url,
            db_schema=self.db_schema,
            data_root=self._local.data_root if self._local else None,
            full_local_rows=None,
            params=FolderListParams(limit=None, offset=0, sort="updated", sort_dir="desc"),
        )
        return result.items

    def _pages_from_db(self, folder_id: str) -> list[tuple[int, str]]:
        """(page_number, page_name) from page_list, ordered for the viewer."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT p.page_number, p.page_name
                          FROM page_list p
                          JOIN chart_list c ON c.id = p.chart_id
                         WHERE c.chart_name = %s
                         ORDER BY p.page_number NULLS LAST, p.id
                        """,
                        (folder_id,),
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            logger.warning("page_list lookup failed for %s: %s", folder_id, exc)
            return []
        out: list[tuple[int, str]] = []
        for idx, (num, name) in enumerate(rows, start=1):
            out.append((int(num or idx), str(name)))
        return out

    def _folder_from_db(self, folder_id: str) -> FolderDetail | None:
        """Build FolderDetail in one DB connection (pages + OCR flags + chart meta)."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, status, current_stage, run_id, batch_id
                          FROM chart_list
                         WHERE chart_name = %s
                         LIMIT 1
                        """,
                        (folder_id,),
                    )
                    chart = cur.fetchone()
                    if not chart:
                        return None
                    chart_id, status_raw, stage_raw, run_id, batch_id = chart

                    cur.execute(
                        """
                        SELECT page_number, page_name
                          FROM page_list
                         WHERE chart_id = %s
                         ORDER BY page_number NULLS LAST, id
                        """,
                        (chart_id,),
                    )
                    page_rows = cur.fetchall()
                    if not page_rows:
                        return None

                    cur.execute(
                        """
                        SELECT p.page_name, o.ocr_type
                          FROM ocr_results o
                          JOIN page_list p ON p.id = o.page_id
                         WHERE o.chart_id = %s
                           AND o.raw_text IS NOT NULL
                           AND length(trim(o.raw_text)) > 0
                        """,
                        (chart_id,),
                    )
                    flag_rows = cur.fetchall()

                    # Pages done in every stage of the chain (completed or
                    # skipped) — the "Imaging N" count and the page dots.
                    cur.execute(
                        """
                        SELECT p.page_name
                          FROM page_list p
                         WHERE p.chart_id = %s
                           AND NOT EXISTS (
                               SELECT 1 FROM pipeline_stage s
                                WHERE s.is_phase1
                                  AND NOT EXISTS (
                                      SELECT 1 FROM page_stage_status ps
                                       WHERE ps.page_id = p.id
                                         AND ps.stage_name = s.stage_name
                                         AND ps.pass_no = s.pass_no
                                         AND ps.status IN ('completed', 'skipped')
                                  )
                           )
                        """,
                        (chart_id,),
                    )
                    imaging_done = {str(r[0]) for r in cur.fetchall()}

                    cur.execute(
                        """
                        SELECT member_name, member_dob, external_member_id
                          FROM manifest_member_list
                         WHERE record_id = %s
                         ORDER BY id
                         LIMIT 1
                        """,
                        (folder_id,),
                    )
                    man = cur.fetchone()
        except Exception as exc:
            logger.warning("folder_from_db failed for %s: %s", folder_id, exc)
            return None

        pages: list[tuple[int, str]] = []
        for idx, (num, name) in enumerate(page_rows, start=1):
            pages.append((int(num or idx), str(name)))

        flags: dict[str, dict[str, bool]] = {}
        for page_name, ocr_type in flag_rows:
            flags.setdefault(str(page_name), {})[str(ocr_type)] = True

        page_summaries = [
            PageSummary(
                page_number=num,
                filename=name,
                image_url=f"/api/folders/{folder_id}/pages/{num}/image",
                has_preliminary_ocr=bool(flags.get(name, {}).get("tesseract")),
                has_final1_ocr=bool(flags.get(name, {}).get("docling")),
                has_final2_ocr=bool(flags.get(name, {}).get("azuredocintel")),
                has_imaging=name in imaging_done
                or str(status_raw or "").lower() in _IMAGING_DONE_STATUSES,
            )
            for num, name in pages
        ]
        kinds_present = {
            k
            for page_flags in flags.values()
            for k, ok in page_flags.items()
            if ok
        }
        ocr_processed = sum(
            1
            for k in ("tesseract", "docling", "azuredocintel")
            if k in kinds_present
        )

        status = str(status_raw or "").lower()
        stage = str(stage_raw or "").lower()
        if status == "processing" and stage:
            chart_status = stage
        else:
            chart_status = status or None
        ocr_status = CHART_STATUS_TO_OCR.get(
            str(chart_status or "").lower(), "QUEUED"
        )
        if ocr_status == "IN_PROGRESS" and ocr_processed == 3:
            ocr_status = "IMAGING_IN_PROGRESS"
        elif ocr_status == "QUEUED" and ocr_processed > 0:
            ocr_status = "IN_PROGRESS"

        manifest = None
        if man:
            name, dob, member_id = man
            manifest = ImagingManifestDetails(
                member=name,
                dob=_fmt_date(dob),
                memberId=member_id,
            )

        shown_run, shown_batch = with_run_batch_default(
            str(run_id) if run_id else None,
            str(batch_id) if batch_id else None,
        )
        return FolderDetail(
            id=folder_id,
            name=folder_id,
            page_count=len(pages),
            ocr_processed=ocr_processed,
            imaging_processed=sum(1 for p in page_summaries if p.has_imaging),
            ocr_status=ocr_status,  # type: ignore[arg-type]
            last_updated_at=None,
            run_id=shown_run,
            batch_id=shown_batch,
            manifest=manifest,
            pages=page_summaries,
        )

    def get_folder(self, folder_id: str) -> FolderDetail:
        """Pages + OCR flags from Postgres first — avoid a local disk walk on open.

        Local workspace is only a fallback when the chart is absent from page_list
        (pure Local Mode folders with no DB row).
        """
        db_detail = self._folder_from_db(folder_id)
        if db_detail is not None and db_detail.pages:
            return db_detail

        local = self._optional_local()
        detail: FolderDetail | None = None
        if local is not None:
            try:
                detail = local.get_folder(folder_id)
            except HTTPException:
                detail = None
            except Exception:
                detail = None
        if detail is None or not detail.pages:
            raise HTTPException(
                status_code=404,
                detail=f"Chart not found: {folder_id}",
            )
        return detail

    def get_page_image_path(self, folder_id: str, page_number: int) -> Path:
        """Local path when a workspace copy exists (fallback for the image route)."""
        local = self._optional_local()
        if local is None:
            raise HTTPException(
                status_code=404,
                detail="No local page image; Production Mode serves images from blob",
            )
        return local.get_page_image_path(folder_id, page_number)

    def _page_rows(self, folder_id: str) -> dict[int, tuple] | None:
        """page_number → (container, blob_path, output_path, page_name,
        use_corrected, image_path, ingest_index), cached per chart.

        Every page image request needs this; querying the whole chart's
        page_list per image made a filmstrip of N thumbnails cost N queries.
        """
        now = time.monotonic()
        cached = self._page_rows_cache.get(folder_id)
        if cached and (now - cached[0]) < _BLOB_LIST_TTL_SEC:
            return cached[1]
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT c.blob_container, c.blob_path, c.output_path,
                               p.page_name, p.page_number, p.use_corrected, p.image_path
                          FROM page_list p
                          JOIN chart_list c ON c.id = p.chart_id
                         WHERE c.chart_name = %s
                         ORDER BY p.page_number NULLS LAST, p.id
                        """,
                        (folder_id,),
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            logger.warning("resolve_page_blob DB failed for %s: %s", folder_id, exc)
            return None

        by_num: dict[int, tuple] = {}
        for idx, (container, blob_path, output_path, page_name, db_num, use_corrected, image_path) in enumerate(
            rows, start=1
        ):
            num = int(db_num or idx)
            by_num.setdefault(
                num,
                (
                    (container or "").strip(),
                    blob_path,
                    output_path,
                    str(page_name),
                    bool(use_corrected),
                    str(image_path or ""),
                    idx,
                ),
            )
        self._page_rows_cache[folder_id] = (now, by_num)
        return by_num

    def resolve_page_blob(
        self, folder_id: str, page_number: int
    ) -> dict[str, str] | None:
        """Locate the Azure blob for a page using chart_list + page_list.

        Preference order:
          1. Raw_Input ``blob_path`` entry at the same ingest index as ``page_number``
          2. Processed ``output_path/corrected-pages/…`` when ``use_corrected``
          3. Processed ``output_path/pages/{page_name}``

        Returns ``container``, ``key``, ``filename`` and ``etag`` (empty when
        unknown). Results are cached per page for ``_BLOB_LIST_TTL_SEC``.
        """
        now = time.monotonic()
        cache_key = (folder_id, page_number)
        cached = self._resolved_blobs.get(cache_key)
        if cached and (now - cached[0]) < _BLOB_LIST_TTL_SEC:
            return cached[1]

        rows = self._page_rows(folder_id)
        hit = rows.get(page_number) if rows else None
        if hit is None:
            return None
        container, blob_path, output_path, page_name, use_corrected, image_path, idx = hit
        if not container:
            return None

        # Raw_Input comes from a live listing, so the key is known to exist —
        # no existence probe needed.
        raw = self._raw_input_blob(folder_id, container, blob_path, idx)
        if raw:
            key, etag = raw
            out_loc = {"container": container, "key": key, "filename": Path(key).name or page_name, "etag": etag}
            self._resolved_blobs[cache_key] = (now, out_loc)
            return out_loc

        out = (output_path or "").strip().strip("/")
        stem = Path(page_name).stem
        candidates: list[str] = []
        if use_corrected and out:
            candidates.append(f"{out}/corrected-pages/{stem}.jpg")
            candidates.append(f"{out}/corrected-pages/{page_name}")
            rel = image_path.replace("\\", "/").lstrip("/")
            if rel.startswith("corrected-pages/"):
                candidates.append(f"{out}/{rel}")
        if out:
            candidates.append(f"{out}/pages/{page_name}")

        ordered = list(dict.fromkeys(k.lstrip("/") for k in candidates if k.lstrip("/")))
        if not ordered:
            return None

        found = self._first_existing_blob(container, ordered)
        key, etag = found if found else (ordered[0], "")
        out_loc = {"container": container, "key": key, "filename": Path(key).name or page_name, "etag": etag}
        self._resolved_blobs[cache_key] = (now, out_loc)
        return out_loc

    def _first_existing_blob(self, container: str, keys: list[str]) -> tuple[str, str] | None:
        """First key that exists, with its ETag (fallback path only)."""
        try:
            from app.services.blob_store import _blob_service_client

            client = _blob_service_client()
            for key in keys:
                blob = client.get_blob_client(container=container, blob=key)
                try:
                    props = blob.get_blob_properties()
                    return key, str(props.etag or "").strip('"')
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("blob existence probe skipped: %s", exc)
            return (keys[0], "") if keys else None
        return None

    def _raw_input_blob(
        self,
        folder_id: str,
        container: str,
        blob_path: str | None,
        page_index_1based: int,
    ) -> tuple[str, str] | None:
        """Map page index → (Raw_Input blob key, etag), same sort order as ingest."""
        prefix = (blob_path or "").strip()
        if not container or not prefix:
            return None
        now = time.monotonic()
        cached = self._raw_blob_lists.get(folder_id)
        if cached and (now - cached[0]) < _BLOB_LIST_TTL_SEC:
            blobs = cached[1]
        else:
            try:
                from app.services.blob_store import list_image_blobs

                blobs = list_image_blobs(container, prefix)
                self._raw_blob_lists[folder_id] = (now, blobs)
            except Exception as exc:
                logger.warning(
                    "Raw_Input blob list failed for %s (%s/%s): %s",
                    folder_id,
                    container,
                    prefix,
                    exc,
                )
                return None
        if page_index_1based < 1 or page_index_1based > len(blobs):
            return None
        return blobs[page_index_1based - 1]

    def get_ocr_text(self, folder_id: str, kind: str) -> OcrTextResponse:
        """Assemble OCR from ocr_results into ===== page ===== marker text for the UI."""
        from app.adapters.local.repository import FINAL2_QUALITY_SKIP_MESSAGE

        normalized = _normalize_kind(kind)
        ocr_type = KIND_TO_OCR_TYPE[normalized]

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if ocr_type in {"azuredocintel", "docling"}:
                        stage_name = (
                            "ocr_final2" if ocr_type == "azuredocintel" else "ocr_final1"
                        )
                        cur.execute(
                            """
                            SELECT p.page_name, o.raw_text, pss.skip_reason
                            FROM page_list p
                            JOIN chart_list c ON c.id = p.chart_id
                            LEFT JOIN ocr_results o
                              ON o.page_id = p.id AND o.ocr_type = %s
                            LEFT JOIN page_stage_status pss
                              ON pss.page_id = p.id
                             AND pss.stage_name = %s
                             AND pss.pass_no = 1
                            WHERE c.chart_name = %s
                            ORDER BY p.page_number NULLS LAST, p.id
                            """,
                            (ocr_type, stage_name, folder_id),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT p.page_name, o.raw_text, NULL::text AS skip_reason
                            FROM ocr_results o
                            JOIN chart_list c ON c.id = o.chart_id
                            JOIN page_list p ON p.id = o.page_id
                            WHERE c.chart_name = %s
                              AND o.ocr_type = %s
                            ORDER BY p.page_number NULLS LAST, p.id
                            """,
                            (folder_id, ocr_type),
                        )
                    rows = cur.fetchall()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Postgres OCR lookup failed: {exc}",
            ) from exc

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No ocr_results for chart={folder_id!r} ocr_type={ocr_type!r}",
            )

        chunks: list[str] = []
        headers_by_file: dict[str, list[OcrSectionHeader]] = {}
        for page_name, raw_text, skip_reason in rows:
            body = (raw_text or "").strip()
            parsed: dict[str, Any] | None = None
            if ocr_type in {"azuredocintel", "docling"} and body.startswith("{"):
                try:
                    import json

                    maybe = json.loads(body)
                    if isinstance(maybe, dict) and (
                        "content" in maybe or "markdown" in maybe
                    ):
                        parsed = maybe
                        body = str(
                            maybe.get("content")
                            or maybe.get("markdown")
                            or ""
                        ).strip()
                except Exception:
                    pass
            if (
                ocr_type in {"azuredocintel", "docling"}
                and not body
                and parsed is not None
            ):
                reason = str(
                    parsed.get("skippedReason") or parsed.get("skipped_reason") or ""
                ).strip().lower()
                if reason == "high_quality_printed":
                    body = FINAL2_QUALITY_SKIP_MESSAGE
                elif reason in {
                    "blank_junk_pass1",
                    "blank_junk",
                    "blank_junk_pass2",
                }:
                    from app.adapters.local.repository import (
                        FINAL_OCR_BLANK_JUNK_SKIP_MESSAGE,
                    )

                    body = FINAL_OCR_BLANK_JUNK_SKIP_MESSAGE
            if (
                ocr_type == "azuredocintel"
                and not body
                and str(skip_reason or "").strip().lower() == "high_quality_printed"
            ):
                body = FINAL2_QUALITY_SKIP_MESSAGE
            if (
                ocr_type in {"azuredocintel", "docling"}
                and not body
                and str(skip_reason or "").strip().lower()
                in {"blank_junk_pass1", "blank_junk", "blank_junk_pass2"}
            ):
                from app.adapters.local.repository import (
                    FINAL_OCR_BLANK_JUNK_SKIP_MESSAGE,
                )

                body = FINAL_OCR_BLANK_JUNK_SKIP_MESSAGE
            if body and ocr_type in {"azuredocintel", "docling"}:
                from app.adapters.local.repository import clean_ocr_display_text

                if not body.startswith("Skipped for "):
                    body = clean_ocr_display_text(body)
            if ocr_type in {"docling", "azuredocintel"} and parsed is not None:
                from app.adapters.local.repository import (
                    _filter_headers_against_canon,
                    _section_headers_from_page,
                )

                headers = _filter_headers_against_canon(
                    _section_headers_from_page(parsed)
                )
                if headers:
                    headers_by_file[str(page_name)] = headers
                    headers_by_file[str(page_name).lower()] = headers
            chunks.append(f"===== {page_name} =====\n{body}".rstrip())

        if ocr_type == "azuredocintel" and not any(
            (c.split("\n", 1)[-1] if "\n" in c else "").strip() for c in chunks
        ):
            raise HTTPException(
                status_code=404,
                detail=f"No ocr_results for chart={folder_id!r} ocr_type={ocr_type!r}",
            )

        return OcrTextResponse(
            folder_id=folder_id,
            kind=normalized,
            text="\n\n".join(chunks),
            section_headers_by_file=headers_by_file,
        )

    def _manifest_from_db(self, folder_id: str) -> ImagingManifestDetails | None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT m.member_name, m.member_dob, m.external_member_id
                        FROM manifest_member_list m
                        WHERE m.record_id = %s
                        ORDER BY m.id
                        LIMIT 1
                        """,
                        (folder_id,),
                    )
                    row = cur.fetchone()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Postgres manifest lookup failed: {exc}",
            ) from exc

        if not row:
            return None
        name, dob, member_id = row
        return ImagingManifestDetails(
            member=name,
            dob=_fmt_date(dob),
            memberId=member_id,
        )

    def _verification_from_db(self, folder_id: str) -> ImagingVerificationDetails | None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT mv.final_status, mv.matched_name, m.external_member_id,
                               mv.confidence, mv.pages_matched, mv.pages_checked,
                               mv.decision_reason
                        FROM member_verification_summary mv
                        JOIN chart_list c ON c.id = mv.chart_id
                        LEFT JOIN manifest_member_list m ON m.id = mv.matched_member_list_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    row = cur.fetchone()
        except Exception:
            return None
        if not row:
            return None
        return ImagingVerificationDetails(
            finalStatus=row[0],
            matchedName=row[1],
            matchedMemberId=row[2],
            matchedConfidence=float(row[3]) if row[3] is not None else None,
            pagesMatched=row[4],
            pagesChecked=row[5],
            decisionReason=row[6],
        )

    def _page_imaging_from_db(self, folder_id: str) -> dict[str, dict[str, Any]]:
        """page_name → field dict from quality / BJ / member / DOS tables."""
        by_page: dict[str, dict[str, Any]] = {}

        def _ensure(name: str) -> dict[str, Any]:
            return by_page.setdefault(name, {})

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    # Quality
                    cur.execute(
                        """
                        SELECT p.page_name, q.printed_or_handwritten, q.hw_confidence,
                               q.quality_score, q.quality_tag,
                               q.orientation_angle, q.tilt_angle, q.mirrored
                        FROM ocr_quality_results q
                        JOIN page_list p ON p.id = q.page_id
                        JOIN chart_list c ON c.id = q.chart_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    for (
                        page_name, hw, conf, qscore, qtag, orient, tilt, mirrored
                    ) in cur.fetchall():
                        fields = _ensure(str(page_name))
                        if hw:
                            label = str(hw).strip().lower()
                            if "hand" in label:
                                fields["handwrittenOrPrinted"] = "Handwritten"
                            elif "uncertain" in label:
                                fields["handwrittenOrPrinted"] = "Uncertain"
                            elif "mix" in label:
                                fields["handwrittenOrPrinted"] = "Mixed"
                            else:
                                fields["handwrittenOrPrinted"] = "Printed"
                        if conf is not None:
                            fields["handwrittenOrPrintedConfidence"] = float(conf)
                        if qscore is not None:
                            fields["pageQualityConfidence"] = float(qscore)
                        if qtag:
                            fields["pageQualityTag"] = str(qtag)
                        if orient is not None:
                            fields["orientationAngle"] = float(orient)
                        if tilt is not None:
                            fields["tiltAngle"] = float(tilt)
                        if mirrored is not None:
                            fields["mirrored"] = bool(mirrored)

                    # Blank/junk — v7 stamps exactly one final row per page,
                    # so precedence is not re-derived here any more.
                    cur.execute(
                        """
                        SELECT p.page_name, b.blank_junk_flag, b.junk_subtype,
                               b.confidence
                        FROM v_page_blank_junk_final b
                        JOIN page_list p ON p.id = b.page_id
                        JOIN chart_list c ON c.id = b.chart_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    for page_name, flag, subtype, conf in cur.fetchall():
                        fields = _ensure(str(page_name))
                        blank, dup, page_type = _blank_junk_ui(flag, subtype)
                        fields["blankOrJunk"] = blank
                        fields["isDuplicate"] = dup
                        fields["pageType"] = page_type
                        if conf is not None:
                            fields["pageTypeConfidence"] = float(conf)

                    # Member extraction
                    cur.execute(
                        """
                        SELECT p.page_name,
                               m.extracted_name, m.extracted_dob, m.extracted_member_id,
                               m.confidence
                        FROM member_extraction_results m
                        JOIN chart_list c ON c.id = m.chart_id
                        JOIN page_list p ON p.id = m.page_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    for page_name, name, dob, mid, conf in cur.fetchall():
                        if not page_name:
                            continue
                        fields = _ensure(str(page_name))
                        fields["memberName"] = name
                        fields["memberDob"] = _fmt_date(dob)
                        fields["memberId"] = mid
                        if conf is not None:
                            fields["memberConfidence"] = float(conf)

                    # DOS. There is one schema (schema/schema.sql), so there
                    # is one set of column names — no runtime sniffing.
                    cur.execute(
                        """
                        SELECT p.page_name,
                               d.date_of_service_from, d.date_of_service_to,
                               d.date_of_service_from_doclevel, d.date_of_service_to_doclevel,
                               d.confidence
                        FROM dos_extraction_results d
                        JOIN page_list p ON p.id = d.page_id
                        JOIN chart_list c ON c.id = d.chart_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    for page_name, d_from, d_to, doc_from, doc_to, conf in cur.fetchall():
                        fields = _ensure(str(page_name))
                        fields["dosFrom"] = _fmt_date(d_from)
                        fields["dosTo"] = _fmt_date(d_to)
                        fields["docDosFrom"] = _fmt_date(doc_from)
                        fields["docDosTo"] = _fmt_date(doc_to)
                        if conf is not None:
                            fields["dosConfidence"] = float(conf)

                    # Codeable / non-codeable / discharge (page_classification)
                    cur.execute(
                        """
                        SELECT p.page_name, pc.page_subtype,
                               pc.classification_category, pc.confidence
                        FROM page_classification pc
                        JOIN page_list p ON p.id = pc.page_id
                        JOIN chart_list c ON c.id = pc.chart_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    codeable_display = {
                        "codeable": "Codeable",
                        "non_codeable": "Non Codeable",
                        "discharge_summary": "Discharge Frequency",
                        "discharge_frequency": "Discharge Frequency",
                        "not_sure": "Not Sure",
                    }
                    for page_name, subtype, cat, conf in cur.fetchall():
                        fields = _ensure(str(page_name))
                        key = str(cat or "").strip()
                        subtype_s = str(subtype or "").strip()
                        if subtype_s:
                            fields["pageType"] = display_page_type(subtype_s)
                        fields["isCodeable"] = codeable_display.get(key, key or "Not Sure")
                        if conf is not None:
                            fields["pageTypeConfidence"] = float(conf)

                    # Main pages with no page_classification row keep junk's
                    # pageType=\"Not Available\" — stamp Not Sure for codeability.
                    for fields in by_page.values():
                        pt = str(fields.get("pageType") or "").strip()
                        if (
                            pt == "Not Available"
                            and not fields.get("isCodeable")
                        ):
                            fields["isCodeable"] = "Not Sure"

                    # Encounter type (DOS-wide TF)
                    cur.execute(
                        """
                        SELECT p.page_name, e.encounter_type, e.confidence
                        FROM encounter_type_results e
                        JOIN page_list p ON p.id = e.page_id
                        JOIN chart_list c ON c.id = e.chart_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    enc_display = {
                        "outpatient_f2f": "Outpatient (F2F)",
                        "outpatient_tele": "Outpatient (Tele)",
                        "inpatient": "Inpatient",
                        "home": "Home",
                    }
                    for page_name, et, conf in cur.fetchall():
                        fields = _ensure(str(page_name))
                        key = str(et or "").strip()
                        fields["encounterType"] = enc_display.get(key, key)

                    # Page sequencing
                    cur.execute(
                        """
                        SELECT p.page_name, p.page_number,
                               s.original_page_number, s.seq, s.confidence
                        FROM page_sequencing_results s
                        JOIN page_list p ON p.id = s.page_id
                        JOIN chart_list c ON c.id = s.chart_id
                        WHERE c.chart_name = %s
                        """,
                        (folder_id,),
                    )
                    for page_name, page_num, orig, seq, conf in cur.fetchall():
                        fields = _ensure(str(page_name))
                        current = orig if orig is not None else page_num
                        if current is not None:
                            fields["currentSequence"] = int(current)
                        if seq is not None:
                            fields["actualSequence"] = int(seq)
        except Exception:
            return by_page

        return by_page

    def get_imaging(self, folder_id: str) -> ImagingDocumentResponse:
        """Build imaging panel entirely from Postgres; page skeleton from page_list.

        Prefer DB page names over a local disk walk — scanning DATA_ROOT for every
        chart open was a major cost on large workspaces / external drives.
        """
        page_files = [(num, Path(name)) for num, name in self._pages_from_db(folder_id)]
        if not page_files:
            local = self._optional_local()
            if local is not None:
                try:
                    folder_dir = local._folder_dir(folder_id)  # noqa: SLF001
                    page_files = local._page_files(folder_dir)  # noqa: SLF001
                except Exception:
                    page_files = []
        pages = empty_imaging_pages(page_files)

        db_fields = self._page_imaging_from_db(folder_id)
        merged: list[ImagingPageResult] = []
        for page in pages:
            hit = db_fields.get(page.fileName) or db_fields.get(page.fileName.lower())
            if not hit:
                # try stem match
                stem = Path(page.fileName).stem
                for key, val in db_fields.items():
                    if Path(key).stem == stem:
                        hit = val
                        break
            if hit:
                merged.append(page.model_copy(update=hit))
            else:
                merged.append(page)
        merged = attach_ground_truth(
            merged, folder_id, self.database_url, self.db_schema
        )

        manifest = self._manifest_from_db(folder_id) or ImagingManifestDetails()
        verification = self._verification_from_db(folder_id)
        # "Processed" means the stage wrote rows — not that it found a value.
        # Using non-empty memberName/dosFrom here made empty-but-finished charts
        # show "Yet to Process" instead of "Not Found".
        stage_flags = self._sections_from_db(folder_id)
        sections = ImagingSectionsProcessed(
            member=stage_flags["member"] or verification is not None,
            dos=stage_flags["dos"],
            hw=stage_flags["hw"]
            or any(p.handwrittenOrPrinted for p in merged),
            quality=stage_flags["quality"]
            or any(
                p.pageQualityTag or p.pageQualityConfidence is not None for p in merged
            ),
            rotation=stage_flags["rotation"]
            or any(
                p.orientationAngle is not None or p.tiltAngle is not None for p in merged
            ),
            junk=stage_flags["junk"]
            or any(p.blankOrJunk is not None or p.isDuplicate is not None for p in merged),
            codeable=stage_flags["codeable"]
            or any(p.isCodeable is not None for p in merged),
            encounter=stage_flags["encounter"]
            or any(p.encounterType is not None for p in merged),
            sequencing=stage_flags["sequencing"]
            or any(p.actualSequence is not None for p in merged),
            verification=verification is not None,
        )

        return ImagingDocumentResponse(
            folder_id=folder_id,
            manifest=manifest,
            verification=verification,
            verifications=[verification] if verification else [],
            pages=merged,
            sectionsProcessed=sections,
        )

    def _sections_from_db(self, folder_id: str) -> dict[str, bool]:
        """Which imaging sections have any result rows for this chart.

        One round-trip with EXISTS (chart_id resolved once) instead of up to ten
        sequential queries — the previous loop dominated get_imaging latency.
        """
        flags = {
            "member": False,
            "dos": False,
            "hw": False,
            "quality": False,
            "rotation": False,
            "junk": False,
            "codeable": False,
            "encounter": False,
            "sequencing": False,
        }
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM chart_list WHERE chart_name = %s LIMIT 1",
                        (folder_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return flags
                    chart_id = row[0]
                    cur.execute(
                        """
                        SELECT
                          EXISTS(
                            SELECT 1 FROM member_extraction_results
                             WHERE chart_id = %s LIMIT 1
                          )
                          OR EXISTS(
                            SELECT 1 FROM member_verification_summary
                             WHERE chart_id = %s LIMIT 1
                          ),
                          EXISTS(
                            SELECT 1 FROM dos_extraction_results
                             WHERE chart_id = %s LIMIT 1
                          ),
                          EXISTS(
                            SELECT 1 FROM ocr_quality_results
                             WHERE chart_id = %s LIMIT 1
                          ),
                          EXISTS(
                            SELECT 1 FROM blank_junk_classification
                             WHERE chart_id = %s LIMIT 1
                          ),
                          EXISTS(
                            SELECT 1 FROM page_classification
                             WHERE chart_id = %s LIMIT 1
                          ),
                          EXISTS(
                            SELECT 1 FROM encounter_type_results
                             WHERE chart_id = %s LIMIT 1
                          )
                          -- Unresolved visits write no row, so a finished
                          -- stage counts even when every page is unresolved.
                          OR EXISTS(
                            SELECT 1 FROM page_stage_status
                             WHERE chart_id = %s
                               AND stage_name = 'encounter_type'
                               AND status IN ('completed', 'skipped')
                             LIMIT 1
                          ),
                          EXISTS(
                            SELECT 1 FROM page_sequencing_results
                             WHERE chart_id = %s LIMIT 1
                          )
                        """,
                        (chart_id,) * 9,
                    )
                    hit = cur.fetchone()
                    if not hit:
                        return flags
                    (
                        member,
                        dos,
                        quality_hw_rot,
                        junk,
                        codeable,
                        encounter,
                        sequencing,
                    ) = hit
                    flags["member"] = bool(member)
                    flags["dos"] = bool(dos)
                    # hw / quality / rotation all come from ocr_quality_results
                    q = bool(quality_hw_rot)
                    flags["hw"] = q
                    flags["quality"] = q
                    flags["rotation"] = q
                    flags["junk"] = bool(junk)
                    flags["codeable"] = bool(codeable)
                    flags["encounter"] = bool(encounter)
                    flags["sequencing"] = bool(sequencing)
        except Exception:
            return flags
        return flags
