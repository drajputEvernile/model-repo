"""Stage: manifest verification of the extracted member details.

Runs the ported V1 engine (``stages/lib/member``) — the verification rules, the
wrong-member check, and the what-if Accept/Reject threshold. See
``stages/lib/member/engine.py`` for the algorithm and how it maps to the
reference.

The member name, DOB and ID are not extracted here: the key/value extraction stage
(``stages/lib/extraction``) read every page once, right after OCR, and staged them.
This stage is responsible for the plumbing around the engine:

  * pick the manifest row for the chart (by record_id, so a manifest swept
    before ingest still counts),
  * choose which pages to feed it (blank/junk/duplicate pages are excluded),
  * feed it each page's staged extraction,
  * persist page rows, the summary, and the two CSVs the review UI reads.

``wrong_member_on_page`` decides from the member names the extraction found on a
page, so a page the extraction could not read (no word boxes) can be neither
verified nor wrong_member.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

from db import (
    connect,
    get_blank_junk_flags,
    list_manifest_members,
    source_record_id,
    upsert_member_extraction,
    upsert_member_summary,
    upsert_manifest_member,
)
from db.paths import imaging_csv, write_csv
from stages._support import (
    BJ_EXCLUDE,
    mark_completed,
    mark_failed,
    mark_processing,
    mark_skipped,
    stage_run,
)

from stages.lib.extraction.stage import ensure_staging
from stages.lib.member import (
    DETECTION_SOURCE_DB,
    detect_name_mode,
    expected_from_manifest,
    page_result_to_v1_row,
    summary_status,
    verify_record,
)

logger = logging.getLogger(__name__)

STAGE = "member_verify"

MEMBER_EXTRACT_COLS = [
    "chart_name",
    "page_name",
    "page_number",
    "extracted_name",
    "extracted_dob",
    "extracted_member_id",
    "detection_source_name",
    "detection_source_dob",
    "detection_source_member_id",
    "ner_key_source_name",
    "ner_key_source_dob",
    "ner_key_source_member_id",
    "page_status",
    "page_verified",
    "confidence",
    "provided_name",
    "provided_dob",
    "provided_external_member_id",
    "matched",
    "ner_enabled",
]

MEMBER_SUMMARY_COLS = [
    "chart_name",
    "final_status",
    "document_decision",
    "matched_name",
    "name_mode",
    "confidence",
    "pages_checked",
    "pages_matched",
    "wrong_member_pages",
    "reject_threshold",
    "decision_reason",
    "ner_enabled",
]

# V1 reports "N/A" for a field it did not find; the DB stores NULL.
NA = {"", "n/a", "na", "none"}


def _clean(value: Optional[str]) -> Optional[str]:
    text = (value or "").strip()
    return None if text.casefold() in NA else text


def _iso_dob(value: Optional[str]) -> Optional[str]:
    """V1 emits DummyDOB as MM/DD/YYYY; the DB column is DATE."""
    text = _clean(value)
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _page_confidence(page: Any) -> float:
    """A transparent score from how many fields were found and how.

    The reference carried no numeric confidence — it reported the detection
    source per field instead, which is what the review UI actually shows. This
    keeps the column populated without inventing a model score: 0.5 for a
    verified page plus 1/6 per field found, rules weighted above NER.
    """
    score = 0.5 if page.page_verified else 0.1
    for value, source in (
        (page.detected_name, page.detection_source_name),
        (page.detected_dob, page.detection_source_dob),
        (page.detected_member_id, page.detection_source_member_id),
    ):
        if not _clean(value):
            continue
        score += 0.1667 if source == "rule based" else 0.1
    return round(min(score, 0.99), 4)


def _pick_manifest_row(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The manifest row to verify against.

    A batch file carries one row per chart, so normally there is exactly one.
    When a chart has several candidates, prefer the most complete one — a row
    with a MemberID and a DOB gives the rules the most to match on.
    """
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]

    def completeness(row: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            1 if (row.get("external_member_id") or "").strip() else 0,
            1 if row.get("member_dob") else 0,
            1 if (row.get("middle_name") or "").strip() else 0,
            -int(row.get("id") or 0),
        )

    return max(rows, key=completeness)


def run(chart_id: int, *, force: bool = False) -> dict[str, Any]:
    with stage_run(chart_id, STAGE, force=force) as ctx:
        chart_name = ctx.chart_name

        with connect() as conn:
            manifest_rows = list_manifest_members(conn, chart_id=chart_id)
            bj_flags = get_blank_junk_flags(conn, chart_id, final_only=True)

        # Every page blank/junk/duplicate → nothing to verify; skip and complete.
        all_blank_junk = bool(ctx.pages) and all(
            bj_flags.get(p["id"], "not_blank_junk") in BJ_EXCLUDE for p in ctx.pages
        )
        if all_blank_junk:
            logger.info(
                "chart %s: all %d page(s) blank/junk/duplicate — member stage skipped",
                chart_id,
                len(ctx.pages),
            )
            with connect() as conn:
                mark_skipped(conn, ctx, sorted(ctx.todo), "blank_junk")
                # final_status is triage (verified|failed|needs_review|skipped),
                # not chart_list.status. All-junk → skipped; chart still completes.
                upsert_member_summary(
                    conn,
                    chart_id=chart_id,
                    final_status="skipped",
                    document_decision=None,
                    matched_member_list_id=None,
                    matched_name=None,
                    name_mode=None,
                    confidence=None,
                    pages_checked=0,
                    pages_matched=0,
                    wrong_member_pages=0,
                    reject_threshold=None,
                    decision_reason="all_blank_junk",
                )
            write_csv(
                imaging_csv(chart_name, "member_extraction"),
                MEMBER_EXTRACT_COLS,
                [],
            )
            write_csv(
                imaging_csv(chart_name, "member_verification"),
                MEMBER_SUMMARY_COLS,
                [
                    {
                        "chart_name": chart_name,
                        "final_status": "skipped",
                        "decision_reason": "all_blank_junk",
                        "ner_enabled": True,
                        "pages_checked": 0,
                        "pages_matched": 0,
                    }
                ],
            )
            return {
                "chart_id": chart_id,
                "status": "completed",
                "reason": "all_blank_junk",
                "pages_checked": 0,
                "skipped": ctx.skipped,
            }

        if not manifest_rows:
            # Test mode / empty MemoryStore / unswept Postgres: same CSVs
            # review-ui Local Mode reads under data/metadata or folders/manifest.
            from jobs.manifest_sweeper import lookup_manifest_members_on_disk

            rid = source_record_id(chart_name)
            disk_rows = lookup_manifest_members_on_disk(rid)
            if not disk_rows and rid != chart_name:
                disk_rows = lookup_manifest_members_on_disk(chart_name)
            if disk_rows:
                logger.info(
                    "chart %s: loaded %d manifest row(s) from disk for record_id=%s",
                    chart_id,
                    len(disk_rows),
                    rid,
                )
                # Seed the in-memory / DB store so later lookups (and review-ui
                # Production Mode) see the same row without re-scanning CSVs.
                with connect() as conn:
                    for row in disk_rows:
                        try:
                            upsert_manifest_member(
                                conn,
                                record_id=str(row.get("record_id") or rid),
                                member_name=str(row.get("member_name") or ""),
                                first_name=row.get("first_name"),
                                middle_name=row.get("middle_name"),
                                last_name=row.get("last_name"),
                                member_dob=row.get("member_dob"),
                                external_member_id=row.get("external_member_id"),
                                run_id=row.get("run_id"),
                                batch_id=row.get("batch_id"),
                                source_file=row.get("source_file"),
                                source_path=row.get("source_path"),
                            )
                        except Exception:
                            logger.exception(
                                "Failed to cache disk manifest row for %s", rid
                            )
                    manifest_rows = list_manifest_members(conn, chart_id=chart_id)
                if not manifest_rows:
                    # Upsert shape differed; use the disk dicts directly.
                    manifest_rows = disk_rows

        manifest = _pick_manifest_row(manifest_rows)
        if manifest is None:
            # No roster to verify against. Blank/junk pages are still skipped so
            # they do not leave the stage pending; remaining main pages wait on
            # a manifesto sweep.
            logger.warning(
                "chart %s: no manifest rows for record_id=%s; member stage cannot run",
                chart_id, chart_name,
            )
            excluded = [
                pid
                for pid in ctx.todo
                if bj_flags.get(pid, "not_blank_junk") in BJ_EXCLUDE
            ]
            with connect() as conn:
                mark_skipped(conn, ctx, excluded, "blank_junk")
                upsert_member_summary(
                    conn,
                    chart_id=chart_id,
                    final_status="needs_review",
                    document_decision=None,
                    matched_member_list_id=None,
                    matched_name=None,
                    name_mode=None,
                    confidence=None,
                    pages_checked=0,
                    pages_matched=0,
                    wrong_member_pages=0,
                    reject_threshold=None,
                    decision_reason="manifest_missing",
                )
            write_csv(
                imaging_csv(chart_name, "member_verification"),
                MEMBER_SUMMARY_COLS,
                [
                    {
                        "chart_name": chart_name,
                        "final_status": "needs_review",
                        "decision_reason": "manifest_missing",
                        "ner_enabled": True,
                    }
                ],
            )
            return {
                "chart_id": chart_id,
                "status": "needs_review",
                "reason": "manifest_missing",
                "pages_checked": 0,
                "skipped": ctx.skipped,
            }

        expected = expected_from_manifest(manifest)
        name_mode = detect_name_mode(expected)
        if not name_mode:
            logger.warning(
                "chart %s: manifest row %s has no usable first/last name; "
                "every page will fail verification",
                chart_id, manifest.get("id"),
            )

        # Blank / junk / duplicate pages carry no member details worth reading.
        page_map = ctx.page_map()
        excluded = [
            pid for pid in ctx.todo
            if bj_flags.get(pid, "not_blank_junk") in BJ_EXCLUDE
        ]
        with connect() as conn:
            mark_skipped(conn, ctx, excluded, "blank_junk")

        todo_pages = [p for p in ctx.pages if p["id"] in ctx.todo]
        if not todo_pages:
            # Nothing left to verify. An empty page list is Reject in the
            # what-if rules, which stored final_status=failed for a chart
            # whose only page was already blank/junk.
            logger.info(
                "chart %s: no pages left to verify — member stage skipped",
                chart_id,
            )
            with connect() as conn:
                upsert_member_summary(
                    conn,
                    chart_id=chart_id,
                    final_status="skipped",
                    document_decision=None,
                    matched_member_list_id=None,
                    matched_name=None,
                    name_mode=None,
                    confidence=None,
                    pages_checked=0,
                    pages_matched=0,
                    wrong_member_pages=0,
                    reject_threshold=None,
                    decision_reason="all_blank_junk",
                )
            write_csv(
                imaging_csv(chart_name, "member_verification"),
                MEMBER_SUMMARY_COLS,
                [
                    {
                        "chart_name": chart_name,
                        "final_status": "skipped",
                        "decision_reason": "all_blank_junk",
                        "ner_enabled": True,
                        "pages_checked": 0,
                        "pages_matched": 0,
                    }
                ],
            )
            return {
                "chart_id": chart_id,
                "status": "completed",
                "reason": "all_blank_junk",
                "pages_checked": 0,
                "skipped": ctx.skipped,
            }

        # What the extraction found on each page. A page it could not read has no entry.
        staged = ensure_staging(chart_id, chart_name)
        engine_pages = [
            {
                "page_no": p.get("page_number") or 0,
                "page_name": p["page_name"],
                "staged": staged.page(p["page_name"]),
                "_page_id": p["id"],
            }
            for p in todo_pages
        ]

        with connect() as conn:
            for page in todo_pages:
                mark_processing(conn, ctx, page["id"])

        # The reject threshold is a proportion of the whole document, so the
        # engine gets the chart's real page count, not just the subset here.
        result = verify_record(
            record_id=chart_name,
            pages=engine_pages,
            expected=expected,
            name_mode=name_mode,
            model_id=staged.model_version,
            total_pages=len(ctx.pages),
        )

        by_page_no = {p["page_no"]: p["_page_id"] for p in engine_pages}
        provided_dob = manifest.get("member_dob")
        if isinstance(provided_dob, date):
            provided_dob = provided_dob.isoformat()

        csv_rows: list[dict[str, Any]] = []
        with connect() as conn:
            for page in result.pages:
                page_id = by_page_no.get(page.page_no)
                if page_id is None:
                    continue
                row = page_map.get(page_id, {})
                try:
                    upsert_member_extraction(
                        conn,
                        chart_id=chart_id,
                        page_id=page_id,
                        extracted_name=_clean(page.detected_name),
                        extracted_dob=_iso_dob(page.detected_dob),
                        extracted_member_id=_clean(page.detected_member_id),
                        detection_source_name=DETECTION_SOURCE_DB.get(
                            page.detection_source_name, ""
                        ),
                        detection_source_dob=DETECTION_SOURCE_DB.get(
                            page.detection_source_dob, ""
                        ),
                        detection_source_member_id=DETECTION_SOURCE_DB.get(
                            page.detection_source_member_id, ""
                        ),
                        ner_key_source_name=_clean(page.ner_key_source_name),
                        ner_key_source_dob=_clean(page.ner_key_source_dob),
                        ner_key_source_member_id=_clean(page.ner_key_source_member_id),
                        page_status=page.db_page_status,
                        page_verified=page.page_verified,
                        confidence=_page_confidence(page),
                        provided_name=manifest.get("member_name"),
                        provided_dob=provided_dob,
                        provided_external_member_id=manifest.get("external_member_id"),
                        matched_member_list_id=(
                            manifest.get("id") if page.page_verified else None
                        ),
                    )
                    mark_completed(conn, ctx, page_id)
                except Exception as exc:  # one bad page must not sink the chart
                    logger.exception("member persist failed for %s", page.page_name)
                    mark_failed(conn, ctx, page_id, str(exc), page.page_name)
                    continue

                csv_rows.append(
                    {
                        "chart_name": chart_name,
                        "page_name": page.page_name,
                        "page_number": row.get("page_number") or page.page_no,
                        "extracted_name": _clean(page.detected_name) or "",
                        "extracted_dob": _clean(page.detected_dob) or "",
                        "extracted_member_id": _clean(page.detected_member_id) or "",
                        "detection_source_name": page.detection_source_name,
                        "detection_source_dob": page.detection_source_dob,
                        "detection_source_member_id": page.detection_source_member_id,
                        "ner_key_source_name": page.ner_key_source_name,
                        "ner_key_source_dob": page.ner_key_source_dob,
                        "ner_key_source_member_id": page.ner_key_source_member_id,
                        "page_status": page.db_page_status,
                        "page_verified": page.page_verified,
                        "confidence": _page_confidence(page),
                        "provided_name": manifest.get("member_name") or "",
                        "provided_dob": provided_dob or "",
                        "provided_external_member_id": manifest.get("external_member_id") or "",
                        "matched": page.page_verified,
                        "ner_enabled": result.ner_enabled,
                    }
                )

        final_status, reason = summary_status(result)

        with connect() as conn:
            upsert_member_summary(
                conn,
                chart_id=chart_id,
                final_status=final_status,
                document_decision=result.db_document_decision,
                matched_member_list_id=manifest.get("id"),
                matched_name=manifest.get("member_name"),
                name_mode=result.name_mode or None,
                confidence=(
                    round(result.pages_verified / result.pages_checked, 4)
                    if result.pages_checked
                    else None
                ),
                pages_checked=result.pages_checked,
                pages_matched=result.pages_verified,
                wrong_member_pages=result.pages_wrong_member,
                reject_threshold=result.reject_threshold,
                decision_reason=reason[:50],
            )

        write_csv(imaging_csv(chart_name, "member_extraction"), MEMBER_EXTRACT_COLS, csv_rows)
        write_csv(
            imaging_csv(chart_name, "member_verification"),
            MEMBER_SUMMARY_COLS,
            [
                {
                    "chart_name": chart_name,
                    "final_status": final_status,
                    "document_decision": result.db_document_decision,
                    "matched_name": manifest.get("member_name") or "",
                    "name_mode": result.name_mode,
                    "confidence": (
                        round(result.pages_verified / result.pages_checked, 4)
                        if result.pages_checked
                        else ""
                    ),
                    "pages_checked": result.pages_checked,
                    "pages_matched": result.pages_verified,
                    "wrong_member_pages": result.pages_wrong_member,
                    "reject_threshold": result.reject_threshold,
                    "decision_reason": reason,
                    "ner_enabled": result.ner_enabled,
                }
            ],
        )

        # The reference's own CSV shape, for diffing a port against a V1 run.
        write_csv(
            imaging_csv(chart_name, "member_v1_compare"),
            [
                "RecordId", "Total_Page_Count", "Page_No",
                "Detection_Source_Name", "ner_key_source_Name", "Detected_Full_Name",
                "Detection_Source_DOB", "ner_key_source_DOB", "Detected_DOB",
                "Detection_Source_MemberID", "ner_key_source_MemberID", "Detected_MemberID",
                "Page_Verified", "Document_Verified",
                "Page_Detection_Correct", "Page_Detection_InCorrect",
            ],
            [page_result_to_v1_row(result, p) for p in result.pages],
        )

        return {
            "chart_id": chart_id,
            "final_status": final_status,
            "document_decision": result.db_document_decision,
            "name_mode": result.name_mode,
            "pages_checked": result.pages_checked,
            "pages_verified": result.pages_verified,
            "pages_wrong_member": result.pages_wrong_member,
            "reject_threshold": result.reject_threshold,
            "ner_enabled": result.ner_enabled,
            "decision_reason": reason,
        }
