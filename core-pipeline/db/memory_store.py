"""In-process store for ``--test-mode`` / ``--skip-db-write`` local runs (no Postgres).

Implements the same operations as ``db`` helpers so stages and CSV rebuilds
keep working. State lives only for the process lifetime.

Workspace charts are stored under ``data/folders/<chart_name>-test`` so a
test run never overwrites a production chart folder.
"""
from __future__ import annotations

import copy
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

TEST_CHART_SUFFIX = "-test"


def test_chart_name(name: str) -> str:
    """Append ``-test`` so workspace lands under ``data/folders/<name>-test``."""
    n = (name or "").strip()
    if not n:
        return n
    if n.endswith(TEST_CHART_SUFFIX):
        return n
    return f"{n}{TEST_CHART_SUFFIX}"


def source_record_id(chart_name: str) -> str:
    """Manifest ``record_id`` for a chart — strips the test-mode ``-test`` suffix.

    Manifest CSVs key on the real chart / RecordId. Test-mode workspaces are
    named ``<chart>-test``, so member verify must look up without the suffix.
    """
    n = (chart_name or "").strip()
    if n.endswith(TEST_CHART_SUFFIX) and len(n) > len(TEST_CHART_SUFFIX):
        return n[: -len(TEST_CHART_SUFFIX)]
    return n


# Seeded to match schema/v1.sql pipeline_stage (phase-1 rows).
DEFAULT_PIPELINE_STAGES: list[dict[str, Any]] = [
    {"stage_name": "ocr_quality", "pass_no": 1, "seq": 10,
     "label": "Rotation + Quality + Handwriting", "is_phase1": True},
    {"stage_name": "ocr_prelim", "pass_no": 1, "seq": 20,
     "label": "Preliminary OCR (Tesseract)", "is_phase1": True},
    {"stage_name": "blank_junk", "pass_no": 1, "seq": 30,
     "label": "Blank/Junk/Duplicate — pass 1", "is_phase1": True},
    {"stage_name": "ocr_final1", "pass_no": 1, "seq": 40,
     "label": "Final OCR 1 (RapidOCR)", "is_phase1": True},
    {"stage_name": "ocr_final2", "pass_no": 1, "seq": 50,
     "label": "Final OCR 2 (Azure DocIntel)", "is_phase1": True},
    {"stage_name": "kv_extract", "pass_no": 1, "seq": 55,
     "label": "Key/Value Extraction", "is_phase1": True},
    {"stage_name": "blank_junk", "pass_no": 2, "seq": 60,
     "label": "Blank/Junk/Duplicate — pass 2", "is_phase1": True},
    {"stage_name": "member_verify", "pass_no": 1, "seq": 70,
     "label": "Member Extraction + Verify", "is_phase1": True},
    {"stage_name": "dos_extract", "pass_no": 1, "seq": 80,
     "label": "Date-of-Service Extraction", "is_phase1": True},
    {"stage_name": "page_subtype", "pass_no": 1, "seq": 85,
     "label": "Codeable / Non Codeable (TF)", "is_phase1": True},
    {"stage_name": "encounter_type", "pass_no": 1, "seq": 90,
     "label": "Encounter Type (TF)", "is_phase1": True},
    {"stage_name": "page_sequencing", "pass_no": 1, "seq": 95,
     "label": "Page Sequencing", "is_phase1": True},
]

DONE_STATUSES = frozenset({"completed", "skipped"})
PAGE_STAGE_STATUSES = ("pending", "processing", "completed", "failed", "skipped")

CHART_RESULT_KEYS = (
    "member_summaries",
    "member_extractions",
    "dos",
    "page_classifications",
    "encounters",
    "sequencing",
    "blank_junk",
    "quality",
    "ocr",
    "page_stages",
)

_lock = threading.RLock()
_enabled = False
_store: Optional["MemoryStore"] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def is_skip_db_write() -> bool:
    with _lock:
        return (
            _enabled
            or _truthy_env("SKIP_DB_WRITE")
            or _truthy_env("TEST_MODE")
        )

def enable_skip_db_write(*, reset: bool = True) -> "MemoryStore":
    """Activate the in-memory backend. Never opens Postgres."""
    global _enabled, _store
    with _lock:
        _enabled = True
        if reset or _store is None:
            _store = MemoryStore()
        return _store


def disable_skip_db_write() -> None:
    global _enabled, _store
    with _lock:
        _enabled = False
        _store = None


def get_memory_store() -> "MemoryStore":
    global _store
    with _lock:
        if _store is None:
            _store = MemoryStore()
        return _store


def _confidence_level(confidence: Optional[float]) -> Optional[str]:
    if confidence is None:
        return None
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


class MemoryStore:
    """Process-local stand-in for the V1 working-set tables."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_id = 1
        self.pipeline_stages = [dict(s) for s in DEFAULT_PIPELINE_STAGES]
        self.charts: dict[int, dict[str, Any]] = {}
        self.charts_by_name: dict[str, int] = {}
        self.pages: dict[int, dict[str, Any]] = {}  # page_id -> row
        self.pages_by_chart: dict[int, list[int]] = {}
        self.page_stages: dict[tuple[int, str, int], dict[str, Any]] = {}
        self.jobs: dict[int, dict[str, Any]] = {}
        self.ocr: dict[tuple[int, str], dict[str, Any]] = {}  # (page_id, type)
        self.quality: dict[int, dict[str, Any]] = {}  # page_id
        self.blank_junk: dict[tuple[int, int], dict[str, Any]] = {}  # (page_id, pass)
        self.dos: dict[int, dict[str, Any]] = {}
        self.page_classifications: dict[int, dict[str, Any]] = {}
        self.encounters: dict[int, dict[str, Any]] = {}
        self.sequencing: dict[int, dict[str, Any]] = {}
        self.member_extractions: dict[int, dict[str, Any]] = {}
        self.member_summaries: dict[int, dict[str, Any]] = {}
        self.manifest_members: dict[int, dict[str, Any]] = {}
        self.manifest_by_record: dict[str, list[int]] = {}

    def _alloc(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _copy(self, row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return copy.deepcopy(row) if row is not None else None

    # -- stages --------------------------------------------------------------

    def list_stages(self, *, phase1_only: bool = True) -> list[dict[str, Any]]:
        rows = self.pipeline_stages
        if phase1_only:
            rows = [r for r in rows if r.get("is_phase1")]
        return [dict(r) for r in sorted(rows, key=lambda r: r["seq"])]

    # -- charts / pages ------------------------------------------------------

    def upsert_chart(
        self,
        *,
        chart_name: str,
        page_count: Optional[int] = None,
        status: Optional[str] = None,
        current_stage: Optional[str] = None,
        current_pass: Optional[int] = None,
        source: Optional[str] = None,
        blob_container: Optional[str] = None,
        blob_path: Optional[str] = None,
        output_path: Optional[str] = None,
        run_id: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            existing_id = self.charts_by_name.get(chart_name)
            if existing_id is None:
                cid = self._alloc()
                row = {
                    "id": cid,
                    "chart_name": chart_name,
                    "page_count": page_count,
                    "status": status or "received",
                    "current_stage": current_stage,
                    "current_pass": current_pass,
                    "source": source or "blob",
                    "blob_container": blob_container,
                    "blob_path": blob_path,
                    "output_path": output_path,
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "created_at": _now(),
                    "updated_at": _now(),
                }
                self.charts[cid] = row
                self.charts_by_name[chart_name] = cid
                self.pages_by_chart.setdefault(cid, [])
                return self._copy(row)  # type: ignore[return-value]

            row = self.charts[existing_id]
            if page_count is not None:
                row["page_count"] = page_count
            if status is not None:
                row["status"] = status
            if current_stage is not None:
                row["current_stage"] = current_stage
            if current_pass is not None:
                row["current_pass"] = current_pass
            new_source = source or row["source"]
            if row["source"] == "manifest" and new_source != "manifest":
                row["source"] = new_source
            elif row["source"] != "manifest" and source is not None:
                row["source"] = source
            if blob_container is not None:
                row["blob_container"] = blob_container
            if blob_path is not None:
                row["blob_path"] = blob_path
            if output_path is not None:
                row["output_path"] = output_path
            if run_id is not None:
                row["run_id"] = run_id
            if batch_id is not None:
                row["batch_id"] = batch_id
            row["updated_at"] = _now()
            return self._copy(row)  # type: ignore[return-value]

    def set_chart_output_path(self, chart_id: int, output_path: str) -> None:
        with self._lock:
            row = self.charts.get(chart_id)
            if row:
                row["output_path"] = output_path
                row["updated_at"] = _now()

    def set_chart_status(
        self,
        chart_id: int,
        status: str,
        *,
        current_stage: Optional[str] = None,
        current_pass: Optional[int] = None,
    ) -> None:
        with self._lock:
            row = self.charts.get(chart_id)
            if not row:
                return
            row["status"] = status
            row["current_stage"] = current_stage
            row["current_pass"] = current_pass
            row["updated_at"] = _now()

    def get_chart(self, chart_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._copy(self.charts.get(chart_id))

    def get_chart_by_name(self, chart_name: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cid = self.charts_by_name.get(chart_name)
            return self._copy(self.charts.get(cid)) if cid else None

    def upsert_pages(
        self, chart_id: int, pages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not pages:
            return self.list_pages(chart_id)
        with self._lock:
            by_name = {
                self.pages[pid]["page_name"]: pid
                for pid in self.pages_by_chart.get(chart_id, [])
                if pid in self.pages
            }
            for page in pages:
                page_name = page["page_name"]
                image_path = page.get("image_path") or f"pages/{page_name}"
                use_corrected = bool(page.get("use_corrected", False))
                existing = by_name.get(page_name)
                if existing is None:
                    pid = self._alloc()
                    row = {
                        "id": pid,
                        "chart_id": chart_id,
                        "page_name": page_name,
                        "page_number": page.get("page_number"),
                        "image_sha256": page.get("image_sha256"),
                        "file_size_bytes": page.get("file_size_bytes"),
                        "use_corrected": use_corrected,
                        "image_path": image_path,
                        "created_at": _now(),
                        "updated_at": _now(),
                    }
                    self.pages[pid] = row
                    self.pages_by_chart.setdefault(chart_id, []).append(pid)
                else:
                    row = self.pages[existing]
                    row["page_number"] = page.get("page_number")
                    if page.get("image_sha256") is not None:
                        row["image_sha256"] = page.get("image_sha256")
                    if page.get("file_size_bytes") is not None:
                        row["file_size_bytes"] = page.get("file_size_bytes")
                    if not row.get("image_path"):
                        row["image_path"] = image_path
                    row["updated_at"] = _now()
        return self.list_pages(chart_id)

    def set_page_image_source(
        self, page_id: int, *, use_corrected: bool, image_path: str
    ) -> None:
        with self._lock:
            row = self.pages.get(page_id)
            if row:
                row["use_corrected"] = bool(use_corrected)
                row["image_path"] = image_path
                row["updated_at"] = _now()

    def list_pages(self, chart_id: int) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self.pages_by_chart.get(chart_id, []))
            rows = [self.pages[pid] for pid in ids if pid in self.pages]

            def sort_key(r: dict[str, Any]) -> tuple[Any, str]:
                num = r.get("page_number")
                return (num is None, num if num is not None else 0, r["page_name"])

            rows.sort(key=sort_key)
            return [self._copy(r) for r in rows]  # type: ignore[misc]

    # -- page_stage_status ---------------------------------------------------

    def init_page_stages(
        self,
        chart_id: int,
        *,
        stages: Optional[Sequence[tuple[str, int]]] = None,
    ) -> int:
        if stages is None:
            stages = [(s["stage_name"], s["pass_no"]) for s in self.list_stages()]
        count = 0
        with self._lock:
            for pid in self.pages_by_chart.get(chart_id, []):
                for stage_name, pass_no in stages:
                    key = (pid, stage_name, pass_no)
                    if key in self.page_stages:
                        continue
                    self.page_stages[key] = {
                        "chart_id": chart_id,
                        "page_id": pid,
                        "stage_name": stage_name,
                        "pass_no": pass_no,
                        "status": "pending",
                        "attempt": 0,
                        "error_message": None,
                        "skip_reason": None,
                        "started_at": None,
                        "completed_at": None,
                    }
                    count += 1
        return count

    def set_page_stage(
        self,
        *,
        chart_id: int,
        page_id: int,
        stage_name: str,
        status: str,
        pass_no: int = 1,
        error_message: Optional[str] = None,
        skip_reason: Optional[str] = None,
    ) -> None:
        if status not in PAGE_STAGE_STATUSES:
            raise ValueError(f"Invalid page stage status: {status}")
        with self._lock:
            key = (page_id, stage_name, pass_no)
            prev = self.page_stages.get(key)
            attempt = int((prev or {}).get("attempt") or 0)
            if status == "processing":
                attempt += 1
            started = (prev or {}).get("started_at")
            completed = (prev or {}).get("completed_at")
            if status == "processing":
                started = _now()
            if status in ("completed", "failed", "skipped"):
                completed = _now()
            self.page_stages[key] = {
                "chart_id": chart_id,
                "page_id": page_id,
                "stage_name": stage_name,
                "pass_no": pass_no,
                "status": status,
                "attempt": attempt,
                "error_message": error_message,
                "skip_reason": skip_reason,
                "started_at": started,
                "completed_at": completed,
            }

    def set_pages_stage(
        self,
        *,
        chart_id: int,
        page_ids: Sequence[int],
        stage_name: str,
        status: str,
        pass_no: int = 1,
        skip_reason: Optional[str] = None,
    ) -> None:
        for pid in page_ids:
            self.set_page_stage(
                chart_id=chart_id,
                page_id=int(pid),
                stage_name=stage_name,
                status=status,
                pass_no=pass_no,
                skip_reason=skip_reason,
            )

    def get_stage_status_map(
        self, chart_id: int, stage_name: str, pass_no: int = 1
    ) -> dict[int, str]:
        with self._lock:
            out: dict[int, str] = {}
            for (pid, sn, pn), row in self.page_stages.items():
                if row["chart_id"] == chart_id and sn == stage_name and pn == pass_no:
                    out[pid] = row["status"]
            return out

    def pages_needing_stage(
        self,
        chart_id: int,
        stage_name: str,
        pass_no: int = 1,
        *,
        force: bool = False,
    ) -> set[int]:
        pages = self.list_pages(chart_id)
        if force:
            return {int(p["id"]) for p in pages}
        status_map = self.get_stage_status_map(chart_id, stage_name, pass_no)
        return {
            int(p["id"])
            for p in pages
            if status_map.get(int(p["id"])) not in DONE_STATUSES
        }

    def reset_stage(
        self, chart_id: int, stage_name: str, pass_no: int = 1
    ) -> None:
        with self._lock:
            for key, row in list(self.page_stages.items()):
                if (
                    row["chart_id"] == chart_id
                    and row["stage_name"] == stage_name
                    and row["pass_no"] == pass_no
                ):
                    row["status"] = "pending"
                    row["error_message"] = None
                    row["skip_reason"] = None
                    row["completed_at"] = None

    def reset_pages_stage(
        self,
        chart_id: int,
        page_ids: Sequence[int],
        stage_name: str,
        pass_no: int = 1,
    ) -> int:
        n = 0
        for pid in page_ids:
            self.set_page_stage(
                chart_id=chart_id,
                page_id=int(pid),
                stage_name=stage_name,
                status="pending",
                pass_no=pass_no,
                error_message=None,
                skip_reason=None,
            )
            with self._lock:
                key = (int(pid), stage_name, pass_no)
                if key in self.page_stages:
                    self.page_stages[key]["completed_at"] = None
            n += 1
        return n

    def clear_stuck_processing(self, chart_id: int) -> int:
        n = 0
        with self._lock:
            for row in self.page_stages.values():
                if row["chart_id"] == chart_id and row["status"] == "processing":
                    row["status"] = "pending"
                    row["error_message"] = None
                    row["skip_reason"] = None
                    row["completed_at"] = None
                    n += 1
        return n

    # -- jobs ----------------------------------------------------------------

    def create_job(
        self,
        *,
        chart_id: Optional[int],
        stage_name: str,
        status: str = "queued",
        pass_no: int = 1,
        queue_name: str = "default",
        worker_id: Optional[str] = None,
        pages_total: Optional[int] = None,
    ) -> int:
        with self._lock:
            jid = self._alloc()
            self.jobs[jid] = {
                "id": jid,
                "chart_id": chart_id,
                "stage_name": stage_name,
                "pass_no": pass_no,
                "status": status,
                "queue_name": queue_name,
                "worker_id": worker_id,
                "pages_total": pages_total,
                "pages_done": None,
                "pages_failed": None,
                "pages_skipped": None,
                "error_message": None,
                "started_at": None,
                "completed_at": None,
                "heartbeat_at": None,
                "lease_expires_at": None,
            }
            return jid

    def update_job(
        self,
        job_id: int,
        *,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        started: bool = False,
        completed: bool = False,
        chart_id: Optional[int] = None,
        pages_total: Optional[int] = None,
        pages_done: Optional[int] = None,
        pages_failed: Optional[int] = None,
        pages_skipped: Optional[int] = None,
    ) -> None:
        with self._lock:
            row = self.jobs.get(job_id)
            if not row:
                return
            if status:
                row["status"] = status
            if error_message is not None:
                row["error_message"] = error_message
            if chart_id is not None:
                row["chart_id"] = chart_id
            for col, val in (
                ("pages_total", pages_total),
                ("pages_done", pages_done),
                ("pages_failed", pages_failed),
                ("pages_skipped", pages_skipped),
            ):
                if val is not None:
                    row[col] = val
            if started:
                row["started_at"] = _now()
                row["heartbeat_at"] = _now()
            if completed:
                row["completed_at"] = _now()

    def heartbeat_job(self, job_id: int, *, lease_seconds: int = 300) -> None:
        with self._lock:
            row = self.jobs.get(job_id)
            if row:
                row["heartbeat_at"] = _now()

    # -- OCR / quality -------------------------------------------------------

    def upsert_ocr_result(
        self,
        *,
        chart_id: int,
        page_id: int,
        ocr_type: str,
        raw_text: Optional[str],
    ) -> None:
        import hashlib

        text = raw_text or ""
        with self._lock:
            self.ocr[(page_id, ocr_type)] = {
                "chart_id": chart_id,
                "page_id": page_id,
                "ocr_type": ocr_type,
                "raw_text": raw_text,
                "char_count": len(text),
                "text_sha256": hashlib.sha256(
                    text.encode("utf-8", "replace")
                ).hexdigest(),
                "updated_at": _now(),
            }

    def get_ocr_texts(self, chart_id: int, ocr_type: str) -> dict[int, str]:
        with self._lock:
            return {
                pid: (row.get("raw_text") or "")
                for (pid, ot), row in self.ocr.items()
                if row["chart_id"] == chart_id and ot == ocr_type
            }

    def has_ocr_results(self, chart_id: int) -> bool:
        with self._lock:
            for row in self.ocr.values():
                if row["chart_id"] == chart_id and int(row.get("char_count") or 0) > 0:
                    return True
            return False

    def upsert_quality(
        self,
        *,
        chart_id: int,
        page_id: int,
        printed_or_handwritten: Optional[str],
        orientation_angle: Optional[float] = None,
        tilt_angle: Optional[float] = None,
        mirrored: Optional[bool] = None,
        rotation_applied: bool = False,
        hw_method: Optional[str] = None,
        hw_confidence: Optional[float] = None,
        quality_tag: Optional[str] = None,
        quality_score: Optional[float] = None,
        quality_detail: Optional[dict[str, Any]] = None,
        input_dpi: Optional[float] = None,
    ) -> None:
        with self._lock:
            self.quality[page_id] = {
                "chart_id": chart_id,
                "page_id": page_id,
                "quality_tag": quality_tag,
                "quality_score": quality_score,
                "quality_detail": quality_detail if quality_detail is not None else {},
                "input_dpi": input_dpi,
                "printed_or_handwritten": printed_or_handwritten,
                "hw_method": hw_method,
                "hw_confidence": hw_confidence,
                "orientation_angle": orientation_angle,
                "tilt_angle": tilt_angle,
                "mirrored": mirrored,
                "rotation_applied": rotation_applied,
                "updated_at": _now(),
            }

    def get_quality_map(self, chart_id: int) -> dict[int, dict[str, Any]]:
        with self._lock:
            return {
                pid: self._copy(row)  # type: ignore[misc]
                for pid, row in self.quality.items()
                if row["chart_id"] == chart_id
            }

    def handwritten_page_ids(self, chart_id: int) -> set[int]:
        return {
            pid
            for pid, row in self.get_quality_map(chart_id).items()
            if str(row.get("printed_or_handwritten") or "").lower() == "handwritten"
        }

    def non_printed_page_ids(self, chart_id: int) -> set[int]:
        return {
            pid
            for pid, row in self.get_quality_map(chart_id).items()
            if str(row.get("printed_or_handwritten") or "").lower()
            in {"handwritten", "uncertain", "mixed"}
        }

    def low_quality_page_ids(self, chart_id: int) -> set[int]:
        return {
            pid
            for pid, row in self.get_quality_map(chart_id).items()
            if str(row.get("quality_tag") or "").lower() == "low"
        }

    def high_quality_printed_page_ids(self, chart_id: int) -> set[int]:
        return {
            pid
            for pid, row in self.get_quality_map(chart_id).items()
            if str(row.get("quality_tag") or "").lower() == "high"
            and str(row.get("printed_or_handwritten") or "").lower() == "printed"
        }

    def list_quality_for_csv(self, chart_id: int) -> list[dict[str, Any]]:
        pages = {p["id"]: p for p in self.list_pages(chart_id)}
        rows: list[dict[str, Any]] = []
        with self._lock:
            for pid, q in self.quality.items():
                if q["chart_id"] != chart_id:
                    continue
                p = pages.get(pid)
                if not p:
                    continue
                rows.append(
                    {
                        "page_name": p["page_name"],
                        "page_number": p.get("page_number"),
                        "printed_or_handwritten": q.get("printed_or_handwritten"),
                        "orientation_angle": q.get("orientation_angle"),
                        "tilt_angle": q.get("tilt_angle"),
                        "mirrored": q.get("mirrored"),
                        "rotation_applied": q.get("rotation_applied"),
                        "hw_confidence": q.get("hw_confidence"),
                        "hw_method": q.get("hw_method"),
                        "quality_tag": q.get("quality_tag"),
                        "quality_score": q.get("quality_score"),
                        "input_dpi": q.get("input_dpi"),
                    }
                )

        def sort_key(r: dict[str, Any]) -> tuple[Any, str]:
            num = r.get("page_number")
            return (num is None, num if num is not None else 0, r["page_name"])

        rows.sort(key=sort_key)
        return rows

    # -- blank / junk --------------------------------------------------------

    def upsert_blank_junk(
        self,
        *,
        chart_id: int,
        page_id: int,
        blank_junk_flag: str,
        pass_no: int,
        ocr_source: str,
        junk_subtype: Optional[str] = None,
        duplicate_of_page_id: Optional[int] = None,
        confidence: Optional[float] = None,
        reason: Optional[str] = None,
        is_final: bool = False,
    ) -> None:
        with self._lock:
            key = (page_id, pass_no)
            prev = self.blank_junk.get(key)
            bid = (prev or {}).get("id") or self._alloc()
            self.blank_junk[key] = {
                "id": bid,
                "chart_id": chart_id,
                "page_id": page_id,
                "pass_no": pass_no,
                "blank_junk_flag": blank_junk_flag,
                "junk_subtype": junk_subtype,
                "duplicate_of_page_id": duplicate_of_page_id,
                "ocr_source": ocr_source,
                "confidence": confidence,
                "reason": reason,
                "is_final": bool(is_final),
                "updated_at": _now(),
            }

    def delete_blank_junk(
        self,
        chart_id: int,
        pass_no: int,
        page_ids: Sequence[int],
    ) -> None:
        if not page_ids:
            return
        ids = {int(pid) for pid in page_ids}
        with self._lock:
            for key, row in list(self.blank_junk.items()):
                if (
                    row["chart_id"] == chart_id
                    and int(row["pass_no"]) == int(pass_no)
                    and int(row["page_id"]) in ids
                ):
                    del self.blank_junk[key]

    def mark_blank_junk_final(self, chart_id: int) -> None:
        with self._lock:
            best: dict[int, tuple[int, datetime, int]] = {}
            for row in self.blank_junk.values():
                if row["chart_id"] != chart_id:
                    continue
                pid = row["page_id"]
                key = (int(row["pass_no"]), row["updated_at"] or _now(), int(row["id"]))
                if pid not in best or key > best[pid]:
                    best[pid] = key
            for row in self.blank_junk.values():
                if row["chart_id"] != chart_id:
                    continue
                pid = row["page_id"]
                key = (int(row["pass_no"]), row["updated_at"] or _now(), int(row["id"]))
                row["is_final"] = best.get(pid) == key

    def get_blank_junk_flags(
        self,
        chart_id: int,
        *,
        pass_no: Optional[int] = None,
        final_only: bool = False,
    ) -> dict[int, str]:
        if final_only:
            return {
                pid: row["blank_junk_flag"]
                for pid, row in self.get_blank_junk_final(chart_id).items()
            }
        with self._lock:
            if pass_no is not None:
                return {
                    row["page_id"]: row["blank_junk_flag"]
                    for row in self.blank_junk.values()
                    if row["chart_id"] == chart_id and row["pass_no"] == pass_no
                }
            best: dict[int, dict[str, Any]] = {}
            for row in self.blank_junk.values():
                if row["chart_id"] != chart_id:
                    continue
                pid = row["page_id"]
                prev = best.get(pid)
                if prev is None or (
                    row["pass_no"],
                    row["updated_at"] or _now(),
                ) > (prev["pass_no"], prev.get("updated_at") or _now()):
                    best[pid] = row
            return {pid: r["blank_junk_flag"] for pid, r in best.items()}

    def get_blank_junk_final(self, chart_id: int) -> dict[int, dict[str, Any]]:
        with self._lock:
            out: dict[int, dict[str, Any]] = {}
            for row in self.blank_junk.values():
                if row["chart_id"] == chart_id and row.get("is_final"):
                    out[int(row["page_id"])] = {
                        "page_id": row["page_id"],
                        "blank_junk_flag": row["blank_junk_flag"],
                        "junk_subtype": row.get("junk_subtype"),
                        "confidence": row.get("confidence"),
                    }
            if out:
                return out
            # Fall back to highest pass when mark_final has not run yet.
            best: dict[int, dict[str, Any]] = {}
            for row in self.blank_junk.values():
                if row["chart_id"] != chart_id:
                    continue
                pid = int(row["page_id"])
                prev = best.get(pid)
                if prev is None or row["pass_no"] > prev["pass_no"]:
                    best[pid] = row
            return {
                pid: {
                    "page_id": r["page_id"],
                    "blank_junk_flag": r["blank_junk_flag"],
                    "junk_subtype": r.get("junk_subtype"),
                    "confidence": r.get("confidence"),
                }
                for pid, r in best.items()
            }

    def list_blank_junk_for_csv(self, chart_id: int) -> list[dict[str, Any]]:
        chart = self.get_chart(chart_id) or {}
        pages = {p["id"]: p for p in self.list_pages(chart_id)}
        rows: list[dict[str, Any]] = []
        with self._lock:
            for row in self.blank_junk.values():
                if row["chart_id"] != chart_id:
                    continue
                p = pages.get(row["page_id"])
                if not p:
                    continue
                rows.append(
                    {
                        "chart_name": chart.get("chart_name"),
                        "page_name": p["page_name"],
                        "page_number": p.get("page_number"),
                        "blank_junk_flag": row["blank_junk_flag"],
                        "junk_subtype": row.get("junk_subtype"),
                        "confidence": row.get("confidence"),
                        "reason": row.get("reason"),
                        "ocr_source": row.get("ocr_source"),
                        "pass_no": row["pass_no"],
                        "is_final": row.get("is_final"),
                    }
                )

        def sort_key(r: dict[str, Any]) -> tuple[Any, str, int]:
            num = r.get("page_number")
            return (
                num is None,
                num if num is not None else 0,
                r["page_name"],
                int(r.get("pass_no") or 0),
            )

        rows.sort(key=sort_key)
        return rows

    # -- DOS / classification / encounter / sequencing -----------------------

    def upsert_dos(
        self,
        *,
        chart_id: int,
        page_id: int,
        date_of_service_from: Optional[str],
        date_of_service_to: Optional[str],
        date_of_service_from_doclevel: Optional[str],
        date_of_service_to_doclevel: Optional[str],
        confidence: Optional[float],
        all_dates: Optional[Sequence[dict[str, Any]]] = None,
        extraction_method: Optional[str] = None,
    ) -> None:
        dates = [
            {
                "seq": seq,
                "date_of_service_from": item.get("dos_from"),
                "date_of_service_to": item.get("dos_to"),
                "source_keyword": item.get("source_keyword"),
                "confidence": item.get("confidence"),
            }
            for seq, item in enumerate(all_dates or [], start=1)
        ]
        with self._lock:
            self.dos[page_id] = {
                "chart_id": chart_id,
                "page_id": page_id,
                "date_of_service_from": date_of_service_from,
                "date_of_service_to": date_of_service_to,
                "date_of_service_from_doclevel": date_of_service_from_doclevel,
                "date_of_service_to_doclevel": date_of_service_to_doclevel,
                "dates": dates,
                "extraction_method": extraction_method,
                "confidence": confidence,
                "updated_at": _now(),
            }

    def get_dos_map(self, chart_id: int) -> dict[int, dict[str, Any]]:
        with self._lock:
            return {
                pid: self._copy(row)  # type: ignore[misc]
                for pid, row in self.dos.items()
                if row["chart_id"] == chart_id
            }

    def upsert_page_classification(
        self,
        *,
        chart_id: int,
        page_id: int,
        page_subtype: Optional[str],
        classification_category: str,
        confidence: Optional[float] = None,
        duplicate_flag: bool = False,
    ) -> None:
        with self._lock:
            self.page_classifications[page_id] = {
                "chart_id": chart_id,
                "page_id": page_id,
                "page_subtype": (page_subtype or "")[:200] or None,
                "classification_category": classification_category,
                "duplicate_flag": bool(duplicate_flag),
                "confidence": confidence,
                "confidence_level": _confidence_level(confidence),
                "updated_at": _now(),
            }

    def upsert_encounter(
        self,
        *,
        chart_id: int,
        page_id: int,
        encounter_type: str,
        confidence: Optional[float] = None,
        matched_keyword: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.encounters[page_id] = {
                "chart_id": chart_id,
                "page_id": page_id,
                "encounter_type": encounter_type,
                "confidence": confidence,
                "matched_keyword": (matched_keyword or "")[:200] or None,
                "updated_at": _now(),
            }

    def delete_encounter(self, chart_id: int, page_ids: Sequence[int]) -> None:
        ids = {int(pid) for pid in page_ids}
        with self._lock:
            for page_id, row in list(self.encounters.items()):
                if row["chart_id"] == chart_id and int(page_id) in ids:
                    del self.encounters[page_id]

    def upsert_sequencing(
        self,
        *,
        chart_id: int,
        page_id: int,
        original_page_number: Optional[int],
        seq: Optional[int],
        confidence: Optional[float] = None,
        sequence_method: Optional[str] = None,
        review_flag: bool = False,
    ) -> None:
        with self._lock:
            self.sequencing[page_id] = {
                "chart_id": chart_id,
                "page_id": page_id,
                "original_page_number": original_page_number,
                "seq": seq,
                "confidence": confidence,
                "sequence_method": (sequence_method or "")[:64] or None,
                "review_flag": bool(review_flag),
                "updated_at": _now(),
            }

    # -- member --------------------------------------------------------------

    def upsert_member_extraction(self, *, chart_id: int, page_id: int, **fields: Any) -> None:
        with self._lock:
            row = {"chart_id": chart_id, "page_id": page_id, **fields, "updated_at": _now()}
            self.member_extractions[page_id] = row

    def upsert_member_summary(self, *, chart_id: int, **fields: Any) -> None:
        with self._lock:
            self.member_summaries[chart_id] = {
                "chart_id": chart_id,
                **fields,
                "updated_at": _now(),
            }

    def get_member_summary(self, chart_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._copy(self.member_summaries.get(chart_id))

    # -- manifest ------------------------------------------------------------

    def list_manifest_members(
        self,
        *,
        chart_id: Optional[int] = None,
        record_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if record_id:
                ids = self.manifest_by_record.get(record_id, [])
                return [self._copy(self.manifest_members[i]) for i in ids]  # type: ignore[misc]
            if chart_id is not None:
                chart = self.charts.get(chart_id)
                if not chart:
                    return []
                # Manifest CSVs key on the real RecordId; test workspaces are
                # named <chart>-test — strip before joining.
                return self.list_manifest_members(
                    record_id=source_record_id(chart["chart_name"])
                )
            return []

    def count_manifest_members(self, record_id: str) -> int:
        with self._lock:
            return len(self.manifest_by_record.get(record_id, []))

    def upsert_manifest_member(
        self,
        *,
        record_id: str,
        member_name: str,
        first_name: Optional[str],
        middle_name: Optional[str],
        last_name: Optional[str],
        member_dob: Optional[str],
        external_member_id: Optional[str],
        run_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        source_file: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upsert by (record_id, external_member_id) or name+dob; return {id, action}."""
        ext = (external_member_id or "").strip() or None
        with self._lock:
            existing_id = None
            for mid in self.manifest_by_record.get(record_id, []):
                row = self.manifest_members[mid]
                if ext is not None:
                    if row.get("external_member_id") == ext:
                        existing_id = mid
                        break
                else:
                    if (
                        not row.get("external_member_id")
                        and (row.get("member_name") or "").lower()
                        == (member_name or "").lower()
                        and (row.get("member_dob") or None) == (member_dob or None)
                    ):
                        existing_id = mid
                        break
            if existing_id is not None:
                row = self.manifest_members[existing_id]
                row["member_name"] = member_name
                if first_name is not None:
                    row["first_name"] = first_name
                if middle_name is not None:
                    row["middle_name"] = middle_name
                if last_name is not None:
                    row["last_name"] = last_name
                if member_dob is not None:
                    row["member_dob"] = member_dob
                if run_id is not None:
                    row["run_id"] = run_id
                if batch_id is not None:
                    row["batch_id"] = batch_id
                if source_file is not None:
                    row["source_file"] = source_file
                if source_path is not None:
                    row["source_path"] = source_path
                row["updated_at"] = _now()
                return {"id": existing_id, "action": "updated"}
            mid = self._alloc()
            row = {
                "id": mid,
                "record_id": record_id,
                "member_name": member_name,
                "first_name": first_name,
                "middle_name": middle_name,
                "last_name": last_name,
                "member_dob": member_dob,
                "external_member_id": ext,
                "run_id": run_id,
                "batch_id": batch_id,
                "source_file": source_file,
                "source_path": source_path,
                "created_at": _now(),
                "updated_at": _now(),
            }
            self.manifest_members[mid] = row
            self.manifest_by_record.setdefault(record_id, []).append(mid)
            return {"id": mid, "action": "inserted"}

    def upsert_manifest_members(
        self,
        members: list[dict[str, Any]],
        *,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Bulk upsert — in-memory path still goes row-by-row (API parity)."""
        _ = batch_size
        inserted = 0
        updated = 0
        for m in members:
            result = self.upsert_manifest_member(
                record_id=str(m["record_id"]),
                member_name=str(m["member_name"]),
                first_name=m.get("first_name"),
                middle_name=m.get("middle_name"),
                last_name=m.get("last_name"),
                member_dob=m.get("member_dob"),
                external_member_id=m.get("external_member_id"),
                run_id=m.get("run_id"),
                batch_id=m.get("batch_id"),
                source_file=m.get("source_file"),
                source_path=m.get("source_path"),
            )
            if result["action"] == "updated":
                updated += 1
            else:
                inserted += 1
        return {"inserted": inserted, "updated": updated}

    # -- reset / prune / progress --------------------------------------------

    def reset_chart_results(self, chart_id: int) -> dict[str, int]:
        deleted: dict[str, int] = {}
        with self._lock:
            def _purge(mapping: dict[Any, dict[str, Any]], label: str) -> None:
                keys = [k for k, r in mapping.items() if r.get("chart_id") == chart_id]
                if keys:
                    deleted[label] = len(keys)
                    for k in keys:
                        del mapping[k]

            _purge(self.member_summaries, "member_verification_summary")
            _purge(self.member_extractions, "member_extraction_results")
            _purge(self.dos, "dos_extraction_results")
            _purge(self.page_classifications, "page_classification")
            _purge(self.encounters, "encounter_type_results")
            _purge(self.sequencing, "page_sequencing_results")
            bj_keys = [k for k, r in self.blank_junk.items() if r["chart_id"] == chart_id]
            if bj_keys:
                deleted["blank_junk_classification"] = len(bj_keys)
                for k in bj_keys:
                    del self.blank_junk[k]
            q_keys = [k for k, r in self.quality.items() if r["chart_id"] == chart_id]
            if q_keys:
                deleted["ocr_quality_results"] = len(q_keys)
                for k in q_keys:
                    del self.quality[k]
            o_keys = [k for k, r in self.ocr.items() if r["chart_id"] == chart_id]
            if o_keys:
                deleted["ocr_results"] = len(o_keys)
                for k in o_keys:
                    del self.ocr[k]
            ps_keys = [
                k for k, r in self.page_stages.items() if r["chart_id"] == chart_id
            ]
            if ps_keys:
                deleted["page_stage_status"] = len(ps_keys)
                for k in ps_keys:
                    del self.page_stages[k]
            chart = self.charts.get(chart_id)
            if chart:
                chart["status"] = "received"
                chart["current_stage"] = None
                chart["current_pass"] = None
                chart["page_count"] = None
                chart["updated_at"] = _now()
        return deleted

    def clear_chart_outputs(self, chart_id: int, *, keep_ocr: bool) -> dict[str, int]:
        ocr_stages = {"ocr_prelim", "ocr_final1", "ocr_final2"}
        deleted: dict[str, int] = {}
        with self._lock:
            def _purge(mapping: dict[Any, dict[str, Any]], label: str, keep=None) -> None:
                keys = [
                    k for k, r in mapping.items()
                    if r.get("chart_id") == chart_id and not (keep and keep(r))
                ]
                if keys:
                    deleted[label] = len(keys)
                    for k in keys:
                        del mapping[k]

            _purge(self.member_summaries, "member_verification_summary")
            _purge(self.member_extractions, "member_extraction_results")
            _purge(self.dos, "dos_extraction_results")
            _purge(self.page_classifications, "page_classification")
            _purge(self.encounters, "encounter_type_results")
            _purge(self.sequencing, "page_sequencing_results")
            _purge(self.blank_junk, "blank_junk_classification")
            _purge(self.quality, "ocr_quality_results")
            if not keep_ocr:
                _purge(self.ocr, "ocr_results")
            _purge(
                self.page_stages,
                "page_stage_status",
                keep=(lambda r: r.get("stage_name") in ocr_stages) if keep_ocr else None,
            )
        return deleted

    def prune_orphan_pages(
        self, chart_id: int, keep_page_names: list[str]
    ) -> int:
        keep = set(n for n in keep_page_names if n)
        with self._lock:
            ids = list(self.pages_by_chart.get(chart_id, []))
            removed = 0
            keep_ids: list[int] = []
            for pid in ids:
                page = self.pages.get(pid)
                if not page:
                    continue
                if keep and page["page_name"] not in keep:
                    del self.pages[pid]
                    removed += 1
                elif not keep:
                    del self.pages[pid]
                    removed += 1
                else:
                    keep_ids.append(pid)
            self.pages_by_chart[chart_id] = keep_ids if keep else []
            return removed

    def stage_progress_rows(self, chart_id: int) -> list[dict[str, Any]]:
        """Equivalent of v_chart_stage_progress for phase-1 stages."""
        pages = self.list_pages(chart_id)
        pages_total = len(pages)
        page_ids = {int(p["id"]) for p in pages}
        rows: list[dict[str, Any]] = []
        for stage in self.list_stages(phase1_only=True):
            sn, pn = stage["stage_name"], stage["pass_no"]
            counts = {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "skipped": 0,
            }
            with self._lock:
                for pid in page_ids:
                    key = (pid, sn, pn)
                    st = (self.page_stages.get(key) or {}).get("status") or "pending"
                    if st in counts:
                        counts[st] += 1
                    else:
                        counts["pending"] += 1
            rows.append(
                {
                    "stage_name": sn,
                    "pass_no": pn,
                    "seq": stage["seq"],
                    "label": stage["label"],
                    "pages_total": pages_total,
                    **counts,
                }
            )
        return rows
