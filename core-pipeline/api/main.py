"""FastAPI surface for core-pipeline: ingest, status, rerun, manifest sweep.

Deployed on its own (see core-pipeline/docker-compose.yml). The review UI is a
separate deployment and does not call this service — the two share the Postgres
database and the `data/folders` volume, not an HTTP boundary.

Long work runs in a BackgroundTask: every mutating endpoint returns 202 with a
chart_id, and progress is read back from GET /api/charts/{id}. See
docs/API.md for the full request/response reference.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    # .env is the source of truth for local uvicorn. Without override=True a
    # leftover shell export (e.g. EXTRACTION_MODELS_ROOT from an earlier trial)
    # silently wins over the value in the file.
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import (
    API_HOST,
    API_PORT,
    STAGE_WORKERS,
)
from db import close_pool, connect, get_chart, get_chart_by_name, list_pages, list_stages
from db.chart_status import refresh_chart_status
from db.paths import normalize_blob_path, normalize_folder_name, normalize_fs_path
from jobs.manifest_sweeper import run_load
from jobs.export_chart import SKIP_ORIG_PAGES, WRITE_MODES
from orchestrator.runner import (
    STAGE_NAMES,
    ingest_and_run,
    resolve_stage,
    run_pipeline_for_chart,
)

from logging_setup import configure_logging

# Root at INFO for our own per-page progress lines; Azure + Docling quieted to
# WARNING, or they bury [batch#] [chart#] under convert/HTTP spam.
# AZURE_LOG_LEVEL=INFO puts that dump back when debugging.
configure_logging(logging.INFO)
logger = logging.getLogger("core-pipeline")

app = FastAPI(
    title="Advantmed Core Pipeline",
    version="0.2.0",
    description=(
        "Chart intake and the imaging pipeline. Mutating endpoints are "
        "asynchronous: they return 202 and work continues in the background."
    ),
)


def _safe_db_url(url: str) -> str:
    """DATABASE_URL with the password replaced by ***, for logging.

    Worth logging at all because the commonest failure is pointing at the wrong
    database and not knowing it — but the password must never reach a log file.
    """
    import re

    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", url or "")


@app.on_event("startup")
def _startup() -> None:
    from config import DATA_ROOT, DATABASE_URL, METADATA_ROOT, STAGE_WORKERS

    logger.info("core-pipeline starting")
    logger.info("  database    : %s", _safe_db_url(DATABASE_URL))
    logger.info("  data root   : %s", DATA_ROOT)
    logger.info("  metadata    : %s", METADATA_ROOT)
    logger.info("  workers     : %s", STAGE_WORKERS)

    # Short-timeout probe, NOT the pool: the pool retries for 30 seconds, which
    # would hold up startup — and block /docs and /health, the two endpoints
    # whose whole job is to work when the database does not.
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM pipeline_stage")
                count = cur.fetchone()[0]
        logger.info("  schema      : OK, %d stage(s) registered", count)
    except Exception as exc:
        logger.warning("  database    : UNREACHABLE — %s", str(exc).splitlines()[0])
        logger.warning(
            "  Mutating endpoints will return 503 until this is fixed. "
            "DATABASE_URL is read once at startup, so restart after editing .env."
        )

    # Optional features: OK / off only — no path dumps.
    try:
        from capabilities import all_capabilities, startup_lines

        caps = all_capabilities(probe=True)
        logger.info("capabilities")
        for label, value in startup_lines(caps):
            logger.info("  %-14s : %s", label, value)
    except Exception as exc:  # a status probe must never stop the server
        logger.warning("  capabilities: probe failed — %s", exc)


@app.on_event("shutdown")
def _shutdown() -> None:
    close_pool()


# --- request models ---------------------------------------------------------


_STAGE_HELP = (
    "Stage name, 'name' for pass 1 or 'name:2' for pass 2. "
    f"Known: {', '.join(STAGE_NAMES)}"
)


class ChartRunOptions(BaseModel):
    """What run and batch-run share. Everything else is a fixed default.

    Fixed, not settable: every stage reprocesses (``force``), written outputs
    replace what is at the destination (``overwrite``), original pages are not
    re-sent to the source they came from (``skip_orig_pages``), run/batch ids
    come from the path (Run1/Batch1 → R1/B1), batch concurrency is
    ``BATCH_WORKERS``.

    Unknown fields are rejected (422) so a caller still sending a removed
    option (``force``, ``blob_read_path``, …) is told, not silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    input_type: Literal["local", "blob"] = Field(
        ..., description="Where the chart folders are read from."
    )
    container_name: Optional[str] = Field(
        None,
        description="Azure Blob container (input_type=blob only). Used for read and write.",
        examples=["imaging-pipeline"],
    )
    input_path: str = Field(
        ...,
        description=(
            "Folder holding the chart folder(s): a directory ON THE SERVER for "
            "local (Windows \\ or / both fine), a prefix inside the container for blob."
        ),
        examples=["C:/data/inbox", "Raw_Input/Run1/Batch1/DEID_PNGs"],
    )
    output_path: Optional[str] = Field(
        None,
        description=(
            "Where results are written, one sub-folder per chart. Same backend "
            "as input_type. Omit to run without writing."
        ),
        examples=["C:/data/processed", "Processed/Run1"],
    )
    only: Optional[list[str]] = Field(
        None,
        description=f"Run only these stages, against what is already on disk. {_STAGE_HELP}",
        examples=[["dos_extract"]],
    )
    run_through: Optional[str] = Field(
        None,
        description=f"Run the chain from the top and stop after this stage. {_STAGE_HELP}",
        examples=["ocr_final2"],
    )
    skip_ocr: bool = Field(
        False,
        description=(
            "Reuse existing OCR (workspace, else Processed output, else DB) "
            "instead of re-running the OCR engines — no Azure final2 billing. "
            "Quality/rotation and every non-OCR stage still re-run."
        ),
    )
    skip_completed: bool = Field(
        False,
        description=(
            "Skip charts whose status is already finished "
            "(completed / needs_review / rejected)."
        ),
    )
    skip_page_download: bool = Field(
        True,
        description=(
            "Reuse page images already in the workspace (default). false = "
            "wipe pages/ + corrected-pages/ and fetch them again from input_path."
        ),
    )

    # --- derived: what the internals were written against ---------------
    @property
    def force(self) -> bool:
        # skip_ocr is implemented as a resume (force off) that re-runs the
        # non-OCR stages itself; everything else always reprocesses.
        return not self.skip_ocr

    @property
    def through(self) -> Optional[str]:
        return self.run_through

    @property
    def redownload_pages(self) -> bool:
        return not self.skip_page_download

    @property
    def is_blob(self) -> bool:
        return self.input_type == "blob"

    @property
    def blob_container(self) -> Optional[str]:
        return self.container_name if self.is_blob else None

    @property
    def blob_read_path(self) -> Optional[str]:
        return self.input_path if self.is_blob else None

    @property
    def blob_write_path(self) -> Optional[str]:
        return self.output_path if self.is_blob else None

    @property
    def local_read_path(self) -> Optional[str]:
        return None if self.is_blob else self.input_path

    @property
    def local_write_path(self) -> Optional[str]:
        return None if self.is_blob else self.output_path

    write_mode: ClassVar[str] = SKIP_ORIG_PAGES
    overwrite: ClassVar[bool] = True

    def wants_offline(self) -> bool:
        return False

    @model_validator(mode="after")
    def _normalise_paths(self) -> "ChartRunOptions":
        norm = normalize_blob_path if self.is_blob else normalize_fs_path
        self.input_path = norm(self.input_path) or ""
        if self.output_path is not None:
            self.output_path = norm(self.output_path)
        if self.container_name is not None:
            self.container_name = self.container_name.strip().strip("/") or None
        return self


class RunRequest(ChartRunOptions):
    """One chart: ``<input_path>/<chart_name>`` → pipeline → ``<output_path>/<chart_name>``.

    Running a chart that was run before resumes it: its workspace pages are
    reused (``skip_page_download``) and every stage reprocesses.
    """

    chart_name: str = Field(
        ...,
        description="The chart folder under input_path. It is the chart's name everywhere.",
        examples=["52754737_48221214"],
    )

    @field_validator("chart_name", mode="before")
    @classmethod
    def _norm_chart_name(cls, value: Any) -> Any:
        return normalize_folder_name(value) if value is not None else value


class BatchRequest(ChartRunOptions):
    """Every chart folder under ``input_path`` (or just ``chart_list``)."""

    chart_list: Optional[list[str]] = Field(
        None,
        description=(
            "Only these chart folders (list, or one comma-separated string). "
            "Default: every chart folder under input_path. Names not found are "
            "reported as charts_missing."
        ),
        examples=[["52743839_44976074", "52754737_48221214"]],
    )
    sample: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Run at most N charts (not-yet-finished ones first). Default: all."
        ),
        examples=[3],
    )

    def charts_cap(self) -> Optional[int]:
        return int(self.sample) if self.sample is not None else None

    @field_validator("chart_list", mode="before")
    @classmethod
    def _norm_chart_list(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = [p.strip() for p in value.replace(";", ",").split(",")]
        if not isinstance(value, list):
            return value
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            name = normalize_folder_name(item)
            if not name or name.casefold() in seen:
                continue
            seen.add(name.casefold())
            out.append(name)
        return out or None


class WriteRequest(BaseModel):
    """Legacy body for the removed ``/write`` endpoint (returns 410)."""

    chart_name: str = Field(
        ...,
        description="Folder name under data/folders — the chart to write",
        examples=["52743839_44976074"],
    )
    local_write_path: Optional[str] = Field(
        None, description="Destination directory ON THE SERVER"
    )
    blob_container: Optional[str] = None
    blob_write_path: Optional[str] = Field(
        None, description="Destination prefix inside the container"
    )
    write_mode: str = Field(SKIP_ORIG_PAGES)
    overwrite: bool = False

    @field_validator("local_write_path", mode="before")
    @classmethod
    def _norm_local_paths(cls, value: Any) -> Any:
        return normalize_fs_path(value) if value is not None else value

    @field_validator("blob_write_path", mode="before")
    @classmethod
    def _norm_blob_paths(cls, value: Any) -> Any:
        return normalize_blob_path(value) if value is not None else value

    @field_validator("chart_name", mode="before")
    @classmethod
    def _norm_folder_names(cls, value: Any) -> Any:
        return normalize_folder_name(value) if value is not None else value


class ManifestSweepRequest(BaseModel):
    """Load a batch manifest into manifest_member_list (upsert)."""

    local_path: Optional[str] = Field(
        None, description="Local CSV/XLSX file or a directory of them"
    )
    blob_container: Optional[str] = None
    blob_prefix: Optional[str] = None
    run_id: Optional[str] = Field(None, description="Overrides the R# in the filename")
    batch_id: Optional[str] = Field(None, description="Overrides the B# in the filename")
    mirror_local: bool = Field(
        True, description="Copy blob manifests into review-ui/data/metadata"
    )

    @field_validator("local_path", mode="before")
    @classmethod
    def _norm_local_paths(cls, value: Any) -> Any:
        return normalize_fs_path(value) if value is not None else value

    @field_validator("blob_prefix", mode="before")
    @classmethod
    def _norm_blob_paths(cls, value: Any) -> Any:
        return normalize_blob_path(value) if value is not None else value


class GroundTruthLoadRequest(BaseModel):
    """Load the client page spreadsheet into page_ground_truth (upsert)."""

    local_path: str = Field(
        ..., min_length=1, description="Local CSV or XLSX file, or a directory of them"
    )

    @field_validator("local_path", mode="before")
    @classmethod
    def _norm_gt_path(cls, value: Any) -> Any:
        return normalize_fs_path(value) if value is not None else value


# --- background wrappers ----------------------------------------------------


# A BackgroundTask runs after the 202 has been sent, so its outcome can only
# ever reach the operator through the log. Logging failures alone left a
# successful run indistinguishable from one that never started.
def _is_credential_failure(exc: BaseException) -> bool:
    """Azure could not authenticate at all, as opposed to anything else.

    Matched by name rather than by import so this works whether or not the
    identity package is installed.
    """
    seen: BaseException | None = exc
    while seen is not None:
        if type(seen).__name__ == "ClientAuthenticationError":
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _join(prefix: Optional[str], folder: str) -> str:
    """`<prefix>/<folder>`, tolerant of missing / slash-wrapped / Windows ``\\``."""
    clean = (prefix or "").replace("\\", "/").strip().strip("/")
    name = (folder or "").replace("\\", "/").strip().strip("/")
    return f"{clean}/{name}" if clean else name


def _write_summary(
    body: "RunRequest", folder: str, write_to: Optional[str]
) -> Optional[dict[str, Any]]:
    """What the run will write, echoed back so the caller can see it was read."""
    if not write_to:
        return None
    destination = (
        f"{body.blob_container}/{_join(body.blob_write_path, folder)}"
        if body.blob_write_path
        else str(Path(write_to) / folder)
    )
    return {
        "destination": destination,
        "write_mode": body.write_mode,
        "overwrite": body.overwrite,
    }


def _write_after_run(payload: "RunRequest", chart_name: str) -> None:
    """Write the chart out, if a write path was given.

    Deliberately separate from the pipeline call: a write failure must not
    make a completed run look failed. Retry by calling /run again with
    chart_id/chart_name and the write path (sync skips files already there).
    """
    if not (payload.blob_write_path or payload.local_write_path):
        return

    write = payload.blob_write_path or payload.local_write_path
    try:
        from db import connect, set_chart_output_path
        from db.path_ids import resolve_output_path

        out_path = resolve_output_path(chart_name, write_path=write)
        if out_path:
            with connect() as conn:
                row = conn.execute(
                    "SELECT id FROM chart_list WHERE chart_name = %s",
                    (chart_name,),
                ).fetchone()
                if row:
                    # Explicit write path wins as-is (no Raw_Input→Processed remap).
                    set_chart_output_path(conn, int(row["id"]), out_path)
    except Exception:
        logger.debug(
            "could not persist output_path for %s", chart_name, exc_info=True
        )

    from jobs.export_chart import write_chart

    # After skip_ocr, quality refreshed corrected-pages/; push outputs over
    # whatever is already on Processed (ocr / corrected-pages / imaging).
    overwrite = bool(payload.overwrite) or bool(payload.skip_ocr)

    try:
        result = write_chart(
            chart_name,
            local_path=payload.local_write_path,
            blob_container=payload.blob_container,
            blob_path=payload.blob_write_path,
            overwrite=overwrite,
            write_mode=payload.write_mode,
        )
        logger.info(
            "Wrote %s -> %s (%d written, %d skipped, %s)",
            chart_name,
            result["destination"],
            result["files_written"],
            result.get("files_skipped", 0),
            result["write_mode"],
        )
    except Exception:
        logger.exception(
            "Chart %s ran, but writing it out FAILED. The results are in the "
            "workspace; retry with POST /api/charts/run "
            '{"chart_name":"%s","local_write_path"|"blob_write_path":...}.',
            chart_name,
            chart_name,
        )


def _bg_pipeline_then_write(
    chart_id: int, chart_name: str, payload: "RunRequest"
) -> None:
    try:
        _bg_pipeline(
            chart_id,
            payload.force,
            payload.only,
            payload.through,
            skip_ocr=payload.skip_ocr,
            redownload_pages=payload.redownload_pages,
        )
        _write_after_run(payload, chart_name)
    finally:
        if getattr(payload, "wants_offline", lambda: False)():
            from db import disable_skip_db_write

            disable_skip_db_write()


def _bg_run(payload: "RunRequest") -> None:
    folder = payload.chart_name
    blob_path = _join(payload.blob_read_path, folder)
    source = f"{payload.blob_container}/{blob_path}"
    logger.info("Background run starting: %s", source)
    try:
        result = ingest_and_run(
            blob_container=payload.blob_container,
            blob_path=blob_path,
            chart_name=folder,
            force=payload.force,
            only=payload.only,
            through=payload.through,
            skip_ocr=payload.skip_ocr,
            redownload_pages=payload.redownload_pages,
        )
        logger.info(
            "Background run finished: %s -> chart_id=%s",
            source, (result or {}).get("chart_id"),
        )
        _write_after_run(payload, result.get("chart_name") or folder)
    except Exception as exc:
        if _is_credential_failure(exc):
            # The traceback is fifteen frames of SDK plumbing and the message
            # is the same nine-source list already in the log. Neither tells
            # the operator the one thing to do.
            logger.error(
                "Background run FAILED for %s — no usable Azure Storage "
                "credential. Set AZURE_STORAGE_AUTH=managed_identity with "
                "AZURE_CLIENT_ID (user-assigned MI on a VM), or "
                "AZURE_STORAGE_AUTH=key with AZURE_STORAGE_ACCOUNT_KEY, or "
                "make a managed identity / `az login` available to THIS "
                "process (it reads PATH at start, so a terminal opened "
                "before installing the CLI will not see it). Local runs "
                "with local_path are unaffected.",
                source,
            )
        else:
            logger.exception("Background run FAILED for %s", source)


def _bg_pipeline(
    chart_id: int,
    force: bool,
    only: Optional[list[str]],
    through: Optional[str] = None,
    skip_ocr: Optional[bool] = None,
    redownload_pages: bool = False,
) -> None:
    logger.info(
        "Background pipeline starting: chart_id=%s force=%s only=%s through=%s "
        "skip_ocr=%s redownload_pages=%s",
        chart_id,
        force,
        only or "all stages",
        through or "end of chain",
        skip_ocr if skip_ocr is not None else "env",
        redownload_pages,
    )
    try:
        run_pipeline_for_chart(
            chart_id,
            force=force,
            only=only,
            through=through,
            skip_ocr=skip_ocr,
            redownload_pages=redownload_pages,
        )
        logger.info("Background pipeline finished: chart_id=%s", chart_id)
    except Exception:
        logger.exception("Background pipeline FAILED for chart %s", chart_id)


def _bg_batch(payload: "BatchRequest") -> None:
    from jobs.batch_intake import run_batch

    source = (
        payload.local_read_path
        or f"{payload.blob_container}/{payload.blob_read_path}"
    )
    logger.info("Background batch starting: %s", source)
    try:
        result = run_batch(
            local_read_path=payload.local_read_path,
            blob_container=payload.blob_container,
            blob_read_path=payload.blob_read_path,
            local_write_path=payload.local_write_path,
            blob_write_path=payload.blob_write_path,
            write_mode=payload.write_mode,
            overwrite=payload.overwrite,
            force=payload.force,
            only=payload.only,
            through=payload.through,
            skip_ocr=payload.skip_ocr,
            redownload_pages=payload.redownload_pages,
            sample=payload.sample,
            chart_names=payload.chart_list,
            skip_completed=payload.skip_completed,
        )
        logger.info(
            "Background batch finished: %s -> %d/%d completed, %d failed "
            "(workers=%s) in %.1fs",
            source, result["completed"], result["charts_found"],
            result["failed"], result.get("workers"), result["duration_seconds"],
        )
        for chart in result["charts"]:
            if chart["status"] == "failed":
                logger.warning("  failed: %s — %s", chart["name"], chart.get("error"))
    except Exception:
        logger.exception("Background batch FAILED: %s", source)


def _bg_manifest(payload: ManifestSweepRequest) -> None:
    source = payload.local_path or f"{payload.blob_container}/{payload.blob_prefix}"
    logger.info("Background manifest sweep starting: %s", source)
    try:
        result = run_load(
            local_path=payload.local_path,
            blob_container=payload.blob_container,
            blob_prefix=payload.blob_prefix,
            run_id=payload.run_id,
            batch_id=payload.batch_id,
            mirror_local=payload.mirror_local,
        )
        logger.info(
            "Background manifest sweep finished: %s -> %s file(s), "
            "+%s inserted, ~%s updated, %s skipped%s",
            source,
            result["files"], result["inserted"], result["updated"], result["skipped"],
            f", {len(result['errors'])} error(s)" if result.get("errors") else "",
        )
        for err in result.get("errors") or []:
            logger.warning("  manifest error: %s", err)
    except Exception:
        logger.exception("Background manifest sweep FAILED: %s", source)


def _require_db() -> None:
    """Fail fast if Postgres is unreachable, instead of accepting work we cannot do.

    The mutating endpoints return 202 and hand off to a BackgroundTask. Without
    this check a bad DATABASE_URL produced a cheerful 202, then a PoolTimeout 30
    seconds later that only ever appeared in the server log — so the caller saw
    "accepted" and no rows, with nothing connecting the two. A short-timeout
    probe turns that into an immediate 503 naming the real error.
    """
    import psycopg

    from config import DATABASE_URL

    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"database unavailable: {exc} "
                "(check DATABASE_URL in core-pipeline/.env, and restart the API "
                "after editing it — the value is read once at startup)"
            ),
        ) from exc


# --- endpoints --------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Liveness plus the toggles that change what a run actually does."""
    # Same source as the startup banner, so the two cannot disagree. No network
    # here: /health must stay fast and must not hang when Azure is down.
    try:
        from capabilities import all_capabilities

        caps = all_capabilities()
    except Exception as exc:  # never let a probe fail on an optional feature
        caps = {"extraction": {"ready": False, "reason": str(exc)}}

    return {
        "status": "ok",
        # extraction.ready=false means the key/value stage will fail the chart:
        # its packages or weights are missing (extraction.reason names which).
        # blob.ready=false means run/batch-run work locally but not from blob.
        **caps,
        "stage_workers": STAGE_WORKERS,
    }


@app.get("/ready", tags=["ops"])
def ready() -> dict[str, Any]:
    """Readiness — verifies the database is reachable and schema v7 is applied."""
    try:
        with connect() as conn:
            stages = list_stages(conn)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")
    if not stages:
        raise HTTPException(
            status_code=503,
            detail="pipeline_stage is empty — apply schema/schema.sql",
        )
    return {"status": "ready", "stages": len(stages)}


@app.get("/api/stages", tags=["ops"])
def get_stages() -> dict[str, Any]:
    """The pipeline's shape, straight from the pipeline_stage table."""
    with connect() as conn:
        return {"stages": list_stages(conn, phase1_only=False)}


def _validate_stages(body: "ChartRunOptions") -> None:
    """Reject an unknown stage name with a 400 naming the known ones.

    Without this a typo reaches the background task, where it becomes a log
    line the caller never sees — the run just does nothing.
    """
    for token in ([body.through] if body.through else []) + list(body.only or []):
        try:
            resolve_stage(token)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def _check_source(body: ChartRunOptions) -> None:
    """Shape rules shared by run and batch-run (400 with a readable detail)."""
    if body.is_blob and not body.container_name:
        raise HTTPException(
            status_code=400, detail="input_type=blob needs container_name"
        )
    if not body.is_blob and body.container_name:
        raise HTTPException(
            status_code=400,
            detail="container_name is for input_type=blob only; remove it for local",
        )
    if not body.input_path:
        raise HTTPException(status_code=400, detail="input_path is required")
    _validate_stages(body)


def _chart_finished(chart_name: str) -> Optional[str]:
    """The chart's status when it is already finished, else None."""
    from jobs.batch_intake import FINISHED_CHART_STATUSES

    with connect() as conn:
        chart = get_chart_by_name(conn, chart_name)
    status = str((chart or {}).get("status") or "")
    return status if status in FINISHED_CHART_STATUSES else None


def _options_echo(body: ChartRunOptions) -> dict[str, Any]:
    return {
        "only": body.only,
        "run_through": body.run_through,
        "skip_ocr": body.skip_ocr,
        "skip_completed": body.skip_completed,
        "skip_page_download": body.skip_page_download,
    }


@app.post("/api/charts/run", status_code=202, tags=["charts"])
def run_chart(
    body: RunRequest, background_tasks: BackgroundTasks, response: Response
) -> dict[str, Any]:
    """One chart: ``<input_path>/<chart_name>`` → pipeline → ``<output_path>/<chart_name>``.

    Re-running a chart resumes it: workspace pages are reused and every stage
    reprocesses. With ``skip_completed`` a finished chart is not run (200).

    Returns immediately (202). Poll GET /api/charts/by-name/{chart_name}.
    """
    _check_source(body)
    _require_db()
    folder = body.chart_name

    if body.skip_completed:
        finished = _chart_finished(folder)
        if finished:
            response.status_code = 200
            return {
                "status": "skipped",
                "reason": f"skip_completed: chart status is {finished}",
                "chart_name": folder,
                "poll": f"/api/charts/by-name/{folder}",
            }

    write_to = body.output_path
    base = {
        "status": "accepted",
        "input_type": body.input_type,
        "chart_name": folder,
        **_options_echo(body),
        "write": _write_summary(body, folder, write_to),
    }

    if not body.is_blob:
        from stages.utilities.download_blob import import_local_folder

        source = str(Path(body.input_path) / folder)
        try:
            result = import_local_folder(
                source,
                chart_name=folder,
                force=body.force,
                redownload_pages=body.redownload_pages,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(
            _bg_pipeline_then_write, result["chart_id"], folder, body
        )
        return {
            **base,
            "chart_id": result["chart_id"],
            "source": source,
            "imported": result["imported"],
            "page_count": result["page_count"],
            "poll": f"/api/charts/{result['chart_id']}",
        }

    background_tasks.add_task(_bg_run, body)
    return {
        **base,
        "source": f"{body.container_name}/{_join(body.input_path, folder)}",
        "poll": f"/api/charts/by-name/{folder}",
    }


@app.post(
    "/api/charts/write",
    status_code=410,
    tags=["charts"],
    deprecated=True,
    include_in_schema=False,
)
def write_chart_out(body: WriteRequest) -> dict[str, Any]:
    """Removed — write is part of ``POST /api/charts/run`` and ``/batch-run``.

    Pass ``local_write_path`` or ``blob_write_path`` on run/batch-run. Existing
    destination files are skipped; missing ones are written (``overwrite=true``
    replaces all).
    """
    raise HTTPException(
        status_code=410,
        detail=(
            "POST /api/charts/write was removed. Pass local_write_path or "
            "blob_write_path on POST /api/charts/run (or /batch-run). "
            f"Example resume+write: "
            f'{{"chart_name":"{body.chart_name}","local_write_path":"..."}}'
        ),
    )


@app.post("/api/charts/batch-run", status_code=202, tags=["charts"])
@app.post(
    "/api/charts/batch",
    status_code=202,
    tags=["charts"],
    deprecated=True,
    include_in_schema=False,
)
def batch_run(
    body: BatchRequest, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Every chart folder under a path: register, run, write if a path is set.

    Canonical path: ``POST /api/charts/batch-run``. ``/batch`` is a deprecated
    alias (not listed in /docs). Write is part of this call when
    ``local_write_path`` / ``blob_write_path`` is set (sync: missing files
    written, existing skipped).
    """
    return _batch_run_impl(body, background_tasks)


def _batch_run_impl(
    body: BatchRequest, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Register every chart under input_path, then run them with a worker pool."""
    from jobs.batch_intake import (
        filter_sources_by_chart_names,
        find_local_chart_folders,
        finished_chart_names,
        resolve_batch_workers,
    )

    _check_source(body)
    try:
        workers = resolve_batch_workers(None)  # BATCH_WORKERS, checked against DB_POOL_MAX
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _require_db()

    found: Optional[int] = None
    queued: Optional[int] = None
    missing: Optional[list[str]] = None
    skipped: Optional[list[str]] = None
    cap = body.charts_cap()
    if not body.is_blob:
        # Local drops can be counted up front; blob is listed in the background.
        try:
            folders = find_local_chart_folders(body.input_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        sources = [(str(f), f.name, "local") for f in folders]
        found = len(sources)
        if not found:
            raise HTTPException(
                status_code=400,
                detail=f"No chart folders with images under {body.input_path}",
            )
        if body.chart_list:
            sources, missing = filter_sources_by_chart_names(sources, body.chart_list)
            if not sources:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"None of chart_list exists under {body.input_path}"
                        + (f" (missing: {', '.join(missing[:20])})" if missing else "")
                    ),
                )
        if body.skip_completed:
            finished = finished_chart_names([n for _s, n, _m in sources])
            skipped = [n for _s, n, _m in sources if n in finished]
            sources = [x for x in sources if x[1] not in finished]
        queued = min(len(sources), cap) if cap else len(sources)

    background_tasks.add_task(_bg_batch, body)
    source = (
        f"{body.container_name}/{body.input_path}" if body.is_blob else body.input_path
    )
    return {
        "status": "accepted",
        "input_type": body.input_type,
        "source": source,
        "charts_found": found,
        "charts_queued": queued,
        "charts_missing": missing,
        "charts_skipped_completed": skipped,
        "chart_list": body.chart_list,
        "sample": body.sample,
        "workers": workers,
        **_options_echo(body),
        "write": (
            {
                "destination": (
                    f"{body.container_name}/{body.output_path}"
                    if body.is_blob
                    else body.output_path
                ),
                "note": "each chart is written under its own folder name, replacing existing files",
            }
            if body.output_path
            else None
        ),
        "poll": "/api/charts/by-name/{chart_name}",
    }


def _chart_payload(conn: Any, chart_id: int, include_pages: bool) -> dict[str, Any]:
    progress = refresh_chart_status(conn, chart_id)
    chart = get_chart(conn, chart_id)
    summary = conn.execute(
        "SELECT * FROM member_verification_summary WHERE chart_id = %s", (chart_id,)
    ).fetchone()
    payload: dict[str, Any] = {
        "chart": chart,
        "progress": progress,
        "member_verification": summary,
    }
    if include_pages:
        payload["pages"] = list_pages(conn, chart_id)
    return payload


@app.get("/api/charts/{chart_id}", tags=["charts"])
def get_chart_status(
    chart_id: int,
    include_pages: bool = Query(True, description="Include the page_list rows"),
) -> dict[str, Any]:
    """Chart row, per-stage progress, and the member verification outcome."""
    with connect() as conn:
        if not get_chart(conn, chart_id):
            raise HTTPException(status_code=404, detail="chart not found")
        return _chart_payload(conn, chart_id, include_pages)


@app.get("/api/charts/by-name/{chart_name}", tags=["charts"])
def get_chart_status_by_name(
    chart_name: str,
    include_pages: bool = Query(True),
) -> dict[str, Any]:
    """Same as GET /api/charts/{id} keyed on the folder name.

    Useful right after ingest, when the caller knows the blob folder but not the
    id the database assigned.
    """
    with connect() as conn:
        chart = get_chart_by_name(conn, chart_name)
        if not chart:
            raise HTTPException(status_code=404, detail="chart not found")
        return _chart_payload(conn, chart["id"], include_pages)


@app.post(
    "/api/charts/{chart_id}/rerun",
    status_code=410,
    tags=["charts"],
    deprecated=True,
    include_in_schema=False,
)
def rerun_chart(chart_id: int) -> dict[str, Any]:
    """Removed — resume via ``POST /api/charts/run`` with ``chart_id``."""
    raise HTTPException(
        status_code=410,
        detail=(
            "POST /api/charts/{id}/rerun was removed. Resume with "
            f'POST /api/charts/run and {{"chart_id":{chart_id}}} '
            "(optional: through, only, force, skip_ocr, local_write_path / "
            "blob_write_path)."
        ),
    )


@app.post("/api/manifest/sweep", status_code=202, tags=["manifest"])
def manifest_sweep(
    body: ManifestSweepRequest, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Load a batch manifest (local path or blob prefix) with upsert semantics.

    Independent of chart ingest: a manifest can be swept before the charts it
    describes exist. Rows are keyed on record_id and linked to a chart when that
    chart is ingested.
    """
    if not body.local_path and not (body.blob_container and body.blob_prefix):
        raise HTTPException(
            status_code=400,
            detail="Provide local_path, or both blob_container and blob_prefix",
        )
    if body.local_path and (body.blob_container or body.blob_prefix):
        raise HTTPException(
            status_code=400, detail="Pass either local_path or blob_*, not both"
        )
    _require_db()
    background_tasks.add_task(_bg_manifest, body)
    return {
        "status": "accepted",
        "mode": "local" if body.local_path else "blob",
        "local_path": body.local_path,
        "blob_container": body.blob_container,
        "blob_prefix": body.blob_prefix,
        "run_id": body.run_id,
        "batch_id": body.batch_id,
    }


@app.post("/api/ground-truth/load", tags=["ground-truth"])
def ground_truth_load(body: GroundTruthLoadRequest) -> dict[str, Any]:
    """Load client page labels. Chart_Name + Id (1.jpg / 1.png / 1.tif) are the keys."""
    from jobs.ground_truth_load import run_load

    _require_db()
    if not body.local_path or not Path(body.local_path).exists():
        raise HTTPException(status_code=400, detail="local_path does not exist")
    try:
        return run_load(body.local_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/manifest/{record_id}", tags=["manifest"])
def get_manifest(record_id: str) -> dict[str, Any]:
    """Manifest rows for one record id (= chart folder name).

    Prefers ``manifest_member_list`` in Postgres. If that has nothing for this
    id, scans ``METADATA_ROOT`` (default ``review-ui/data/metadata``) for
    matching rows in every CSV/XLSX there — the same files Local Mode reads.
    """
    from jobs.manifest_sweeper import lookup_manifest_members_on_disk

    rid = (record_id or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="record_id is required")

    rows: list[dict[str, Any]] = []
    source = "postgres"
    try:
        with connect() as conn:
            rows = list(
                conn.execute(
                    "SELECT * FROM manifest_member_list WHERE record_id = %s ORDER BY id",
                    (rid,),
                ).fetchall()
            )
    except Exception:
        logger.exception(
            "Postgres unavailable for manifest %s — falling back to METADATA_ROOT",
            rid,
        )
        rows = []

    if not rows:
        rows = lookup_manifest_members_on_disk(rid)
        source = "metadata"
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no manifest rows for record_id={rid!r} "
                "(checked Postgres and METADATA_ROOT)"
            ),
        )
    return {"record_id": rid, "source": source, "members": rows}


@app.get("/api/jobs", tags=["ops"])
def list_jobs(
    chart_id: Optional[int] = Query(None),
    limit: int = Query(50, le=500),
) -> dict[str, Any]:
    """Recent pipeline_jobs rows — the run log for a chart or the whole service."""
    with connect() as conn:
        if chart_id is not None:
            rows = conn.execute(
                """
                SELECT * FROM pipeline_jobs WHERE chart_id = %s
                 ORDER BY id DESC LIMIT %s
                """,
                (chart_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pipeline_jobs ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
    return {"jobs": rows}


def main() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
