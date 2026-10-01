"""Pipeline orchestrator.

Runs the phase-1 stages in order and recomputes chart status after each one.

Two behaviours worth knowing:

* **Resumable.** Each stage asks ``pages_needing_stage`` which pages are still
  outstanding and does only those. A chart that died at DOS on page 400 of 500
  re-runs DOS for the pages that never finished — it does not re-OCR the chart
  or re-bill Azure Document Intelligence. ``force=True`` overrides that.

* **Continue-on-stage-failure is deliberate off.** A stage that raises aborts the
  chain, because every later stage reads what the failed one produced. Per-page
  failures are different: those are recorded on the page and the stage carries
  on, so one unreadable page does not stop a 500-page chart.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Optional

from db import connect, create_job, get_chart, set_chart_status, update_job
from db.chart_status import refresh_chart_status
from stages._support import stage_label
# Each stage runner lives next to its engine in stages/lib/<module>/.
from stages.lib.blank_junk import stage as blank_junk_classify
from stages.lib.dos import stage as dos_extract
from stages.lib.encounter import stage as encounter_type
from stages.lib.extraction import stage as kv_extract
from stages.lib.image_preprocess import stage as quality_rotation_hw
from stages.lib.member import stage as member_extract_verify
from stages.lib.ocr import stage_final1 as ocr_final1_docling
from stages.lib.ocr import stage_final2 as ocr_final2_azure
from stages.lib.ocr import stage_prelim as ocr_prelim_tesseract
from stages.lib.page_classify import stage as page_subtype
from stages.lib.sequencing import stage as page_sequencing
from stages.utilities.download_blob import run_download

logger = logging.getLogger(__name__)

StageFn = Callable[..., Any]

# (stage_name, pass_no, callable). Order matches pipeline_stage.seq; that table
# is the source of truth for progress reporting, this list for execution.
STAGE_CHAIN: list[tuple[str, int, StageFn]] = [
    # Rotation runs FIRST so every OCR pass reads an upright page. It needs no
    # OCR output of its own — it works from pixels — and a 90-degree page OCRs
    # badly in all three engines, so measuring before correcting was costing
    # accuracy on every rotated scan.
    ("ocr_quality", 1, quality_rotation_hw.run),
    ("ocr_prelim", 1, ocr_prelim_tesseract.run),
    ("blank_junk", 1, blank_junk_classify.run_pass1),
    ("ocr_final1", 1, ocr_final1_docling.run),
    ("ocr_final2", 1, ocr_final2_azure.run),
    ("kv_extract", 1, kv_extract.run),
    ("blank_junk", 2, blank_junk_classify.run_pass2),
    ("member_verify", 1, member_extract_verify.run),
    ("dos_extract", 1, dos_extract.run),
    ("page_subtype", 1, page_subtype.run),
    ("encounter_type", 1, encounter_type.run),
    ("page_sequencing", 1, page_sequencing.run),
]

STAGE_NAMES = [f"{name}:{pass_no}" for name, pass_no, _ in STAGE_CHAIN]


def _stage_key(name: str, pass_no: int) -> str:
    return f"{name}:{pass_no}"


def resolve_stage(token: str) -> int:
    """Index into STAGE_CHAIN for a stage token, or raise ValueError.

    Accepts "blank_junk:2" for an explicit pass and "dos_extract" for pass 1 —
    the same spelling `only` takes, so callers learn one vocabulary, not two.
    A bare name that exists only at pass 2 is NOT silently promoted: naming a
    stage that does not exist should be an error the caller sees immediately,
    not a run that quietly does something else.
    """
    token = (token or "").strip()
    if not token:
        raise ValueError("empty stage name")
    if ":" in token:
        name, _, raw_pass = token.partition(":")
        try:
            pass_no = int(raw_pass)
        except ValueError:
            raise ValueError(
                f"unknown stage {token!r} — pass must be a number, e.g. blank_junk:2"
            ) from None
    else:
        name, pass_no = token, 1
    for index, (chain_name, chain_pass, _) in enumerate(STAGE_CHAIN):
        if chain_name == name and chain_pass == pass_no:
            return index
    raise ValueError(
        f"unknown stage {token!r} — known: {', '.join(STAGE_NAMES)}"
    )


def _clear_for_rerun(chart_id: int, chart_name: str, *, keep_ocr: bool) -> dict[str, Any]:
    """Clear DB results and workspace outputs so a full re-run starts clean.

    ``pages/`` is always kept (``skip_page_download=false`` clears it at intake).
    ``keep_ocr`` (skip_ocr) also keeps ``ocr/`` and ``ocr_results``.
    """
    from db import clear_chart_outputs
    from db.paths import clear_chart_subdirs

    with connect() as conn:
        db_rows = clear_chart_outputs(conn, chart_id, keep_ocr=keep_ocr)
    subdirs = ("imaging", "corrected-pages") if keep_ocr else ("ocr", "imaging", "corrected-pages")
    files = clear_chart_subdirs(chart_name, subdirs)
    logger.info(
        "Re-run %s: cleared DB %s and files %s (kept pages/%s)",
        chart_name,
        db_rows or "{}",
        files or "{}",
        " + ocr/" if keep_ocr else "",
    )
    return {"db": db_rows, "files": files, "kept_ocr": keep_ocr}


def run_pipeline_for_chart(
    chart_id: int,
    *,
    force: bool = True,
    only: Optional[list[str]] = None,
    through: Optional[str] = None,
    skip_ocr: Optional[bool] = None,
    redownload_pages: bool = False,
) -> dict[str, Any]:
    """Run the stage chain for one chart.

    Two independent ways to run less than the whole chain:

    * ``through="ocr_final2"`` runs the chain from the top and stops after that
      stage — "everything up to here".
    * ``only=["dos_extract"]`` runs just those stages, whatever came before.
      Useful when an earlier stage's output is already on disk and only the
      last step changed.

    They compose: ``through`` bounds the chain, ``only`` filters within it.
    Neither disturbs the recorded progress of the stages it does not run.

    ``skip_ocr`` overrides the ``SKIP_OCR`` env for this run (``None`` = env).
    With ``skip_ocr`` active and ``force=false``: reuse OCR artifacts, always
    refresh quality, and **force-re-run every non-OCR stage** (blank/junk,
    headers, member, DOS, codeable, encounter, sequencing). OCR engines only
    re-run for pages gate-delta marks pending (missing artifacts or a
    rotation/HW/quality path change that invalidates reused text).

    Before stages run, ``ensure_chart_images`` prefers workspace ``pages/``,
    then hydrates from ``output_path``, then Raw_Input — and does not
    re-download when local pages already exist unless ``redownload_pages``.
    """
    stop_at = resolve_stage(through) if through else None
    with connect() as conn:
        chart = get_chart(conn, chart_id)
        if not chart:
            raise RuntimeError(f"chart_id={chart_id} not found")
        if not force:
            from db import clear_stuck_processing

            stuck = clear_stuck_processing(conn, chart_id)
            if stuck:
                logger.info(
                    "Resume chart %s: cleared %d stuck processing page-stage row(s)",
                    chart["chart_name"],
                    stuck,
                )
        job_id = create_job(
            conn, chart_id=chart_id, stage_name="pipeline_full", status="running"
        )
        update_job(conn, job_id, started=True)
        progress = refresh_chart_status(conn, chart_id)

    from stages.utilities.download_blob import ensure_chart_images

    try:
        image_info = ensure_chart_images(
            chart["chart_name"],
            chart_id=chart_id,
            blob_container=chart.get("blob_container"),
            blob_path=chart.get("blob_path"),
            output_path=chart.get("output_path"),
            force_redownload_pages=redownload_pages,
            register=True,
            source=chart.get("source") or "blob",
            run_id=chart.get("run_id"),
            batch_id=chart.get("batch_id"),
        )
    except Exception as exc:
        with connect() as conn:
            set_chart_status(conn, chart_id, "failed")
            update_job(
                conn,
                job_id,
                status="failed",
                error_message=str(exc),
                completed=True,
            )
        raise

    wanted = set(only or [])
    # Only a whole-chain run rebuilds the chart; `only` / `through` runs read
    # what earlier stages left, so they must not clear it.
    full_run = not wanted and stop_at is None
    chain = STAGE_CHAIN if stop_at is None else STAGE_CHAIN[: stop_at + 1]
    results: dict[str, Any] = {
        "chart_id": chart_id,
        "chart_name": chart["chart_name"],
        "stages": {},
        "skipped_stages": [],
        "progress": progress,
        "images": {
            "source": image_info.get("image_source"),
            "page_count": image_info.get("page_count"),
            "pages_reused": image_info.get("pages_reused"),
            "pages_downloaded": image_info.get("pages_downloaded"),
        },
    }
    if stop_at is not None:
        results["through"] = STAGE_NAMES[stop_at]
        # The stages past the stop are not failures and not "skipped by filter"
        # either — they were never in scope. Naming them keeps a partial run
        # distinguishable from a chain that died early.
        results["not_run"] = STAGE_NAMES[stop_at + 1 :]
        logger.info(
            "Chart %s: running through [%s] — %d of %d stage(s)",
            chart["chart_name"], stage_label(*STAGE_CHAIN[stop_at][:2]), len(chain),
            len(STAGE_CHAIN),
        )

    from logging_setup import reset_current_chart, set_current_chart

    chart_token = set_current_chart(chart["chart_name"])
    try:
        total_stages = len(chain)
        skip_ocr_active = False
        ocr_hydrated = False
        adaptive_gates = False
        old_gates: dict[int, Any] = {}
        old_presence: dict[int, Any] = {}
        gate_delta_applied = False

        from config import SKIP_OCR
        from stages.lib.ocr.reuse import apply_skip_ocr, should_skip_ocr_stages

        skip_requested = (SKIP_OCR if skip_ocr is None else bool(skip_ocr)) and not force
        will_reuse_ocr = should_skip_ocr_stages(
            chart_name=chart["chart_name"],
            chart_id=chart_id,
            force=force,
            skip_ocr=skip_ocr,
        )
        force_quality_for_skip = False

        # skip_ocr always refreshes quality/rotation/corrected-pages.
        if skip_requested:
            force_quality_for_skip = True
            from stages.utilities.gate_delta import snapshot_gates, snapshot_ocr_presence

            with connect() as conn:
                old_gates = snapshot_gates(conn, chart_id)
                old_presence = snapshot_ocr_presence(conn, chart_id)

            # After the snapshot, so gate-delta still sees the old rotation.
            if full_run:
                results["cleared"] = _clear_for_rerun(
                    chart_id, chart["chart_name"], keep_ocr=True
                )

            if will_reuse_ocr:
                results["ocr_reuse"] = apply_skip_ocr(chart_id, chart["chart_name"])
                ocr_hydrated = True
                if (results["ocr_reuse"] or {}).get("source") == "none":
                    will_reuse_ocr = False
                    logger.info(
                        "skip_ocr for chart %s — no OCR to reuse; engines will "
                        "run after quality",
                        chart["chart_name"],
                    )
                else:
                    skip_ocr_active = True
                    adaptive_gates = True
                    logger.info(
                        "skip_ocr for chart %s — reuse OCR; re-run quality + "
                        "all non-OCR stages",
                        chart["chart_name"],
                    )
            else:
                logger.info(
                    "skip_ocr for chart %s — no OCR on disk/Processed/DB; "
                    "quality then full OCR",
                    chart["chart_name"],
                )
        elif full_run and force:
            results["cleared"] = _clear_for_rerun(
                chart_id, chart["chart_name"], keep_ocr=False
            )

        for index, (name, pass_no, fn) in enumerate(chain, start=1):
            key = f"{name}:{pass_no}"
            # Under skip_ocr, quality always runs even when `only` omits it.
            if (
                wanted
                and key not in wanted
                and name not in wanted
                and not (force_quality_for_skip and name == "ocr_quality")
            ):
                results["skipped_stages"].append(key)
                continue

            # Always re-measure quality under skip_ocr; with reuse, gate-delta
            # reopens only pages whose path changed.
            if (
                (adaptive_gates or force_quality_for_skip)
                and name == "ocr_quality"
                and not gate_delta_applied
            ):
                label = stage_label(name, pass_no)
                logger.info(
                    "=== [%s]  stage %d of %d  —  chart %s (force quality) ===",
                    label, index, total_stages, chart["chart_name"],
                )
                results["stages"][key] = fn(chart_id, force=True)
                if adaptive_gates:
                    from stages.utilities.gate_delta import apply_adaptive_gate_delta

                    with connect() as conn:
                        plan = apply_adaptive_gate_delta(
                            conn,
                            chart_id,
                            old_gates=old_gates,
                            old_presence=old_presence,
                        )
                    results["gate_delta"] = {
                        "pages_reopened": len(plan.reasons),
                        "reasons": {str(k): v for k, v in plan.reasons.items()},
                        "force_prelim": sorted(plan.force_prelim),
                        "force_final1": sorted(plan.force_final1),
                        "force_final2": sorted(plan.force_final2),
                        "invalidate": {
                            f"{s}:{p}": sorted(pids)
                            for (s, p), pids in plan.invalidate.items()
                        },
                    }
                    logger.info(
                        "[%s] done — gate-delta reopened %d page(s)",
                        label,
                        len(plan.reasons),
                    )
                else:
                    logger.info("[%s] done — quality refreshed (OCR will re-run)", label)
                gate_delta_applied = True
                with connect() as conn:
                    progress = refresh_chart_status(conn, chart_id)
                results["progress"] = progress
                continue

            if name in {"ocr_prelim", "ocr_final1", "ocr_final2"}:
                if not ocr_hydrated:
                    skip_ocr_active = should_skip_ocr_stages(
                        chart_name=chart["chart_name"],
                        chart_id=chart_id,
                        force=force,
                        skip_ocr=skip_ocr,
                    )
                    if skip_ocr_active:
                        results["ocr_reuse"] = apply_skip_ocr(
                            chart_id, chart["chart_name"]
                        )
                        if (results["ocr_reuse"] or {}).get("source") == "none":
                            skip_ocr_active = False
                    ocr_hydrated = True
                if skip_ocr_active and adaptive_gates:
                    # Per-page: only pages reset to pending by gate-delta run.
                    label = stage_label(name, pass_no)
                    logger.info(
                        "=== [%s]  stage %d of %d  —  chart %s "
                        "(skip_ocr + gate-delta pending only) ===",
                        label, index, total_stages, chart["chart_name"],
                    )
                    results["stages"][key] = fn(chart_id, force=False)
                    with connect() as conn:
                        progress = refresh_chart_status(conn, chart_id)
                    results["progress"] = progress
                    logger.info(
                        "[%s] done — chart status=%s, next=%s",
                        label, progress.get("status"),
                        progress.get("current_stage") or "finished",
                    )
                    continue
                if skip_ocr_active:
                    reason = (results.get("ocr_reuse") or {}).get("source") or "reuse"
                    results["skipped_stages"].append(key)
                    results["stages"][key] = {
                        "skipped": True,
                        "reason": f"skip_ocr_{reason}",
                    }
                    logger.info(
                        "=== [%s]  skipped (skip_ocr, %s)  —  chart %s ===",
                        stage_label(name, pass_no), reason, chart["chart_name"],
                    )
                    continue

            label = stage_label(name, pass_no)
            # skip_ocr skips OCR engines only — blank/junk and every later
            # stage re-run (force=True). Plain resume keeps force as passed.
            stage_force = True if skip_requested else force
            force_note = ""
            if skip_requested and stage_force:
                force_note = " (skip_ocr: force non-OCR)"
            logger.info(
                "=== [%s]  stage %d of %d  —  chart %s%s ===",
                label, index, total_stages, chart["chart_name"], force_note,
            )
            results["stages"][key] = fn(chart_id, force=stage_force)

            with connect() as conn:
                progress = refresh_chart_status(conn, chart_id)
            results["progress"] = progress
            logger.info(
                "[%s] done — chart status=%s, next=%s",
                label, progress.get("status"),
                progress.get("current_stage") or "finished",
            )

        with connect() as conn:
            progress = refresh_chart_status(conn, chart_id)
            update_job(conn, job_id, status="completed", completed=True)

        results["progress"] = progress
        results["status"] = progress.get("status")
        # The extraction's staging has done its job once the whole chain has run: every
        # stage that reads it has. A partial run (`only` / `through`) leaves it for the next.
        if full_run and progress.get("status") in {"completed", "needs_review"}:
            from stages.lib.extraction import staging

            if staging.drop(chart["chart_name"]):
                logger.info("Chart %s: extraction staging dropped", chart["chart_name"])
        return results

    except Exception as exc:
        logger.error("Pipeline failed for chart %s: %s", chart["chart_name"], exc)
        logger.debug(traceback.format_exc())
        with connect() as conn:
            progress = refresh_chart_status(conn, chart_id)
            # refresh_chart_status only sees page-level state; a stage that blew
            # up before touching any page would otherwise leave the chart
            # looking merely "processing".
            set_chart_status(
                conn,
                chart_id,
                "failed",
                current_stage=progress.get("current_stage"),
                current_pass=progress.get("current_pass"),
            )
            update_job(
                conn, job_id, status="failed", error_message=str(exc), completed=True
            )
        results["status"] = "failed"
        results["progress"] = progress
        results["error"] = str(exc)
        raise
    finally:
        reset_current_chart(chart_token)


def ingest_and_run(
    *,
    blob_container: Optional[str] = None,
    blob_path: Optional[str] = None,
    local_path: Optional[str] = None,
    chart_name: Optional[str] = None,
    run_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    run_pipeline: bool = True,
    force: bool = True,
    only: Optional[list[str]] = None,
    through: Optional[str] = None,
    skip_ocr: Optional[bool] = None,
    redownload_pages: bool = False,
    skip_db_write: bool = False,
) -> dict[str, Any]:
    """Fetch one chart's pages into the workspace, then run the chain on it.

    The source is either a blob prefix or a local directory; both end with the
    pages under data/folders/<chart>/pages as 1.jpg, 2.jpg …, so everything
    downstream is identical either way. This is the whole of what /api/charts/run
    does, and what batch calls once per folder.

    Page images prefer the workspace; missing ones hydrate from output_path then
    Raw_Input. OCR/imaging are not fetched here — stages reconstruct them
    (or ``skip_ocr`` reuses OCR artifacts). ``force`` reprocesses stages without
    wiping ``pages/``; ``redownload_pages`` is the escape hatch to re-fetch images.

    ``skip_db_write`` / test mode runs against an in-memory store (no Postgres)
    and writes the workspace under ``data/folders/<chart>-test``. Local path
    only — blob sources are rejected.
    """
    if bool(blob_path) == bool(local_path):
        raise ValueError("Provide exactly one of blob_path (+ blob_container) or local_path")
    if skip_db_write and not local_path:
        raise ValueError("test_mode / skip_db_write requires local_path (no Postgres / blob)")

    from db import (
        disable_skip_db_write,
        enable_skip_db_write,
        is_skip_db_write,
        test_chart_name,
    )
    from db.path_ids import resolve_run_batch
    from logging_setup import reset_current_chart, set_current_chart
    from pathlib import Path as _Path

    # Own the memory store only when we turned it on (batch may enable once).
    owned_memory = False
    if skip_db_write and not is_skip_db_write():
        enable_skip_db_write(reset=True)
        owned_memory = True
    elif skip_db_write:
        enable_skip_db_write(reset=False)

    # Bind the folder name as early as possible so intake/download lines show it.
    hint = (chart_name or "").strip()
    if not hint and local_path:
        hint = _Path(local_path).name
    elif not hint and blob_path:
        hint = _Path(str(blob_path).rstrip("/")).name
    if skip_db_write and hint:
        hint = test_chart_name(hint)
        chart_name = hint
    chart_token = set_current_chart(hint or None)

    run_id, batch_id = resolve_run_batch(
        run_id, batch_id, blob_path, local_path, blob_container
    )

    # Intake resets results only for a whole-chain run; `only` / `run_through`
    # read what earlier stages left in the DB.
    intake_reset = force and not only and not through

    try:
        if local_path:
            from stages.utilities.download_blob import import_local_folder

            intake = import_local_folder(
                local_path,
                chart_name=chart_name,
                force=intake_reset,
                redownload_pages=redownload_pages,
                run_id=run_id,
                batch_id=batch_id,
            )
            out: dict[str, Any] = {
                "chart_id": intake["chart_id"],
                "chart_name": intake["chart_name"],
                "page_count": intake["page_count"],
                "source": intake["source"],
                "imported": intake["imported"],
                "manifest": intake["manifest"],
            }
        else:
            if not blob_container:
                raise ValueError("blob_container is required with blob_path")
            download = run_download(
                blob_container=blob_container,
                blob_path=blob_path,
                run_id=run_id,
                batch_id=batch_id,
                force=intake_reset,
                redownload_pages=redownload_pages,
            )
            out = {
                "chart_id": download["chart_id"],
                "chart_name": download["chart_name"],
                "page_count": download["page_count"],
                "source": f"{blob_container}/{blob_path}",
                "image_source": download.get("image_source"),
            }

        # Prefer the registered chart name once intake finishes.
        set_current_chart(out.get("chart_name") or hint or None)

        if run_pipeline:
            out["pipeline"] = run_pipeline_for_chart(
                out["chart_id"],
                force=force,
                only=only,
                through=through,
                skip_ocr=skip_ocr,
                redownload_pages=False,  # intake already ensured images
            )
        if skip_db_write:
            out["skip_db_write"] = True
        return out
    finally:
        reset_current_chart(chart_token)
        if owned_memory:
            disable_skip_db_write()
