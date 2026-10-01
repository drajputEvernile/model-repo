"""Run the extraction on one chart's pages.

The module was built and trained as a standalone tool (``run.py`` read OCR JSON from a
folder, wrote CSVs, an Excel workbook and overlays). The pipeline keeps the part that
extracts and drops the rest: this is ``run.py::_write_document`` without the writing.

  1. Every extractor runs once per page on the OCR word boxes (pipeline.py).
  2. The candidate log is scored by the trained version (``training/model.py``): the rules
     find every candidate, the version picks among them.
  3. The picks are pushed back into the rows, which staging.py then writes.

Only the trained version pinned in config.py (EXTRACTION_MODEL_VERSION, v002) runs. It is not
selectable per run, and one that is not on disk is an error — never a silent fall back to
rules or to another version, which would change what every later stage reads.
"""
from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import EXTRACTION_MODEL_VERSION

from .training import registry
from .util import config
from .util.documents import Document
from .util.model_setup import missing_models

if TYPE_CHECKING:
    from . import pipeline

logger = logging.getLogger(__name__)

# Import names the extraction needs. Checked with find_spec, so asking never imports torch.
REQUIRED_PACKAGES = ("torch", "transformers", "gliner", "lightgbm", "pandas", "numpy", "PIL")


class ExtractionNotReady(RuntimeError):
    """The weights or packages the extraction needs are not there."""


def model_version() -> str:
    return EXTRACTION_MODEL_VERSION


def readiness() -> dict[str, Any]:
    """Whether the extraction can run, and the first thing to fix. Loads nothing.

    Never raises — it feeds /health and the startup banner as well as the stage.
    """
    version = model_version()
    status: dict[str, Any] = {
        "model_version": version,
        "models_root": str(config.Models_Root),
        "ready": False,
        "reason": None,
    }
    absent = [name for name in REQUIRED_PACKAGES if find_spec(name) is None]
    if absent:
        status["reason"] = f"packages not installed: {', '.join(absent)} (requirements-extraction.txt)"
        return status
    folders = [Path(config.Ner_Model_Path), *(Path(path) for path in config.Heading_Models.values())]
    gaps = missing_models(folders)
    if gaps:
        status["reason"] = "model files missing under " + ", ".join(sorted(str(folder) for folder in gaps))
        return status
    if not registry.is_runnable(version):
        status["reason"] = f"trained version {version} not found at {config.Model_Registry / version}"
        return status
    status["ready"] = True
    return status


def require_ready() -> str:
    """The pinned version, or ExtractionNotReady naming what is missing."""
    version = model_version()
    status = readiness()
    if not status["ready"]:
        raise ExtractionNotReady(f"key/value extraction cannot run: {status['reason']}")
    return version


def extract_chart(record_id: str, pages: list[dict[str, Any]]) -> "pipeline.DocumentResult":
    """Extract every page of one chart. ``pages`` come from ocr_input.extraction_page."""
    # pipeline pulls in pandas and the feature code; the orchestrator imports this module
    # at startup and must not need them until a chart is actually extracted.
    from . import pipeline

    version = require_ready()
    result = pipeline.extract_document(Document(record_id=record_id, pages=pages, source=""))
    from .training.model import load_version

    log = load_version(version).apply(result.candidates(record_id, version))
    pipeline.apply_model_selection(result, log)
    logger.info(
        "extraction %s: %d page(s) in %.1fs (model %s)",
        record_id, len(result.pages), result.time_seconds, version,
    )
    return result
