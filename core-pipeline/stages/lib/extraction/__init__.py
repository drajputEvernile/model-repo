"""Key/value extraction: member name / DOB / ID, DOS, provider, e-signature, page number, headings.

Runs once per chart, right after OCR (``stage.py``), and stages what it found
(``staging.py``) for the stages that use it. Public surface used by those stages.
"""
from .engine import ExtractionNotReady, extract_chart, model_version, readiness, require_ready
from .ocr_input import extraction_page, has_word_boxes
from .staging import Staged, StagedPage, drop, read, staging_path

__all__ = [
    "ExtractionNotReady",
    "Staged",
    "StagedPage",
    "drop",
    "extract_chart",
    "extraction_page",
    "has_word_boxes",
    "model_version",
    "read",
    "readiness",
    "require_ready",
    "staging_path",
]
