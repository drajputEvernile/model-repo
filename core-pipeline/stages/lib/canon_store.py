"""Editable ``*_canon.json`` config files that reload themselves.

Every editable keyword / catalog JSON lives in ``stages/lib/keyword-canon/`` and
is read through a :class:`CanonFile`, so editing one takes effect on the next
page processed — no pipeline restart, no re-OCR:

    junk_keywords_canon.json     blank/junk detector phrases + patterns
    dos_canon.json               date-of-service span resolution: page types, default date
    member_keywords_canon.json   member key groups / ignore / label words
    section_header_canon.json    section-header catalog (stage 6)
    codeable_canon.json          page type / codeability keywords
    encounter_canon.json         encounter-type keywords

A ``CanonFile`` pairs a path with a ``build`` function that turns the parsed
JSON into whatever the caller needs (compiled regexes, frozensets, …). The
build runs once per file version. The file's mtime is checked at most every
``CHECK_INTERVAL_SEC`` so hot loops pay a clock read, not a ``stat``.

A broken edit (invalid JSON, missing key) is logged and the last good version
keeps serving, so a typo cannot take a running batch down. With no good
version yet — first load — the error is raised.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

# stages/lib/keyword-canon/ — one folder for every editable keyword file.
CANON_DIR = Path(__file__).resolve().parent / "keyword-canon"
CHECK_INTERVAL_SEC = 1.0

T = TypeVar("T")


def _identity(data: Any) -> Any:
    return data


class CanonFile(Generic[T]):
    def __init__(
        self,
        path: Path | str,
        build: Callable[[Any], T] = _identity,  # type: ignore[assignment]
    ) -> None:
        self.path = Path(path)
        self._build = build
        self._lock = threading.Lock()
        self._value: T | None = None
        self._loaded = False
        self._mtime_ns: int | None = None
        self._checked_at = 0.0

    def get(self) -> T:
        now = time.monotonic()
        if self._loaded and now - self._checked_at < CHECK_INTERVAL_SEC:
            return self._value  # type: ignore[return-value]
        with self._lock:
            now = time.monotonic()
            if self._loaded and now - self._checked_at < CHECK_INTERVAL_SEC:
                return self._value  # type: ignore[return-value]
            self._checked_at = now
            try:
                mtime_ns = self.path.stat().st_mtime_ns
            except OSError:
                if self._loaded:
                    logger.warning("%s disappeared — keeping last good version", self.path.name)
                    return self._value  # type: ignore[return-value]
                raise
            if self._loaded and mtime_ns == self._mtime_ns:
                return self._value  # type: ignore[return-value]
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                value = self._build(data)
            except Exception:
                if not self._loaded:
                    raise
                logger.exception(
                    "%s changed but could not be loaded — keeping last good version",
                    self.path.name,
                )
                self._mtime_ns = mtime_ns  # don't retry the same broken file every second
                return self._value  # type: ignore[return-value]
            if self._loaded:
                logger.info("%s changed — reloaded", self.path.name)
            self._value = value
            self._mtime_ns = mtime_ns
            self._loaded = True
            return value

    def reset(self) -> None:
        """Forget the cached version (tests)."""
        with self._lock:
            self._value = None
            self._loaded = False
            self._mtime_ns = None
            self._checked_at = 0.0
