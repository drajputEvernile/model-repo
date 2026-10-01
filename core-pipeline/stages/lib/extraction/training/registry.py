"""Extraction model versions a run can use.

v0 is the hand-written rules. Trained versions live in config.Model_Registry/vNNN (written
by training/train.py): the rules still find every candidate, the version's models pick among
them for the fields it trained, and the other fields keep the rules' choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..util import config

RULES_VERSION = "v0"


def _manifest(folder: Path) -> dict[str, Any]:
    try:
        return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def list_versions() -> list[dict[str, Any]]:
    versions: list[dict[str, Any]] = [
        {
            "id": RULES_VERSION,
            "label": "v0 · Rules",
            "description": "Hand-written key and location rules pick every value.",
            "trained_at": None,
            "runnable": True,
        }
    ]
    root = config.Model_Registry
    if root.is_dir():
        for folder in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("v")):
            manifest = _manifest(folder)
            versions.append(
                {
                    "id": folder.name,
                    "label": f"{folder.name} · Trained ranker",
                    "description": str(manifest.get("description") or "Rules find candidates; a trained model picks."),
                    "trained_at": manifest.get("trained_at"),
                    "runnable": is_runnable(folder.name),
                }
            )
    return versions


def is_runnable(version: str) -> bool:
    if version == RULES_VERSION:
        return True
    folder = config.Model_Registry / version
    return bool(_manifest(folder).get("fields_trained")) and (folder / "thresholds.json").is_file()
