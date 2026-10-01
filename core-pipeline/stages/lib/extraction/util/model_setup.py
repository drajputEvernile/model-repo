"""Make sure every model a run needs is on disk before it starts.

The public models in config.Model_Sources (GLiNER, the Heron layout detector) are
downloaded from Hugging Face at their pinned revision when their folder is missing or
incomplete; only model files come down, nothing is sent. A trained version is built locally
from reviews and has to be copied with the repo, so a missing one is an error.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

_ATTEMPTS = 3


def _missing(folder: Path, source: config.ModelSource) -> list[str]:
    target = folder / source.subfolder if source.subfolder else folder
    return [name for name in source.required if not (target / name).is_file()]


def _download(folder: Path, source: config.ModelSource) -> None:
    from huggingface_hub import snapshot_download

    target = folder / source.subfolder if source.subfolder else folder
    target.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, _ATTEMPTS + 1):
        logger.info("downloading %s@%s -> %s (attempt %s/%s)", source.repo_id, source.revision[:10], target, attempt, _ATTEMPTS)
        try:
            snapshot_download(
                repo_id=source.repo_id,
                revision=source.revision,
                local_dir=str(target),
                allow_patterns=list(source.allow) or None,
            )
            return
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            if attempt == _ATTEMPTS:
                raise
            logger.warning("download failed (%s); retrying", exc)
            time.sleep(5 * attempt)


def missing_models(folders: list[Path]) -> dict[Path, list[config.ModelSource]]:
    """Folder -> the sources whose required files are not all there."""
    out: dict[Path, list[config.ModelSource]] = {}
    for folder in folders:
        gaps = [source for source in config.Model_Sources.get(folder, []) if _missing(folder, source)]
        if gaps:
            out[folder] = gaps
    return out


def ensure_models(model_version: str, rules_version: str = "v0") -> None:
    """Download any missing public model, then check the trained version is present.
    Raises SystemExit with what is missing when a model cannot be had."""
    folders = [Path(config.Ner_Model_Path), *(Path(path) for path in config.Heading_Models.values())]
    missing = missing_models(folders)
    if missing:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise SystemExit(f"models missing and HF_HUB_OFFLINE=1: {sorted(map(str, missing))}")
        for folder, sources in missing.items():
            for source in sources:
                try:
                    _download(folder, source)
                except Exception as exc:  # noqa: BLE001 - reported with the folder it was for
                    raise SystemExit(
                        f"could not download {source.repo_id} into {folder}: {exc}\n"
                        f"Copy the folder from a machine that has it, or check the network."
                    ) from exc
        still = missing_models(folders)
        if still:
            detail = {str(folder): [name for s in sources for name in _missing(folder, s)] for folder, sources in still.items()}
            raise SystemExit(f"models still incomplete after download: {detail}")
    else:
        logger.info("models present: %s", ", ".join(str(folder) for folder in folders))

    if model_version != rules_version:
        folder = Path(config.Model_Registry) / model_version
        if not (folder / "manifest.json").is_file():
            raise SystemExit(
                f"trained version {model_version} not found at {folder}. It is trained locally from "
                f"reviews: copy Models/kv_ranker/{model_version} with the repo, or run with --model-version {rules_version}."
            )


if __name__ == "__main__":
    # python -m stages.lib.extraction.util.model_setup   (from core-pipeline/)
    from config import EXTRACTION_MODEL_VERSION

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_models(EXTRACTION_MODEL_VERSION)
