"""Load GLiNER from util/config.py Ner_Model_Path. No network calls."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

_MODEL = None


def _weights_present(model_dir: Path) -> bool:
    return (model_dir / "pytorch_model.bin").is_file() or (model_dir / "model.safetensors").is_file()


def _retarget_local_encoder(model_dir: Path) -> None:
    """Point GLiNER at the encoder beside the weights, not an old path.

    Also bumps a stale ``transformers_version`` on non-Mistral encoders. Older
    checkpoints ship ``\"4.0.0\"``, and transformers 5.x then falsely warns about
    a Mistral regex when ``is_local`` is not detected during tokenizer init.
    """
    encoder = (model_dir / "encoder").resolve()
    config_path = model_dir / "gliner_config.json"
    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if data.get("model_name") != str(encoder):
            data["model_name"] = str(encoder)
            config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for name in ("config.json", "tokenizer_config.json"):
        path = encoder / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for key in ("model_name", "name_or_path", "_name_or_path"):
            if key in data and data.get(key) != str(encoder):
                data[key] = str(encoder)
                changed = True
        if name == "config.json":
            model_type = str(data.get("model_type") or "").casefold()
            mistral_types = {"mistral", "mistral3", "voxtral", "ministral", "pixtral"}
            version = str(data.get("transformers_version") or "")
            if model_type and model_type not in mistral_types:
                # Any declared 5.x version skips the false-positive Mistral regex path.
                if not version.startswith("5."):
                    data["transformers_version"] = "5.13.1"
                    changed = True
        if changed:
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _patch_torch_jit_script_eager() -> None:
    """Run Deberta helpers eagerly instead of deprecated TorchScript.

    GLiNER pulls in transformers Deberta-v2, which still decorates a few tiny
    expand/bucket helpers with ``@torch.jit.script``. Torch 2.14+ emits a
    FutureWarning for that API. Those helpers do not need scripting for our
    inference path, so bind them as plain Python callables before GLiNER loads.
    """
    import torch

    if getattr(torch.jit.script, "_kv_extraction_eager", False):
        return

    original = torch.jit.script

    def eager_script(obj=None, *args, **kwargs):
        # @torch.jit.script
        if callable(obj) and not args and not kwargs:
            return obj

        # @torch.jit.script(...)
        if obj is None:

            def decorate(fn):
                return fn

            return decorate

        return original(obj, *args, **kwargs)

    eager_script._kv_extraction_eager = True  # type: ignore[attr-defined]
    torch.jit.script = eager_script  # type: ignore[assignment]


# Charts run side by side in a batch, so two of them can ask for the model at once.
_LOAD_LOCK = threading.Lock()
_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is None:
            _MODEL = _load_model()
    return _MODEL


def _load_model():
    model_dir = Path(config.Ner_Model_Path)
    if not _weights_present(model_dir):
        raise FileNotFoundError(f"GLiNER weights not found at {model_dir}")
    # Offline only while GLiNER loads: the rest of the pipeline (Docling, MiniLM) may still
    # need the hub, and an environment variable set here would outlive this call.
    saved = {name: os.environ.get(name) for name in _OFFLINE_ENV}
    os.environ.update(_OFFLINE_ENV)
    try:
        _retarget_local_encoder(model_dir)
        logger.info("loading GLiNER from %s", model_dir)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                _patch_torch_jit_script_eager()
                from gliner import GLiNER

                return GLiNER.from_pretrained(str(model_dir), local_files_only=True)
            except OSError as exc:
                last_error = exc
                logger.warning("GLiNER load attempt %s/3 failed (%s)", attempt, exc)
                for name in list(sys.modules):
                    if name == "torch" or name.startswith(("torch.", "gliner", "transformers")):
                        del sys.modules[name]
                time.sleep(2)
            except Exception as exc:
                raise RuntimeError(f"GLiNER could not be loaded from {model_dir}: {exc}") from exc
        raise RuntimeError(f"GLiNER could not be loaded from {model_dir}: {last_error}") from last_error
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@lru_cache(maxsize=2048)
def _predict_cached(snippet: str, labels: tuple[str, ...], threshold: float) -> tuple[tuple, ...]:
    hits = get_model().predict_entities(snippet, list(labels), threshold=threshold) or []
    return tuple(
        (
            str(hit.get("text") or "").strip(),
            str(hit.get("label") or ""),
            float(hit.get("score") or 0),
            hit.get("start"),
            hit.get("end"),
        )
        for hit in hits
    )


def predict(text: str, labels: list[str], threshold: float = 0.25) -> list[dict]:
    """GLiNER entities; the same sentence + labels on a page is only run once."""
    snippet = (text or "").strip()
    if not snippet:
        return []
    return [
        {"text": found, "label": label, "score": score, "start": start, "end": end}
        for found, label, score, start, end in _predict_cached(snippet, tuple(labels), threshold)
    ]
