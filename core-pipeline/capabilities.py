"""What the optional features can actually do right now.

Every optional dependency degrades a stage rather than failing it — no Azure
Document Intelligence means final2 produces no text, no extraction weights means the key/value stage
fails the chart, no blob credentials means `run` works from a local
path and not from a container. The run records that (see the working rules in
CLAUDE.md), but only after it has happened. This module answers the same
question *before* a chart is submitted.

One source, two readers: the startup log and `GET /health`. They disagreed
before this existed — the log named the database and the worker count, /health
named NER and the DOS LLM, and neither mentioned blob at all.

**Configuration, not reachability.** Everything here is local: environment
variables and `find_spec`. Nothing opens a socket, so /health stays fast and
cannot hang on a network that is down. `probe_blob()` is the one exception and
is called only from startup, bounded, once — the same shape as the database
probe next to it.
"""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

from config import (
    AZURE_CLIENT_ID,
    AZURE_DI_FEATURES,
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
    AZURE_DOCUMENT_INTELLIGENCE_KEY,
    AZURE_PRINCIPAL_ID,
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_AUTH,
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_CONTAINER,
    SKIP_OCR,
)


def blob_status() -> dict[str, Any]:
    """Can we reach a blob container at all, and as whom?

    Mirrors the precedence in `db.blob_store.get_blob_service_client` — connection
    string, then Entra / managed identity, then account key — so this cannot say
    "ready" for a credential that function would not use.
    """
    status: dict[str, Any] = {
        "container": AZURE_STORAGE_CONTAINER,
        "account": AZURE_STORAGE_ACCOUNT_NAME or None,
        "auth": None,
        "ready": False,
        "client_id": AZURE_CLIENT_ID or None,
        "principal_id": AZURE_PRINCIPAL_ID or None,
    }

    if find_spec("azure.storage.blob") is None:
        status["reason"] = "azure-storage-blob not installed"
        return status

    if AZURE_STORAGE_CONNECTION_STRING:
        status.update(auth="connection_string", ready=True)
        return status

    if not AZURE_STORAGE_ACCOUNT_NAME:
        status["reason"] = (
            "AZURE_STORAGE_ACCOUNT_NAME not set (and no AZURE_STORAGE_CONNECTION_STRING)"
        )
        return status

    from db.blob_store import (
        ENTRA_INTERACTIVE_MODES,
        ENTRA_MODES,
        MANAGED_IDENTITY_MODES,
    )

    if AZURE_STORAGE_AUTH in MANAGED_IDENTITY_MODES:
        if find_spec("azure.identity") is None:
            status["auth"] = "managed_identity"
            status["reason"] = (
                "AZURE_STORAGE_AUTH=managed_identity but azure-identity not installed"
            )
            return status
        status.update(auth="managed_identity", ready=True)
        return status

    if AZURE_STORAGE_AUTH in ENTRA_MODES | ENTRA_INTERACTIVE_MODES:
        interactive = AZURE_STORAGE_AUTH in ENTRA_INTERACTIVE_MODES
        label = "entra_interactive" if interactive else "entra"
        if find_spec("azure.identity") is None:
            status["auth"] = label
            status["reason"] = (
                f"AZURE_STORAGE_AUTH={label} but azure-identity not installed"
            )
            return status
        status.update(auth=label, ready=True)
        if interactive:
            # The probe must not be the thing that opens a browser. It runs at
            # startup with nobody necessarily watching, and a prompt there would
            # block a server start on a human.
            status["probe"] = "skipped — would prompt"
        return status

    if AZURE_STORAGE_ACCOUNT_KEY:
        status.update(auth="key", ready=True)
        return status

    status["reason"] = (
        f"AZURE_STORAGE_AUTH={AZURE_STORAGE_AUTH!r} but no AZURE_STORAGE_ACCOUNT_KEY"
    )
    return status


PROBE_DEADLINE_SECONDS = 5


def _blob_round_trip(status: dict[str, Any], timeout: int) -> None:
    """One call to the container, with the SDK's own commentary suppressed.

    When no credential is available `DefaultAzureCredential` logs a ~20-line
    WARNING listing all nine sources it tried. That is genuinely useful the
    first time you see it and pure noise on every start after — and it is
    redundant here, because the banner line this probe feeds says the same
    thing in one line, with the reason attached.

    Only for the duration of the probe. A credential failure during real chart
    processing still warns in full, which is where the detail earns its space.
    """
    import logging

    identity = logging.getLogger("azure.identity")
    previous = identity.level
    identity.setLevel(logging.ERROR)
    try:
        from db.blob_store import get_container_client

        client = get_container_client(AZURE_STORAGE_CONTAINER)
        client.get_container_properties(timeout=timeout)
        status["reachable"] = True
    except Exception as exc:
        status["reachable"] = False
        status["reason"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    finally:
        identity.setLevel(previous)


def probe_blob(timeout: int = PROBE_DEADLINE_SECONDS) -> dict[str, Any]:
    """`blob_status()` plus one round trip to the container, under a hard deadline.

    Credentials being present is not the same as the role being assigned: an
    Entra identity without **Storage Blob Data Reader** authenticates and then
    fails with a 403 on the first real call. That is a chart-time failure this
    turns into a startup line.

    **The deadline is the point.** Passing `timeout` to the SDK call bounds only
    the HTTP request; `DefaultAzureCredential` runs first, tries nine credential
    sources with its own retries and backoff, and ignores it completely. On a
    machine with no managed identity and no `az login` that took 96 seconds —
    during which startup blocked, which is precisely what the database probe
    beside it was rewritten to stop doing. A daemon thread we stop waiting on
    cannot hold the server up, whatever the SDK decides to do next.

    Called once, from startup. Never raises.
    """
    status = blob_status()
    if not status["ready"]:
        return status
    if status.get("probe") == "skipped — would prompt":
        # Verifying reachability would trigger the browser prompt this mode
        # exists to allow — at startup, where nobody asked for it. The first
        # real chart does the login instead, which is when a human is present.
        return status

    import threading

    worker = threading.Thread(
        target=_blob_round_trip, args=(status, timeout), daemon=True
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        # Still going. Leave it to finish into a dict nobody reads and move on:
        # the answer is not worth delaying every start for.
        status["reachable"] = None
        status["reason"] = (
            f"reachability unverified — probe exceeded {timeout}s "
            "(credentials are configured; the round trip did not finish in time)"
        )
    return status


def azure_di_status() -> dict[str, Any]:
    """Final OCR 2. Absent, handwritten pages get no pass-2 verdict."""
    ready = bool(
        AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY
    )
    features = [
        p.strip()
        for p in (AZURE_DI_FEATURES or "").split(",")
        if p.strip() and p.strip().casefold() not in {"0", "false", "off", "none", "-"}
    ]
    status: dict[str, Any] = {
        "endpoint": AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or None,
        "ready": ready,
        "features": features,
    }
    if not ready:
        missing = [
            name
            for name, value in (
                ("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT),
                ("AZURE_DOCUMENT_INTELLIGENCE_KEY", AZURE_DOCUMENT_INTELLIGENCE_KEY),
            )
            if not value
        ]
        status["reason"] = f"not set: {', '.join(missing)}"
    return status


def extraction_status() -> dict[str, Any]:
    """Key/value extraction: the packages and weights it needs. Never raises, never loads a model."""
    try:
        from stages.lib.extraction import readiness

        return readiness()
    except Exception as exc:
        return {"ready": False, "reason": str(exc)}


def hw_model_status() -> dict[str, Any]:
    """Stage-1 handwriting classifier weights on disk (ConvNeXt preferred, RF backup)."""
    from config import HW_MODEL_PATH, CORE_ROOT

    convnext = Path(HW_MODEL_PATH)
    if not convnext.is_file():
        hw = CORE_ROOT / "models" / "hw"
        for name in (
            "handwritten_printed_convnext_tiny.pth",
            "handwritten_printed_convnext_tiny_backup.pth",
        ):
            candidate = hw / name
            if candidate.is_file():
                convnext = candidate
                break
    rf = CORE_ROOT / "models" / "hw" / "image_type_classification.pkl"
    torch_ok = find_spec("torch") is not None and find_spec("torchvision") is not None

    if convnext.is_file() and torch_ok:
        return {
            "ready": True,
            "engine": "convnext",
            "path": str(convnext),
            "backup": str(rf) if rf.is_file() else None,
        }
    if rf.is_file():
        reason = None
        if convnext.is_file() and not torch_ok:
            reason = "ConvNeXt present but torch/torchvision missing — using RF"
        return {
            "ready": True,
            "engine": "random_forest",
            "path": str(rf),
            "reason": reason,
        }
    return {
        "ready": False,
        "engine": "fallback",
        "reason": (
            f"no HW weights under models/hw/ "
            f"(expected {convnext.name} or {rf.name})"
        ),
    }


def blank_junk_model_status() -> dict[str, Any]:
    """TF-IDF blank/junk model. Never raises; never loads the model."""
    try:
        import sys

        junk = str(Path(__file__).resolve().parent / "stages" / "lib" / "blank_junk")
        if junk not in sys.path:
            sys.path.insert(0, junk)
        from model_bridge import model_status

        return model_status()
    except Exception as exc:
        return {"ready": False, "loaded": False, "reason": str(exc)}


def rapidocr_models_status() -> dict[str, Any]:
    """Local RapidOCR .pth files used by Docling final1."""
    from stages.lib.ocr.docling_ocr import (
        missing_model_files,
        model_paths,
        rapid_models_dir,
    )

    root = rapid_models_dir()
    missing = missing_model_files(root)
    present = [p.name for p in model_paths(root).values() if p.is_file()]
    return {
        "ready": not missing,
        "models_dir": str(root),
        "present": present,
        "missing": [p.name for p in missing],
        "reason": (
            None
            if not missing
            else "missing: " + ", ".join(p.name for p in missing)
        ),
    }


def all_capabilities(*, probe: bool = False) -> dict[str, Any]:
    """Every optional feature at once. `probe=True` allows one blob round trip."""
    from stages.lib.ocr.docling_ocr import docling_status

    return {
        "blob": probe_blob() if probe else blob_status(),
        "azure_document_intelligence": azure_di_status(),
        "extraction": extraction_status(),
        "docling_final1": docling_status(),
        "hw_model": hw_model_status(),
        "blank_junk_model": blank_junk_model_status(),
        "rapidocr_models": rapidocr_models_status(),
        "skip_ocr": {
            "enabled": SKIP_OCR,
            "ready": True,
            "reason": (
                "reuse ocr/ when present, else materialize from ocr_results"
                if SKIP_OCR
                else "SKIP_OCR=false"
            ),
        },
    }


def _one_line(status: dict[str, Any], on: str) -> str:
    """`on` when ready, otherwise the single precondition to fix."""
    if status.get("ready"):
        if status.get("reachable") is False:
            return f"configured but UNREACHABLE — {status.get('reason')}"
        # Only the blob probe sets ``reachable``. Other features may carry a
        # non-fatal ``reason`` (e.g. HW using RF because torch is missing) that
        # is already folded into ``on`` by the caller — do not append twice.
        if "reachable" in status and status.get("reachable") is None and status.get("reason"):
            return f"{on} ({status['reason']})"
        return on
    return f"off — {status.get('reason') or 'not configured'}"


def startup_lines(caps: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, value) pairs for the startup banner — OK / off, no paths."""
    blob = caps["blob"]
    di = caps["azure_document_intelligence"]
    extraction = caps["extraction"]
    docling = caps.get("docling_final1") or {}
    skip_ocr = caps.get("skip_ocr") or {}
    hw = caps.get("hw_model") or {}
    rapid = caps.get("rapidocr_models") or {}
    bj = caps.get("blank_junk_model") or {}

    blob_on = f"OK — {blob.get('auth')}"
    if blob.get("account"):
        blob_on += f", account={blob['account']}"
    if blob.get("client_id"):
        blob_on += f", client_id={blob['client_id']}"
    blob_on += f", container={blob.get('container')}"

    di_on = "OK"
    if di.get("features"):
        di_on += f" — features={','.join(di['features'])}"

    extraction_on = f"OK — {extraction.get('model_version') or 'ready'}"
    docling_on = "OK"
    hw_on = f"OK — {hw.get('engine') or 'ready'}"
    rapid_on = "OK"
    bj_on = f"OK — {bj.get('model_version') or 'ready'}"
    skip_on = "ON"

    skip_line = (
        skip_on
        if skip_ocr.get("enabled")
        else f"off — {skip_ocr.get('reason') or 'SKIP_OCR=false'}"
    )

    return [
        ("blob", _one_line(blob, blob_on)),
        ("HW model", _one_line(hw, hw_on)),
        ("RapidOCR", _one_line(rapid, rapid_on)),
        ("blank/junk model", _one_line(bj, bj_on)),
        ("final1 Docling", _one_line(docling, docling_on)),
        ("final2 OCR", _one_line(di, di_on)),
        ("key/value extraction", _one_line(extraction, extraction_on)),
        ("skip OCR", skip_line),
    ]
