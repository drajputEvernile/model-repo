"""Shared file/path helpers for chart workspace."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from config import IMAGE_SUFFIXES, chart_dir, imaging_dir, ocr_dir, pages_dir

PAGE_MARKER_RE = re.compile(r"^=====\s*(.+?)\s*=====\s*$", re.MULTILINE)
PAGE_NUM_RE = re.compile(r"^(\d+)\.(jpe?g|png|webp|tif{1,2})$", re.IGNORECASE)


def normalize_fs_path(value: Optional[str]) -> Optional[str]:
    """Local filesystem path from an API/CLI string — accept ``\\`` or ``/``.

    Windows clients often send ``C:\\\\data\\\\inbox`` (or mixed separators).
    ``pathlib`` on every platform accepts forward slashes, so we normalize to
    ``/`` here once. Empty / whitespace → ``None``. Trailing separators are
    stripped (``/`` alone stays as root).
    """
    if value is None:
        return None
    s = str(value).strip().strip('"').strip("'")
    if not s:
        return None
    s = s.replace("\\", "/")
    if s.startswith("//"):
        # UNC: \\server\share\… → //server/share/…
        rest = re.sub(r"/{2,}", "/", s[2:])
        s = "//" + rest.rstrip("/")
        return s or None
    s = re.sub(r"/{2,}", "/", s).rstrip("/")
    return s or "/"


def normalize_blob_path(value: Optional[str]) -> Optional[str]:
    """Azure blob prefix — always ``/``, no leading/trailing slash."""
    if value is None:
        return None
    s = str(value).strip().strip('"').strip("'").replace("\\", "/")
    s = re.sub(r"/{2,}", "/", s).strip("/")
    return s or None


def normalize_folder_name(value: Optional[str]) -> Optional[str]:
    """Chart folder name only — basename if a path was pasted by mistake."""
    if value is None:
        return None
    s = str(value).strip().strip('"').strip("'").replace("\\", "/")
    s = s.strip("/")
    if not s:
        return None
    return s.rsplit("/", 1)[-1]


def list_local_pages(chart_name: str) -> list[Path]:
    root = pages_dir(chart_name)
    if not root.is_dir():
        return []
    files = [
        p for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        # macOS AppleDouble stubs. `._1.jpg` carries the resource fork of
        # `1.jpg`, not an image — copying onto exFAT/SMB creates one per page,
        # so an intake that filtered them at the source (download_blob,
        # batch_intake both do) still finds them here, on the destination.
        # Registering them doubles page_count and fails every stage on them.
        and not p.name.startswith("._")
    ]

    def sort_key(p: Path):
        m = PAGE_NUM_RE.match(p.name)
        if m:
            return (0, int(m.group(1)))
        return (1, p.name.casefold())

    return sorted(files, key=sort_key)


def clear_chart_workspace(chart_name: str) -> dict[str, int]:
    """Delete on-disk outputs under data/folders/<chart>/ for a re-submit.

    Clears ``pages/``, ``ocr/``, ``imaging/``, and ``corrected-pages/`` files.
    Does not touch Postgres — callers reset result tables separately and keep
    ``chart_list`` / ``page_list`` rows.
    """
    return clear_chart_subdirs(
        chart_name, ("pages", "ocr", "imaging", "corrected-pages")
    )


def clear_chart_subdirs(
    chart_name: str, subdirs: tuple[str, ...]
) -> dict[str, int]:
    """Delete files under selected chart workspace subfolders."""
    removed: dict[str, int] = {}
    root = chart_dir(chart_name)
    if not root.is_dir():
        return removed
    folder_for = {
        "pages": pages_dir(chart_name),
        "ocr": ocr_dir(chart_name),
        "imaging": imaging_dir(chart_name),
        "corrected-pages": root / "corrected-pages",
        "staging": root / "staging",
    }
    for label in subdirs:
        folder = folder_for.get(label)
        if folder is None or not folder.is_dir():
            continue
        n = 0
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
                n += 1
        if n:
            removed[label] = n
    return removed


def clear_page_image_dirs(chart_name: str) -> dict[str, int]:
    """Wipe only ``pages/`` and ``corrected-pages/`` (redownload escape hatch)."""
    return clear_chart_subdirs(chart_name, ("pages", "corrected-pages"))


def write_combined_ocr_txt(
    chart_name: str, kind: str, page_texts: list[tuple[str, str]]
) -> Path:
    """kind: prelim — page_texts: [(page_filename, text), ...]"""
    out = ocr_dir(chart_name) / f"{chart_name}_{kind}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for page_name, text in page_texts:
        blocks.append(f"===== {page_name} =====\n{(text or '').rstrip()}\n")
    out.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    return out


def write_final1_json(
    chart_name: str,
    pages: list[dict[str, Any]],
    *,
    model: str = "docling+rapidocr",
) -> Path:
    """Final OCR 1 on disk — same pages[].content shape as final2 for the UI."""
    out = ocr_dir(chart_name) / f"{chart_name}_final1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "recordId": chart_name,
        "model": model,
        "pageCount": len(pages),
        "pages": pages,
    }
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return out


def write_final2_json(
    chart_name: str, pages: list[dict[str, Any]]
) -> Path:
    out = ocr_dir(chart_name) / f"{chart_name}_final2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "recordId": chart_name,
        "model": "prebuilt-read",
        "pageCount": len(pages),
        "pages": pages,
    }
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def parse_combined_ocr_txt(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = PAGE_MARKER_RE.split(text)
    # parts: [preamble, name1, body1, name2, body2, ...]
    result: dict[str, str] = {}
    i = 1
    while i + 1 < len(parts):
        name = parts[i].strip()
        body = parts[i + 1].strip()
        result[name] = body
        i += 2
    return result


def parse_ocr_json(path: Path) -> dict[str, str]:
    """final1 / final2 JSON → {fileName: content}."""
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for page in doc.get("pages") or []:
        name = str(page.get("fileName") or f"{page.get('pageNumber')}.jpg")
        content = page.get("content")
        if content is None:
            content = page.get("markdown") or ""
        out[name] = str(content or "")
    return out


def parse_final2_json(path: Path) -> dict[str, str]:
    return parse_ocr_json(path)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def append_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def imaging_csv(chart_name: str, suffix: str) -> Path:
    return imaging_dir(chart_name) / f"{chart_name}_{suffix}.csv"


def folder_progress_path(chart_name: str) -> Path:
    return imaging_dir(chart_name) / "progress.txt"


def _progress_text(
    action: str, current: int, total: int, unit: str, detail: str = ""
) -> str:
    total = max(int(total), 0)
    current = max(0, min(int(current), total) if total else int(current))
    line = f"{action} {current}/{total} {unit} in the folder"
    return f"{line} — {detail}" if detail else line


def write_batch_folder_progress(
    batch_dir: str | Path,
    current: int,
    total: int,
    *,
    action: str = "processing",
    detail: str = "",
) -> Path:
    """``processing N/X charts in the folder`` → ``<batch_dir>/progress.txt``."""
    root = Path(batch_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "progress.txt"
    path.write_text(
        _progress_text(action, current, total, "charts", detail) + "\n",
        encoding="utf-8",
    )
    return path


def write_folder_progress(
    chart_name: str,
    current: int,
    total: int,
    *,
    action: str = "processing",
    detail: str = "",
    unit: str = "files",
) -> Path:
    """``processing N/X files in the folder`` → ``imaging/progress.txt``."""
    from config import ensure_chart_dirs

    ensure_chart_dirs(chart_name)
    path = folder_progress_path(chart_name)
    path.write_text(
        _progress_text(action, current, total, unit, detail) + "\n",
        encoding="utf-8",
    )
    return path


def load_ocr_text_for_page(
    chart_name: str,
    page_name: str,
    *,
    prefer: str = "final2",
) -> str:
    """prefer: final2 | final1 | prelim"""
    order = {
        "final2": ["final2", "final1", "prelim"],
        "final1": ["final1", "prelim"],
        "prelim": ["prelim"],
    }.get(prefer, ["final2", "final1", "prelim"])
    for kind in order:
        if kind in ("final2", "final1"):
            json_path = ocr_dir(chart_name) / f"{chart_name}_{kind}.json"
            texts = parse_ocr_json(json_path)
            if not texts and kind == "final1":
                # Older runs wrote final1 as combined .txt
                texts = parse_combined_ocr_txt(
                    ocr_dir(chart_name) / f"{chart_name}_final1.txt"
                )
        else:
            texts = parse_combined_ocr_txt(
                ocr_dir(chart_name) / f"{chart_name}_{kind}.txt"
            )
        if page_name in texts:
            return texts[page_name]
        # try bare number match
        for k, v in texts.items():
            if Path(k).stem == Path(page_name).stem:
                return v
    return ""


def iso_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
