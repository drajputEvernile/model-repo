"""Heading candidates from the Heron layout detector, reviewed as the field heading_heron.

The detector boxes the regions of the page image by class. Its
title, section-header and page-header boxes become heading candidates: the OCR words whose
centres fall inside the box are the heading text. Overlapping boxes that cover the same
words are one candidate (the higher score wins).

Rules on top of the detector (v0, until a trained classifier replaces them):
  score      boxes at or above ACCEPT_SCORE are headings; FLOOR_SCORE..ACCEPT_SCORE are
             shown to reviewers as rejected near-misses
  KV key     a box whose words are all a trusted KV key ('Patient:', 'DOB') is not a
             heading; the reviewer can still tick it, which is how the exceptions are learnt
  page head  a page-header box needs two words of letters (the clock or a page number is not
             a running title)
  inline     'Assessment: stable ...' keeps only the label words up to the colon
  level      titles and page headers are Heading; a section header is Heading when it is
             clearly taller than the body text or in capitals, else Subheading
  text label colon labels outside every detector box ('HPI:', 'Family Hx:' mid-line) are added
             as rejected candidates (class text_label); only a trained version selects them
  common     a detector box rejected only for its low score (FLOOR_SCORE..ACCEPT_SCORE) is
             accepted when its text is in common_headings.txt. The list is never searched for
             on the page; it only verifies boxes the detector already found. Text labels are
             not accepted this way (specialty lists like 'CARDIOLOGY:' match it too); for
             them, as for every candidate, the match is a feature a trained version learns from.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from PIL import Image

from ..util import config
from ..util.geometry import Box, Word, group_lines, median_height, union_boxes
from ..util.keys import KeyHit

logger = logging.getLogger(__name__)

LEVELS = ("Heading", "Subheading")

ACCEPT_SCORE = 0.5
FLOOR_SCORE = 0.3


@dataclass(frozen=True)
class Detector:
    label: str
    # detector class -> kind (title / section / page_header)
    classes: dict[str, str]


DETECTORS: dict[str, Detector] = {
    "heading_heron": Detector(
        "Headings",
        {"title": "title", "section_header": "section", "page_header": "page_header"},
    ),
}
DETECTORS_KIND: dict[str, str] = {
    det_class: kind for detector in DETECTORS.values() for det_class, kind in detector.classes.items()
}


def heading_fields() -> list[str]:
    """Heading fields whose detector is configured, in DETECTORS order."""
    return [name for name in DETECTORS if name in config.Heading_Models]


COMMON_HEADINGS_FILE = Path(__file__).with_name("common_headings.txt")
# Shorter keys (HPI, EKG, Plan) must match exactly; longer ones may differ by OCR / spelling slips.
_FUZZY_MIN_CHARS = 6
_FUZZY_RATIO = 0.88
_FILLER = frozenset({"and", "of", "the", "for", "to"})


def _common_key(text: str) -> str:
    """'PROGRESS NOTES:' == 'Progress note'; 'History & Physical' == 'History and Physical'."""
    words = re.findall(r"[a-z0-9]+", (text or "").casefold())
    words = [word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word for word in words]
    return " ".join(word for word in words if word not in _FILLER)


@lru_cache(maxsize=1)
def common_headings() -> frozenset[str]:
    if not COMMON_HEADINGS_FILE.is_file():
        return frozenset()
    lines = COMMON_HEADINGS_FILE.read_text(encoding="utf-8-sig").splitlines()
    return frozenset(key for key in map(_common_key, lines) if key)


@lru_cache(maxsize=4096)
def common_heading(text: str) -> float:
    """How closely the whole text matches a common heading: 1.0 exact, the similarity for a
    near match, 0.0 when it is not in the list."""
    key = _common_key(text)
    known = common_headings()
    if not key:
        return 0.0
    if key in known:
        return 1.0
    if len(key) < _FUZZY_MIN_CHARS:
        return 0.0
    best = max(
        (
            SequenceMatcher(None, key, item).ratio()
            for item in known
            if len(item) >= _FUZZY_MIN_CHARS and abs(len(item) - len(key)) <= 0.25 * len(key) + 2
        ),
        default=0.0,
    )
    return round(best, 3) if best >= _FUZZY_RATIO else 0.0


@dataclass
class HeadingRow:
    text: str
    words: list[Word]
    box: Box
    det_class: str
    kind: str
    score: float
    level: str
    accepted: bool
    note: str = ""
    key_field: str = ""
    key_overlap: float = 0.0
    value_overlap: float = 0.0
    height_ratio: float = 1.0
    whole_line: bool = False
    line_count: int = 1
    # where the heading starts: line_start, after_stop (after '.', ';', ':') or inline
    position: str = ""
    # words after the heading on its line
    tail_words: int = 0
    # match with common_headings.txt (0 = not in the list)
    common: float = 0.0

    @property
    def selected(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class Detection:
    det_class: str
    score: float
    box: tuple[float, float, float, float]


@lru_cache(maxsize=None)
def _load(name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    path = Path(config.Heading_Models[name])
    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForObjectDetection.from_pretrained(path).eval()
    torch.set_grad_enabled(False)
    return processor, model


def load_detectors() -> dict[str, str | None]:
    """Load every configured detector; field -> error (None when it loaded)."""
    out: dict[str, str | None] = {}
    for name in heading_fields():
        try:
            _load(name)
            out[name] = None
        except Exception as exc:  # noqa: BLE001 - reported in the run queue
            out[name] = str(exc)
    return out


def detect(name: str, image: Image.Image) -> list[Detection]:
    """Heading-class boxes (image pixels) at or above FLOOR_SCORE."""
    import torch

    processor, model = _load(name)
    classes = DETECTORS[name].classes
    inputs = processor(images=[image], return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs, threshold=FLOOR_SCORE, target_sizes=[(image.height, image.width)]
    )[0]
    found: list[Detection] = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        det_class = model.config.id2label[int(label)]
        if det_class in classes:
            found.append(Detection(det_class, round(float(score), 4), tuple(float(v) for v in box)))
    return found


def _words_in(box: tuple[float, float, float, float], words: list[Word], slack: float) -> list[Word]:
    left, top, right, bottom = box
    return [
        word for word in words
        if left - slack <= word.box.cx <= right + slack and top - slack <= word.box.cy <= bottom + slack
    ]


_COLON_LABEL = re.compile(r"[A-Za-z][^:]*:$")


def _trim_inline(words: list[Word]) -> list[Word]:
    """'Assessment: stable, continue ...' -> 'Assessment:' (label words only)."""
    for n, word in enumerate(words[:6]):
        if _COLON_LABEL.search(word.content) and n < len(words) - 1:
            return words[: n + 1]
    return words


def _level(kind: str, height_ratio: float, text: str) -> str:
    if kind in {"title", "page_header"}:
        return "Heading"
    letters = [ch for ch in text if ch.isalpha()]
    capitals = len(letters) >= 4 and sum(ch.isupper() for ch in letters) / len(letters) >= 0.8
    return "Heading" if height_ratio >= 1.25 or capitals else "Subheading"


def page_headings(
    detections: list[Detection],
    image_size: tuple[int, int],
    words: list[Word],
    hits: list[KeyHit],
    page_w: float,
    page_h: float,
) -> list[HeadingRow]:
    """Heading candidates for one detector on one page, in reading order."""
    if not words:
        return []
    sx = page_w / image_size[0] if image_size[0] else 1.0
    sy = page_h / image_size[1] if image_size[1] else 1.0
    body_h = median_height(words) or 10.0
    lines = group_lines(words)
    line_of = {word.index: n for n, line in enumerate(lines) for word in line}
    trusted_keys: dict[int, str] = {}
    any_keys: set[int] = set()
    trusted_values: set[int] = set()
    for hit in hits:
        any_keys.update(hit.word_indexes)
        if hit.trusted:
            trusted_keys.update({i: hit.field for i in hit.word_indexes})
            trusted_values.update(word.index for word in hit.value_words)

    taken: list[set[int]] = []
    rows: list[HeadingRow] = []
    for det in sorted(detections, key=lambda item: -item.score):
        box = (det.box[0] * sx, det.box[1] * sy, det.box[2] * sx, det.box[3] * sy)
        inside = _words_in(box, words, body_h * 0.25)
        if not inside:
            continue
        ordered = [word for line in group_lines(inside) for word in line]
        kind = DETECTORS_KIND.get(det.det_class, "section")
        if kind == "section":
            ordered = _trim_inline(ordered)
        indexes = {word.index for word in ordered}
        if any(len(indexes & other) >= 0.5 * min(len(indexes), len(other)) for other in taken):
            continue
        taken.append(indexes)

        text = _label_text(ordered)
        height = median_height(ordered) or body_h
        height_ratio = round(height / body_h, 3)
        key_fields = {trusted_keys[i] for i in indexes if i in trusted_keys}
        is_key = bool(key_fields) and indexes <= set(trusted_keys)
        line_ids = {line_of[i] for i in indexes if i in line_of}
        whole_line = all(
            {word.index for word in lines[n]} <= indexes for n in line_ids
        )
        letter_words = [word for word in ordered if sum(ch.isalpha() for ch in word.content) >= 2]

        problems = [
            name for name, bad in (
                ("KV key", is_key),
                ("page header noise", kind == "page_header" and len(letter_words) < 2),
            ) if bad
        ]
        common = common_heading(text)
        note = "low score" if det.score < ACCEPT_SCORE else (problems[0] if problems else "")
        accepted = not note
        if note == "low score" and common and not problems:
            accepted, note = True, "common heading (low score)"
        rows.append(
            HeadingRow(
                text=text,
                words=ordered,
                box=union_boxes([word.box for word in ordered]) or Box(*box),
                det_class=det.det_class,
                kind=kind,
                score=det.score,
                level=_level(kind, height_ratio, text),
                accepted=accepted,
                note=note,
                common=common,
                key_field=sorted(key_fields)[0] if key_fields else "",
                key_overlap=round(len(indexes & any_keys) / len(indexes), 3),
                value_overlap=round(len(indexes & trusted_values) / len(indexes), 3),
                height_ratio=height_ratio,
                whole_line=whole_line,
                line_count=len(line_ids) or 1,
                **_placement(ordered, lines, line_of),
            )
        )
    rows.extend(text_labels(words, lines, taken, trusted_keys, any_keys, trusted_values, body_h))
    rows.sort(key=lambda row: (row.box.top, row.box.left))
    return rows


def _placement(span: list[Word], lines: list[list[Word]], line_of: dict[int, int]) -> dict:
    """position and tail_words of a span within its (first / last) line."""
    first, last = span[0], span[-1]
    line = lines[line_of[first.index]] if first.index in line_of else span
    at = next((n for n, word in enumerate(line) if word.index == first.index), 0)
    end_line = lines[line_of[last.index]] if last.index in line_of else span
    end = next((n for n, word in enumerate(end_line) if word.index == last.index), len(end_line) - 1)
    if at == 0:
        position = "line_start"
    elif line[at - 1].content.endswith((".", ";", ":")):
        position = "after_stop"
    else:
        position = "inline"
    return {"position": position, "tail_words": len(end_line) - end - 1}


TEXT_LABEL_CLASS = "text_label"
_LABEL_WORDS = 4
_LABEL_STOP = (":", ".", ",", ";")
_QUALIFIER = re.compile(r"\s*\([^()]*\)\s*:?\s*$")


def _label_text(words: list[Word]) -> str:
    """The words joined; a colon label drops a trailing qualifier ('Specialty Meds (Initial):')."""
    text = " ".join(word.content for word in words)
    if text.endswith(":") and "(" in text:
        return _QUALIFIER.sub("", text) or text
    return text


def _starts_label(word: Word) -> bool:
    return word.content[:1].isupper() or (word.content[:1] == "(" and word.content[1:2].isupper())


def text_labels(
    words: list[Word],
    lines: list[list[Word]],
    taken: list[set[int]],
    trusted_keys: dict[int, str],
    any_keys: set[int],
    trusted_values: set[int],
    body_h: float,
) -> list[HeadingRow]:
    """Colon labels the layout detector did not box ('HPI:', 'Family Hx:' mid-line), as
    candidates the trained model decides on: up to four capitalized words ending in a colon,
    not a trusted KV key and not inside a detector box. A trailing qualifier is dropped
    ('Specialty Meds (Initial):' -> 'Specialty Meds')."""
    line_of = {word.index: n for n, line in enumerate(lines) for word in line}
    boxed = set().union(*taken) if taken else set()
    blocked = boxed | set(trusted_keys)
    rows: list[HeadingRow] = []
    for line in lines:
        for end, word in enumerate(line):
            text = word.content
            if not text.endswith(":") or " " in text or word.index in blocked:
                continue
            start = end
            while (
                start > 0
                and end - start < _LABEL_WORDS - 1
                and line[start - 1].index not in blocked
                and " " not in line[start - 1].content
                and not line[start - 1].content.endswith(_LABEL_STOP)
                and _starts_label(line[start - 1])
            ):
                start -= 1
            span = line[start : end + 1]
            if not _starts_label(span[0]):
                continue
            label = _label_text(span)
            if sum(ch.isalpha() for ch in label) < 2:
                continue
            height = median_height(span) or body_h
            height_ratio = round(height / body_h, 3)
            common = common_heading(label)
            rows.append(
                HeadingRow(
                    text=label,
                    words=span,
                    box=union_boxes([item.box for item in span]),
                    det_class=TEXT_LABEL_CLASS,
                    kind="label",
                    score=0.0,
                    level=_level("section", height_ratio, label),
                    accepted=False,
                    note="text label",
                    common=common,
                    key_overlap=round(sum(item.index in any_keys for item in span) / len(span), 3),
                    value_overlap=round(sum(item.index in trusted_values for item in span) / len(span), 3),
                    height_ratio=height_ratio,
                    whole_line=len(span) == len(line),
                    **_placement(span, lines, line_of),
                )
            )
    return rows


def page_image(record_id: str, page: dict) -> Image.Image | None:
    """The page image the OCR words were read from: the page's imagePath when the pipeline set
    one (the corrected page when rotation changed it), else {Raw_Input}/{RecordId}/{fileName}."""
    path = Path(page.get("imagePath") or config.Raw_Input / record_id / str(page.get("fileName") or ""))
    if not path.is_file():
        logger.warning("no raw image for headings: %s", path)
        return None
    return Image.open(path).convert("RGB")


def extract_page(
    record_id: str,
    page: dict,
    words: list[Word],
    hits: list[KeyHit],
    page_w: float,
    page_h: float,
) -> dict[str, list[HeadingRow]]:
    """Heading candidates of every configured detector for one page (image opened once)."""
    names = heading_fields()
    image = page_image(record_id, page) if names else None
    if image is None:
        return {name: [] for name in names}
    return {
        name: page_headings(detect(name, image), image.size, words, hits, page_w, page_h)
        for name in names
    }
