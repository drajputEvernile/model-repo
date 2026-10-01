"""Read one DOB value from the sentence inside a key's overlay box.

Improvements from Data/examp/dob.csv:
  - Extra keys / OCR spellings (DOB#, Date of Birthe, BirthDate, PATIENT DOB, …)
  - Dotted month dates (26.Nov.1953) via Util.dates
  - Value must normalize to a real calendar date (noise like names/page labels rejected)
  - Keyless DOB in header/footer (of the image or the text's extent) only for a date at least
    two years older than the page's other dates, so visit / print dates are not read as a DOB
  - Similar-key matching comes from Util.keys OCR fuzzy (all fields)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..util.dates import canonical_date, find_dates, normalize_date
from ..util.keys import KeyHit
from ..util.geometry import (
    Word,
    edge_band,
    is_edge_region,
    region_priority,
    union_boxes,
)
from ..util.model import predict

LABELS = ["date", "date of birth"]
_CLEAN = re.compile(r"\s+")
_HAS_DIGIT = re.compile(r"\d")
_TIME_AFTER = re.compile(r"\s+\d{1,2}:\d{2}\b")
_MIN_AGE_DAYS = 2 * 365
_LONE_MIN_AGE_DAYS = 10 * 365
_FUZZY_DOB = re.compile(r"(?i)d[o0][bg]")
# Reject obvious non-DOB phrases that sometimes sit near dates.
_NOISE = re.compile(
    r"(?i)\b("
    r"page|pg|pago|of\s+\d+|visit|encounter|appointment|dos\b|printed"
    r")\b"
)


@dataclass
class DobHit:
    key: str
    region: str
    sentence: str
    ner_text: str = ""
    value: str = ""
    score: float = 0.0
    accepted: bool = False
    selected: bool = False
    source: str = "ner"
    key_hit: KeyHit | None = field(default=None, repr=False, compare=False)


def is_dob_value(text: str) -> bool:
    """True when text is (or contains) a normalizable calendar date, not page/name noise."""
    return bool(accept_dob_value(text))


def accept_dob_value(text: str) -> str:
    cleaned = _CLEAN.sub(" ", (text or "")).strip(" #.,;:|")
    if not cleaned or not _HAS_DIGIT.search(cleaned):
        return ""
    if _NOISE.search(cleaned) and not find_dates(cleaned):
        return ""
    matches = find_dates(cleaned)
    if not matches:
        return ""
    # Prefer the first date span; require normalize success.
    value = normalize_date(matches[0].group(1))
    return value


def _accept(text: str) -> str:
    """Legacy NER helper — keep digit-bearing spans that contain a date."""
    cleaned = _CLEAN.sub(" ", text).strip(" #.,;:|")
    if not cleaned or not _HAS_DIGIT.search(cleaned):
        return ""
    if accept_dob_value(cleaned):
        return cleaned
    return cleaned if find_dates(cleaned) else ""


def _key_span(sentence: str, key: str) -> tuple[int, int] | None:
    found = re.search(re.escape(key), sentence, flags=re.IGNORECASE)
    if found:
        return found.start(), found.end()
    compact = re.sub(r"[^a-z0-9]", "", key.casefold())
    if compact in {"dob", "dobage", "dobagesex"} or compact.startswith("dob"):
        fuzzy = _FUZZY_DOB.search(sentence)
        if fuzzy:
            return fuzzy.start(), fuzzy.end()
    return None


def _gap(sentence: str, key_at: tuple[int, int], date_at: tuple[int, int]) -> tuple[int, int, int]:
    """Word gap, then a penalty if a time follows the date, then character gap."""
    key_start, key_end = key_at
    date_start, date_end = date_at
    if date_end <= key_start:
        middle = sentence[date_end:key_start]
        char_gap = key_start - date_end
    elif date_start >= key_end:
        middle = sentence[key_end:date_start]
        char_gap = date_start - key_end
    else:
        middle = ""
        char_gap = 0
    followed_by_time = 1 if _TIME_AFTER.match(sentence[date_end:]) else 0
    return len(middle.split()), followed_by_time, char_gap


def nearest_date(sentence: str, key: str, raws: list[dict] | None = None) -> tuple[str, float]:
    """Date closest to the key (numeric or month-name). Time after the date loses a tie."""
    text = sentence or ""
    key_at = _key_span(text, key)
    dates = find_dates(text)
    if not dates or key_at is None:
        return "", 0.0
    chosen = min(dates, key=lambda match: _gap(text, key_at, (match.start(), match.end())))
    value = normalize_date(chosen.group(1))
    if not value or not is_dob_value(value):
        return "", 0.0
    score = 0.0
    for raw in raws or []:
        span = str(raw.get("text") or "")
        if value.casefold() in span.casefold() or chosen.group(0).casefold() in span.casefold():
            score = max(score, float(raw.get("score") or 0))
    if score <= 0:
        score = max(0.55, 0.92 - 0.12 * _gap(text, key_at, (chosen.start(), chosen.end()))[0])
    return value, round(score, 4)


def extract_box(hit: KeyHit) -> DobHit:
    """One overlay box produces one row. Accepted only for real date values."""
    raws = predict(hit.value_text, LABELS)
    value, score = nearest_date(hit.value_text, hit.key, raws)
    if value and not is_dob_value(value):
        value, score = "", 0.0
    best_text = ""
    best_score = -1.0
    for raw in raws:
        ner_text = str(raw.get("text") or "").strip()
        accepted = _accept(ner_text)
        if not value or not accepted:
            continue
        if value.casefold() not in accepted.casefold() and accepted.casefold() not in value.casefold():
            continue
        raw_score = float(raw.get("score") or 0)
        if raw_score > best_score:
            best_text = ner_text
            best_score = raw_score
    if best_text and is_dob_value(value):
        return DobHit(
            key=hit.key,
            region=hit.region,
            sentence=hit.value_text,
            ner_text=best_text,
            value=value,
            score=best_score,
            accepted=True,
            source="ner",
        )
    if value and is_dob_value(value):
        return DobHit(
            key=hit.key,
            region=hit.region,
            sentence=hit.value_text,
            ner_text=value,
            value=value,
            score=score,
            accepted=True,
            source="geometry",
        )
    return DobHit(key=hit.key, region=hit.region, sentence=hit.value_text)


def _band_sentence(words: list[Word], page_h: float, *, header: bool) -> tuple[str, str]:
    if page_h <= 0 or not words:
        return "", "mid"
    text = union_boxes([word.box for word in words])
    wanted = "header" if header else "footer"
    band = [word for word in words if edge_band(word.box, page_h, text) == wanted]
    band.sort(key=lambda item: (item.box.cy, item.box.left))
    sentence = " ".join(word.content for word in band).strip()
    region = "keyless_header" if header else "keyless_footer"
    return sentence, region


def _as_date(value: str) -> date | None:
    try:
        return date.fromisoformat(canonical_date(value))
    except ValueError:
        return None


def _old_enough(born: date, page_dates: list[date]) -> bool:
    """A keyless date is a birth date only when it is years before the page's own dates
    (visit, print, signature). With no other date on the page it must be a decade old: a
    scanned record can be years old, so its visit date is often years before today."""
    others = [day for day in page_dates if day != born]
    if others:
        return (max(others) - born).days >= _MIN_AGE_DAYS
    return (date.today() - born).days >= _LONE_MIN_AGE_DAYS


def extract_keyless_edge_dates(words: list[Word], page_h: float) -> list[DobHit]:
    """Keyless DOB: header/footer dates at least two years older than the page's other dates."""
    rows: list[DobHit] = []
    seen: set[str] = set()
    page_text = " ".join(word.content for word in words)
    page_dates = [day for day in (_as_date(m.group(1)) for m in find_dates(page_text)) if day]
    for header in (True, False):
        sentence, region = _band_sentence(words, page_h, header=header)
        for match in find_dates(sentence):
            value = normalize_date(match.group(1))
            if not value or not is_dob_value(value):
                continue
            born = _as_date(value)
            if born is None or not _old_enough(born, page_dates):
                continue
            if value.casefold() in seen:
                continue
            seen.add(value.casefold())
            rows.append(
                DobHit(
                    key="No Key Header" if header else "No Key Footer",
                    region=region,
                    sentence=sentence,
                    ner_text=match.group(0),
                    value=value,
                    score=0.7,
                    accepted=True,
                    source="keyless_edge",
                )
            )
    return rows


def extract_page(hits: list[KeyHit], words: list[Word], page_w: float, page_h: float) -> list[DobHit]:
    """Keyed DOB at every DOB key, then keyless header/footer dates if no key gave one."""
    del page_w
    from ..util.keys import extract_field_keys

    rows = extract_field_keys(hits, "dob", extract_box)
    if not any(row.accepted and row.value for row in rows):
        rows.extend(extract_keyless_edge_dates(words, page_h))
    mark_selected(rows)
    return rows


def mark_selected(rows: list[DobHit]) -> None:
    """Prefer header/footer hits; only then mid. Highest score breaks a tie."""
    accepted = [row for row in rows if row.accepted and row.value and is_dob_value(row.value)]
    if not accepted:
        return
    edge = [row for row in accepted if is_edge_region(row.region)]
    pool = edge if edge else accepted
    counts: dict[str, int] = {}
    for row in pool:
        counts[row.value.casefold()] = counts.get(row.value.casefold(), 0) + 1
    best_count = max(counts.values())
    tied = [row for row in pool if counts[row.value.casefold()] == best_count]
    winner = max(tied, key=lambda row: (region_priority(row.region), row.score, -len(row.value)))
    winner.selected = True
