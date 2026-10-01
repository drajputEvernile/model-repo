"""Read one member ID from the sentence inside a key's overlay box.

Acceptance (all member_id keys):
  - Value is never a pure alphabetic word.
  - Allowed shapes (from Data/examp/id.csv):
      * digits only: 462124, 279
      * letters + digits: HG758668, EMA20577880, G06330
      * digits/letters with hyphens: 134-26-2945, 000859374-1473, D24-58988
      * masked SSN only when last 4 digits exist: XXX-XX-0594 (reject xxx-xx-xxxx)
  - Reject phones and slash-dates.

Chart is hash-required (Chart# / Chart # only). Similar OCR keys match via fuzzy matcher.
No keyless member-id extraction — only catalog key hits. An untrusted key counts only when
it reads an ID a trusted key already read on the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..util.keys import KeyHit, extract_field_keys, linked
from ..util.geometry import Word, is_edge_region, region_priority
from ..util.model import predict

LABELS = ["ID", "identifier"]
_PHONE = re.compile(r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
_SLASH_DATE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_EDGE = re.compile(r"^[#.,;:|()\[\]]+|[#.,;:|()\[\]]+$")
# Must contain at least one digit; letters and internal hyphens optional.
_ID_SHAPE = re.compile(r"^(?=.*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
_ALPHA_ONLY = re.compile(r"^[A-Za-z]+$")
# Masked SSN-style: xxx-xx-1234 / XXX-XX-0594 — keep only when last 4 are digits.
_MASKED_SSN = re.compile(r"^[Xx]{2,}-[Xx]{2,}-(\d{4})$")
_MIN_LEN = 3


@dataclass
class IdHit:
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


def _normalize(text: str) -> str:
    return text.strip().strip("#.,;:|")


def is_member_id_value(text: str) -> bool:
    """True when text matches member-id value patterns (digit-bearing, not alpha-only)."""
    return bool(accept(text))


def accept(text: str) -> str:
    """Normalize and keep only ID-shaped tokens.

    Masked forms like xxx-xx-xxxx are rejected unless the last 4 digits are
    present (e.g. XXX-XX-0594 → keep).
    """
    cleaned = _normalize(text)
    if not cleaned or any(char.isspace() for char in cleaned):
        return ""
    if _ALPHA_ONLY.fullmatch(cleaned):
        return ""
    if _PHONE.fullmatch(text.strip()) or _SLASH_DATE.fullmatch(cleaned):
        return ""
    masked = _MASKED_SSN.fullmatch(cleaned)
    if masked:
        return cleaned  # last-4 digits available
    # Fully masked / placeholder with no real last-4 (xxx-xx-xxxx).
    compact = cleaned.casefold().replace("-", "")
    if compact and set(compact) <= {"x"}:
        return ""
    if not _ID_SHAPE.fullmatch(cleaned):
        return ""
    if len(cleaned.replace("-", "")) < _MIN_LEN:
        return ""
    return cleaned


def key_span(sentence: str, words, indexes: list[int]) -> tuple[int, int] | None:
    """Character span of the key inside the overlay sentence."""
    del sentence
    wanted = set(indexes)
    offset = 0
    start = end = None
    for word in words:
        if offset:
            offset += 1
        if word.index in wanted:
            if start is None:
                start = offset
            end = offset + len(word.content)
        offset += len(word.content)
    if start is None:
        return None
    return start, end


def _word_gap(sentence: str, key_at: tuple[int, int], token_at: tuple[int, int]) -> tuple[int, int, int]:
    key_start, key_end = key_at
    token_start, token_end = token_at
    if token_end <= key_start:
        middle = sentence[token_end:key_start]
        after = 1
        char_gap = key_start - token_end
    elif token_start >= key_end:
        middle = sentence[key_end:token_start]
        after = 0
        char_gap = token_start - key_end
    else:
        middle = ""
        after = 0
        char_gap = 0
    return len(middle.split()), after, char_gap


def _id_token(raw: str) -> str:
    token = _EDGE.sub("", raw.strip())
    return accept(token)


def geometry_confidence(word_gap: int) -> float:
    """A delivered geometry value is never 0. Adjacent to the key is 0.92."""
    return round(max(0.55, 0.92 - 0.12 * word_gap), 4)


def nearest(sentence: str, key_at: tuple[int, int] | None, raws: list[dict] | None = None) -> tuple[str, float]:
    """ID token closest to the key. A token after the key wins a tie. Leading zeros stay."""
    text = sentence or ""
    if key_at is None:
        return "", 0.0
    found = []
    offset = 0
    for part in text.split(" "):
        if offset:
            offset += 1
        token = _id_token(part)
        if token:
            found.append((token, offset, offset + len(part)))
        offset += len(part)
    if not found:
        return "", 0.0
    token, start, end = min(found, key=lambda item: _word_gap(text, key_at, (item[1], item[2])))
    word_gap = _word_gap(text, key_at, (start, end))[0]
    score = 0.0
    for raw in raws or []:
        span = str(raw.get("text") or "")
        if token.casefold() in span.casefold() or token.casefold() in _normalize(span).casefold():
            score = max(score, float(raw.get("score") or 0))
    if score <= 0:
        score = geometry_confidence(word_gap)
    return token, score


def _in_header_row(hit: KeyHit, hits: list[KeyHit]) -> bool:
    """Other keys share the key's line and the next word right of it is one of them (or
    nothing): 'Name  Patient ID  SSN', not the inline 'MRN: 123  DOB: ...'."""
    height = max(hit.box.height(), 1.0)
    own = set(hit.word_indexes)
    row_keys = {
        index
        for other in hits
        if other is not hit and abs(other.box.cy - hit.box.cy) < 0.6 * height
        for index in other.word_indexes
        if index not in own
    }
    if not row_keys:
        return False
    right = [
        word for word in hit.value_words
        if word.index not in own and abs(word.box.cy - hit.box.cy) < 0.6 * height and word.box.left >= hit.box.right - 1
    ]
    return not right or min(right, key=lambda word: word.box.left).index in row_keys


def _column_value(hit: KeyHit, words: list[Word]) -> str:
    """The ID under a table column header ('Patient ID' over 'E2694028'), else ''. A line read
    as one OCR word ('(330) 376-7416 EMA24002442') gives the token under the key's centre."""
    height = max(hit.box.height(), 1.0)
    below = [
        word for word in words
        if 0 <= word.box.top - hit.box.bottom + 0.3 * height < 2.5 * height
        and min(word.box.right, hit.box.right) - max(word.box.left, hit.box.left) > 0
    ]
    if not below:
        return ""
    top = min(word.box.top for word in below)
    found: list[tuple[float, str]] = []
    for word in below:
        if word.box.top - top > height:
            continue
        text = word.content
        per_char = word.box.width() / max(len(text), 1)
        for part in re.finditer(r"\S+", text):
            token = _id_token(part.group())
            if token:
                centre = word.box.left + per_char * (part.start() + part.end()) / 2
                found.append((abs(centre - hit.box.cx), token))
    return min(found)[1] if found else ""


def extract_box(hit: KeyHit, hits: list[KeyHit] | None = None, words: list[Word] | None = None) -> IdHit:
    """One overlay box → one row. Accepted only when value matches ID patterns. A key in a
    table's header row reads its column: the value under it, never the next header."""
    if hits and _in_header_row(hit, hits):
        value = _column_value(hit, words or hit.value_words)
        return IdHit(
            key=hit.key,
            region=hit.region,
            sentence=hit.value_text,
            ner_text=value,
            value=value,
            score=geometry_confidence(0) if value else 0.0,
            accepted=bool(value),
            source="column" if value else "ner",
        )
    raws = predict(hit.value_text, LABELS)
    value, score = nearest(hit.value_text, key_span(hit.value_text, hit.value_words, hit.word_indexes), raws)
    if value and not is_member_id_value(value):
        value, score = "", 0.0
    best_text = ""
    best_score = -1.0
    for raw in raws:
        ner_text = str(raw.get("text") or "").strip()
        accepted = accept(ner_text)
        if not value or not accepted or value.casefold() not in accepted.casefold():
            continue
        raw_score = float(raw.get("score") or 0)
        if raw_score > best_score:
            best_text = ner_text
            best_score = raw_score
    if best_text and is_member_id_value(value):
        return IdHit(
            key=hit.key,
            region=hit.region,
            sentence=hit.value_text,
            ner_text=best_text,
            value=value,
            score=best_score,
            accepted=True,
            source="ner",
        )
    if value and is_member_id_value(value):
        return IdHit(
            key=hit.key,
            region=hit.region,
            sentence=hit.value_text,
            ner_text=value,
            value=value,
            score=score,
            accepted=True,
            source="geometry",
        )
    return IdHit(key=hit.key, region=hit.region, sentence=hit.value_text)


def select_per_key(rows: list[IdHit]) -> list[IdHit]:
    """Keep every key occurrence; select the best one per distinct key (header/footer first,
    then higher confidence), and only header/footer ones when the page has any."""
    best: dict[str, IdHit] = {}
    for row in rows:
        if row.accepted and row.value and not is_member_id_value(row.value):
            row.accepted = False
            row.value = ""
        current = best.get(row.key.casefold())
        if current is None or (
            row.accepted,
            region_priority(row.region),
            row.score,
        ) > (
            current.accepted,
            region_priority(current.region),
            current.score,
        ):
            best[row.key.casefold()] = row
    chosen = [row for row in best.values() if row.accepted and row.value and is_member_id_value(row.value)]
    chosen_ids = {id(row) for row in chosen}
    edge_chosen = any(is_edge_region(row.region) for row in chosen)
    for row in rows:
        row.selected = id(row) in chosen_ids and (is_edge_region(row.region) or not edge_chosen)
    return rows


def confirmed_repeats(hits: list[KeyHit], rows: list[IdHit], words: list[Word] | None) -> list[IdHit]:
    """Untrusted keys whose ID is one a trusted key already read on the page: a lone mid-page
    '(MRN 20766549)' repeats the header's MRN."""
    known = {_normalize(row.value).casefold() for row in rows if row.accepted and row.value}
    repeats = []
    for hit in hits:
        if hit.trusted or hit.field != "member_id" or not hit.value_text:
            continue
        row = extract_box(hit, hits, words)
        if row.accepted and _normalize(row.value).casefold() in known:
            repeats.append(linked(row, hit))
    return repeats


def extract_page(hits: list[KeyHit], words: list[Word] | None = None) -> list[IdHit]:
    rows = extract_field_keys(hits, "member_id", lambda hit: extract_box(hit, hits, words))
    return select_per_key(rows + confirmed_repeats(hits, rows, words))
