"""Read provider name + signature date from an electronic-signature key box.

Rules from Data/examp/esign.csv:
  - Words are read in raw OCR order (sorting by line position interleaves lines).
  - Provider name is returned as printed, from the first name word to the last
    credential: "Agati, Alicia, APRN.CNP", "Reilly, Thomas J MD", "LAUBERT CNP, MATTHEW".
  - Name right after the key: credential optional (any key).
  - Name right before the key (nothing usable after it): credential optional for
    strong keys ("... by", "Electronically/Digitally ..."), required for weak keys.
  - Name elsewhere in the box: credential required.
  - Signature date: first date after the name, else nearest to the key; blank if none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..provider_name.extract import _NON_NAME, geometry_confidence, load_suffixes
from ..util.dates import find_dates, normalize_date
from ..util.keys import KeyHit, extract_field_keys
from ..util.geometry import is_edge_region, median_height, region_priority
from ..util.model import predict
from ..util.tokens import Token, key_token_span, ordered_tokens, same_line
from ..util.window import expand_for_key

LABELS = ["person", "date"]
_CLEAN = re.compile(r"\s+")
_NAME_WORD = re.compile(r"^[A-Z][A-Za-z'\-]*\.?$")
_PUNCT_ONLY = re.compile(r"^[,;:\-â€“|/]+$")
_CONNECTORS = frozenset({"by", "by:", ":", "-", "â€“", "|"})
_CRED_MODIFIERS = frozenset({"bc", "c", "s"})
_NOT_NAME = _NON_NAME | frozenset({"am", "pm", "edt", "est", "cst", "pst", "on", "at", "seen", "office", "visit"})
_MAX_CORE = 5
_BEFORE_SPAN = 8


@dataclass
class ESigHit:
    key: str
    region: str
    sentence: str
    scale: str = ""
    ner_text: str = ""
    provider_name: str = ""
    signature_date: str = ""
    score: float = 0.0
    accepted: bool = False
    selected: bool = False
    source: str = "ner"
    key_hit: KeyHit | None = field(default=None, repr=False, compare=False)


@dataclass
class _Name:
    first: int
    last: int
    core: list[str]
    has_credential: bool


def is_strong_key(key: str) -> bool:
    folded = _CLEAN.sub(" ", key.strip().casefold())
    return folded.endswith(" by") or folded.startswith(("electronically", "digitally"))


def _norm(text: str) -> str:
    return re.sub(r"[\s.]+", "", text or "").casefold()


def is_credential(text: str) -> bool:
    cleaned = text.strip(",;:()")
    if not any(ch.isupper() for ch in cleaned):
        return False
    catalog = load_suffixes()
    if _norm(cleaned) in catalog:
        return True
    parts = [part for part in re.split(r"[.\-]", cleaned) if part]
    if len(parts) < 2 or _norm(parts[0]) not in catalog:
        return False
    return all(_norm(part) in catalog or part.casefold() in _CRED_MODIFIERS for part in parts[1:])


def _is_name_word(text: str) -> bool:
    if not _NAME_WORD.match(text):
        return False
    bare = text.rstrip(".")
    return len(bare) == 1 or bare.casefold() not in _NOT_NAME


def _next_meaningful(tokens: list[Token], index: int, stop: int) -> int | None:
    while index < stop and _PUNCT_ONLY.match(tokens[index].text):
        index += 1
    return index if index < stop else None


def _parse_name(tokens: list[Token], start: int, stop: int, line_h: float) -> _Name | None:
    """Printed provider name starting at tokens[start] (name words, then credentials)."""
    core: list[str] = []
    first = last = start
    has_credential = False
    index = start
    while index < stop:
        token = tokens[index]
        stripped = token.text.strip(",;")
        if index > start and not same_line(tokens[index - 1], token, line_h):
            if not (has_credential or len(core) >= 2) or not is_credential(stripped):
                break
        if _PUNCT_ONLY.match(token.text):
            nxt = _next_meaningful(tokens, index, stop)
            if nxt is None or not is_credential(tokens[nxt].text.strip(",;")):
                break
            index += 1
            continue
        if core and is_credential(stripped):
            has_credential = True
            last = index
            index += 1
            continue
        if has_credential:
            # "LAUBERT CNP, MATTHEW": one surname, credential, comma, first name.
            if len(core) == 1 and tokens[index - 1].text.endswith(",") and _is_name_word(stripped):
                core.append(stripped)
                last = index
            break
        if _is_name_word(stripped) and len(core) < _MAX_CORE:
            core.append(stripped)
            last = index
            index += 1
            continue
        break
    full_words = [word for word in core if len(word.rstrip(".")) >= 2]
    if len(full_words) < 2 or not core:
        return None
    return _Name(first=first, last=last, core=core, has_credential=has_credential)


def _printed(tokens: list[Token], name: _Name, text: str) -> str:
    return text[tokens[name.first].start : tokens[name.last].end].strip(" ,;:-")


def _after_key(tokens: list[Token], key_last: int, line_h: float) -> _Name | None:
    index = key_last + 1
    while index < len(tokens) and tokens[index].text.casefold() in _CONNECTORS:
        index += 1
    if index >= len(tokens):
        return None
    return _parse_name(tokens, index, len(tokens), line_h)


def _before_key(tokens: list[Token], key_first: int, line_h: float) -> _Name | None:
    """Longest name that ends right before the key."""
    best: _Name | None = None
    for start in range(key_first - 1, max(-1, key_first - 1 - _BEFORE_SPAN), -1):
        name = _parse_name(tokens, start, key_first, line_h)
        if name is None:
            continue
        gap = tokens[name.last + 1 : key_first]
        if any(not _PUNCT_ONLY.match(token.text) for token in gap):
            continue
        if best is None or name.first < best.first:
            best = name
    return best


def _in_window(tokens: list[Token], key_first: int, key_last: int, line_h: float) -> _Name | None:
    """Nearest credentialed name anywhere else in the box (after the key preferred)."""
    found: list[tuple[int, int, _Name]] = []
    index = 0
    while index < len(tokens):
        if key_first <= index <= key_last:
            index = key_last + 1
            continue
        name = _parse_name(tokens, index, key_first if index < key_first else len(tokens), line_h)
        if name is None or not name.has_credential:
            index += 1
            continue
        before = 1 if name.last < key_first else 0
        gap = key_first - name.last if before else name.first - key_last
        found.append((before, gap, name))
        index = name.last + 1
    if not found:
        return None
    return min(found, key=lambda item: (item[0], item[1]))[2]


def _date_gap(
    sentence: str,
    key_at: tuple[int, int],
    date_at: tuple[int, int],
    after: int | None,
) -> tuple[int, int, int]:
    """Prefer dates after the provider name when known; then nearest to the key."""
    start, end = date_at
    after_penalty = 1 if after is not None and start < after else 0
    key_start, key_end = key_at
    if end <= key_start:
        middle = sentence[end:key_start]
        char_gap = key_start - end
    elif start >= key_end:
        middle = sentence[key_end:start]
        char_gap = start - key_end
    else:
        middle = ""
        char_gap = 0
    return after_penalty, len(middle.split()), char_gap


def nearest_signature_date(
    sentence: str,
    key_at: tuple[int, int] | None,
    after: int | None = None,
) -> tuple[str, float]:
    text = sentence or ""
    if key_at is None:
        return "", 0.0
    matches = list(find_dates(text))
    if not matches:
        return "", 0.0
    if after is not None:
        following = [match for match in matches if match.start() >= after]
        if following:
            chosen = min(following, key=lambda match: match.start())
            value = normalize_date(chosen.group(1))
            words_between = len(text[after : chosen.start()].split())
            return value, geometry_confidence(words_between)
    chosen = min(
        matches,
        key=lambda match: _date_gap(text, key_at, (match.start(), match.end()), after),
    )
    value = normalize_date(chosen.group(1))
    if not value:
        return "", 0.0
    score = geometry_confidence(_date_gap(text, key_at, (chosen.start(), chosen.end()), after)[1])
    return value, score


def _ner_support(raws: list[dict], start: int, end: int) -> tuple[str, float]:
    best_text, best_score = "", 0.0
    for raw in raws:
        if str(raw.get("label") or "").casefold() != "person":
            continue
        raw_start, raw_end = raw.get("start"), raw.get("end")
        if not isinstance(raw_start, int) or not isinstance(raw_end, int):
            continue
        if raw_start < end and raw_end > start and float(raw.get("score") or 0) > best_score:
            best_text, best_score = str(raw.get("text") or ""), float(raw.get("score") or 0)
    return best_text, best_score


def extract_box(hit: KeyHit) -> ESigHit:
    """One overlay box â†’ printed provider name + signature date when present."""
    scale = f"{expand_for_key(hit.key):g}x"
    tokens, text = ordered_tokens(hit.value_words)
    blank = ESigHit(key=hit.key, region=hit.region, sentence=text, scale=scale)
    positions = key_token_span(tokens, hit.word_indexes, hit.key)
    if positions is None:
        return blank
    key_first, key_last = positions
    key_at = (tokens[key_first].start, tokens[key_last].end)
    line_h = median_height([token.word for token in tokens])
    strong = is_strong_key(hit.key)

    name = _after_key(tokens, key_last, line_h)
    source, geometry = "printed_after", 0.92
    if name is None:
        name = _before_key(tokens, key_first, line_h)
        source, geometry = "printed_before", 0.8
        if name is not None and not (strong or name.has_credential):
            name = None
    if name is None:
        name = _in_window(tokens, key_first, key_last, line_h)
        source, geometry = "printed_window", 0.65

    provider = _printed(tokens, name, text) if name else ""
    name_end = tokens[name.last].end if name else None
    signature_date, date_score = nearest_signature_date(text, key_at, after=name_end)
    if not provider and not signature_date:
        return blank

    ner_text, ner_score = "", 0.0
    if name:
        raws = predict(text, LABELS)
        ner_text, ner_score = _ner_support(raws, tokens[name.first].start, tokens[name.last].end)
    score = max(ner_score, geometry) if provider else date_score
    return ESigHit(
        key=hit.key,
        region=hit.region,
        sentence=text,
        scale=scale,
        ner_text=ner_text or provider,
        provider_name=provider,
        signature_date=signature_date,
        score=round(score, 4),
        accepted=bool(provider),
        source=source if provider else "date_only",
    )


def mark_selected(rows: list[ESigHit]) -> None:
    """Prefer header/footer; then full name+date; then highest score."""
    accepted = [row for row in rows if row.accepted and row.provider_name]
    if not accepted:
        return
    edge = [row for row in accepted if is_edge_region(row.region)]
    pool = edge if edge else accepted

    def rank(row: ESigHit) -> tuple[int, int, float, int]:
        both = 1 if row.provider_name and row.signature_date else 0
        return (region_priority(row.region), both, row.score, -len(row.provider_name))

    winner = max(pool, key=rank)
    winner.selected = True


def extract_page(hits: list[KeyHit]) -> list[ESigHit]:
    rows = extract_field_keys(hits, "electronic_signature", extract_box)
    mark_selected(rows)
    return rows
