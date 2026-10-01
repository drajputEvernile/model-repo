"""Read a provider name from the sentence inside a key's overlay box.

Applies to EVERY provider_name catalog key (Provider, PCP, Author, Bill Under,
Physician, Progress Notes by, OCR near-misses, …):

  - Similar/OCR-split keys may match freely (see Util.keys OCR fuzzy matcher).
  - A key is only accepted when the value is a person name, with optional
    credential/designation tokens around it (MD, APRN, OD, …). Non-person
    values are noise and are rejected.

Profiles (search geometry / gates):
  - standard: near-key window; credential preferred but optional when NER is strong
  - role/designation: anywhere; require ID in full-width ±5% band; value usually above
  - Bill Under: header/footer only
  - Progress Note (no "by"): require credential suffix
  - A key after its value ('Allison Moosally, MD (Primary Provider)', 'Lauren N Burns, DO
    as PCP'): the same-line words before it, back to another key, are the sentence

No keyless provider extraction yet. Role blocks may emit a keyless member-name hint
(name above the provider in the same vicinity) for the Member_Name pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..util.keys import KeyHit, linked
from ..util.geometry import Word, is_edge_region, median_height, region_priority
from ..util.model import predict
from ..util.window import (
    ROLE_ID_BAND_FRAC,
    expand_for_key,
    full_width_band,
    upward_box,
    words_in_box,
)

LABELS = ["person"]
_WORD = re.compile(r"[A-Za-z]+")
_CLEAN = re.compile(r"\s+")
_EDGE = re.compile(r"^[\s:,;\-|]+|[\s:,;\-|]+$")
_SPLIT = re.compile(r"[,]+|(?<=[A-Za-z0-9])\.(?=\s*[A-Za-z])")
_MIN_WORDS = 2
_MAX_WORDS = 4
_SUFFIXES_PATH = Path(__file__).resolve().parent / "suffixes.txt"
_ROLES_PATH = Path(__file__).resolve().parent / "roles.json"
_EDGE_ONLY_KEYS = frozenset({"bill under"})
_STRICT_SUFFIX_KEYS = frozenset({"progress note", "progress notes"})

# ID-like token in the designation band (MRN, bracket staff id, long digit run).
_ID_LIKE = re.compile(
    r"(?:"
    r"\[\s*[A-Za-z]?\d{4,}\s*\]"
    r"|\bMRN\b\s*[:#]?\s*\d{4,}"
    r"|\b\d{6,}\b"
    r"|\b[A-Z]\d{6,}\b"
    r")",
    re.IGNORECASE,
)

_NON_NAME = frozenset(
    {
        "provider",
        "physician",
        "doctor",
        "attending",
        "ordering",
        "primary",
        "care",
        "verified",
        "performed",
        "by",
        "the",
        "and",
        "for",
        "a",
        "of",
        "signature",
        "author",
        "resident",
        "surgeon",
        "pathologist",
        "psychologist",
        "anesthesiologist",
        "optometrist",
        "nurse",
        "practitioner",
        "patient",
        "name",
        "dob",
        "mrn",
        "date",
        "page",
        "pcp",
        "phys",
        "bill",
        "under",
        "referring",
        "medicare",
        "summa",
        "dermatology",
        "dermatologie",
        "dermatologic",
        "surgery",
        "assistant",
        "scribe",
        "signed",
        "electronically",
        "admission",
        "information",
        "internal",
        "medicine",
        "specialty",
        "wellness",
        "exam",
        "general",
        "meding",
        "certer",
        "center",
        "nertheast",
        "northeast",
        "akron",
        "street",
        "market",
        "work",
        "fax",
        "location",
        "preop",
        "postop",
        "diagnosis",
        "progress",
        "note",
        "notes",
        "printed",
        "encounter",
        "time",
        "discharge",
        "disch",
        "admit",
        "specimen",
        "reviewed",
        "am",
        "pm",
    }
)
# Lowercase words allowed inside a name; any other lowercase word is prose.
_NAME_PARTICLES = frozenset({"van", "von", "de", "da", "del", "der", "di", "du", "la", "le", "st"})


@dataclass
class ProviderHit:
    key: str
    region: str
    sentence: str
    scale: str = ""
    ner_text: str = ""
    value: str = ""
    score: float = 0.0
    accepted: bool = False
    selected: bool = False
    source: str = "ner"
    profile: str = "standard"
    member_hint: str = ""
    key_hit: KeyHit | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class RoleMemberHint:
    """Keyless member name found above a designation provider block."""

    key: str
    region: str
    sentence: str
    value: str
    score: float
    source: str = "keyless_role_block"


def _norm_suffix(text: str) -> str:
    return re.sub(r"[\s.]+", "", text or "").casefold()


@lru_cache(maxsize=1)
def load_suffixes() -> dict[str, str]:
    """normalized form -> preferred display form from suffixes.txt."""
    mapping: dict[str, str] = {}
    if not _SUFFIXES_PATH.is_file():
        return mapping
    for raw in _SUFFIXES_PATH.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        mapping[_norm_suffix(text)] = text
    return mapping


@lru_cache(maxsize=1)
def load_role_aliases() -> frozenset[str]:
    """All designation/role key spellings (casefolded)."""
    if not _ROLES_PATH.is_file():
        return frozenset()
    try:
        data = json.loads(_ROLES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    aliases: set[str] = set()
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            for alias in item.get("aliases") or []:
                text = str(alias).strip()
                if text:
                    aliases.add(text.casefold())
            role = str(item.get("role") or "").strip()
            if role:
                aliases.add(role.casefold())
    return frozenset(aliases)


def is_role_key(key: str) -> bool:
    return key.strip().casefold() in load_role_aliases()


def provider_profile(key: str) -> str:
    folded = key.strip().casefold()
    if folded in _EDGE_ONLY_KEYS:
        return "edge_only"
    if is_role_key(key):
        return "role"
    return "standard"


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def accept_name(text: str) -> str:
    """Person name part only: 2-4 words, no label/credential junk."""
    catalog = load_suffixes()
    tokens = _tokens(text)
    while tokens and tokens[0].casefold() in _NON_NAME:
        tokens.pop(0)
    while tokens and tokens[-1].casefold() in _NON_NAME:
        tokens.pop()
    while tokens and _norm_suffix(tokens[0]) in catalog:
        tokens.pop(0)
    while tokens and _norm_suffix(tokens[-1]) in catalog:
        tokens.pop()
    if not tokens:
        return ""
    if any(token.casefold() in _NON_NAME for token in tokens):
        return ""
    if any(_norm_suffix(token) in catalog for token in tokens):
        return ""
    if any(token[0].islower() and token not in _NAME_PARTICLES for token in tokens):
        return ""
    if not _MIN_WORDS <= len(tokens) <= _MAX_WORDS:
        return ""
    return _CLEAN.sub(" ", " ".join(tokens)).strip()


def key_span(sentence: str, words, indexes: list[int], key: str = "") -> tuple[int, int] | None:
    """The key's characters in the sentence; inside a line read as one OCR word
    ('Author : Davis, Alfred H III, PA') only the key's own letters."""
    del sentence
    wanted = set(indexes)
    offset = 0
    start = end = None
    spans: list[tuple[int, str]] = []
    for word in words:
        if offset:
            offset += 1
        if word.index in wanted:
            if start is None:
                start = offset
            end = offset + len(word.content)
            spans.append((offset, word.content))
        offset += len(word.content)
    if start is None:
        return None
    parts = _WORD.findall(key.casefold())
    if len(spans) == 1 and parts:
        at, content = spans[0]
        found = re.search(r"[^A-Za-z0-9]*".join(map(re.escape, parts)), content, flags=re.IGNORECASE)
        if found:
            return at + found.start(), at + found.end()
    return start, end


def _word_gap(sentence: str, key_at: tuple[int, int], span_at: tuple[int, int]) -> tuple[int, int]:
    key_start, key_end = key_at
    span_start, span_end = span_at
    if span_end <= key_start:
        middle = sentence[span_end:key_start]
        after = 1
    elif span_start >= key_end:
        middle = sentence[key_end:span_start]
        after = 0
    else:
        middle = ""
        after = 0
    return len(middle.split()), after


def geometry_confidence(word_gap: int) -> float:
    return round(max(0.55, 0.92 - 0.12 * word_gap), 4)


def _looks_like_credential(piece: str) -> bool:
    letters = [ch for ch in piece if ch.isalpha()]
    if not letters:
        return False
    return any(ch.isupper() for ch in letters)


def parse_trailing_suffixes(tail: str) -> str:
    catalog = load_suffixes()
    if not catalog:
        return ""
    cleaned = _EDGE.sub("", tail or "")
    if not cleaned:
        return ""
    chunks = [part.strip() for part in _SPLIT.split(cleaned) if part.strip()]
    tokens: list[str] = []
    for chunk in chunks:
        tokens.extend(part for part in re.split(r"\s+", chunk) if part)

    collected: list[str] = []
    index = 0
    while index < len(tokens):
        matched: tuple[int, str] | None = None
        for width in range(min(4, len(tokens) - index), 0, -1):
            piece = " ".join(tokens[index : index + width])
            key = _norm_suffix(piece)
            if key in catalog and _looks_like_credential(piece):
                matched = (width, catalog[key])
                break
        if matched is None:
            break
        collected.append(matched[1])
        index += matched[0]
    if not collected:
        return ""
    deduped: list[str] = []
    for item in collected:
        if not deduped or deduped[-1].casefold() != item.casefold():
            deduped.append(item)
    return ", ".join(deduped)


def with_suffixes(sentence: str, name: str, name_at: tuple[int, int]) -> str:
    start, end = name_at
    name_text = name or sentence[start:end]
    suffixes = parse_trailing_suffixes(sentence[end:])
    if not suffixes:
        return ""
    return f"{name_text}, {suffixes}"


def is_person_provider_value(value: str) -> bool:
    """True only when value is a person name, with optional credential designation(s)."""
    return bool(person_core_from_provider_value(value))


def person_core_from_provider_value(value: str) -> str:
    """Return the person-name core, or '' if the value is not a provider person name."""
    text = _CLEAN.sub(" ", (value or "").strip())
    if not text:
        return ""
    # Direct accept (strips leading/trailing designation tokens).
    core = accept_name(text)
    if core:
        return core
    # "Last, First, MD" / "Name, APRN-CNP" — strip trailing credential chunks then retry.
    catalog = load_suffixes()
    pieces = [part.strip() for part in re.split(r"[,]+", text) if part.strip()]
    while pieces:
        tail = pieces[-1]
        tail_tokens = re.split(r"[\s.]+", tail)
        if tail_tokens and all(_norm_suffix(tok) in catalog for tok in tail_tokens if tok):
            pieces.pop()
            continue
        # Mixed "MAHAJAN MD" last chunk — drop trailing credential tokens.
        toks = _tokens(tail)
        while toks and _norm_suffix(toks[-1]) in catalog:
            toks.pop()
        if toks:
            pieces[-1] = " ".join(toks)
        break
    if not pieces:
        return ""
    return accept_name(" ".join(pieces))


def _name_span_in_sentence(sentence: str, name: str, hint_start: int | None = None) -> tuple[int, int] | None:
    if hint_start is not None and hint_start >= 0:
        return hint_start, hint_start + len(name)
    found = re.search(re.escape(name), sentence, flags=re.IGNORECASE)
    if not found:
        return None
    return found.start(), found.end()


def _widen(text: str, start: int, end: int, key_at: tuple[int, int]) -> tuple[int, int]:
    """Grow a partial NER name over the rest of a 'Last, First' name: the surname before it
    ('Davis, | Alfred H III, PA') or the first name after it when credentials follow
    ('Susoiu Tcaciuc | , Daniela, MD')."""
    before = re.search(r"([A-Z][A-Za-z'\-]+),\s*$", text[:start])
    if before and before.start(1) >= key_at[1]:
        start = before.start(1)
    after = re.match(r",\s*[A-Z][a-z][A-Za-z'\-]*(?=,)", text[end:])
    if after and parse_trailing_suffixes(text[end + after.end():]):
        end += after.end()
    return start, end


def nearest_provider(
    sentence: str,
    key_at: tuple[int, int] | None,
    raws: list[dict] | None = None,
    *,
    require_suffix: bool = True,
    prefer_before_key: bool = False,
) -> tuple[str, float, str, str]:
    """Nearest person name (+ optional designation). Non-person values are rejected."""
    text = sentence or ""
    if key_at is None:
        return "", 0.0, "", ""

    candidates: list[tuple[str, str, int, int, float, str]] = []
    for raw in raws or []:
        ner_text = str(raw.get("text") or "").strip()
        name = accept_name(ner_text)
        if not name:
            continue
        start = raw.get("start")
        end = raw.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            span = _name_span_in_sentence(text, ner_text)
            if span is None:
                continue
            start, end = span
        wide = _widen(text, start, end, key_at)
        if wide != (start, end):
            start, end = wide
            name = accept_name(text[start:end]) or name
        full = with_suffixes(text, name, (start, end))
        if require_suffix and not full:
            continue
        if not full:
            full = name
        if not is_person_provider_value(full):
            continue
        candidates.append((full, ner_text, start, end, float(raw.get("score") or 0), "ner"))

    if candidates:

        def rank(item: tuple[str, str, int, int, float, str]) -> tuple:
            gap, after = _word_gap(text, key_at, (item[2], item[3]))
            before = 0 if prefer_before_key and item[3] <= key_at[0] else 1
            return (before, gap, after, -item[4])

        full, ner_text, start, end, score, source = min(candidates, key=rank)
        if score <= 0:
            score = geometry_confidence(_word_gap(text, key_at, (start, end))[0])
        return full, round(score, 4), ner_text, source

    entries = [(found.group(), found.start(), found.end()) for found in re.finditer(r"\S+", text)]
    line_of = [text.count("\n", 0, item[1]) for item in entries]
    geometry: list[tuple[str, str, int, int]] = []
    for start in range(len(entries)):
        for width in range(_MIN_WORDS, _MAX_WORDS + 1):
            end = start + width
            if end > len(entries) or line_of[end - 1] != line_of[start]:
                break
            chunk = " ".join(item[0] for item in entries[start:end])
            name = accept_name(chunk)
            if not name:
                continue
            span = (entries[start][1], entries[end - 1][2])
            full = with_suffixes(text, name, span)
            if require_suffix and not full:
                continue
            value = full or name
            if not is_person_provider_value(value):
                continue
            geometry.append((value, name, span[0], span[1]))
    if not geometry:
        return "", 0.0, "", ""

    def geo_rank(item: tuple[str, str, int, int]) -> tuple:
        gap, after = _word_gap(text, key_at, (item[2], item[3]))
        before = 0 if prefer_before_key and item[3] <= key_at[0] else 1
        return (before, gap, after)

    full, ner_text, start, end = min(geometry, key=geo_rank)
    score = geometry_confidence(_word_gap(text, key_at, (start, end))[0])
    return full, score, ner_text, "geometry"


def band_has_id(text: str) -> bool:
    return bool(_ID_LIKE.search(text or ""))


def _require_suffix_for_key(key: str) -> bool:
    folded = key.strip().casefold()
    if folded in _STRICT_SUFFIX_KEYS:
        return True
    if folded.endswith(" by"):
        return False
    if is_role_key(key):
        return False
    return False


def _role_windows(
    hit: KeyHit,
    words: list[Word],
    page_w: float,
    page_h: float,
) -> tuple[list[Word], list[Word], str, str]:
    """Return (value_words upward+key, id_band_words, value_sentence, band_sentence)."""
    key_words = [word for word in words if word.index in set(hit.word_indexes)]
    if not key_words:
        key_words = hit.value_words
    if not key_words:
        return [], [], "", ""
    left = min(word.box.left for word in key_words)
    top = min(word.box.top for word in key_words)
    right = max(word.box.right for word in key_words)
    bottom = max(word.box.bottom for word in key_words)
    up = upward_box(left, top, right, bottom, page_w, page_h, expand_for_key(hit.key))
    # Include a thin strip through the key line so key_span still resolves.
    value_box = (up[0], up[1], up[2], max(up[3], bottom))
    value_words = words_in_box(words, value_box)
    band = full_width_band(top, bottom, page_w, page_h, ROLE_ID_BAND_FRAC)
    band_words = words_in_box(words, band)
    value_text = " ".join(word.content for word in value_words).strip()
    band_text = " ".join(word.content for word in band_words).strip()
    return value_words, band_words, value_text, band_text


def _member_above_provider(
    sentence: str,
    provider_value: str,
    key_at: tuple[int, int] | None,
    raws: list[dict],
) -> str:
    """Second person name in the block, preferring text above the provider/key."""
    provider_fold = accept_name(provider_value).casefold() or provider_value.casefold()
    key_start = key_at[0] if key_at else len(sentence)
    best = ""
    best_score = -1.0
    best_end = key_start + 1
    for raw in raws or []:
        ner_text = str(raw.get("text") or "").strip()
        name = accept_name(ner_text)
        if not name or name.casefold() == provider_fold:
            continue
        # Skip if this NER span is just the provider with credentials stripped.
        if provider_fold and name.casefold() in provider_fold:
            continue
        start = raw.get("start")
        end = raw.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            span = _name_span_in_sentence(sentence, ner_text)
            if span is None:
                continue
            start, end = span
        if end > key_start:
            continue
        score = float(raw.get("score") or 0)
        if end < best_end or (end == best_end and score > best_score):
            best = name
            best_score = score
            best_end = end
    return best


def _line_text(words: list[Word]) -> str:
    """The words joined like ' '.join, with a line break where the next word starts a new
    line (same offsets), so a name is never read across two lines."""
    parts: list[str] = []
    for n, word in enumerate(words):
        if n:
            prev = words[n - 1]
            height = max(min(prev.box.height(), word.box.height()), 1.0)
            parts.append(" " if abs(word.box.cy - prev.box.cy) < 0.6 * height else "\n")
        parts.append(word.content)
    return "".join(parts)


def _split_at_lines(raws: list[dict], sentence: str) -> list[dict]:
    """NER spans crossing a line break become one span per line ('Emma' / 'Susoiu Tcaciuc')."""
    out: list[dict] = []
    for raw in raws:
        start, end = raw.get("start"), raw.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or "\n" not in sentence[start:end]:
            out.append(raw)
            continue
        offset = start
        for piece in sentence[start:end].split("\n"):
            text = piece.strip()
            if text:
                at = offset + piece.index(text)
                out.append({**raw, "text": text, "start": at, "end": at + len(text)})
            offset += len(piece) + 1
    return out


def _key_words(hit: KeyHit, words: list[Word]) -> list[Word]:
    indexes = set(hit.word_indexes)
    return sorted((word for word in words if word.index in indexes), key=lambda word: word.box.left)


def _written_after(hit: KeyHit, words: list[Word]) -> bool:
    """The key follows its value: bracketed ('Allison Moosally, MD (Primary Provider)') or
    after 'as' ('Lauren N Burns, DO as PCP')."""
    key_words = _key_words(hit, words)
    if not key_words:
        return False
    if key_words[0].content[:1] in "({[":
        return True
    before = next((word for word in words if word.index == key_words[0].index - 1), None)
    return (
        before is not None
        and before.content.casefold() == "as"
        and abs(before.box.cy - key_words[0].box.cy) < 0.6 * max(key_words[0].box.height(), 1.0)
    )


def _before_on_line(hit: KeyHit, words: list[Word], hits: list[KeyHit]) -> list[Word]:
    """The same-line words right before the key, back to another key or a wide gap."""
    key_words = _key_words(hit, words)
    first = key_words[0]
    line_h = median_height(words)
    other_keys = {index for other in hits if other is not hit for index in other.word_indexes}
    line = sorted(
        (
            word for word in words
            if abs(word.box.cy - first.box.cy) <= 0.6 * line_h
            and word.box.right <= first.box.left + 0.5 * line_h
            and word.index not in hit.word_indexes
        ),
        key=lambda word: word.box.left,
        reverse=True,
    )
    run: list[Word] = []
    edge = first.box.left
    for word in line:
        if word.index in other_keys or edge - word.box.right > 3.0 * line_h:
            break
        run.append(word)
        edge = word.box.left
    return run[::-1]


def extract_box(
    hit: KeyHit,
    words: list[Word] | None = None,
    page_w: float = 0.0,
    page_h: float = 0.0,
    hits: list[KeyHit] | None = None,
) -> ProviderHit:
    """One provider_name key → one row. Accepted only if value is a person name."""
    profile = provider_profile(hit.key)
    scale = f"{expand_for_key(hit.key):g}x"

    if profile == "edge_only" and not is_edge_region(hit.region):
        return ProviderHit(
            key=hit.key,
            region=hit.region,
            sentence=hit.value_text,
            scale=scale,
            profile=profile,
        )

    sentence = hit.value_text
    value_words = hit.value_words
    member_hint = ""
    prefer_before = False
    require_suffix = _require_suffix_for_key(hit.key)

    if profile == "role" and words is not None:
        value_words, _band_words, value_text, band_text = _role_windows(hit, words, page_w, page_h)
        sentence = value_text or hit.value_text
        if not band_has_id(band_text):
            return ProviderHit(
                key=hit.key,
                region=hit.region,
                sentence=sentence,
                scale=scale,
                profile=profile,
                source="role_id_gate",
            )
        prefer_before = True
        require_suffix = False
    elif words is not None and _written_after(hit, words):
        before = _before_on_line(hit, words, hits or [])
        if not before:
            return ProviderHit(
                key=hit.key,
                region=hit.region,
                sentence=hit.value_text,
                scale=scale,
                profile=profile,
                source="no_value_before_key",
            )
        value_words = before + _key_words(hit, words)
        sentence = " ".join(word.content for word in value_words)
        prefer_before = True

    if value_words and sentence == " ".join(word.content for word in value_words):
        sentence = _line_text(value_words)
    raws = _split_at_lines(predict(sentence, LABELS), sentence)
    stored = sentence.replace("\n", " ")
    span = key_span(sentence, value_words, hit.word_indexes, hit.key)
    if span is None and value_words:
        span = key_span(
            sentence,
            value_words,
            [word.index for word in value_words if word.index in set(hit.word_indexes)],
            hit.key,
        )
    # Inside a line read as one OCR word: 'Allison Moosally, MD (Primary Provider)'.
    if span is not None and re.search(r"(?:[({\[]|\bas)\s*$", sentence[: span[0]]):
        prefer_before = True
    value, score, ner_text, source = nearest_provider(
        sentence,
        span,
        raws,
        require_suffix=require_suffix,
        prefer_before_key=prefer_before,
    )

    # If a strict-suffix key failed, still allow a clear person name (all keys).
    if not value and require_suffix:
        soft, soft_score, soft_ner, soft_source = nearest_provider(
            sentence,
            span,
            raws,
            require_suffix=False,
            prefer_before_key=prefer_before,
        )
        if soft and soft_score >= 0.55 and is_person_provider_value(soft):
            value, score, ner_text, source = soft, soft_score, soft_ner, soft_source

    if value and not is_person_provider_value(value):
        value, score, ner_text, source = "", 0.0, "", ""

    if value and profile == "role":
        hint = _member_above_provider(sentence, value, span, raws)
        member_hint = hint if is_person_provider_value(hint) else ""

    if value and is_person_provider_value(value):
        return ProviderHit(
            key=hit.key,
            region=hit.region,
            sentence=stored,
            scale=scale,
            ner_text=(ner_text or value).replace("\n", " "),
            value=value.replace("\n", " "),
            score=score,
            accepted=True,
            source=source or "ner",
            profile=profile,
            member_hint=member_hint,
        )
    return ProviderHit(
        key=hit.key,
        region=hit.region,
        sentence=stored,
        scale=scale,
        profile=profile,
        source=source or ("rejected_non_person" if (ner_text or sentence) else ""),
    )


def extract_provider_hits(
    hits: list[KeyHit],
    words: list[Word],
    page_w: float,
    page_h: float,
) -> list[ProviderHit]:
    """Role keys anywhere; every other key, header/footer first (selection prefers them);
    Bill Under edge-only."""
    candidates = [hit for hit in hits if hit.trusted and hit.field == "provider_name" and hit.value_text]
    role_hits = [hit for hit in candidates if provider_profile(hit.key) == "role"]
    other_hits = [hit for hit in candidates if provider_profile(hit.key) != "role"]
    edge = [hit for hit in other_hits if is_edge_region(hit.region)]
    mid = [hit for hit in other_hits if not is_edge_region(hit.region)]
    rows = [linked(extract_box(hit, words, page_w, page_h, hits), hit) for hit in role_hits + edge + mid]
    mark_selected(rows)
    return rows


def role_member_hints(rows: list[ProviderHit]) -> list[RoleMemberHint]:
    """Member names discovered above accepted designation provider blocks."""
    hints: list[RoleMemberHint] = []
    seen: set[str] = set()
    for row in rows:
        if row.profile != "role" or not row.accepted or not row.member_hint:
            continue
        text = row.member_hint.casefold()
        if text in seen:
            continue
        seen.add(text)
        hints.append(
            RoleMemberHint(
                key=f"near:{row.key}",
                region=row.region if is_edge_region(row.region) else f"role_{row.region or 'mid'}",
                sentence=row.sentence,
                value=row.member_hint,
                score=max(0.55, row.score),
            )
        )
    return hints


def extract_role_block_members(
    hits: list[KeyHit],
    words: list[Word],
    page_w: float,
    page_h: float,
) -> list[RoleMemberHint]:
    """Run designation provider path only; return keyless member names above providers."""
    role_hits = [
        hit
        for hit in hits
        if hit.trusted and hit.field == "provider_name" and hit.value_text and is_role_key(hit.key)
    ]
    rows = [extract_box(hit, words, page_w, page_h) for hit in role_hits]
    return role_member_hints(rows)


def mark_selected(rows: list[ProviderHit]) -> None:
    """Prefer header/footer person-name hits; only then mid. Non-persons never win."""
    accepted = [
        row
        for row in rows
        if row.accepted and row.value and is_person_provider_value(row.value)
    ]
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
