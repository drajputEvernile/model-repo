"""Read the member (patient) name next to a name key, or keyless beside DOB/ID in header/footer.

Rules from Data/examp/mname.csv:
  - Words are read in raw OCR order (sorting by line position interleaves lines).
  - Value is the name as printed ("Williams, David E", "MARKO, Nicole J", "Thomas Emma Jr."),
    minus leading titles (MR./MRS./MS./DR.) and quoted nicknames ("Russ").
  - Keyed: name right after the key on its line, else the column-aligned name just below
    (table headers). Keyed "Last + initial" ("Anderson S") is allowed. Mid-page keyed names
    need NER support.
  - The name stops at a line end, a digit, "(", a word ending in ":", or a word that belongs
    to another key on the page ("Thomas David | Admil Dale").
  - Last Name + First Name keys on one page combine into one name.
  - Insured Name only wins when no patient-type key gave a name.
  - Keyless (header/footer only, beside a DOB/ID key): the name must sit right before the
    anchor on its line, or be NER-backed; provider names (credential after, "by:"/"Dr"
    before) and clinic/header words are rejected.
  - Keyless running header: the name that starts a header line with 'Page N' and a date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..electronic_signature.extract import is_credential
from ..util.keys import KeyHit, linked
from ..util.model import predict
from ..util.geometry import (
    FOOTER_FRAC,
    HEADER_FRAC,
    KEYLESS_BAND_FRAC,
    boxes_overlap,
    is_edge_region,
    edge_band,
    median_height,
    region_priority,
    union_boxes,
)
from ..util.dates import find_dates
from ..util.tokens import Token, key_token_span, ordered_tokens, same_line
from ..util.window import expand_for_key, words_in_box

LABELS = ["person"]
_NAME_WORD = re.compile(r"^[A-Z][A-Za-z'\-]*\.?$")
_MAX_CORE = 4
_NER_MIN = 0.5
_PAGE_NUMBER = re.compile(r"#?\d{1,3}")

# Keyless name zones — same independent maxes as keyed header/footer tracks.
KEYLESS_TOP_FRAC = HEADER_FRAC
KEYLESS_BOTTOM_FRAC = FOOTER_FRAC
ANCHOR_FIELDS = frozenset({"dob", "member_id"})

_TITLES = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr"})
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})
_CONNECTORS = frozenset({":", "-", "–", "|", "#", "="})
_QUOTES = "\"'“”‘’("
_SPLIT_KEYS = {"last name": "last", "first name": "first"}
_INSURED_KEYS = frozenset({"insured name"})
_PROVIDER_BEFORE = frozenset(
    {"by", "provider", "physician", "dr", "pcp", "attending", "author", "signed", "seen"}
)

_NOT_NAME = frozenset(
    {
        # labels / demographics
        "patient", "name", "ptnt", "pt", "the", "dob", "mrn", "date", "birth", "male",
        "female", "legal", "account", "acct", "record", "number", "member", "chart",
        "policy", "insurance", "insured", "id", "demographics", "gender", "identity",
        "admit", "discharge", "phone", "sex", "age", "page", "refer", "doctor", "visit",
        "note", "notes", "of", "and", "for", "a", "an", "preferred", "fin", "har",
        "location", "position", "time", "status", "information", "service", "document",
        "primary", "care", "office", "perform", "auth", "result", "verified", "client",
        "data", "first", "last", "middle", "ssn", "address", "city", "state", "zip",
        # clinic / header words
        "dermatology", "surgery", "skin", "allied", "clinic", "center", "centre",
        "hospital", "health", "medical", "medicine", "associates", "group", "eye", "eyes",
        "family", "practice", "services", "imaging", "radiology", "laboratory", "lab",
        "pharmacy", "university", "institute", "physicians", "specialists", "orthopedics",
        "pediatrics", "cardiology", "oncology", "urology", "neurology", "internal",
        "urgent", "northeast", "ohio", "summa", "fax", "from", "www", "com",
        # document words
        "encounter", "progress", "report", "summary", "history", "assessment", "plan",
        "impression", "chief", "complaint", "reason", "preview", "printed", "prin",
        "subjective", "objective", "exam", "xr", "ct", "mri", "dsc", "pms",
        "right", "left", "bilateral", "chest", "ribs", "views", "view", "include", "lung",
        "spine", "knee", "hip", "shoulder", "ankle", "wrist", "contrast", "without", "with",
        # time words
        "am", "pm", "pdt", "edt", "est", "pst", "cst", "cdt",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    }
)


@dataclass
class NameHit:
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
    key_hit: KeyHit | None = field(default=None, repr=False, compare=False)


@dataclass
class _Parsed:
    first: int
    last: int
    core: list[str]
    credential_after: bool


def _all_caps(text: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= 2 and letters.isupper()


def _title_case(text: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= 2 and not letters.isupper()


def _is_name_word(text: str) -> bool:
    if not _NAME_WORD.match(text):
        return False
    bare = text.rstrip(".")
    if len(bare) == 1:
        return True
    return bare.casefold() not in _NOT_NAME and not is_credential(text)


def _parse(
    tokens: list[Token],
    start: int,
    line_h: float,
    stops: set[int],
    *,
    allow_initial: bool,
    min_full: int = 2,
) -> _Parsed | None:
    """Printed member name starting at tokens[start] (leading titles skipped)."""
    index = start
    while index < len(tokens) and tokens[index].text.strip(",;.").casefold() in _TITLES:
        index += 1
    first = index
    core: list[str] = []
    last: int | None = None
    while index < len(tokens) and index not in stops:
        token = tokens[index]
        if last is not None and not same_line(tokens[index - 1], token, line_h):
            break
        raw = token.text
        if raw[:1] in _QUOTES or raw.rstrip(",;").endswith(":"):
            break
        stripped = raw.strip(",;")
        if core and stripped.rstrip(".").casefold() in _SUFFIXES:
            last = index
            index += 1
            break
        if not _is_name_word(stripped) or len(core) >= _MAX_CORE:
            break
        if len(core) >= 2 and _title_case(core[-1]) and _all_caps(stripped):
            # 'Benjamin David DAGENE' ends the name; 'Young Margaret ANO SHIN ...' is a caps heading.
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if (
                following is not None
                and same_line(token, following, line_h)
                and _all_caps(following.text.strip(",;"))
                and _is_name_word(following.text.strip(",;"))
            ):
                break
        core.append(stripped)
        last = index
        index += 1
    if last is None:
        return None
    full = sum(1 for word in core if len(word.rstrip(".")) >= 2)
    if full < min_full and not (allow_initial and full == 1 and len(core) >= 2):
        return None
    credential_after = (
        index < len(tokens)
        and same_line(tokens[index - 1], tokens[index], line_h)
        and is_credential(tokens[index].text)
    )
    return _Parsed(first=first, last=last, core=core, credential_after=credential_after)


def _printed(tokens: list[Token], parsed: _Parsed, text: str) -> str:
    return text[tokens[parsed.first].start : tokens[parsed.last].end].strip(" ,;:")


def _stop_positions(tokens: list[Token], hits: list[KeyHit], skip: KeyHit | None) -> set[int]:
    """Token positions spelling any other key on the page."""
    present = {token.word.index for token in tokens}
    stops: set[int] = set()
    for other in hits:
        if other is skip or not present.intersection(other.word_indexes):
            continue
        span = key_token_span(tokens, other.word_indexes, other.key)
        if span is not None:
            stops.update(range(span[0], span[1] + 1))
    return stops


def _ner_support(raws: list[dict], start: int, end: int) -> tuple[str, float]:
    """Best person span covering most of [start, end) ('Young Margaret ANO' ≠ 'ANO SHIN')."""
    best_text, best_score = "", 0.0
    for raw in raws:
        raw_start, raw_end = raw.get("start"), raw.get("end")
        if not isinstance(raw_start, int) or not isinstance(raw_end, int):
            continue
        overlap = min(end, raw_end) - max(start, raw_start)
        union = max(end, raw_end) - min(start, raw_start)
        score = float(raw.get("score") or 0)
        if union > 0 and overlap / union >= 0.5 and score > best_score:
            best_text, best_score = str(raw.get("text") or ""), score
    return best_text, best_score


def _same_row(left: Token, right: Token, line_h: float) -> bool:
    """Right token is on the left token's row (any gap: tabbed key/value)."""
    return (
        abs(left.word.box.cy - right.word.box.cy) <= 0.6 * line_h
        and right.word.box.left >= left.word.box.left
    )


def _row_start(tokens: list[Token], key_last: int, stops: set[int], line_h: float) -> int | None:
    """Nearest token right of the key on its row (stacked key columns read out of order)."""
    key_box = tokens[key_last].word.box
    row = sorted(
        (token.word.box.left, pos)
        for pos, token in enumerate(tokens)
        if pos != key_last
        and abs(token.word.box.cy - key_box.cy) <= 0.6 * line_h
        and token.word.box.left >= key_box.right - 1
        and token.text not in _CONNECTORS
    )
    if not row or row[0][1] in stops:
        return None
    return row[0][1]


def _below_starts(tokens: list[Token], key_first: int, key_last: int, line_h: float) -> list[int]:
    """Leftmost token in the key's column on each line just below it, nearest line first
    (a centred header sits over the middle of its value: 'Name' over 'Miller Jennifer')."""
    key_left = tokens[key_first].word.box.left
    key_right = tokens[key_last].word.box.right
    key_bottom = max(tokens[pos].word.box.bottom for pos in range(key_first, key_last + 1))
    best: dict[int, tuple[float, int]] = {}
    for pos, token in enumerate(tokens):
        box = token.word.box
        if box.top <= key_bottom - 0.3 * line_h or box.top - key_bottom > 3.0 * line_h:
            continue
        if box.right < key_left - line_h or box.left > key_right + 2.0 * line_h:
            continue
        line = round((box.top - key_bottom) / line_h)
        rank = (box.left, pos)
        if line not in best or rank < best[line]:
            best[line] = rank
    return [best[line][1] for line in sorted(best)]


def extract_box(hit: KeyHit, hits: list[KeyHit] | None = None) -> NameHit:
    """One name key → the printed name right after it (or just below it)."""
    scale = f"{expand_for_key(hit.key):g}x"
    tokens, text = ordered_tokens(hit.value_words)
    blank = NameHit(key=hit.key, region=hit.region, sentence=text, scale=scale)
    span = key_token_span(tokens, hit.word_indexes, hit.key)
    if span is None:
        return blank
    key_first, key_last = span
    line_h = median_height([token.word for token in tokens])
    stops = _stop_positions(tokens, hits or [], hit)
    split = hit.key.strip().casefold() in _SPLIT_KEYS
    min_full = 1 if split else 2

    start = key_last + 1
    while start < len(tokens) and tokens[start].text in _CONNECTORS:
        start += 1
    def parse_at(pos: int | None) -> _Parsed | None:
        if pos is None:
            return None
        found = _parse(tokens, pos, line_h, stops, allow_initial=True, min_full=min_full)
        return None if found is None or found.credential_after else found

    parsed = None
    source, geometry = "after_key", 0.92
    if start < len(tokens) and _same_row(tokens[key_last], tokens[start], line_h):
        parsed = parse_at(start)
    if parsed is None:
        parsed = parse_at(_row_start(tokens, key_last, stops, line_h))
    if parsed is None:
        source, geometry = "below_key", 0.8
        for below in _below_starts(tokens, key_first, key_last, line_h):
            parsed = parse_at(below)
            if parsed is not None:
                break
    if parsed is None:
        return blank
    if split:
        parsed = _Parsed(parsed.first, parsed.first, parsed.core[:1], False)

    value = _printed(tokens, parsed, text)
    raws = predict(text, LABELS)
    ner_text, ner_score = _ner_support(raws, tokens[parsed.first].start, tokens[parsed.last].end)
    if source == "below_key" and not is_edge_region(hit.region) and ner_score <= 0 and not split:
        return blank
    return NameHit(
        key=hit.key,
        region=hit.region,
        sentence=text,
        scale=scale,
        ner_text=ner_text or value,
        value=value,
        score=round(max(ner_score, geometry), 4),
        accepted=True,
        source="split_part" if split else source,
    )


def _anchor_zone(cy: float, page_h: float) -> str | None:
    """Keyless only in header/footer max bands. Mid is out of scope."""
    if page_h <= 0:
        return None
    frac = cy / page_h
    if frac <= KEYLESS_TOP_FRAC:
        return "keyless_header"
    if frac >= 1.0 - KEYLESS_BOTTOM_FRAC:
        return "keyless_footer"
    return None


def keyless_band(anchor: KeyHit, page_w: float, page_h: float) -> tuple[float, float, float, float]:
    """±KEYLESS_BAND_FRAC page height around the key, full page width horizontally."""
    pad_y = KEYLESS_BAND_FRAC * page_h if page_h else 8.0
    left = 0.0
    right = page_w if page_w else anchor.box.right + 200.0
    top = max(0.0, anchor.box.top - pad_y)
    bottom = min(page_h, anchor.box.bottom + pad_y) if page_h else anchor.box.bottom + pad_y
    return left, top, right, bottom


def extract_keyless(
    anchor: KeyHit,
    words,
    page_w: float,
    page_h: float,
    hits: list[KeyHit] | None = None,
) -> NameHit | None:
    """Name with no name key: beside a DOB/ID anchor that sits in the header/footer."""
    if not anchor.trusted or anchor.field not in ANCHOR_FIELDS:
        return None
    zone = _anchor_zone(anchor.box.cy, page_h)
    if zone is None:
        return None
    band_words = words_in_box(words, keyless_band(anchor, page_w, page_h))
    tokens, text = ordered_tokens(band_words)
    label = f"near:{anchor.key}"
    pct = f"±{KEYLESS_BAND_FRAC * 100:g}%"
    if not tokens:
        return None
    span = key_token_span(tokens, anchor.word_indexes, anchor.key)
    if span is None:
        return None
    anchor_first = span[0]
    line_h = median_height([token.word for token in tokens])
    stops = _stop_positions(tokens, hits or [], None) | set(range(span[0], span[1] + 1))
    raws = predict(text, LABELS)
    anchor_cy = tokens[anchor_first].word.box.cy

    best: tuple[tuple, _Parsed, str, float] | None = None
    for start in range(len(tokens)):
        if start in stops:
            continue
        parsed = _parse(tokens, start, line_h, stops, allow_initial=False)
        if parsed is None or parsed.credential_after:
            continue
        before = tokens[parsed.first - 1].text.strip(":,.").casefold() if parsed.first > 0 else ""
        if before in _PROVIDER_BEFORE:
            continue
        ner_text, ner_score = _ner_support(raws, tokens[parsed.first].start, tokens[parsed.last].end)
        same_row = abs(tokens[parsed.last].word.box.cy - anchor_cy) <= 0.6 * line_h
        # Words of other lines read in between ('Stephen V Valerie Fuller, D.O. DOB') don't count.
        between = sum(
            1 for token in tokens[parsed.last + 1 : anchor_first]
            if abs(token.word.box.cy - anchor_cy) <= 0.6 * line_h
        ) if parsed.last < anchor_first else -1
        adjacent = same_row and 0 <= between <= 2
        if not adjacent and ner_score < _NER_MIN:
            continue
        rank = (
            adjacent,
            same_row,
            round(ner_score, 2),
            -abs(tokens[parsed.last].word.box.cy - anchor_cy),
            len(parsed.core),
        )
        if best is None or rank > best[0]:
            best = (rank, parsed, ner_text, ner_score)
    if best is None:
        return NameHit(key=label, region=zone, sentence=text, scale=pct, source="keyless")
    rank, parsed, ner_text, ner_score = best
    value = _printed(tokens, parsed, text)
    return NameHit(
        key=label,
        region=zone,
        sentence=text,
        scale=pct,
        ner_text=ner_text or value,
        value=value,
        score=round(max(ner_score, 0.85 if rank[0] else 0.0), 4),
        accepted=True,
        source="keyless_adjacent" if rank[0] else "keyless_ner",
    )


def extract_running_header(words, page_h: float, hits: list[KeyHit]) -> NameHit | None:
    """A printed record's running header: the name that starts the header line holding a
    'Page N' label and a date ('Sarah S Phillips DD 04/01/2024 Page #3')."""
    text_box = union_boxes([word.box for word in words])
    for at, word in enumerate(words[:-1]):
        if word.content.strip(":").casefold() != "page" or not _PAGE_NUMBER.fullmatch(words[at + 1].content):
            continue
        if edge_band(word.box, page_h, text_box) != "header":
            continue
        height = max(word.box.height(), 1.0)
        line = sorted(
            (other for other in words if abs(other.box.cy - word.box.cy) < 0.6 * height),
            key=lambda other: other.box.left,
        )
        tokens, text = ordered_tokens(line)
        if not tokens or not find_dates(text):
            continue
        stops = _stop_positions(tokens, hits, None)
        parsed = _parse(tokens, 0, median_height(line), stops, allow_initial=True)
        if parsed is None or parsed.first != 0 or parsed.credential_after:
            continue
        value = _printed(tokens, parsed, text)
        return NameHit(
            key="near:Page",
            region="keyless_header",
            sentence=text,
            ner_text=value,
            value=value,
            score=0.8,
            accepted=True,
            source="keyless_running_header",
        )
    return None


def _canon(value: str) -> str:
    """Order-free comparison key: 'Gonzalez, Nicole' == 'Nicole Gonzalez'."""
    words = [word.casefold() for word in re.findall(r"[A-Za-z]{2,}", value)]
    return " ".join(sorted(words))


def combine_split(rows: list[NameHit]) -> list[NameHit]:
    """Last Name + First Name keys on one page → one 'Last First' name row."""
    parts = {_SPLIT_KEYS[row.key.strip().casefold()]: row for row in rows
             if row.accepted and row.source == "split_part"}
    if "last" not in parts or "first" not in parts:
        return rows
    last, first = parts["last"], parts["first"]
    rows.append(
        NameHit(
            key=f"{last.key} + {first.key}",
            region=last.region,
            sentence=f"{last.sentence} | {first.sentence}",
            ner_text=f"{last.value} {first.value}",
            value=f"{last.value} {first.value}",
            score=round(min(last.score, first.score), 4),
            accepted=True,
            source="split_combined",
            key_hit=last.key_hit,
        )
    )
    return rows


def merge_page_names(keyed: list[NameHit], keyless: list[NameHit]) -> list[NameHit]:
    """Keep keyed rows. Drop keyless duplicates of a keyed accepted value."""
    known = {_canon(row.value) for row in keyed if row.accepted and row.value}
    extra: list[NameHit] = []
    seen_keyless: set[str] = set()
    for row in keyless:
        if row.accepted and row.value:
            marker = _canon(row.value)
            if marker in known or marker in seen_keyless:
                continue
            seen_keyless.add(marker)
        extra.append(row)
    return keyed + extra


def mark_selected(rows: list[NameHit]) -> None:
    """Header/footer first, then the most common name, then score. Insured Name is a fallback."""
    accepted = [
        row for row in rows if row.accepted and row.value and row.source != "split_part"
    ]
    patient = [row for row in accepted if row.key.strip().casefold() not in _INSURED_KEYS]
    accepted = patient or accepted
    if not accepted:
        return
    edge = [row for row in accepted if is_edge_region(row.region)]
    pool = edge if edge else accepted
    counts: dict[str, int] = {}
    for row in pool:
        counts[_canon(row.value)] = counts.get(_canon(row.value), 0) + 1
    best_count = max(counts.values())
    tied = [row for row in pool if counts[_canon(row.value)] == best_count]
    winner = max(tied, key=lambda row: (region_priority(row.region), row.score, len(row.value)))
    winner.selected = True


def extract_page(
    hits: list[KeyHit],
    words,
    page_w: float,
    page_h: float,
    role_hints: list | None = None,
) -> list[NameHit]:
    """Keyed: every name key, header/footer first (selection prefers them). Keyless:
    header/footer only + role blocks.

    role_hints: member names above provider designation blocks, when the caller has
    already run Provider extraction on this page (else they are computed here).
    """
    if role_hints is None:
        from ..provider_name.extract import extract_role_block_members

        role_hints = extract_role_block_members(hits, words, page_w, page_h)

    name_keys = [hit for hit in hits if hit.trusted and hit.field == "name" and hit.value_text]
    edge_keys = [hit for hit in name_keys if is_edge_region(hit.region)]
    mid_keys = [hit for hit in name_keys if not is_edge_region(hit.region)]
    keyed = combine_split([linked(extract_box(hit, hits), hit) for hit in edge_keys + mid_keys])
    keyed_boxes = [hit.value_box for hit in name_keys if hit.value_box is not None]

    keyless: list[NameHit] = []
    for hit in hits:
        if not (hit.trusted and hit.field in ANCHOR_FIELDS):
            continue
        band = keyless_band(hit, page_w, page_h)
        if any(boxes_overlap(band, box) for box in keyed_boxes):
            continue
        row = extract_keyless(hit, words, page_w, page_h, hits)
        if row is not None:
            keyless.append(linked(row, hit))

    running = extract_running_header(words, page_h, hits)
    if running is not None:
        keyless.append(running)

    for hint in role_hints:
        keyless.append(
            NameHit(
                key=hint.key,
                region=hint.region,
                sentence=hint.sentence,
                ner_text=hint.value,
                value=hint.value,
                score=hint.score,
                accepted=True,
                source=hint.source,
            )
        )

    rows = merge_page_names(keyed, keyless)
    mark_selected(rows)
    return rows
