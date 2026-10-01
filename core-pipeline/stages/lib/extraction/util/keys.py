"""Find catalog keys on an OCR page and decide which ones are trusted.

Each field keeps its own keys.json under {field folder}/keys.json.
Missing key files are skipped so fields can be added one at a time.

{FieldFolder}/key_blocklist.json lists, per key, words that make it prose rather than a label
when they come right before or after it ("The patient", "Patient has"). Reviewers grow it
from the Review UI ("Not a real key here").

One spot, one key: keys are matched longest first, a word belongs to one key, and a shorter
key is skipped next to a longer key of the same field that contains it ("Provider" under
"Attending Provider"). OCR-variant keys ("Pro vider") never take a word that is spelled
exactly like another key of the field.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .geometry import (
    WEAK_KEYS,
    Box,
    Word,
    is_edge_region,
    median_height,
    near,
    near_keyless,
    region_of,
    union_boxes,
)
from .window import box_words

KV_ROOT = Path(__file__).resolve().parents[1]

# Folder name under the extraction package -> field id used in hits / window profiles.
FIELD_FOLDERS = {
    "dob": "member_dob",
    "member_id": "member_id",
    "name": "member_name",
    "provider_name": "provider_name",
    "dos": "dos",
    "electronic_signature": "electronic_signature",
    "page_no": "page_no",
}

_PARTS = re.compile(r"[a-z0-9]+")
_CATALOG: dict[str, list[str]] | None = None
_ROLE_KEYS: frozenset[str] | None = None
_FUZZY = {
    "dob": re.compile(r"(?i)^d[o0][bg][:#.]?$"),
    # MRN OCR near-misses seen in id.csv: MAN, MEN, MRK, MARK, M?N?
    "mrn": re.compile(r"(?i)^m(?:rn|[ae]n|rk|ark)[:#.]?$"),
}

# Keys whose alphanumeric form must appear with '#' (Chart / Chart# / Chart #) or their colon (Chart:).
_HASH_REQUIRED = frozenset({"chart"})
# Common words that are a key only when printed with their colon ('For: Diaz Sarah').
_COLON_REQUIRED = frozenset({"for", "scribe", "technician"})
# E-signature keys shorter than this (letters only) never fuzzy-match.
_ESIG_FUZZY_MIN = 10
_DOS_FUZZY_MIN = 5

# Bare "Birth" is not DOB when followed by these, or when "age of first birth".
_DOB_BAD_AFTER = frozenset({"place", "order", "sex"})
# Bare "Name" is only a name key when the prior word is one of these (or absent).
_NAME_OK_BEFORE = frozenset({"legal", "preferred", "person", "patient", "id", "mrn"})
# Bare "Name" is also a key in a header row with one of these columns.
_TABLE_COLUMNS = frozenset({"dob", "birth", "chart", "mrn", "id", "ssn", "gender", "sex", "age", "acct", "account"})

@dataclass
class KeyHit:
    field: str
    key: str
    word_indexes: list[int]
    box: Box
    weak: bool
    region: str = ""
    trusted: bool = False
    value_words: list[Word] = field(default_factory=list)
    value_box: Box | None = None
    value_text: str = ""
    # exact / regex / fuzzy / lookalike, and the edit distance of a fuzzy match.
    match: str = "exact"
    edit_distance: int = 0
    trusted_reason: str = ""
    cluster_size: int = 1


@dataclass(frozen=True)
class _Match:
    indexes: list[int]
    kind: str = "exact"
    distance: int = 0


def linked(row, hit: KeyHit):
    """Attach the key a field row was read from (feature logging needs its geometry)."""
    row.key_hit = hit
    return row


def _read_keys(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]
    if isinstance(data, dict):
        raw = data.get("keys")
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        # Tiered catalog ({"service": [...], "weak": [...]}): every tier is a key list.
        return [
            str(item).strip()
            for tier in data.values()
            if isinstance(tier, list)
            for item in tier
            if str(item).strip()
        ]
    return []


@lru_cache(maxsize=None)
def load_key_tiers(field_id: str) -> dict[str, str]:
    """casefolded key -> tier name for a field whose keys.json is tiered (else empty)."""
    folder = FIELD_FOLDERS.get(field_id)
    path = KV_ROOT / folder / "keys.json" if folder else None
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "keys" in data:
        return {}
    return {
        str(item).strip().casefold(): tier
        for tier, items in data.items()
        if isinstance(items, list)
        for item in items
        if str(item).strip()
    }


def load_catalog() -> dict[str, list[str]]:
    """Load keys from each field folder that already has a keys.json."""
    global _CATALOG
    if _CATALOG is None:
        catalog: dict[str, list[str]] = {}
        for field_id, folder in FIELD_FOLDERS.items():
            path = KV_ROOT / folder / "keys.json"
            if not path.is_file():
                continue
            keys = _read_keys(path)
            if keys:
                catalog[field_id] = keys
        _CATALOG = catalog
    return _CATALOG


@lru_cache(maxsize=1)
def _catalog_forms() -> dict[str, frozenset[tuple[str, ...]]]:
    """field -> every key of the field as printed parts ('Provider' -> ('provider',))."""
    return {field: frozenset(tuple(_parts(key)) for key in keys) for field, keys in load_catalog().items()}


def _blocklist_path(field_id: str) -> Path | None:
    folder = FIELD_FOLDERS.get(field_id)
    return KV_ROOT / folder / "key_blocklist.json" if folder else None


def _read_blocklist(path: Path | None) -> dict[str, dict[str, list[str]]]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=None)
def load_key_blocklist(field_id: str) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """Key letters ('Pro vider' and 'Provider' -> 'provider') -> (blocked words before, after)."""
    out: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for key, sides in _read_blocklist(_blocklist_path(field_id)).items():
        if not isinstance(sides, dict):
            continue
        before = frozenset(block_word(str(word), before=True) for word in sides.get("before") or [])
        after = frozenset(block_word(str(word), before=False) for word in sides.get("after") or [])
        out["".join(_parts(str(key)))] = (before - {""}, after - {""})
    return out


def block_word(text: str, *, before: bool) -> str:
    """The part of a neighbour word compared against the blocklist ('and/or' -> 'and')."""
    parts = _parts(text)
    if not parts:
        return ""
    return parts[-1] if before else parts[0]


def add_key_blocks(field_id: str, key: str, before: str = "", after: str = "") -> bool:
    """Add neighbour words that mark `key` as prose. True when the list changed."""
    path = _blocklist_path(field_id)
    words = {"before": block_word(before, before=True), "after": block_word(after, before=False)}
    if path is None or not key.strip() or not any(words.values()):
        return False
    data = _read_blocklist(path)
    name = next((existing for existing in data if existing.casefold() == key.strip().casefold()), key.strip())
    sides = data.setdefault(name, {})
    changed = False
    for side, word in words.items():
        current = [str(item) for item in sides.get(side) or []]
        if word and word not in {block_word(item, before=side == "before") for item in current}:
            sides[side] = sorted([*current, word])
            changed = True
    if changed:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        load_key_blocklist.cache_clear()
    return changed


def _blocked_by_context(words: list[Word], indexes: list[int], field: str, key: str) -> bool:
    compact = "".join(_parts(key))
    blocklist = load_key_blocklist(field)
    # OCR-variant catalog keys ('Pro viler') share their key's entry ('Provider').
    blocks = blocklist.get(compact) or next(
        (sides for listed, sides in blocklist.items() if _edit_distance(listed, compact) <= 1), None
    )
    if not blocks:
        return False
    positions = {word.index: position for position, word in enumerate(words)}
    first, last = positions.get(indexes[0]), positions.get(indexes[-1])
    if first is None or last is None:
        return False
    key_parts = _parts(key)
    # A key printed with its colon ("Patient:") is a label whatever follows it; a qualifier
    # before it still makes it another label ('Print Date:', 'Report Request ID:').
    last_text = words[last].content.rstrip()
    labelled = last_text.endswith(":") or bool(
        key_parts and re.search(rf"(?i){re.escape(key_parts[-1])}\s*:", last_text)
    )
    # Parts glued to the key inside its own tokens count as neighbours ("Patient-Reported").
    span = [part for position in range(first, last + 1) for part in _parts(words[position].content)]
    width = len(key_parts)
    start = next((n for n in range(len(span) - width + 1) if span[n : n + width] == key_parts), None)
    if start is None:
        # An OCR typo of the key glued to its neighbour ('Patlent Demographics', 'Providier reviewed').
        target = "".join(key_parts)
        max_dist = 1 if len(target) <= 5 else 2
        start, width = next(
            (
                (n, size)
                for n in range(len(span))
                for size in range(1, len(key_parts) + 2)
                if n + size <= len(span) and _edit_distance("".join(span[n : n + size]), target) <= max_dist
            ),
            (None, width),
        )
    inner_before = span[:start] if start is not None else []
    inner_after = span[start + width :] if start is not None else []
    before_words, after_words = blocks
    if inner_before:
        before = inner_before[-1]
    else:
        before = block_word(words[first - 1].content, before=True) if first > 0 else ""
    if inner_after:
        after = inner_after[0]
    else:
        after = block_word(words[last + 1].content, before=False) if last + 1 < len(words) else ""
    return bool(before and before in before_words) or bool(not labelled and after and after in after_words)


def load_provider_role_keys() -> frozenset[str]:
    """Designation keys that may stand alone anywhere on the page."""
    global _ROLE_KEYS
    if _ROLE_KEYS is not None:
        return _ROLE_KEYS
    path = KV_ROOT / "provider_name" / "roles.json"
    aliases: set[str] = set()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = []
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
    _ROLE_KEYS = frozenset(aliases)
    return _ROLE_KEYS


def is_provider_role_key(key: str) -> bool:
    return key.strip().casefold() in load_provider_role_keys()


def _parts(text: str) -> list[str]:
    return _PARTS.findall(text.casefold())


def _word_stream(words: list[Word]) -> list[tuple[str, int]]:
    stream: list[tuple[str, int]] = []
    for word in words:
        for part in _parts(word.content):
            stream.append((part, word.index))
    return stream


def _span_has_hash(words: list[Word], indexes: list[int]) -> bool:
    by_index = {word.index: word for word in words}
    return any("#" in by_index[index].content for index in indexes if index in by_index)


def _span_has_colon(words: list[Word], indexes: list[int], key: str) -> bool:
    """The key's last word is printed with a colon right after it ('Chart: 521867')."""
    parts = _parts(key)
    if not parts:
        return False
    pattern = re.compile(rf"(?i){re.escape(parts[-1])}\s*:")
    by_index = {word.index: word for word in words}
    if any(pattern.search(by_index[index].content) for index in indexes if index in by_index):
        return True
    positions = {word.index: position for position, word in enumerate(words)}
    last = positions.get(indexes[-1])
    return last is not None and last + 1 < len(words) and words[last + 1].content.startswith(":")


def _with_hash(words: list[Word], indexes: list[int], need_hash: bool, key: str = "") -> list[int] | None:
    if not need_hash:
        return indexes
    if key and "".join(_parts(key)) in _COLON_REQUIRED:
        return indexes if _span_has_colon(words, indexes, key) else None
    if _span_has_hash(words, indexes):
        return indexes
    # A hash-required word key ('Chart') is a label just as clearly when printed with its colon.
    if key and "".join(_parts(key)) in _HASH_REQUIRED and _span_has_colon(words, indexes, key):
        return indexes
    positions = {word.index: position for position, word in enumerate(words)}
    last = positions.get(indexes[-1])
    if last is None or last + 1 >= len(words):
        return None
    nxt = words[last + 1]
    if "#" in nxt.content:
        return indexes + [nxt.index]
    return None


def _same_line(words: list[Word], indexes: list[int]) -> bool:
    selected = [word for word in words if word.index in set(indexes)]
    if len(selected) <= 1:
        return True
    height = median_height(selected)
    centers = [word.box.cy for word in selected]
    return max(centers) - min(centers) <= max(height * 0.8, 4.0)


def _wrapped(words: list[Word], indexes: list[int]) -> bool:
    """A two-word key broken by the line end: '(Collection' closes a line, 'Date' opens the next."""
    selected = [word for word in words if word.index in set(indexes)]
    if len(selected) != 2:
        return False
    first, second = selected
    height = max(first.box.height(), 1.0)
    return second.box.left < first.box.left and 0.5 * height < second.box.cy - first.box.cy < 1.8 * height


def _neighbor_parts(words: list[Word], indexes: list[int]) -> tuple[list[str], list[str]]:
    """Token parts immediately before / after the matched key span."""
    positions = {word.index: position for position, word in enumerate(words)}
    first = positions.get(indexes[0])
    last = positions.get(indexes[-1])
    before: list[str] = []
    after: list[str] = []
    if first is not None and first > 0:
        before = _parts(words[first - 1].content)
    if last is not None and last + 1 < len(words):
        after = _parts(words[last + 1].content)
    # Also peek two words back for "age of first birth".
    if first is not None and first >= 3:
        before = _parts(words[first - 3].content) + _parts(words[first - 2].content) + before
    elif first is not None and first >= 2:
        before = _parts(words[first - 2].content) + before
    return before, after


def _dob_context_ok(words: list[Word], indexes: list[int], key: str) -> bool:
    """Reject Birth Place / Birth Order / Birth Sex / age of first birth."""
    parts = _parts(key)
    before, after = _neighbor_parts(words, indexes)
    # Only bare "Birth" (and fuzzy) is blocked by Place/Order/Sex.
    if parts == ["birth"] and after and after[0] in _DOB_BAD_AFTER:
        return False
    if "birth" in parts:
        tail = before[-3:] if len(before) >= 3 else before
        if tail == ["age", "of", "first"]:
            return False
        if "age" in before and "first" in before:
            return False
    return True


def _key_capitalized(content: str, part: str) -> bool:
    """True when the key's first letter inside the OCR word is upper-case. The whole key word
    is looked for first: in a line read as one word ('Follow up in 8 days for: ...') the
    first two letters alone would find another word."""
    found = re.search(rf"(?i)(?<![a-z]){re.escape(part)}(?![a-z])", content)
    if found is None:
        found = re.search(rf"(?i)(?<![a-z]){re.escape(part[:2])}", content)
    if found is None:
        found = re.search(r"[A-Za-z]", content)
    return bool(found) and content[found.start()].isupper()


def _name_context_ok(words: list[Word], indexes: list[int], key: str) -> bool:
    """Name keys must be capitalized; bare Name needs Legal/Preferred/... before it."""
    parts = _parts(key)
    positions = {word.index: position for position, word in enumerate(words)}
    if parts:
        position = positions.get(indexes[0])
        if position is None or not _key_capitalized(words[position].content, parts[0]):
            return False
    if parts != ["name"]:
        return True
    first = positions.get(indexes[0])
    if first is None or first <= 0:
        return True
    prior = _parts(words[first - 1].content)
    if not prior:
        return True
    return prior[-1] in _NAME_OK_BEFORE or _in_header_row(words, words[first])


def _in_header_row(words: list[Word], key_word: Word) -> bool:
    """The word sits in a table's column-header row ('Name  Patient ID  SSN  Birth Date')."""
    height = max(key_word.box.height(), 1.0)
    return any(
        part in _TABLE_COLUMNS
        for word in words
        if word is not key_word and abs(word.box.cy - key_word.box.cy) < 0.6 * height
        for part in _parts(word.content)
    )


def _fuzzy_matches(words: list[Word], key: str) -> list[list[int]]:
    """Whole-token OCR typos such as DOG for DOB, including dotted D.O.B keys."""
    pattern = _FUZZY.get("".join(_parts(key)))
    if pattern is None:
        return []
    return [[word.index] for word in words if pattern.fullmatch(word.content.strip())]


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance with early exit when lengths diverge too much."""
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 2:
        return 99
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, start=1):
        curr = [i]
        for j, ch_right in enumerate(right, start=1):
            cost = 0 if ch_left == ch_right else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


_DIGIT_AS_LETTER = (
    str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b"}),
    str.maketrans({"0": "o", "1": "i", "5": "s", "8": "b"}),
)


def _lookalike_match(exact_parts: list[str], target: str) -> bool:
    """OCR look-alikes or a split word are the ONLY difference from the key.

    'D0B', 'S1gned', '55N', '1D' (digit for letter), 'Narne', 'Adrnit' (rn for m),
    'Da te', 'Sign ed' (split word, keys of 4+ letters).
    """
    if not all(re.search(r"[a-z]", part) for part in exact_parts):
        return False
    raw = "".join(exact_parts)
    if raw == target:
        return len(exact_parts) > 1 and len(target) >= 4
    for table in _DIGIT_AS_LETTER:
        folded = raw.translate(table).replace("rn", "m").replace("vv", "w")
        if folded == target:
            return True
    return False


def _requires_hash(key: str) -> bool:
    """True when the key must include '#' (in-token or following word), or its colon."""
    if "#" in key:
        return True
    return "".join(_parts(key)) in _HASH_REQUIRED | _COLON_REQUIRED


def _ocr_fuzzy_matches(
    words: list[Word],
    stream: list[tuple[str, int]],
    key: str,
    field: str,
) -> list[_Match]:
    """Match spelling / OCR variants of ANY catalog key.

    Examples: 'Pro vider'/'Provi der'/'Pro viler' → Provider, 'Patlent #' → Patient #,
    'MAN'/'MEN'/'MRK' → MRN, 'Primry Provider' → Primary Provider.

    Joins consecutive OCR tokens and allows a small edit distance to the catalog
    key (spaces/punctuation removed). Field-specific value gates (person name,
    ID shape, etc.) still reject noisy values after a key match.
    """
    parts = _parts(key)
    target = "".join(parts)
    if len(target) < 2:
        return []
    # Keys too short for spelling edits still accept look-alikes and splits ('D0B', '1D', 'Da te'):
    # 'ID'/'PN' (< 3), short e-sign ('Seen by' ≈ 'sign by'), short DOS ('Date' ≈ 'Data').
    spelling_ok = not (
        len(target) < 3
        or (field == "electronic_signature" and len(target) < _ESIG_FUZZY_MIN)
        or (field == "dos" and len(target) < _DOS_FUZZY_MIN)
    )
    if len(target) <= 5:
        max_dist = 1
        max_width = min(max(len(parts) + 3, 3), 8)
    else:
        max_dist = 2
        max_width = min(max(len(parts) + 4, 3), 10)
    found: list[_Match] = []
    need_hash = _requires_hash(key)
    for start in range(len(stream)):
        for width in range(1, max_width + 1):
            if start + width > len(stream):
                break
            exact_parts = [stream[start + offset][0] for offset in range(width)]
            chunk = "".join(exact_parts)
            if abs(len(chunk) - len(target)) > max_dist:
                if len(chunk) > len(target) + max_dist:
                    break
                continue
            if exact_parts == parts or tuple(exact_parts) in _catalog_forms().get(field, ()):
                continue
            kind, distance = "lookalike", 0
            if not _lookalike_match(exact_parts, target):
                if not spelling_ok or chunk[0] != target[0]:
                    continue
                # Single-token typos: anchor both ends ('provided' ≠ 'provider').
                # Multi-token OCR splits: only require the same start letter.
                if width == 1 and chunk[-1] != target[-1]:
                    continue
                # A short key cut short is another word: 'D.O.' (credential) ≠ 'D.O.B.'.
                if len(target) <= 5 and target.startswith(chunk):
                    continue
                folded = chunk.replace("rn", "m").replace("vv", "w")
                kind = "fuzzy"
                distance = min(_edit_distance(chunk, target), _edit_distance(folded, target))
                if distance > max_dist:
                    continue
                # An extra whole word is not an OCR split: 'Patient ID Name' ≠ 'Patient Name'.
                if width > 1 and any(
                    len(exact_parts[skip]) >= 2
                    and "".join(exact_parts[:skip] + exact_parts[skip + 1 :]) == target
                    for skip in range(width)
                ):
                    continue
            indexes: list[int] = []
            for offset in range(width):
                word_index = stream[start + offset][1]
                if not indexes or indexes[-1] != word_index:
                    indexes.append(word_index)
            accepted = _with_hash(words, indexes, need_hash, key)
            if accepted is None or not _same_line(words, accepted):
                continue
            at_word_start = start == 0 or stream[start - 1][1] != stream[start][1]
            if kind == "fuzzy" and not _fuzzy_words_ok(words, accepted, parts, field, target, at_word_start):
                continue
            if field == "dob" and not _dob_context_ok(words, accepted, key):
                continue
            if field == "name" and not _name_context_ok(words, accepted, key):
                continue
            found.append(_Match(accepted, kind, distance))
    return found


def _fuzzy_words_ok(
    words: list[Word], indexes: list[int], parts: list[str], field: str, target: str, at_word_start: bool
) -> bool:
    """A spelling variant is a key only in words that could be one: not in the middle of a
    whole line read as one OCR word ('Pälient: Thomas David' starts one, a sentence doesn't),
    not a possessive ('patient's' ≠ 'Patient ID'), and a short ID acronym only in capitals
    ('MAN' for MRN, never 'Main')."""
    contents = [word.content for word in words if word.index in set(indexes)]
    if not at_word_start and any(len(_parts(content)) > len(parts) + 2 for content in contents):
        return False
    if any(re.search(r"['’]s\b", content) for content in contents):
        return False
    if field == "member_id" and len(target) <= 3:
        letters = [char for content in contents for char in content if char.isalpha()]
        return bool(letters) and all(char.isupper() for char in letters)
    return True


def _match_key(words: list[Word], stream: list[tuple[str, int]], key: str, field: str) -> list[_Match]:
    parts = _parts(key)
    if not parts:
        return []
    need_hash = _requires_hash(key)
    # Bare "Chart" without hash support is never matched (too many false positives).
    if "".join(parts) == "chart" and not need_hash and "#" not in key:
        need_hash = True
    found: list[_Match] = []
    limit = len(stream) - len(parts) + 1
    for start in range(max(limit, 0)):
        if not all(stream[start + offset][0] == parts[offset] for offset in range(len(parts))):
            continue
        indexes: list[int] = []
        for offset in range(len(parts)):
            word_index = stream[start + offset][1]
            if not indexes or indexes[-1] != word_index:
                indexes.append(word_index)
        accepted = _with_hash(words, indexes, need_hash, key)
        if accepted is None or not (_same_line(words, accepted) or (field == "dos" and _wrapped(words, accepted))):
            continue
        if field == "dob" and not _dob_context_ok(words, accepted, key):
            continue
        if field == "name" and not _name_context_ok(words, accepted, key):
            continue
        # An ID acronym in lower case is prose: 'orally 2 times a day prn diarrhea'.
        if field == "member_id" and key.isupper():
            first_word = next(word.content for word in words if word.index == accepted[0])
            if not _key_capitalized(first_word, parts[0]):
                continue
        found.append(_Match(accepted))
    forms = _catalog_forms().get(field, frozenset())
    by_index = {word.index: word for word in words}
    for indexes in _fuzzy_matches(words, key):
        if tuple(_parts(by_index[indexes[0]].content)) in forms - {tuple(parts)}:
            continue
        if field == "dob" and not _dob_context_ok(words, indexes, key):
            continue
        if need_hash:
            accepted = _with_hash(words, indexes, True, key)
            if accepted is None:
                continue
            indexes = accepted
        found.append(_Match(indexes, "regex"))
    found.extend(_ocr_fuzzy_matches(words, stream, key, field))
    return found


def _longest_first(catalog: dict[str, list[str]]) -> list[tuple[str, str]]:
    pairs = [(field, key) for field, keys in catalog.items() for key in keys]
    pairs.sort(key=lambda item: (len(_parts(item[1])), len(item[1])), reverse=True)
    return pairs


def _same_spot(a: Box, b: Box) -> bool:
    """b is on a's line (close by) or on the line just above / below, overlapping it."""
    height = max(a.height(), b.height(), 1.0)
    across = max(a.left, b.left) - min(a.right, b.right)
    down = max(a.top, b.top) - min(a.bottom, b.bottom)
    same_line = abs(a.cy - b.cy) < height * 0.8 and across < height * 3
    stacked = down < height * 1.5 and across < 0
    return same_line or stacked


def _part_of_longer_key(hits: list[KeyHit], field: str, key: str, box: Box) -> bool:
    """A longer key of this field containing `key` was found at the same spot."""
    compact = "".join(_parts(key))
    for hit in hits:
        parts = _parts(hit.key)
        if hit.field != field or len("".join(parts)) <= len(compact):
            continue
        runs = ("".join(parts[i:j]) for i in range(len(parts)) for j in range(i + 1, len(parts) + 1))
        if compact in runs and _same_spot(hit.box, box):
            return True
    return False


def find_key_hits(words: list[Word], page_w: float, page_h: float) -> list[KeyHit]:
    """All catalog keys, longest match first, with trust flags applied."""
    catalog = load_catalog()
    stream = _word_stream(words)
    by_index = {word.index: word for word in words}
    occupied: set[int] = set()
    hits: list[KeyHit] = []

    for field, key in _longest_first(catalog):
        for found in _match_key(words, stream, key, field):
            indexes = found.indexes
            if any(index in occupied for index in indexes):
                continue
            if _blocked_by_context(words, indexes, field, key):
                continue
            key_words = [by_index[index] for index in indexes if index in by_index]
            box = union_boxes([word.box for word in key_words])
            if box is None or _part_of_longer_key(hits, field, key, box):
                continue
            occupied.update(indexes)
            window = box_words(field, key, key_words, words, page_w, page_h)
            value_box = union_boxes([word.box for word in window])
            weak = key.casefold() in WEAK_KEYS or (
                field == "electronic_signature" and key.casefold() == "signature"
            )
            hits.append(
                KeyHit(
                    field=field,
                    key=key,
                    word_indexes=indexes,
                    box=box,
                    weak=weak,
                    region=region_of(box, value_box, page_h),
                    value_words=window,
                    value_box=value_box,
                    value_text=" ".join(word.content for word in window).strip(),
                    match=found.kind,
                    edit_distance=found.distance,
                )
            )
    _mark_trusted(hits, page_w, page_h, by_index)
    return hits


def trusted_field_hits(hits: list[KeyHit], field: str) -> list[KeyHit]:
    """Trusted keys for a field with value text, header/footer first then mid."""
    candidates = [hit for hit in hits if hit.trusted and hit.field == field and hit.value_text]
    edge = [hit for hit in candidates if is_edge_region(hit.region)]
    mid = [hit for hit in candidates if not is_edge_region(hit.region)]
    return edge + mid


def extract_field_keys(hits: list[KeyHit], field: str, extract_fn):
    """Run extract_fn on every trusted key of the field, header/footer keys first.

    Every key-value pair on the page is a candidate (accuracy counts each one); the field's
    selection still prefers header/footer values for the record output.
    """
    candidates = trusted_field_hits(hits, field)
    edge = [hit for hit in candidates if is_edge_region(hit.region)]
    mid = [hit for hit in candidates if not is_edge_region(hit.region)]
    return [linked(extract_fn(hit), hit) for hit in edge + mid]


def _clusters(hits: list[KeyHit], page_w: float, page_h: float) -> list[list[int]]:
    parent = list(range(len(hits)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(hits)):
        for right in range(left + 1, len(hits)):
            if near(hits[left].box, hits[right].box, page_w, page_h):
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(hits)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _band(region: str) -> str:
    if region in {"header", "footer"}:
        return region
    return "mid"


def _strong_in_same_band(hits: list[KeyHit], group: list[int], hit: KeyHit) -> bool:
    band = _band(hit.region)
    return any(not hits[index].weak and _band(hits[index].region) == band for index in group)


def _labels_in_own_word(hit: KeyHit, by_index: dict[int, Word] | None, fields: set[str]) -> bool:
    """A key of `fields` printed with its colon in the same OCR word after this key: a line
    read as one word ('Patient: OZIOMEK, JAMES DOB: 02/17/1941') holds only one key hit."""
    if not by_index:
        return False
    text = " ".join(by_index[index].content for index in hit.word_indexes if index in by_index)
    found = re.search(r"[^a-z0-9]*".join(map(re.escape, _parts(hit.key))), text, flags=re.IGNORECASE)
    rest = text[found.end():] if found else ""
    if not rest:
        return False
    for field, keys in load_catalog().items():
        if field not in fields:
            continue
        for key in keys:
            parts = _parts(key)
            if parts and re.search(
                rf"(?i)(?<![a-z0-9]){'[^a-z0-9]*'.join(map(re.escape, parts))}\s*[:#]", rest
            ):
                return True
    return False


def _patient_has_nearby_key(
    hits: list[KeyHit], index: int, page_w: float, page_h: float, by_index: dict[int, Word] | None = None
) -> bool:
    """Patient is trusted only when a DOB/ID/other name key sits in the keyless band."""
    patient = hits[index]
    if _labels_in_own_word(patient, by_index, {"dob", "member_id"}):
        return True
    for other_index, other in enumerate(hits):
        if other_index == index:
            continue
        if other.field not in {"dob", "member_id", "name"}:
            continue
        if other.key.casefold() == "patient":
            continue
        if near_keyless(patient.box, other.box, page_w, page_h):
            return True
    return False


def _dos_trusted_alone(key: str) -> bool:
    tier = load_key_tiers("dos").get(key.strip().casefold(), "")
    return tier in {"service", "admit", "discharge"} and len(_parts(key)) >= 2


def _printed_as_label(hit: KeyHit, by_index: dict[int, Word]) -> bool:
    """A multi-word key printed as a label: capitalized and closed by a colon or bracket
    ('(Primary Provider)', 'Referring Provider:'), not prose ('your primary provider')."""
    parts = _parts(hit.key)
    if len(parts) < 2:
        return False
    text = " ".join(by_index[index].content for index in hit.word_indexes if index in by_index)
    pattern = r"[^a-z0-9]*".join(re.escape(part) for part in parts)
    for match in re.finditer(rf"(?i)([(\[{{]\s*)?({pattern})(\s*[:)\]}}])?", text):
        words = re.findall(r"[A-Za-z]+", match.group(2))
        if words and all(word[0].isupper() for word in words) and (match.group(1) or match.group(3)):
            return True
    return False


def _mark_trusted(hits: list[KeyHit], page_w: float, page_h: float, by_index: dict[int, Word] | None = None) -> None:
    if not hits:
        return
    for group in _clusters(hits, page_w, page_h):
        clustered = len(group) >= 2
        for index in group:
            hit = hits[index]
            hit.cluster_size = len(group)
            # Patient: require another key within keyless-band proximity.
            if hit.key.casefold() == "patient" and hit.field == "name":
                if not _patient_has_nearby_key(hits, index, page_w, page_h, by_index):
                    continue
                if hit.region == "mid":
                    hit.region = "mid_cluster"
                hit.trusted, hit.trusted_reason = True, "patient_near_key"
                continue
            if hit.weak and not _strong_in_same_band(hits, group, hit):
                continue
            # Designation keys (Physician, NP, …) are trusted alone anywhere.
            if hit.field == "provider_name" and is_provider_role_key(hit.key):
                hit.trusted, hit.trusted_reason = True, "role_key"
                continue
            if hit.field == "provider_name" and by_index and _printed_as_label(hit, by_index):
                if hit.region == "mid":
                    hit.region = "mid_cluster"
                hit.trusted, hit.trusted_reason = True, "provider_label"
                continue
            # Multi-word DOS service/admit/discharge keys are trusted alone mid-page.
            if hit.field == "dos" and _dos_trusted_alone(hit.key):
                if hit.region == "mid":
                    hit.region = "mid_cluster"
                hit.trusted, hit.trusted_reason = True, "dos_strong_key"
                continue
            # Multi-word e-signature phrases are trusted even alone mid-page.
            if hit.field == "electronic_signature" and not hit.weak:
                if hit.region == "mid":
                    hit.region = "mid_cluster"
                hit.trusted, hit.trusted_reason = True, "esig_phrase"
                continue
            if hit.region == "mid" and not clustered:
                continue
            if hit.region == "mid":
                hit.region = "mid_cluster"
            hit.trusted = True
            hit.trusted_reason = "cluster" if hit.region == "mid_cluster" else "edge"
