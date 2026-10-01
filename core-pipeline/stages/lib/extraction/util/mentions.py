"""Where a value is written on the page, and the other places it is written again.

locate_value finds the words a candidate value was read from (the candidate log's
value_words). mention_runs finds every occurrence of a value (dates in any format, names in
either order). repeat_mentions turns the other occurrences of an accepted DOB or Name into
keyless copies (source 'repeat', never selected): the reviews label every mention,
and a value repeated in a header, a footer or a table is the same patient's value. A name
inside a sentence ('Emma Young is a 73 year old') is prose, not a mention.
"""

from __future__ import annotations

import dataclasses
import re

from .dates import canonical_date, find_dates
from .geometry import Box, Word, edge_band, union_boxes
from .keys import KeyHit

_PARTS = re.compile(r"[a-z0-9]+")
PERSON_FIELDS = frozenset({"name", "provider_name"})
# Not Member ID: an ID is repeated under its own key (a keyless copy has the wrong key), and a
# number read by a prose key ('2024') would be copied everywhere.
REPEAT_FIELDS = frozenset({"dob", "name"})
REPEAT_KEY = "Repeated Value"
REPEAT_SOURCE = "repeat"


def value_parts(text: str) -> list[str]:
    return _PARTS.findall((text or "").casefold())


def all_runs(words: list[Word], parts: list[str]) -> list[list[Word]]:
    """Every run of words whose alphanumeric parts spell `parts` in order."""
    if not parts:
        return []
    stream = [(part, word) for word in words for part in value_parts(word.content)]
    runs: list[list[Word]] = []
    for start in range(len(stream) - len(parts) + 1):
        if all(stream[start + offset][0] == parts[offset] for offset in range(len(parts))):
            run: list[Word] = []
            for offset in range(len(parts)):
                word = stream[start + offset][1]
                if not run or run[-1] is not word:
                    run.append(word)
            runs.append(run)
    return runs


def find_run(words: list[Word], parts: list[str], anchor: Box | None) -> list[Word]:
    """Words whose alphanumeric parts spell `parts` in order; nearest to the anchor wins."""
    runs = all_runs(words, parts)
    if not runs:
        return []
    if anchor is None:
        return runs[0]

    def distance(run: list[Word]) -> float:
        box = union_boxes([word.box for word in run])
        return abs(box.cy - anchor.cy) * 4 + abs(box.cx - anchor.cx)

    return min(runs, key=distance)


def all_dates(words: list[Word], value: str) -> list[list[Word]]:
    """Every run of words spelling the same calendar date in any format ('26.Nov.1953' for '11/26/1953')."""
    target = canonical_date(value)
    found: list[list[Word]] = []
    if not target:
        return found
    for start in range(len(words)):
        for width in (1, 2, 3):
            window = words[start : start + width]
            text = " ".join(word.content for word in window)
            first_end = len(window[0].content)
            last_start = len(text) - len(window[-1].content)
            # The date must start in the first word and end in the last, so labels
            # before it ('Service: 06/12/2024') stay out of the value.
            if any(
                canonical_date(match.group(1)) == target
                and match.start(1) < first_end
                and match.end(1) > last_start
                for match in find_dates(text)
            ):
                found.append(window)
                break
    return found


def _find_date(words: list[Word], value: str, anchor: Box | None) -> list[Word]:
    found = all_dates(words, value)
    if not found:
        return []
    if anchor is None:
        return found[0]
    return min(found, key=lambda run: abs(union_boxes([w.box for w in run]).cy - anchor.cy))


def locate_value(field: str, row, value: str, page_words: list[Word]) -> list[Word]:
    """The page words the candidate value was read from: the words a repeat was pinned to,
    else the key window first, then the page."""
    pinned = getattr(row, "value_at", None)
    if pinned:
        return pinned
    hit: KeyHit | None = getattr(row, "key_hit", None)
    anchor = hit.box if hit is not None else None
    pools = [hit.value_words, page_words] if hit is not None else [page_words]
    texts = [value, getattr(row, "ner_text", "")]
    for pool in pools:
        for text in texts:
            run = find_run(pool, value_parts(text), anchor)
            if run:
                return run
        dates = find_dates(value) if field in {"dob", "dos"} else []
        if dates:
            run = _find_date(pool, dates[0].group(1), anchor)
            if run:
                return run
    return []


def mention_runs(field: str, value: str, words: list[Word]) -> list[list[Word]]:
    """Every occurrence of a value on the page: dates in any format, person names in either
    order; too short a name or ID (one word, under 4 characters, no digit) has none."""
    if field == "dob":
        return all_dates(words, value)
    parts = value_parts(value)
    if field in PERSON_FIELDS and len(parts) < 2:
        return []
    if field == "member_id" and (len("".join(parts)) < 4 or not any(ch.isdigit() for ch in value)):
        return []
    variants = [parts] + ([parts[-1:] + parts[:-1]] if field in PERSON_FIELDS else [])
    return [run for variant in variants for run in all_runs(words, variant)]


def _marker(field: str, value: str) -> str:
    if field == "dob":
        return canonical_date(value)
    parts = value_parts(value)
    return " ".join(sorted(parts)) if field in PERSON_FIELDS else "".join(parts)


def _in_prose(run: list[Word], words: list[Word]) -> bool:
    """The name is the subject of a sentence: a lowercase word follows it on its line
    ('Emma Young is a 73 year old'), unlike a label line ('Young Emma (MRN 20766549)')."""
    position = {word.index: at for at, word in enumerate(words)}.get(run[-1].index)
    if position is None or position + 1 >= len(words):
        return False
    last, after = run[-1], words[position + 1]
    same_line = abs(after.box.cy - last.box.cy) < 0.6 * max(last.box.height(), 1.0)
    return same_line and after.content[:1].islower()


def repeat_mentions(field: str, rows: list, words: list[Word], page_h: float) -> list:
    """Keyless copies of the field's accepted values at their other occurrences on the page.

    A copy keeps the row's value, gets key REPEAT_KEY, source REPEAT_SOURCE, the band it sits
    in as region, and value_at = the words it was found at (read by locate_value).
    """
    if field not in REPEAT_FIELDS:
        return []
    taken: set[int] = set()
    for row in rows:
        if getattr(row, "value", ""):
            taken.update(word.index for word in locate_value(field, row, row.value, words))
    text = union_boxes([word.box for word in words])
    copies: list = []
    seen: set[str] = set()
    for row in rows:
        if not (row.accepted and row.value) or row.source == "split_part":
            continue
        marker = _marker(field, row.value)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        for run in mention_runs(field, row.value, words):
            indexes = {word.index for word in run}
            if indexes & taken or (field == "name" and _in_prose(run, words)):
                continue
            taken |= indexes
            band = edge_band(union_boxes([word.box for word in run]), page_h, text)
            copy = dataclasses.replace(
                row,
                key=REPEAT_KEY,
                region=f"keyless_{band}" if band else "mid",
                sentence=" ".join(word.content for word in run),
                selected=False,
                accepted=True,
                source=REPEAT_SOURCE,
                key_hit=None,
            )
            copy.value_at = run
            copies.append(copy)
    return copies
