from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from stages.lib.canon_store import CANON_DIR, CanonFile

WORD = re.compile(r"[A-Za-z0-9]+")
WINDOW = 5

@dataclass(frozen=True)
class MemberKeywords:
    ignore: frozenset[str]
    labels: frozenset[str]
    # Words a member's name never contains: articles, pronouns, connectors,
    # roles, relationships and chart furniture. Used to throw out an NER hit
    # like "my medical assistant" and to stop the key-value reader taking prose
    # after a bare "Patient:" key as a name.
    non_name: frozenset[str]
    key_groups: dict[str, tuple[str, ...]]


def _build_keywords(data: dict[str, Any]) -> MemberKeywords:
    groups = data.get("key_groups") or {}
    return MemberKeywords(
        ignore=frozenset(data["ignore"]),
        labels=frozenset(data["labels"]),
        non_name=frozenset(data["non_name"]),
        key_groups={
            name: tuple(group.get("keys") or []) for name, group in groups.items()
        },
    )


# keyword-canon/member_keywords_canon.json — reloaded when the file changes.
_KEYWORDS: CanonFile[MemberKeywords] = CanonFile(
    CANON_DIR / "member_keywords_canon.json", _build_keywords
)


def member_keywords() -> MemberKeywords:
    return _KEYWORDS.get()


def is_non_name(token: str) -> bool:
    return token.casefold() in member_keywords().non_name


BOTH_FULL = "both_full"
INITIAL = "initial"
MISMATCH = "mismatch"
NONE = "none"
ONE_FULL_WRONG = "one_full_wrong"
ALL_FULL = "all_full"
TWO_FULL = "two_full"
TWO_FULL_WRONG = "two_full_wrong"

TWO_WORD_MATCH = {BOTH_FULL, INITIAL}
THREE_WORD_MATCH = {ALL_FULL, TWO_FULL}


def tokenize(text: str) -> list[str]:
    return WORD.findall(text or "")


def is_ignore(token: str) -> bool:
    return token.casefold() in member_keywords().ignore


def is_label(token: str) -> bool:
    return token.casefold() in member_keywords().labels


def is_full(token: str, name: str) -> bool:
    return bool(name) and token.casefold() == name.casefold()


def is_initial(token: str, name: str) -> bool:
    return bool(name) and len(token) == 1 and token.isalpha() and token.casefold() == name[0].casefold()


def is_any_initial(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def _name_like(token: str) -> bool:
    if not token or is_label(token) or token.isdigit():
        return False
    return token.isalpha() or is_ignore(token)


def _touches(token: str, *names: str) -> bool:
    return any(is_full(token, name) or is_initial(token, name) for name in names if name)


def _extra_full(span: list[str], *names: str) -> list[str]:
    extras: list[str] = []
    for token in span:
        if is_ignore(token) or is_label(token) or is_any_initial(token):
            continue
        if any(is_full(token, name) for name in names if name):
            continue
        if token.isalpha():
            extras.append(token)
    return extras


def classify_two_word_name(span: list[str], first: str, last: str) -> str:
    if not span or not first or not last:
        return NONE
    first_full = any(is_full(token, first) for token in span)
    last_full = any(is_full(token, last) for token in span)
    first_init = (not first_full) and any(is_initial(token, first) for token in span)
    last_init = (not last_full) and any(is_initial(token, last) for token in span)
    extras = _extra_full(span, first, last)
    if first_full and last_full:
        return BOTH_FULL
    if (first_full and last_init) or (first_init and last_full):
        if extras:
            return ONE_FULL_WRONG
        return INITIAL
    if (first_full or last_full) and extras:
        # One of the two names matches and another full name sits beside it:
        # a different member, not merely a missing detection.
        return ONE_FULL_WRONG
    return MISMATCH


def classify_three_word_name(span: list[str], first: str, middle: str, last: str) -> str:
    names = [part for part in (first, middle, last) if part]
    if not span or len(names) < 3:
        return NONE
    full_count = sum(any(is_full(token, name) for token in span) for name in names)
    extras = _extra_full(span, first, middle, last)
    if full_count >= 3:
        return ALL_FULL
    if full_count >= 2:
        if extras:
            return TWO_FULL_WRONG
        return TWO_FULL
    return MISMATCH


def name_matches(found: str, first: str, last: str, middle: str, name_mode: str) -> bool:
    span = tokenize(found)
    if name_mode == "3":
        return classify_three_word_name(span, first, middle, last) in THREE_WORD_MATCH
    return classify_two_word_name(span, first, last) in TWO_WORD_MATCH


def _expand(tokens: list[str], start: int, end: int) -> tuple[int, int]:
    while start > 0 and _name_like(tokens[start - 1]):
        start -= 1
    while end + 1 < len(tokens) and _name_like(tokens[end + 1]):
        end += 1
    return start, end


def _best_span(
    tokens: list[str],
    names: tuple[str, ...],
    classify,
    scores: dict[str, int],
) -> str:
    best_key: tuple[int, int, int] | None = None
    best_text = "N/A"
    for index, token in enumerate(tokens):
        if not _touches(token, *names):
            continue
        lo = max(0, index - WINDOW)
        hi = min(len(tokens), index + WINDOW + 1)
        for left in range(lo, index + 1):
            for right in range(index, hi):
                span_left, span_right = _expand(tokens, left, right)
                span_left = max(span_left, lo)
                span_right = min(span_right, hi - 1)
                span = tokens[span_left : span_right + 1]
                kind = classify(span)
                score = scores.get(kind, 0)
                if score <= 0:
                    continue
                key = (score, -(span_right - span_left), -span_left)
                if best_key is None or key > best_key:
                    best_key = key
                    best_text = " ".join(span)
    return best_text


def find_two_word_name(ocr_text: str, first: str, last: str) -> str:
    first = (first or "").strip()
    last = (last or "").strip()
    if not first or not last:
        return "N/A"
    return _best_span(
        tokenize(ocr_text),
        (first, last),
        lambda span: classify_two_word_name(span, first, last),
        {BOTH_FULL: 3, INITIAL: 2},
    )


def find_three_word_name(ocr_text: str, first: str, middle: str, last: str) -> str:
    first = (first or "").strip()
    middle = (middle or "").strip()
    last = (last or "").strip()
    if not first or not middle or not last:
        return "N/A"
    return _best_span(
        tokenize(ocr_text),
        (first, middle, last),
        lambda span: classify_three_word_name(span, first, middle, last),
        {ALL_FULL: 4, TWO_FULL: 3, TWO_FULL_WRONG: 1},
    )
