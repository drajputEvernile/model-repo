"""Raw-OCR-order sub-word tokens for reading a value next to a key.

Sorting words by line position interleaves neighbouring lines, so extractors
that read "key value" phrases walk words in OCR index order instead. Some OCR
words are whole lines ("Electronically signed by Davis, Alfred H III, PA") or
glue names at a comma ("BULLER,BRETT"); both are split into sub-word tokens
whose character offsets index into the printed sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .geometry import Box, Word

_SUB_TOKEN = re.compile(r"[^\s,]+,*|,+")


@dataclass
class Token:
    word: Word
    text: str
    start: int
    end: int


def edit_distance(left: str, right: str) -> int:
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


def _sub_word(word: Word, start: int, end: int) -> Word:
    """Slice of an OCR word with an x-range proportional to its characters."""
    length = max(len(word.content), 1)
    width = word.box.width()
    box = Box(
        word.box.left + width * start / length,
        word.box.top,
        word.box.left + width * end / length,
        word.box.bottom,
    )
    return Word(index=word.index, content=word.content[start:end], box=box)


def ordered_tokens(words: list[Word]) -> tuple[list[Token], str]:
    """Sub-word tokens in raw OCR order plus the printed sentence they index into."""
    tokens: list[Token] = []
    parts: list[str] = []
    cursor = 0
    for word in sorted(words, key=lambda item: item.index):
        if parts:
            cursor += 1
        for match in _SUB_TOKEN.finditer(word.content):
            tokens.append(
                Token(
                    word=_sub_word(word, match.start(), match.end()),
                    text=match.group(0),
                    start=cursor + match.start(),
                    end=cursor + match.end(),
                )
            )
        parts.append(word.content)
        cursor += len(word.content)
    return tokens, " ".join(parts)


def same_line(left: Token, right: Token, line_h: float) -> bool:
    """Right token continues left token's line (same row, moving right, small gap)."""
    return (
        abs(left.word.box.cy - right.word.box.cy) <= 0.6 * line_h
        and right.word.box.left >= left.word.box.left
        and right.word.box.left - left.word.box.right <= 3.0 * line_h
    )


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def key_token_span(tokens: list[Token], indexes: list[int], key: str) -> tuple[int, int] | None:
    """First/last token positions that spell the key inside the key's OCR words."""
    wanted = set(indexes)
    positions = [pos for pos, token in enumerate(tokens) if token.word.index in wanted]
    if not positions:
        return None
    target = _alnum(key)
    width_max = len(key.split()) + 2
    best: tuple[int, int, int] | None = None
    for offset, first in enumerate(positions):
        for width in range(1, width_max + 1):
            if offset + width > len(positions):
                break
            last = positions[offset + width - 1]
            chunk = "".join(_alnum(tokens[pos].text) for pos in range(first, last + 1))
            distance = edit_distance(chunk, target)
            if distance <= 2 and (best is None or distance < best[0]):
                best = (distance, first, last)
        if best is not None and best[0] == 0:
            break
    if best is None:
        # Too garbled to spell ("Admil Dale/DxLe"): assume the key is its first N sub-words.
        return positions[0], positions[min(len(key.split()), len(positions)) - 1]
    return best[1], best[2]
