"""In-memory overlay boxes. Nothing is written to disk.

DOB: short keys 6x, longer keys 5x, key at 40:60, only downward.
Member ID: same base scale, width 1.4x that box, height 0.7x, key at 20:80.
Name: short keys 6x, longer keys 5x, key at 15:85, only downward.
Provider Name / Electronic Signature: same box rules as Name (default downward).
Provider role/designation keys: upward + full-width ID band (see helpers below).
"""

from __future__ import annotations

from .geometry import Word, group_lines

SHORT_EXPAND = 6.0
LONG_EXPAND = 5.0
LEFT_FRAC = 0.40
RIGHT_FRAC = 0.60
# Full-width vertical pad around a designation key when hunting for an ID.
ROLE_ID_BAND_FRAC = 0.05


def expand_for_key(key: str) -> float:
    """Keys of 3-5 characters use 6x. Longer keys use 5x."""
    return SHORT_EXPAND if len(key.strip()) <= 5 else LONG_EXPAND


def _profile(field: str) -> tuple[float, float, float, float]:
    """left, right, width factor, height factor."""
    if field == "member_id":
        return 0.20, 0.80, 2.0 * 0.70, 0.70
    if field in {"name", "provider_name", "electronic_signature"}:
        return 0.15, 0.85, 1.0, 1.0
    return LEFT_FRAC, RIGHT_FRAC, 1.0, 1.0


def expanded_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    page_w: float,
    page_h: float,
    expand: float,
    left_frac: float = LEFT_FRAC,
    right_frac: float = RIGHT_FRAC,
    width_factor: float = 1.0,
    height_factor: float = 1.0,
) -> tuple[float, float, float, float]:
    """Key sits left_frac across the box. Nothing above the key."""
    width = max(right - left, 8.0)
    height = max(bottom - top, 8.0)
    center_x = (left + right) / 2.0
    span = width * expand * width_factor
    box_left = max(0.0, center_x - span * left_frac)
    box_right = min(page_w, center_x + span * right_frac) if page_w else center_x + span * right_frac
    box_top = top
    box_bottom = min(page_h, top + height * expand * height_factor) if page_h else top + height * expand * height_factor
    return box_left, box_top, box_right, box_bottom


def upward_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    page_w: float,
    page_h: float,
    expand: float,
    left_frac: float = 0.15,
    right_frac: float = 0.85,
    width_factor: float = 1.0,
    height_factor: float = 1.0,
) -> tuple[float, float, float, float]:
    """Same scale as expanded_box but grows upward from the key (value above)."""
    width = max(right - left, 8.0)
    height = max(bottom - top, 8.0)
    center_x = (left + right) / 2.0
    span = width * expand * width_factor
    box_left = max(0.0, center_x - span * left_frac)
    box_right = min(page_w, center_x + span * right_frac) if page_w else center_x + span * right_frac
    rise = height * expand * height_factor
    box_top = max(0.0, top - rise) if page_h else max(0.0, top - rise)
    box_bottom = bottom
    return box_left, box_top, box_right, box_bottom


def full_width_band(
    top: float,
    bottom: float,
    page_w: float,
    page_h: float,
    pad_frac: float = ROLE_ID_BAND_FRAC,
) -> tuple[float, float, float, float]:
    """Full page width, pad_frac of page height above and below the key band."""
    pad = pad_frac * page_h if page_h else 8.0
    box_top = max(0.0, top - pad)
    box_bottom = min(page_h, bottom + pad) if page_h else bottom + pad
    right = page_w if page_w else 1.0e9
    return 0.0, box_top, right, box_bottom


def words_in_box(words: list[Word], box: tuple[float, float, float, float]) -> list[Word]:
    """Words whose centers fall inside the box, in reading order."""
    box_left, box_top, box_right, box_bottom = box
    chosen = [
        word
        for word in words
        if box_left <= word.box.cx <= box_right and box_top <= word.box.cy <= box_bottom
    ]
    # By line, then left to right: a value a pixel higher than its key still reads after it.
    return [word for line in group_lines(chosen) for word in line]


def field_box(
    field: str,
    key: str,
    key_left: float,
    key_top: float,
    key_right: float,
    key_bottom: float,
    page_w: float,
    page_h: float,
) -> tuple[float, float, float, float]:
    """Overlay rectangle for a field key, using that field's box rules."""
    left_frac, right_frac, width_factor, height_factor = _profile(field)
    return expanded_box(
        key_left,
        key_top,
        key_right,
        key_bottom,
        page_w,
        page_h,
        expand_for_key(key),
        left_frac,
        right_frac,
        width_factor,
        height_factor,
    )


def box_words(
    field: str,
    key: str,
    key_words: list[Word],
    words: list[Word],
    page_w: float,
    page_h: float,
) -> list[Word]:
    """Words inside this field's overlay box, including the key itself."""
    if not key_words:
        return []
    left = min(word.box.left for word in key_words)
    top = min(word.box.top for word in key_words)
    right = max(word.box.right for word in key_words)
    bottom = max(word.box.bottom for word in key_words)
    left_frac, right_frac, width_factor, height_factor = _profile(field)
    box = expanded_box(
        left,
        top,
        right,
        bottom,
        page_w,
        page_h,
        expand_for_key(key),
        left_frac,
        right_frac,
        width_factor,
        height_factor,
    )
    return words_in_box(words, box)
