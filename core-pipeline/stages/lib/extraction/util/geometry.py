"""Page geometry for header, footer, and mid-page key clusters.

Header and footer tracks are independent (mes.csv correlation ≈ 0).
Priors from Data/examp/mes.csv:
  - Header max 28% (≈ p99); core ~14% (≈ p75)
  - Footer max 20% (covers p99+); core ~9% (≈ p75)
"""

from __future__ import annotations

from dataclasses import dataclass

# Extended / max bands — independent tracks (do not set these equal).
HEADER_FRAC = 0.28
FOOTER_FRAC = 0.20
# Core bands (high-confidence zone nearer the edge).
HEADER_CORE_FRAC = 0.14
FOOTER_CORE_FRAC = 0.09
CLUSTER_Y_FRAC = 0.12
CLUSTER_X_FRAC = 0.40
# Keyless name band: ±5% page height (also used for Patient-key proximity).
KEYLESS_BAND_FRAC = 0.05

WEAK_KEYS = frozenset({"name", "patient"})


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    def width(self) -> float:
        return max(0.0, self.right - self.left)


@dataclass
class Word:
    index: int
    content: str
    box: Box


def polygon_box(polygon: list) -> Box | None:
    if not polygon or len(polygon) < 8:
        return None
    xs = [float(value) for value in polygon[0::2]]
    ys = [float(value) for value in polygon[1::2]]
    return Box(min(xs), min(ys), max(xs), max(ys))


def union_boxes(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return Box(
        min(box.left for box in boxes),
        min(box.top for box in boxes),
        max(box.right for box in boxes),
        max(box.bottom for box in boxes),
    )


def page_size(page: dict, words: list[Word]) -> tuple[float, float]:
    width = float(page.get("width") or 0)
    height = float(page.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    bounds = union_boxes([word.box for word in words])
    if bounds is None:
        return 1.0, 1.0
    return max(bounds.right, 1.0), max(bounds.bottom, 1.0)


def words_from_page(page: dict) -> list[Word]:
    words: list[Word] = []
    for index, raw in enumerate(page.get("words") or []):
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content") or "").strip()
        box = polygon_box(raw.get("polygon") or [])
        if not content or box is None:
            continue
        words.append(Word(index=index, content=content, box=box))
    return words


def group_lines(words: list[Word]) -> list[list[Word]]:
    if not words:
        return []
    heights = sorted(word.box.height() for word in words if word.box.height() > 0)
    median = heights[len(heights) // 2] if heights else 10.0
    tolerance = max(median * 0.6, 1.0)
    ordered = sorted(words, key=lambda word: (word.box.cy, word.box.left))
    lines: list[list[Word]] = []
    current = [ordered[0]]
    for word in ordered[1:]:
        line_cy = sum(item.box.cy for item in current) / len(current)
        if abs(word.box.cy - line_cy) <= tolerance:
            current.append(word)
        else:
            lines.append(sorted(current, key=lambda item: item.box.left))
            current = [word]
    lines.append(sorted(current, key=lambda item: item.box.left))
    return lines


def median_height(words: list[Word]) -> float:
    heights = sorted(word.box.height() for word in words if word.box.height() > 0)
    if not heights:
        return 10.0
    return heights[len(heights) // 2]


def header_score(box: Box | None, page_h: float) -> float:
    """1 at the top edge → 0 at HEADER_FRAC; independent of footer."""
    if box is None or page_h <= 0 or HEADER_FRAC <= 0:
        return 0.0
    frac = box.cy / page_h
    if frac > HEADER_FRAC:
        return 0.0
    return max(0.0, 1.0 - frac / HEADER_FRAC)


def footer_score(box: Box | None, page_h: float) -> float:
    """1 at the bottom edge → 0 at FOOTER_FRAC from bottom; independent of header."""
    if box is None or page_h <= 0 or FOOTER_FRAC <= 0:
        return 0.0
    from_bottom = 1.0 - box.cy / page_h
    if from_bottom > FOOTER_FRAC:
        return 0.0
    return max(0.0, 1.0 - from_bottom / FOOTER_FRAC)


def edge_band(box: Box, page_h: float, text: Box | None) -> str:
    """'header' / 'footer' when the box is in that band of the image or of the text's extent
    (text = union of the page's word boxes), else ''. A screenshot of a document viewer ends
    the page well above the image bottom, so its footer is only a footer of the text. Keys
    keep the image bands (region_of): on the text's extent, print-stamp keys in the footer
    ('Report Request ID') become trusted."""
    fracs = [box.cy / page_h] if page_h > 0 else []
    if text is not None and text.height() > 0:
        fracs.append((box.cy - text.top) / text.height())
    if any(frac <= HEADER_FRAC for frac in fracs):
        return "header"
    if any(frac >= 1.0 - FOOTER_FRAC for frac in fracs):
        return "footer"
    return ""


def region_of(key_box: Box, value_box: Box | None, page_h: float) -> str:
    """Pick header / footer / mid using independent scores; key wins over value."""
    key_h = header_score(key_box, page_h)
    key_f = footer_score(key_box, page_h)
    if key_h > 0 or key_f > 0:
        if key_h >= key_f:
            return "header"
        return "footer"
    val_h = header_score(value_box, page_h)
    val_f = footer_score(value_box, page_h)
    if val_h > 0 or val_f > 0:
        if val_h >= val_f:
            return "header"
        return "footer"
    return "mid"


def is_edge_region(region: str) -> bool:
    """True for header/footer bands (including keyless_*). Mid is never an edge."""
    token = (region or "").casefold()
    return "header" in token or "footer" in token


def region_priority(region: str) -> int:
    """Keyed selection: try header/footer first (1), then mid (0)."""
    return 1 if is_edge_region(region) else 0


def near(left: Box, right: Box, page_w: float, page_h: float) -> bool:
    return (
        abs(left.cx - right.cx) <= CLUSTER_X_FRAC * page_w
        and abs(left.cy - right.cy) <= CLUSTER_Y_FRAC * page_h
    )


def near_keyless(left: Box, right: Box, page_w: float, page_h: float) -> bool:
    """True when boxes fall inside each other's keyless vertical band (full width)."""
    del page_w
    pad = KEYLESS_BAND_FRAC * page_h if page_h else 8.0
    left_top = left.top - pad
    left_bottom = left.bottom + pad
    right_top = right.top - pad
    right_bottom = right.bottom + pad
    return left_top <= right_bottom and right_top <= left_bottom


def boxes_overlap(left: tuple[float, float, float, float] | Box, right: Box | None) -> bool:
    """Axis-aligned overlap between a band tuple/box and a key value box."""
    if right is None:
        return False
    if isinstance(left, Box):
        a_left, a_top, a_right, a_bottom = left.left, left.top, left.right, left.bottom
    else:
        a_left, a_top, a_right, a_bottom = left
    return not (a_right < right.left or a_left > right.right or a_bottom < right.top or a_top > right.bottom)
