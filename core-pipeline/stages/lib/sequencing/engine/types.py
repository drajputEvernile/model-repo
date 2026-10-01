"""Shared data structures for page sequencing."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageFeatures:
    page_id: str
    original_page_number: int
    is_classified: bool
    header_text: str
    footer_text: str
    header_raw: str
    footer_raw: str
    top_lines_text: str
    bottom_lines_text: str
    full_text: str
    word_count: int = 0
    has_structured: bool = False
    identity_text: str = ""
    text_fingerprint: str | None = None
    # The printed page number the key/value extraction found. When the extraction ran
    # (marker_extracted) it replaces the text patterns: a page it found none on has no marker.
    extracted_marker: "ExplicitMarker | None" = None
    marker_extracted: bool = False


@dataclass
class ExplicitMarker:
    page_num: int
    total_pages: int | None
    pattern: str
    confidence: float
    source: str = "body"


@dataclass
class SequenceAssignment:
    page_id: str
    original_page_number: int
    sequence_position: int | None
    sequence_method: str
    sequence_confidence: float
    sequence_review_flag: bool
    explicit_marker_found: bool = False
    marker_page_num: int | None = None
    marker_total_pages: int | None = None
    header_group_id: str | None = None
    continuation_score: float | None = None
    stream_id: str | None = None


@dataclass
class Stream:
    stream_id: str
    total_pages: int
    features: list[PageFeatures] = field(default_factory=list)
    markers: dict[str, ExplicitMarker] = field(default_factory=dict)
    is_conflict: bool = False


@dataclass
class StreamSplit:
    streams: list[Stream] = field(default_factory=list)
    orphans: list[PageFeatures] = field(default_factory=list)
    classified: list[PageFeatures] = field(default_factory=list)
    duplicate_page_ids: set[str] = field(default_factory=set)

    @property
    def is_single_stream(self) -> bool:
        if self.classified or self.duplicate_page_ids:
            return False
        if len(self.streams) > 1:
            return False
        if len(self.streams) == 1 and self.orphans:
            return False
        return True
