"""Gate-delta: reopen OCR stages when quality/rotation gates flip under skip_ocr.

Used by the orchestrator when ``skip_ocr=true`` and ``force=false``: re-run
quality, compare each page's gate signature to the pre-run snapshot, and set
only the affected OCR ``page_stage_status`` rows back to pending so engines
re-fire for those pages. Non-OCR stages (blank/junk, member, DOS, …) are
force-re-run by the orchestrator separately — they do not rely on this module.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

NON_PRINTED = frozenset({"handwritten", "uncertain", "mixed"})

# (stage_name, pass_no)
STAGE_PRELIM = ("ocr_prelim", 1)
STAGE_FINAL1 = ("ocr_final1", 1)
STAGE_FINAL2 = ("ocr_final2", 1)
STAGE_BJ1 = ("blank_junk", 1)
STAGE_BJ2 = ("blank_junk", 2)
STAGE_HEADERS = ("kv_extract", 1)
STAGE_MEMBER = ("member_verify", 1)
STAGE_DOS = ("dos_extract", 1)


@dataclass(frozen=True)
class GateSignature:
    hw_class: str  # printed | non_printed | unknown | ""
    quality_tag: str  # high | medium | low | ""
    rotation_applied: bool
    orientation_bucket: int  # 0/90/180/270

    @property
    def skips_bj_pass1(self) -> bool:
        return self.hw_class == "non_printed" or self.quality_tag == "low"

    @property
    def is_high_quality_printed(self) -> bool:
        return self.hw_class == "printed" and self.quality_tag == "high"

    @property
    def needs_final2(self) -> bool:
        return not self.is_high_quality_printed


@dataclass
class PageOcrPresence:
    has_prelim: bool = False
    has_final1: bool = False
    has_final2: bool = False


@dataclass
class GateDeltaPlan:
    """Per-chart result of comparing old vs new gates."""

    invalidate: dict[tuple[str, int], set[int]] = field(default_factory=dict)
    reasons: dict[int, list[str]] = field(default_factory=dict)
    force_prelim: set[int] = field(default_factory=set)
    force_final1: set[int] = field(default_factory=set)
    force_final2: set[int] = field(default_factory=set)

    def add(
        self,
        page_id: int,
        stages: Iterable[tuple[str, int]],
        reason: str,
    ) -> None:
        self.reasons.setdefault(page_id, [])
        if reason not in self.reasons[page_id]:
            self.reasons[page_id].append(reason)
        for stage in stages:
            self.invalidate.setdefault(stage, set()).add(page_id)
            if stage == STAGE_PRELIM:
                self.force_prelim.add(page_id)
            elif stage == STAGE_FINAL1:
                self.force_final1.add(page_id)
            elif stage == STAGE_FINAL2:
                self.force_final2.add(page_id)


def orientation_bucket(angle: Any) -> int:
    """Nearest 90° in [0, 360). Ties round away from zero toward the next step."""
    try:
        raw = float(angle if angle is not None else 0.0)
    except (TypeError, ValueError):
        raw = 0.0
    # Avoid banker's rounding on *.5 (Python round(0.5)==0).
    return int((raw + 45.0) // 90.0 * 90.0) % 360


def hw_class_from_row(row: Optional[dict[str, Any]]) -> str:
    if not row:
        return ""
    hw = str(row.get("printed_or_handwritten") or "").strip().lower()
    if not hw:
        return ""
    if hw in NON_PRINTED:
        return "non_printed"
    if hw == "printed":
        return "printed"
    return "unknown"


def gate_signature_from_row(row: Optional[dict[str, Any]]) -> Optional[GateSignature]:
    """Build a signature, or None when no quality row exists yet."""
    if not row:
        return None
    hw = hw_class_from_row(row)
    tag = str(row.get("quality_tag") or "").strip().lower()
    # A row with neither HW nor tag is treated as missing for delta purposes.
    if not hw and not tag and row.get("orientation_angle") is None:
        if not row.get("rotation_applied"):
            return None
    return GateSignature(
        hw_class=hw or "unknown",
        quality_tag=tag,
        rotation_applied=bool(row.get("rotation_applied")),
        orientation_bucket=orientation_bucket(row.get("orientation_angle")),
    )


def snapshot_gates(conn: Any, chart_id: int) -> dict[int, GateSignature]:
    """page_id → signature for every page that already has quality."""
    from db import get_quality_map

    out: dict[int, GateSignature] = {}
    for page_id, row in get_quality_map(conn, chart_id).items():
        sig = gate_signature_from_row(row)
        if sig is not None:
            out[int(page_id)] = sig
    return out


def _raw_has_content(raw: str, *, ocr_type: str) -> bool:
    text = (raw or "").strip()
    if not text:
        return False
    if ocr_type in {"docling", "azuredocintel"}:
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return True
        if isinstance(parsed, dict):
            body = str(
                parsed.get("content") or parsed.get("markdown") or ""
            ).strip()
            return bool(body)
    return True


def snapshot_ocr_presence(conn: Any, chart_id: int) -> dict[int, PageOcrPresence]:
    from db import get_ocr_texts, list_pages

    pages = list_pages(conn, chart_id)
    prelim = get_ocr_texts(conn, chart_id, "tesseract")
    final1 = get_ocr_texts(conn, chart_id, "docling")
    final2 = get_ocr_texts(conn, chart_id, "azuredocintel")
    out: dict[int, PageOcrPresence] = {}
    for page in pages:
        pid = int(page["id"])
        out[pid] = PageOcrPresence(
            has_prelim=_raw_has_content(prelim.get(pid, ""), ocr_type="tesseract"),
            has_final1=_raw_has_content(final1.get(pid, ""), ocr_type="docling"),
            has_final2=_raw_has_content(final2.get(pid, ""), ocr_type="azuredocintel"),
        )
    return out


def plan_page_delta(
    old: Optional[GateSignature],
    new: GateSignature,
    presence: PageOcrPresence,
) -> list[tuple[list[tuple[str, int]], str]]:
    """Return ``[(stages, reason), ...]`` for one page."""
    actions: list[tuple[list[tuple[str, int]], str]] = []

    def add(stages: list[tuple[str, int]], reason: str) -> None:
        if stages:
            actions.append((stages, reason))

    # Missing artifacts for the *new* gate — even when signature unchanged.
    if not presence.has_final1:
        add([STAGE_FINAL1, STAGE_HEADERS], "missing_final1")
    if new.needs_final2 and not presence.has_final2:
        add(
            [STAGE_FINAL2, STAGE_BJ2, STAGE_HEADERS],
            "needs_final2_missing",
        )

    if old is None:
        return actions

    if old == new:
        return actions

    rotation_changed = (
        old.rotation_applied != new.rotation_applied
        or old.orientation_bucket != new.orientation_bucket
    )
    if rotation_changed:
        stages = [
            STAGE_PRELIM,
            STAGE_FINAL1,
            STAGE_BJ1,
            STAGE_BJ2,
            STAGE_HEADERS,
            STAGE_MEMBER,
            STAGE_DOS,
        ]
        if new.needs_final2 or presence.has_final2:
            # Image changed — any existing Final2 is stale if present; if the
            # new gate needs Final2, open it even when empty.
            stages.append(STAGE_FINAL2)
        add(stages, "rotation_or_orientation_changed")
        return actions

    path_changed = (
        old.skips_bj_pass1 != new.skips_bj_pass1
        or old.hw_class != new.hw_class
        or old.quality_tag != new.quality_tag
    )
    if not path_changed:
        return actions

    bj_stages = [
        STAGE_BJ1,
        STAGE_BJ2,
        STAGE_HEADERS,
        STAGE_MEMBER,
        STAGE_DOS,
    ]
    add(bj_stages, "quality_or_hw_gate_changed")

    if old.is_high_quality_printed and not new.is_high_quality_printed:
        if not presence.has_final2:
            add([STAGE_FINAL2, STAGE_BJ2, STAGE_HEADERS], "lost_high_printed_need_final2")
        else:
            add([STAGE_BJ2, STAGE_HEADERS], "lost_high_printed_have_final2")
    elif not old.is_high_quality_printed and new.is_high_quality_printed:
        add([STAGE_BJ2, STAGE_HEADERS], "gained_high_printed")

    if not presence.has_final1:
        add([STAGE_FINAL1, STAGE_HEADERS], "gate_change_missing_final1")
    if new.needs_final2 and not presence.has_final2:
        add([STAGE_FINAL2], "gate_change_missing_final2")

    return actions


def compute_gate_delta(
    old_gates: dict[int, GateSignature],
    new_gates: dict[int, GateSignature],
    presence: dict[int, PageOcrPresence],
    page_ids: Sequence[int],
) -> GateDeltaPlan:
    plan = GateDeltaPlan()
    for page_id in page_ids:
        pid = int(page_id)
        new = new_gates.get(pid)
        if new is None:
            continue
        old = old_gates.get(pid)
        page_presence = presence.get(pid) or PageOcrPresence()
        for stages, reason in plan_page_delta(old, new, page_presence):
            plan.add(pid, stages, reason)
    return plan


def invalidate_from_plan(conn: Any, chart_id: int, plan: GateDeltaPlan) -> dict[str, int]:
    """Apply ``page_stage_status`` resets; return counts per stage key."""
    from db import reset_pages_stage

    counts: dict[str, int] = {}
    for (stage_name, pass_no), page_ids in plan.invalidate.items():
        reset_pages_stage(
            conn, chart_id, sorted(page_ids), stage_name, pass_no=pass_no
        )
        counts[f"{stage_name}:{pass_no}"] = len(page_ids)
        logger.info(
            "Gate-delta: reset %s:%s for %d page(s)",
            stage_name,
            pass_no,
            len(page_ids),
        )
    return counts


def apply_adaptive_gate_delta(
    conn: Any,
    chart_id: int,
    *,
    old_gates: dict[int, GateSignature],
    old_presence: dict[int, PageOcrPresence],
) -> GateDeltaPlan:
    """Diff current quality vs snapshot and invalidate affected stages."""
    from db import list_pages

    new_gates = snapshot_gates(conn, chart_id)
    # Prefer pre-hydrate OCR presence; refresh may have filled gaps from disk.
    presence = snapshot_ocr_presence(conn, chart_id)
    for pid, prior in old_presence.items():
        cur = presence.get(pid) or PageOcrPresence()
        # Keep True if either snapshot saw content (hydrate may have rewritten).
        presence[pid] = PageOcrPresence(
            has_prelim=cur.has_prelim or prior.has_prelim,
            has_final1=cur.has_final1 or prior.has_final1,
            has_final2=cur.has_final2 or prior.has_final2,
        )
    page_ids = [int(p["id"]) for p in list_pages(conn, chart_id)]
    plan = compute_gate_delta(old_gates, new_gates, presence, page_ids)
    invalidate_from_plan(conn, chart_id, plan)
    if plan.reasons:
        logger.info(
            "Gate-delta chart %s: %d page(s) reopened — %s",
            chart_id,
            len(plan.reasons),
            {pid: rs for pid, rs in list(plan.reasons.items())[:20]},
        )
    else:
        logger.info("Gate-delta chart %s: no gate changes requiring reopen", chart_id)
    return plan
