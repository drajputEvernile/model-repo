"""Date of Service detail rows and the record summary."""

from __future__ import annotations

from collections import Counter

from ..dos.extract import DosHit
from ..util.dates import canonical_date

COLUMNS = [
    "RecordId",
    "FileName",
    "PageNumber",
    "Key",
    "Region",
    "Tier",
    "Sentence",
    "Ner_Text",
    "Value",
    "DOS_From",
    "DOS_To",
    "Score",
    "Accepted",
    "Selected",
    "Source",
    "Accuracy",
]
SUMMARY_COLUMNS = ["RecordId", "PageCount", "DOS", "DOS_List", "TimeSeconds"]


def to_row(record_id: str, page: dict, hit: DosHit) -> dict[str, str]:
    return {
        "RecordId": record_id,
        "FileName": str(page.get("fileName") or ""),
        "PageNumber": str(page.get("pageNumber") or ""),
        "Key": hit.key,
        "Region": hit.region,
        "Tier": hit.tier,
        "Sentence": hit.sentence,
        "Ner_Text": "",
        "Value": hit.value,
        "DOS_From": hit.dos_from,
        "DOS_To": hit.dos_to,
        "Score": f"{hit.score:.4f}" if hit.accepted else "",
        "Accepted": "yes" if hit.accepted else "no",
        "Selected": "yes" if hit.selected else "no",
        "Source": hit.source,
        "Accuracy": "",
    }


def _page_dos(rows: list[DosHit]) -> list[tuple[str, str]]:
    """(from, to) of a page's selected DOS; its admit and discharge dates make one range."""
    selected = [row for row in rows if row.selected]
    admit = next((row for row in selected if row.tier == "admit"), None)
    discharge = next((row for row in selected if row.tier == "discharge"), None)
    if admit and discharge:
        return [(admit.dos_from, discharge.dos_to)]
    return [(row.dos_from, row.dos_to) for row in selected]


def _record_dos(ranges: list[tuple[str, str]]) -> tuple[str, str]:
    """Most common selected DOS, plus every distinct selected DOS in page order."""
    if not ranges:
        return "", ""
    values = [low if low == high else f"{low} - {high}" for low, high in ranges]
    markers = [canonical_date(low) + "|" + canonical_date(high) for low, high in ranges]
    counts = Counter(markers)
    best = max(counts.values())
    top = next(value for value, marker in zip(values, markers) if counts[marker] == best)
    listed: list[str] = []
    seen: set[str] = set()
    for value, marker in zip(values, markers):
        if marker not in seen:
            seen.add(marker)
            listed.append(value)
    return top, "; ".join(listed)


def summarize(pages: list[list[DosHit]]) -> dict[str, str]:
    dos, dos_list = _record_dos([span for rows in pages for span in _page_dos(rows)])
    return {"DOS": dos, "DOS_List": dos_list}
