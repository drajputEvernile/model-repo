"""Member DOB detail rows and the record summary."""

from __future__ import annotations

from collections import Counter

from ..member_dob.extract import DobHit

COLUMNS = [
    "RecordId",
    "FileName",
    "PageNumber",
    "Key",
    "Region",
    "Sentence",
    "Ner_Text",
    "Value",
    "Score",
    "Accepted",
    "Selected",
    "Source",
    "Accuracy",
]
SUMMARY_COLUMNS = ["RecordId", "PageCount", "DOB", "TimeSeconds"]


def _score(hit: DobHit) -> str:
    if not hit.accepted and not hit.ner_text:
        return ""
    return f"{hit.score:.4f}"


def to_row(record_id: str, page: dict, hit: DobHit) -> dict[str, str]:
    return {
        "RecordId": record_id,
        "FileName": str(page.get("fileName") or ""),
        "PageNumber": str(page.get("pageNumber") or ""),
        "Key": hit.key,
        "Region": hit.region,
        "Sentence": hit.sentence,
        "Ner_Text": hit.ner_text,
        "Value": hit.value,
        "Score": _score(hit),
        "Accepted": "yes" if hit.accepted else "no",
        "Selected": "yes" if hit.selected else "no",
        "Source": hit.source,
        "Accuracy": "",
    }


def _record_dob(rows: list[DobHit]) -> str:
    present = [row for row in rows if row.accepted and row.selected and row.value]
    if not present:
        return ""
    counts = Counter(row.value.casefold() for row in present)
    best_count = max(counts.values())
    tied = {text for text, count in counts.items() if count == best_count}
    winners = [row for row in present if row.value.casefold() in tied]
    winners.sort(key=lambda row: row.score, reverse=True)
    return winners[0].value


def summarize(pages: list[list[DobHit]]) -> dict[str, str]:
    return {"DOB": _record_dob([row for rows in pages for row in rows])}
