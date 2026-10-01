"""Member ID detail rows and the record summary."""

from __future__ import annotations

from ..member_id.extract import IdHit

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
SUMMARY_COLUMNS = ["RecordId", "PageCount", "MemberID", "TimeSeconds"]


def _score(hit: IdHit) -> str:
    if not hit.accepted and not hit.ner_text:
        return ""
    return f"{hit.score:.4f}"


def to_row(record_id: str, page: dict, hit: IdHit) -> dict[str, str]:
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


def _record_ids(rows: list[IdHit]) -> str:
    """Distinct keys across the record; higher score wins when the key repeats."""
    best: dict[str, IdHit] = {}
    for row in rows:
        if not (row.accepted and row.selected and row.value):
            continue
        current = best.get(row.key.casefold())
        if current is None or row.score > current.score:
            best[row.key.casefold()] = row
    ordered = sorted(best.values(), key=lambda row: row.key.casefold())
    return "; ".join(f"{row.key}={row.value}" for row in ordered)


def summarize(pages: list[list[IdHit]]) -> dict[str, str]:
    return {"MemberID": _record_ids([row for rows in pages for row in rows])}
