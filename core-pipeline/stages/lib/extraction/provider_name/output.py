"""Provider Name detail rows and the record summary."""

from __future__ import annotations

from collections import Counter

from ..provider_name.extract import ProviderHit

COLUMNS = [
    "RecordId",
    "FileName",
    "PageNumber",
    "Key",
    "Region",
    "Scale",
    "Sentence",
    "Ner_Text",
    "Value",
    "Score",
    "Accepted",
    "Selected",
    "Source",
    "Profile",
    "Accuracy",
]
SUMMARY_COLUMNS = ["RecordId", "PageCount", "ProviderName", "TimeSeconds"]


def _score(hit: ProviderHit) -> str:
    if not hit.accepted and not hit.ner_text:
        return ""
    return f"{hit.score:.4f}"


def to_row(record_id: str, page: dict, hit: ProviderHit) -> dict[str, str]:
    return {
        "RecordId": record_id,
        "FileName": str(page.get("fileName") or ""),
        "PageNumber": str(page.get("pageNumber") or ""),
        "Key": hit.key,
        "Region": hit.region,
        "Scale": hit.scale,
        "Sentence": hit.sentence,
        "Ner_Text": hit.ner_text,
        "Value": hit.value,
        "Score": _score(hit),
        "Accepted": "yes" if hit.accepted else "no",
        "Selected": "yes" if hit.selected else "no",
        "Source": hit.source if hit.accepted or hit.ner_text else "",
        "Profile": hit.profile,
        "Accuracy": "",
    }


def _record_provider(rows: list[ProviderHit]) -> str:
    present = [row for row in rows if row.accepted and row.selected and row.value]
    if not present:
        return ""
    counts = Counter(row.value.casefold() for row in present)
    best_count = max(counts.values())
    tied = {text for text, count in counts.items() if count == best_count}
    winners = [row for row in present if row.value.casefold() in tied]
    winners.sort(key=lambda row: row.score, reverse=True)
    return winners[0].value


def summarize(pages: list[list[ProviderHit]]) -> dict[str, str]:
    return {"ProviderName": _record_provider([row for rows in pages for row in rows])}
