"""Electronic Signature detail rows and the record summary."""

from __future__ import annotations

from collections import Counter

from ..electronic_signature.extract import ESigHit

COLUMNS = [
    "RecordId",
    "FileName",
    "PageNumber",
    "Key",
    "Region",
    "Scale",
    "Sentence",
    "Ner_Text",
    "ProviderName",
    "SignatureDate",
    "Score",
    "Accepted",
    "Selected",
    "Source",
    "Accuracy",
]
SUMMARY_COLUMNS = ["RecordId", "PageCount", "ProviderName", "SignatureDate", "TimeSeconds"]


def _score(hit: ESigHit) -> str:
    if not hit.accepted and not hit.ner_text and not hit.signature_date:
        return ""
    return f"{hit.score:.4f}"


def to_row(record_id: str, page: dict, hit: ESigHit) -> dict[str, str]:
    return {
        "RecordId": record_id,
        "FileName": str(page.get("fileName") or ""),
        "PageNumber": str(page.get("pageNumber") or ""),
        "Key": hit.key,
        "Region": hit.region,
        "Scale": hit.scale,
        "Sentence": hit.sentence,
        "Ner_Text": hit.ner_text,
        "ProviderName": hit.provider_name,
        "SignatureDate": hit.signature_date,
        "Score": _score(hit),
        "Accepted": "yes" if hit.accepted else "no",
        "Selected": "yes" if hit.selected else "no",
        "Source": hit.source if hit.accepted or hit.ner_text or hit.signature_date else "",
        "Accuracy": "",
    }


def _record_field(rows: list[ESigHit], attr: str) -> str:
    present = [row for row in rows if row.accepted and row.selected and getattr(row, attr)]
    if not present:
        # Fall back: any selected row's companion field, else majority across accepted.
        selected = [row for row in rows if row.accepted and row.selected]
        if selected and getattr(selected[0], attr):
            return getattr(selected[0], attr)
        values = [getattr(row, attr) for row in rows if row.accepted and getattr(row, attr)]
        if not values:
            return ""
        counts = Counter(value.casefold() for value in values)
        best = max(counts.values())
        winners = [value for value in values if counts[value.casefold()] == best]
        return winners[0]
    counts = Counter(getattr(row, attr).casefold() for row in present)
    best_count = max(counts.values())
    tied = {text for text, count in counts.items() if count == best_count}
    winners = [row for row in present if getattr(row, attr).casefold() in tied]
    winners.sort(key=lambda row: row.score, reverse=True)
    return getattr(winners[0], attr)


def summarize(pages: list[list[ESigHit]]) -> dict[str, str]:
    chosen = [row for rows in pages for row in rows]
    return {
        "ProviderName": _record_field(chosen, "provider_name"),
        "SignatureDate": _record_field(chosen, "signature_date"),
    }
