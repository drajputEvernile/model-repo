"""Printed page-number detail rows and the record summary."""

from __future__ import annotations

from collections import Counter

from ..page_no.extract import PageHit

COLUMNS = [
    "RecordId",
    "FileName",
    "PageNumber",
    "Key",
    "Region",
    "Sentence",
    "Ner_Text",
    "Value",
    "PageNo",
    "PageTotal",
    "Score",
    "Accepted",
    "Selected",
    "Source",
    "Accuracy",
]
SUMMARY_COLUMNS = ["RecordId", "PageCount", "PagesWithLabel", "PageTotal", "TimeSeconds"]


def to_row(record_id: str, page: dict, hit: PageHit) -> dict[str, str]:
    return {
        "RecordId": record_id,
        "FileName": str(page.get("fileName") or ""),
        "PageNumber": str(page.get("pageNumber") or ""),
        "Key": hit.key,
        "Region": hit.region,
        "Sentence": hit.sentence,
        "Ner_Text": "",
        "Value": hit.value,
        "PageNo": hit.page_no,
        "PageTotal": hit.page_total,
        "Score": f"{hit.score:.4f}",
        "Accepted": "yes" if hit.accepted else "no",
        "Selected": "yes" if hit.selected else "no",
        "Source": hit.source,
        "Accuracy": "",
    }


def _record_total(rows: list[PageHit]) -> str:
    totals = Counter(row.page_total for row in rows if row.accepted and row.page_total)
    return totals.most_common(1)[0][0] if totals else ""


def summarize(pages: list[list[PageHit]]) -> dict[str, str]:
    return {
        "PagesWithLabel": str(sum(1 for hits in pages if hits)),
        "PageTotal": _record_total([hit for hits in pages for hit in hits]),
    }
