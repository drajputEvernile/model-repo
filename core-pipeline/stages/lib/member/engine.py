"""Member verification engine.

Ported from the V1 ``Member_Verification/run.py``. Function names, call order and decision
logic match the reference so a pipeline run is diffable against a V1 run page for page.

What changed: where the page's name, DOB and member ID come from
------------------------------------------------------------------
V1 searched each page's text for *the manifest's* name, DOB and ID, and sent the pages it
could not read to a NER pass. That extraction is now its own layer (``stages/lib/extraction``):
it reads every page once, finds the member name, DOB and ID wherever the page keeps them, and
stages them. Verification takes what was staged and checks it against the manifest with the
same rules as before.

* **Name** — the extraction's member names for the page. The one that matches the manifest is
  the page's detected name; with none matching, the one the extraction selected.
* **DOB / member ID** — present when an extracted value says what the manifest says
  (``rules/field_match.py``, the V1 comparison). The value shown is the matching one, else the
  one the extraction selected, so a page with someone else's DOB shows that DOB.
* **The wrong-member escalation** — every member name the extraction found on a page that
  failed verification. A page that names someone else and never the expected member is
  ``wrong_member``, and wrong-member pages are what reject a document.
* **what-if thresholds** — unchanged: Accept/Reject on ``wrong pages >= min(5, ceil(10% of
  pages))``, from ``what_if_rules``.

Unchanged: ``name_mode`` comes from the manifest row's populated name parts; results are
dataclasses for the stage to persist; the reference's CSV column names are preserved in
:func:`page_result_to_v1_row` for diffing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..extraction.staging import StagedPage
from ..extraction.util.dates import canonical_date
from .rules.field_match import dob_matches, member_id_matches
from .rules.name_2_words_rules import verify_two_word_name
from .rules.name_3_words_rules import verify_three_word_name
from .rules.name_common import name_matches, tokenize
from .rules.what_if_rules import (
    ACCEPT,
    PAGE_NOT_VERIFIED,
    PAGE_VERIFIED,
    PAGE_WRONG_MEMBER,
    REJECT,
    apply_what_if,
    count_wrong_member,
    page_status,
    reject_threshold,
)
from .rules.wrong_member_rules import wrong_member_on_page

logger = logging.getLogger(__name__)

NA = "N/A"

# V1 page bucket -> member_extraction_results.page_status
PAGE_STATUS_DB = {
    PAGE_VERIFIED: "verified",
    PAGE_WRONG_MEMBER: "wrong_member",
    PAGE_NOT_VERIFIED: "not_verified",
}

# V1 detection source -> member_extraction_results.detection_source_*
DETECTION_SOURCE_DB = {"rule based": "rule_based", "ner": "ner", "": ""}


@dataclass
class PageResult:
    """One page's verification outcome (one member_extraction_results row)."""

    page_no: int
    page_name: str
    detected_name: str
    detected_dob: str
    detected_member_id: str
    detection_source_name: str
    detection_source_dob: str
    detection_source_member_id: str
    ner_key_source_name: str
    ner_key_source_dob: str
    ner_key_source_member_id: str
    page_verified: bool
    page_status: str          # V1 bucket: Verified / Wrong_Member / Not_Verified
    ner_names: list[str] = field(default_factory=list)

    @property
    def db_page_status(self) -> str:
        return PAGE_STATUS_DB.get(self.page_status, "not_verified")


@dataclass
class RecordResult:
    """A chart's verification outcome (one member_verification_summary row)."""

    record_id: str
    name_mode: str
    total_pages: int
    pages_checked: int
    pages_verified: int
    pages_wrong_member: int
    pages_not_verified: int
    reject_threshold: int
    document_decision: str    # V1: Accept / Reject
    ner_enabled: bool
    model_id: Optional[str]
    pages: list[PageResult] = field(default_factory=list)
    expected: dict[str, str] = field(default_factory=dict)

    @property
    def db_document_decision(self) -> str:
        return "accept" if self.document_decision == ACCEPT else "reject"


def expected_from_manifest(row: dict[str, Any]) -> dict[str, str]:
    """manifest_member_list row -> the reference's `expected` dict.

    Key names are the reference's (DummyFirstName, ...) so every ported rule
    reads them unchanged.
    """

    def _s(value: Any) -> str:
        return "" if value is None else str(value).strip()

    dob = row.get("member_dob")
    dob_text = ""
    if dob:
        # The reference parses DummyDOB as MM/DD/YYYY.
        if hasattr(dob, "strftime"):
            dob_text = dob.strftime("%m/%d/%Y")
        else:
            dob_text = _s(dob)

    first = _s(row.get("first_name"))
    middle = _s(row.get("middle_name"))
    last = _s(row.get("last_name"))

    # Fall back to splitting member_name when the parts were never populated.
    if not first and not last:
        parts = [p for p in _s(row.get("member_name")).split() if p]
        if len(parts) == 2:
            first, last = parts
        elif len(parts) >= 3:
            first, middle, last = parts[0], parts[1], parts[-1]
        elif len(parts) == 1:
            first = parts[0]

    return {
        "DummyFirstName": first,
        "DummyMiddleName": middle,
        "DummyLastName": last,
        "DummyDOB": dob_text,
        "MemberID": _s(row.get("external_member_id")),
    }


def detect_name_mode(expected: dict[str, str]) -> str:
    """'3' when a middle name is known, '2' for first+last, '' when unusable.

    The reference read this off the system-input CSV's header set; the manifest
    carries the same information in which name parts are populated.
    """
    first = (expected.get("DummyFirstName") or "").strip()
    middle = (expected.get("DummyMiddleName") or "").strip()
    last = (expected.get("DummyLastName") or "").strip()
    if first and middle and last:
        return "3"
    if first and last:
        return "2"
    return ""


# --- the page's fields, from the extraction ----------------------------------


def _source(row: dict[str, Any]) -> str:
    """V1's detection source: NER when GLiNER read the value, else the rules / page geometry."""
    return "ner" if row.get("ner_text") else "rule based"


def _dob_text(value: str) -> str:
    """MM/DD/YYYY when the extracted DOB is a date we can read, else as extracted."""
    iso = canonical_date(value)
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%m/%d/%Y")
    except ValueError:
        return value


def _chosen_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows the extraction selected first, then the rest in page order."""
    return sorted(rows, key=lambda row: not row.get("selected"))


def staged_page_fields(
    staged: Optional[StagedPage],
    expected: dict[str, str],
    name_mode: str,
) -> tuple[dict[str, str], list[str], bool, bool]:
    """(page fields, member names found, DOB matches the manifest, ID matches the manifest).

    The fields use the reference's keys. ``staged`` is None for a page the extraction could
    not read (no word boxes): nothing is found on it, so it is neither verified nor wrong.
    """
    fields = {
        "Detected_Full_Name": NA,
        "Detection_Source_Name": "",
        "ner_key_source_Name": "",
        "Detected_DOB": NA,
        "Detection_Source_DOB": "",
        "ner_key_source_DOB": "",
        "Detected_MemberID": NA,
        "Detection_Source_MemberID": "",
        "ner_key_source_MemberID": "",
    }
    if staged is None:
        return fields, [], False, False

    names = [row for row in _chosen_first(staged.accepted("name")) if (row.get("value") or "").strip()]
    first = expected.get("DummyFirstName", "")
    middle = expected.get("DummyMiddleName", "")
    last = expected.get("DummyLastName", "")
    ours = next(
        (row for row in names if name_mode and name_matches(row["value"], first, last, middle, name_mode)),
        None,
    )
    pick = ours or (names[0] if names else None)
    if pick is not None:
        fields.update(
            Detected_Full_Name=pick["value"].strip(),
            Detection_Source_Name=_source(pick),
            ner_key_source_Name=pick.get("key") or "",
        )

    dobs = [row for row in _chosen_first(staged.accepted("dob")) if (row.get("value") or "").strip()]
    dob_hit = next((row for row in dobs if dob_matches(row["value"], expected["DummyDOB"])), None)
    dob_pick = dob_hit or (dobs[0] if dobs else None)
    if dob_pick is not None:
        fields.update(
            Detected_DOB=_dob_text(dob_pick["value"]),
            Detection_Source_DOB=_source(dob_pick),
            ner_key_source_DOB=dob_pick.get("key") or "",
        )

    ids = [row for row in _chosen_first(staged.accepted("member_id")) if (row.get("value") or "").strip()]
    id_hit = next((row for row in ids if member_id_matches(row["value"], expected["MemberID"])), None)
    id_pick = id_hit or (ids[0] if ids else None)
    if id_pick is not None:
        fields.update(
            Detected_MemberID=id_pick["value"].strip(),
            Detection_Source_MemberID=_source(id_pick),
            ner_key_source_MemberID=id_pick.get("key") or "",
        )

    return fields, [row["value"].strip() for row in names], dob_hit is not None, id_hit is not None


def verify_page(
    expected: dict[str, str],
    name_mode: str,
    found_name: str,
    dob_ok: bool,
    id_ok: bool,
) -> bool:
    if name_mode == "3":
        status = verify_three_word_name(
            found_name,
            expected["DummyFirstName"],
            expected["DummyMiddleName"],
            expected["DummyLastName"],
            dob_ok,
            id_ok,
        )
    elif name_mode == "2":
        status = verify_two_word_name(
            found_name,
            expected["DummyFirstName"],
            expected["DummyLastName"],
            dob_ok,
            id_ok,
        )
    else:
        status = "Reject"
    return status == "Accept"


def classify_page(
    ner_names: list[str],
    expected: dict[str, str],
    name_mode: str,
    verified: bool,
) -> str:
    """Verified / wrong member / not verified for one page.

    A page counts as wrong member only when the member names found on it are
    someone other than the expected member. A page with nothing detected is
    not considered.
    """
    wrong_member = not verified and wrong_member_on_page(
        expected,
        name_mode,
        ner_names,
    )
    return page_status(verified=verified, wrong_member=wrong_member)


def document_verified(page_statuses: list[str], total: int) -> str:
    return apply_what_if(page_statuses, total)


# --- chart-level driver -----------------------------------------------------


def trim_extracted_name(extracted: str, expected: dict[str, str]) -> str:
    """Drop words before the member's name and after it.

    "Abhinav X Dasgupta alias of" against manifest "Abhinav Dasgupta" becomes
    "Abhinav X Dasgupta". Words between the first and last name stay, and the
    names may sit in either order. A name that is not this member is left
    unchanged.
    """
    raw = (extracted or "").strip()
    if not raw or raw.upper() == NA:
        return extracted
    first = (expected.get("DummyFirstName") or "").strip()
    middle = (expected.get("DummyMiddleName") or "").strip()
    last = (expected.get("DummyLastName") or "").strip()
    if not first or not last:
        return extracted
    anchors = {part.casefold() for part in (first, middle, last) if part}
    tokens = tokenize(raw)
    hits = [index for index, token in enumerate(tokens) if token.casefold() in anchors]
    found = {tokens[index].casefold() for index in hits}
    if first.casefold() not in found or last.casefold() not in found:
        return extracted
    start, end = hits[0], hits[-1]
    if start == 0 and end == len(tokens) - 1:
        return extracted
    return " ".join(tokens[start : end + 1])


def verify_record(
    record_id: str,
    pages: list[dict[str, Any]],
    expected: dict[str, str],
    name_mode: str,
    model_id: Optional[str] = None,
    total_pages: Optional[int] = None,
) -> RecordResult:
    """Verify one chart. `pages` is [{page_no, page_name, staged}, ...].

    Mirrors run.py::verify_record. `staged` is the page's extraction (a
    ``StagedPage``), or None for a page it could not read. `total_pages` is the chart's
    full page count, which can exceed len(pages) once blank/junk pages are excluded — the
    reject threshold is a proportion of the *document*, so it must be computed on the real
    total, not on the subset that reached this stage.
    """
    total = int(total_pages if total_pages is not None else len(pages))
    results: list[PageResult] = []

    for index, page in enumerate(pages, start=1):
        page_no = int(page.get("page_no") or index)

        fields, ner_names, dob_ok, id_ok = staged_page_fields(page.get("staged"), expected, name_mode)
        fields["Detected_Full_Name"] = trim_extracted_name(fields["Detected_Full_Name"], expected)
        page_ok = verify_page(
            expected,
            name_mode,
            fields["Detected_Full_Name"],
            dob_ok,
            id_ok,
        )
        status = classify_page(ner_names, expected, name_mode, page_ok)

        results.append(
            PageResult(
                page_no=page_no,
                page_name=str(page.get("page_name") or ""),
                detected_name=fields["Detected_Full_Name"],
                detected_dob=fields["Detected_DOB"],
                detected_member_id=fields["Detected_MemberID"],
                detection_source_name=fields["Detection_Source_Name"],
                detection_source_dob=fields["Detection_Source_DOB"],
                detection_source_member_id=fields["Detection_Source_MemberID"],
                ner_key_source_name=fields["ner_key_source_Name"],
                ner_key_source_dob=fields["ner_key_source_DOB"],
                ner_key_source_member_id=fields["ner_key_source_MemberID"],
                page_verified=page_ok,
                page_status=status,
                ner_names=ner_names,
            )
        )

    results.sort(key=lambda r: r.page_no)
    statuses = [r.page_status for r in results]
    doc_status = document_verified(statuses, total)
    threshold = reject_threshold(total)

    logger.info(
        "record %s model %s: pages=%s checked=%s verified=%s wrong_member=%s "
        "not_verified=%s reject_at=%s document=%s",
        record_id,
        model_id,
        total,
        len(results),
        statuses.count(PAGE_VERIFIED),
        count_wrong_member(statuses),
        statuses.count(PAGE_NOT_VERIFIED),
        threshold,
        doc_status,
    )

    return RecordResult(
        record_id=record_id,
        name_mode=name_mode,
        total_pages=total,
        pages_checked=len(results),
        pages_verified=statuses.count(PAGE_VERIFIED),
        pages_wrong_member=count_wrong_member(statuses),
        pages_not_verified=statuses.count(PAGE_NOT_VERIFIED),
        reject_threshold=threshold,
        document_decision=doc_status,
        ner_enabled=True,
        model_id=model_id,
        pages=results,
        expected=expected,
    )


def summary_status(result: RecordResult) -> tuple[str, str]:
    """(member_verification_summary.final_status, decision_reason).

    The reference produced Accept/Reject; the schema also carries a
    verified/failed/needs_review triage value for the review UI.
    """
    # No pages were checked — every page was blank/junk/duplicate, or none
    # reached this stage. The what-if rule rejects an empty list; that is not
    # a member failure, and it must not mark a one-page junk chart failed.
    if result.pages_checked == 0:
        return "skipped", "all_blank_junk"
    if result.document_decision == REJECT:
        return "failed", "wrong_member_threshold"
    if result.pages_verified > 0:
        return "verified", "pages_verified"
    if not result.name_mode:
        return "needs_review", "manifest_name_incomplete"
    return "needs_review", "no_page_verified"


def page_result_to_v1_row(
    result: RecordResult, page: PageResult
) -> dict[str, Any]:
    """One row in the reference's Member CSV column order, for diffing."""
    return {
        "RecordId": result.record_id,
        "Total_Page_Count": result.total_pages,
        "Page_No": page.page_no,
        "Detection_Source_Name": page.detection_source_name,
        "ner_key_source_Name": page.ner_key_source_name,
        "Detected_Full_Name": page.detected_name,
        "Detection_Source_DOB": page.detection_source_dob,
        "ner_key_source_DOB": page.ner_key_source_dob,
        "Detected_DOB": page.detected_dob,
        "Detection_Source_MemberID": page.detection_source_member_id,
        "ner_key_source_MemberID": page.ner_key_source_member_id,
        "Detected_MemberID": page.detected_member_id,
        "Page_Verified": page.page_verified,
        "Document_Verified": result.document_decision,
        "Page_Detection_Correct": page.page_status == PAGE_VERIFIED,
        "Page_Detection_InCorrect": page.page_status == PAGE_WRONG_MEMBER,
    }
