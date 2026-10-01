"""Fidelity tests for the ported V1 member verification engine.

These pin the reference's decision logic. If a change here starts failing, the
port has drifted from the V1 ``Member_Verification`` behaviour and the fix is
to bring it back, not to update the expectation.
"""
from __future__ import annotations

import math

import pytest

from stages.lib.member import (
    classify_page,
    detect_name_mode,
    expected_from_manifest,
    summary_status,
    verify_page,
    verify_record,
)
from stages.lib.extraction.staging import StagedPage
from stages.lib.member import staged_page_fields
from stages.lib.member.rules.field_match import dob_matches, member_id_matches
from stages.lib.member.rules.name_common import (
    ALL_FULL,
    BOTH_FULL,
    INITIAL,
    MISMATCH,
    ONE_FULL_WRONG,
    TWO_FULL,
    classify_three_word_name,
    classify_two_word_name,
    name_matches,
    tokenize,
)
from stages.lib.member.rules.base_rules import combine_evidences
from stages.lib.member.rules.what_if_rules import (
    ACCEPT,
    PAGE_NOT_VERIFIED,
    PAGE_VERIFIED,
    PAGE_WRONG_MEMBER,
    REJECT,
    apply_what_if,
    count_wrong_member,
    reject_threshold,
)


def manifest(**overrides):
    row = {
        "id": 1,
        "member_name": "Justin Anderson",
        "first_name": "Justin",
        "middle_name": None,
        "last_name": "Anderson",
        "member_dob": "08/29/1954",
        "external_member_id": "A9000603900",
    }
    row.update(overrides)
    return row


# --- evidence combination (base_rules) --------------------------------------


class TestCombineEvidences:
    def test_no_name_always_rejects(self):
        assert combine_evidences(name_ok=False, dob_ok=True, id_ok=True) == "Reject"

    def test_name_plus_one_corroborator_accepts(self):
        assert combine_evidences(name_ok=True, dob_ok=True, id_ok=False) == "Accept"
        assert combine_evidences(name_ok=True, dob_ok=False, id_ok=True) == "Accept"

    def test_name_alone_rejects(self):
        assert combine_evidences(name_ok=True, dob_ok=False, id_ok=False) == "Reject"

    def test_initial_only_name_needs_both_corroborators(self):
        assert combine_evidences(
            name_ok=True, dob_ok=True, id_ok=True, initial_only=True
        ) == "Accept"
        assert combine_evidences(
            name_ok=True, dob_ok=True, id_ok=False, initial_only=True
        ) == "Reject"


# --- name classification ----------------------------------------------------


class TestTwoWordName:
    def test_both_full(self):
        span = tokenize("Justin Anderson")
        assert classify_two_word_name(span, "Justin", "Anderson") == BOTH_FULL

    def test_first_initial_last_full(self):
        span = tokenize("J Anderson")
        assert classify_two_word_name(span, "Justin", "Anderson") == INITIAL

    def test_different_person_is_mismatch(self):
        span = tokenize("Maria Gonzalez")
        assert classify_two_word_name(span, "Justin", "Anderson") == MISMATCH

    def test_one_match_beside_another_full_name_is_wrong_not_partial(self):
        """A matching surname next to a different given name is a different
        member, not a missed detection — the reference distinguishes these."""
        span = tokenize("Marcus Anderson")
        assert classify_two_word_name(span, "Justin", "Anderson") == ONE_FULL_WRONG

    def test_suffixes_and_titles_ignored(self):
        span = tokenize("Justin Anderson Jr MD")
        assert classify_two_word_name(span, "Justin", "Anderson") == BOTH_FULL

    def test_reversed_name_with_a_middle_initial_matches(self):
        span = tokenize("Lisa X Anderson")
        assert classify_two_word_name(span, "Anderson", "Lisa") == BOTH_FULL
        assert name_matches("Lisa X Anderson", "Anderson", "Lisa", "", "2")


class TestThreeWordName:
    def test_all_three_full(self):
        span = tokenize("Justin Robert Anderson")
        assert classify_three_word_name(span, "Justin", "Robert", "Anderson") == ALL_FULL

    def test_two_of_three_is_still_a_match(self):
        span = tokenize("Justin Anderson")
        assert classify_three_word_name(span, "Justin", "Robert", "Anderson") == TWO_FULL

    def test_one_of_three_is_mismatch(self):
        span = tokenize("Justin Gonzalez Ramirez")
        assert classify_three_word_name(span, "Justin", "Robert", "Anderson") == MISMATCH


# --- DOB and member id ------------------------------------------------------


class TestDobMatch:
    @pytest.mark.parametrize(
        "value",
        ["08/29/1954", "8-29-1954", "1954 08 29", "August 29, 1954"],
    )
    def test_matches_parts_in_any_supported_order(self, value):
        assert dob_matches(value, "08/29/1954")

    def test_rejects_a_different_date(self):
        assert not dob_matches("01/02/1970", "08/29/1954")

    def test_nothing_to_compare_never_matches(self):
        assert not dob_matches("", "08/29/1954")
        assert not dob_matches("08/29/1954", "")


class TestMemberIdMatch:
    def test_exact_value_matches_case_insensitively(self):
        assert member_id_matches("a9000603900", "A9000603900")

    def test_substring_of_a_longer_token_does_not_match(self):
        assert not member_id_matches("XA90006039001", "A9000603900")

    def test_absent_or_na_expected_value(self):
        assert not member_id_matches("12345", "")
        assert not member_id_matches("12345", "N/A")


# --- page verdicts ----------------------------------------------------------


class TestVerifyPage:
    def test_name_and_dob_verifies(self):
        exp = expected_from_manifest(manifest())
        assert verify_page(exp, "2", "Justin Anderson", True, False) is True

    def test_name_alone_does_not_verify(self):
        exp = expected_from_manifest(manifest())
        assert verify_page(exp, "2", "Justin Anderson", False, False) is False

    def test_no_name_mode_never_verifies(self):
        exp = expected_from_manifest(manifest(first_name=None, last_name=None,
                                              member_name="Anderson"))
        assert verify_page(exp, "", "Justin Anderson", True, True) is False


class TestClassifyPage:
    def test_verified_page(self):
        exp = expected_from_manifest(manifest())
        assert classify_page([], exp, "2", True) == PAGE_VERIFIED

    def test_other_member_named_is_wrong_member(self):
        exp = expected_from_manifest(manifest())
        assert classify_page(["Maria Gonzalez"], exp, "2", False) == PAGE_WRONG_MEMBER

    def test_expected_member_named_but_unverified_is_not_wrong(self):
        exp = expected_from_manifest(manifest())
        assert classify_page(["Justin Anderson"], exp, "2", False) == PAGE_NOT_VERIFIED

    def test_nothing_detected_is_not_wrong_member(self):
        """A page with no names read off it is not evidence of another member."""
        exp = expected_from_manifest(manifest())
        assert classify_page([], exp, "2", False) == PAGE_NOT_VERIFIED


# --- document decision (what_if_rules) --------------------------------------


class TestRejectThreshold:
    @pytest.mark.parametrize(
        "pages,expected",
        [(0, 1), (1, 1), (3, 1), (10, 1), (12, 2), (40, 4), (50, 5), (60, 5), (500, 5)],
    )
    def test_min_of_five_or_ten_percent(self, pages, expected):
        assert reject_threshold(pages) == expected

    def test_matches_the_documented_formula(self):
        for pages in range(1, 200):
            assert reject_threshold(pages) == max(
                1, min(5, math.ceil(pages * 0.10))
            )


class TestDocumentDecision:
    def test_below_threshold_accepts(self):
        statuses = [PAGE_WRONG_MEMBER] + [PAGE_VERIFIED] * 11
        assert apply_what_if(statuses, 12) == ACCEPT

    def test_at_threshold_rejects(self):
        statuses = [PAGE_WRONG_MEMBER] * 2 + [PAGE_VERIFIED] * 10
        assert apply_what_if(statuses, 12) == REJECT

    def test_no_pages_rejects(self):
        assert apply_what_if([], 0) == REJECT

    def test_counts_only_wrong_member_pages(self):
        statuses = [PAGE_NOT_VERIFIED] * 20
        assert count_wrong_member(statuses) == 0
        assert apply_what_if(statuses, 20) == ACCEPT


# --- end to end over a record ----------------------------------------------


def staged(*, name=None, dob=None, ids=(), name_extra=()):
    """A page as the extraction stages it: only what it accepted and selected."""

    def row(value, key, *, ner=False):
        return {"key": key, "value": value, "accepted": True, "selected": True,
                "ner_text": value if ner else ""}

    fields = {
        "name": ([row(name, "Patient Name")] if name else []) + [row(n, "Name") for n in name_extra],
        "dob": [row(dob, "DOB", ner=True)] if dob else [],
        "member_id": [row(i, "Member ID") for i in ids],
    }
    return StagedPage("p.jpg", {"fields": fields})


class TestStagedPageFields:
    def test_the_matching_name_is_the_detected_one(self):
        exp = expected_from_manifest(manifest())
        page = staged(name="Maria Gonzalez", name_extra=["Justin Anderson"])
        fields, names, dob_ok, id_ok = staged_page_fields(page, exp, "2")
        assert fields["Detected_Full_Name"] == "Justin Anderson"
        assert names == ["Maria Gonzalez", "Justin Anderson"]
        assert (dob_ok, id_ok) == (False, False)

    def test_dob_and_id_present_only_when_they_are_the_manifests(self):
        exp = expected_from_manifest(manifest())
        fields, _names, dob_ok, id_ok = staged_page_fields(
            staged(name="Justin Anderson", dob="01/02/1970", ids=["999", "A9000603900"]), exp, "2"
        )
        assert dob_ok is False and fields["Detected_DOB"] == "01/02/1970"
        assert id_ok is True and fields["Detected_MemberID"] == "A9000603900"
        assert fields["Detection_Source_DOB"] == "ner" and fields["Detection_Source_Name"] == "rule based"
        assert fields["ner_key_source_DOB"] == "DOB"

    def test_dob_is_reported_as_mm_dd_yyyy(self):
        exp = expected_from_manifest(manifest())
        fields, *_ = staged_page_fields(staged(dob="August 29, 1954"), exp, "2")
        assert fields["Detected_DOB"] == "08/29/1954"

    def test_a_page_the_extraction_could_not_read_finds_nothing(self):
        exp = expected_from_manifest(manifest())
        fields, names, dob_ok, id_ok = staged_page_fields(None, exp, "2")
        assert fields["Detected_Full_Name"] == "N/A" and names == [] and not dob_ok and not id_ok


class TestVerifyRecord:
    def test_identifies_the_verified_page(self):
        exp = expected_from_manifest(manifest())
        pages = [
            {"page_no": 1, "page_name": "1.jpg",
             "staged": staged(name="Justin Anderson", dob="08/29/1954", ids=["A9000603900"])},
            {"page_no": 2, "page_name": "2.jpg", "staged": staged()},
        ]
        result = verify_record("rec1", pages, exp, detect_name_mode(exp))
        assert result.pages_verified == 1
        assert result.pages[0].page_status == PAGE_VERIFIED
        assert result.pages[0].detection_source_name == "rule based"
        assert result.pages[1].page_status == PAGE_NOT_VERIFIED
        assert result.document_decision == ACCEPT

    def test_other_members_pages_reject_the_document(self):
        exp = expected_from_manifest(manifest())
        pages = [{"page_no": 1, "page_name": "1.jpg", "staged": staged(name="Maria Gonzalez")}]
        result = verify_record("rec", pages, exp, "2")
        assert result.pages[0].page_status == PAGE_WRONG_MEMBER
        assert result.pages[0].ner_names == ["Maria Gonzalez"]
        assert result.document_decision == REJECT
        assert summary_status(result) == ("failed", "wrong_member_threshold")

    def test_threshold_uses_the_whole_document_not_the_subset(self):
        """Blank/junk pages are dropped before this stage, but the reject
        threshold is a proportion of the document, so total_pages must win."""
        exp = expected_from_manifest(manifest())
        pages = [{"page_no": 1, "page_name": "1.jpg", "staged": staged()}]
        subset = verify_record("rec", pages, exp, "2", total_pages=100)
        assert subset.reject_threshold == 5
        assert subset.total_pages == 100
        assert subset.pages_checked == 1

    def test_db_status_mapping(self):
        exp = expected_from_manifest(manifest())
        pages = [{"page_no": 1, "page_name": "1.jpg",
                  "staged": staged(name="Justin Anderson", dob="08/29/1954")}]
        result = verify_record("rec", pages, exp, "2")
        assert result.pages[0].db_page_status == "verified"
        assert result.db_document_decision == "accept"

    def test_no_pages_checked_is_skipped_not_failed(self):
        """All-blank/junk leaves nothing to verify. The what-if rule rejects
        an empty list; the chart triage must not store that as failed."""
        exp = expected_from_manifest(manifest())
        result = verify_record("rec", [], exp, "2", total_pages=1)
        assert result.document_decision == REJECT
        assert result.pages_checked == 0
        status, reason = summary_status(result)
        assert status == "skipped"
        assert reason == "all_blank_junk"

    def test_summary_status_reports_manifest_gap(self):
        exp = expected_from_manifest(manifest(first_name=None, last_name=None,
                                              member_name="Anderson"))
        result = verify_record("rec", [{"page_no": 1, "page_name": "1.jpg",
                                        "staged": None}], exp, detect_name_mode(exp))
        status, reason = summary_status(result)
        assert status == "needs_review"
        assert reason == "manifest_name_incomplete"


class TestNameMode:
    def test_three_when_middle_name_known(self):
        exp = expected_from_manifest(manifest(middle_name="Robert"))
        assert detect_name_mode(exp) == "3"

    def test_two_for_first_and_last(self):
        assert detect_name_mode(expected_from_manifest(manifest())) == "2"

    def test_empty_when_unusable(self):
        exp = expected_from_manifest(
            manifest(first_name=None, last_name=None, member_name="Anderson")
        )
        assert detect_name_mode(exp) == ""

    def test_falls_back_to_splitting_a_joined_name(self):
        exp = expected_from_manifest(
            manifest(first_name=None, last_name=None, middle_name=None,
                     member_name="Justin Robert Anderson")
        )
        assert exp["DummyFirstName"] == "Justin"
        assert exp["DummyMiddleName"] == "Robert"
        assert exp["DummyLastName"] == "Anderson"
        assert detect_name_mode(exp) == "3"


class TestTrimExtractedName:
    def test_extra_words_around_the_manifest_name_are_dropped(self):
        from stages.lib.member.engine import trim_extracted_name

        expected = {
            "DummyFirstName": "Abhinav",
            "DummyMiddleName": "",
            "DummyLastName": "Dasgupta",
        }
        assert (
            trim_extracted_name("Abhinav X Dasgupta alias of", expected)
            == "Abhinav X Dasgupta"
        )
        assert trim_extracted_name("alias of Abhinav Dasgupta", expected) == "Abhinav Dasgupta"
        assert trim_extracted_name("Abhinav Dasgupta", expected) == "Abhinav Dasgupta"
        assert trim_extracted_name("Mary Smith", expected) == "Mary Smith"
        reversed_name = {
            "DummyFirstName": "Anderson",
            "DummyMiddleName": "",
            "DummyLastName": "Lisa",
        }
        assert trim_extracted_name("Lisa X Anderson", reversed_name) == "Lisa X Anderson"
        assert (
            trim_extracted_name("alias of Lisa X Anderson", reversed_name)
            == "Lisa X Anderson"
        )

    def test_a_manifest_middle_name_is_kept_when_it_was_extracted(self):
        from stages.lib.member.engine import trim_extracted_name

        expected = {
            "DummyFirstName": "Abhinav",
            "DummyMiddleName": "Kumar",
            "DummyLastName": "Dasgupta",
        }
        assert (
            trim_extracted_name("Abhinav Kumar Dasgupta alias", expected)
            == "Abhinav Kumar Dasgupta"
        )
        assert trim_extracted_name("Abhinav Dasgupta", expected) == "Abhinav Dasgupta"
