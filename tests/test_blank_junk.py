"""Blank/junk duplicate scoping.

Duplicate detection runs after the model, among Main pages only: ±2 neighbor
similarity (≥98%) or one page's text wholly contained in the other's.
Blank / junk / short pages are excluded from comparison; on a match the higher
character-count page stays the original (earlier page on a tie).
UI: similarity 100% → Yes; [98%, 100%) → May Be; else No.
"""
from __future__ import annotations

from stages.lib.blank_junk.stage import SUBTYPE_DB, _classify, _to_db_flag


def page(page_id: int, number: int):
    return {"id": page_id, "page_name": f"{number}.jpg", "page_number": number}


def _body(n: int = 6) -> str:
    return "Office visit note for the patient. Assessment and plan follow. " * n


class TestDuplicateScope:
    def test_duplicate_within_one_pass_is_found(self):
        pages = [page(1, 1), page(2, 2)]
        body = _body()
        texts = {1: body, 2: body}
        rows = _classify(pages, texts, {1, 2})
        flags = {r["page_id"]: r["flag"] for r in rows}
        assert flags[1] == "not_blank_junk"
        assert flags[2] == "duplicate"
        assert next(r for r in rows if r["page_id"] == 2)["duplicate_of"] == 1

    def test_duplicate_of_a_page_judged_in_an_earlier_pass_is_found(self):
        """Pass 2 can still match a neighbor that was main in pass 1."""
        body = _body()
        pages = [page(1, 1), page(2, 2)]
        rows1 = _classify(pages, {1: body, 2: "x"}, {1})
        assert rows1[0]["flag"] == "not_blank_junk"
        rows2 = _classify(
            pages, {1: body, 2: body}, {2}, prior_main_ids={1}
        )
        assert rows2[0]["flag"] == "duplicate"
        assert rows2[0]["duplicate_of"] == 1

    def test_a_page_is_never_a_duplicate_of_itself(self):
        body = _body()
        again = _classify([page(1, 1)], {1: body}, {1})
        assert again[0]["flag"] == "not_blank_junk"

    def test_equal_length_keeps_earlier_as_original(self):
        pages = [page(1, 1), page(2, 2), page(3, 3)]
        body = _body()
        texts = {1: body, 2: body, 3: body}
        rows = _classify(pages, texts, {1, 2, 3})
        by_id = {r["page_id"]: r for r in rows}
        assert by_id[1]["flag"] == "not_blank_junk"
        assert by_id[2]["duplicate_of"] == 1
        # page 3 is within ±2 of page 1 and page 2
        assert by_id[3]["flag"] == "duplicate"
        assert by_id[3]["duplicate_of"] == 1

    def test_longer_later_page_wins_as_original(self):
        pages = [page(1, 1), page(2, 2)]
        long = _body(8)
        # Tiny truncation keeps SequenceMatcher ratio well above 98%.
        short = long[:-3]
        rows = _classify(pages, {1: short, 2: long}, {1, 2})
        by_id = {r["page_id"]: r for r in rows}
        assert by_id[2]["flag"] == "not_blank_junk"
        assert by_id[1]["flag"] == "duplicate"
        assert by_id[1]["duplicate_of"] == 2

    def test_pages_more_than_two_apart_are_not_compared(self):
        pages = [page(1, 1), page(2, 2), page(3, 3), page(4, 4)]
        body = _body()
        # page 1 and page 4 are identical but 3 steps apart (> ±2)
        texts = {
            1: body,
            2: "Completely different progress note content here. " * 6,
            3: "Another unrelated assessment and plan page text. " * 6,
            4: body,
        }
        rows = _classify(pages, texts, {1, 2, 3, 4})
        flags = {r["page_id"]: r["flag"] for r in rows}
        assert flags[1] == "not_blank_junk"
        assert flags[4] == "not_blank_junk"

    def test_similarity_below_threshold_is_not_duplicate(self):
        pages = [page(1, 1), page(2, 2)]
        a = ("Alpha clinical history. " * 20)
        b = ("Beta surgical findings. " * 20)
        rows = _classify(pages, {1: a, 2: b}, {1, 2})
        # Blank/junk is the model's call; this test is only about duplicates.
        for row in rows:
            assert row["flag"] != "duplicate"
            assert row["duplicate_of"] is None

    def test_page_contained_in_a_neighbor_is_duplicate(self):
        pages = [page(1, 1), page(2, 2)]
        inner = _body(6)
        outer = inner + "Addendum with medication reconciliation and labs. " * 12
        rows = _classify(pages, {1: inner, 2: outer}, {1, 2})
        by_id = {r["page_id"]: r for r in rows}
        assert by_id[2]["flag"] == "not_blank_junk"
        assert by_id[1]["flag"] == "duplicate"
        assert by_id[1]["duplicate_of"] == 2
        assert by_id[1]["confidence"] == 1.0
        assert "contained" in by_id[1]["reason"]

    def test_short_identical_pages_are_not_compared(self):
        pages = [page(1, 1), page(2, 2)]
        short = "Office visit note. Assessment and plan follow. " * 2
        rows = _classify(pages, {1: short, 2: short}, {1, 2})
        for row in rows:
            assert row["flag"] != "duplicate"

    def test_page_not_main_in_an_earlier_pass_is_not_compared(self):
        """A prior-pass blank/junk page is never a duplicate original."""
        body = _body()
        pages = [page(1, 1), page(2, 2)]
        rows = _classify(pages, {1: body, 2: body}, {2}, prior_main_ids=set())
        assert rows[0]["flag"] != "duplicate"
        assert rows[0]["duplicate_of"] is None

    def test_blank_and_short_pages_are_not_compared(self):
        pages = [page(1, 1), page(2, 2)]
        body = _body()
        rows = _classify(pages, {1: body, 2: "   "}, {1, 2})
        by_id = {r["page_id"]: r for r in rows}
        assert by_id[1]["flag"] == "not_blank_junk"
        assert by_id[2]["flag"] == "blank"

    def test_only_requested_pages_are_classified(self):
        pages = [page(1, 1), page(2, 2)]
        texts = {1: "alpha content here " * 10, 2: "beta content here " * 10}
        rows = _classify(pages, texts, {2})
        assert [r["page_id"] for r in rows] == [2]

    def test_blank_page_is_flagged_blank(self):
        rows = _classify([page(1, 1)], {1: "   "}, {1})
        assert rows[0]["flag"] == "blank"


class TestClinicalNotJunk:
    """Clinical progress notes must not become Letter/Fax or Invoice junk."""

    def test_progress_note_with_confidentiality_footer_is_main(self):
        import sys
        from pathlib import Path

        junk = Path("core-pipeline/stages/lib/blank_junk").resolve()
        if str(junk) not in sys.path:
            sys.path.insert(0, str(junk))
        from classify import CLASSIFICATION_LABELS, CODE_MAIN, classify_text

        text = (
            "Progress Note\n"
            "Reason for Appointment: 6 month follow up.\n"
            "History of Present Illness: Patient presents with SOB.\n"
            "Current Medications: Amlodipine, Apixaban.\n"
            "Vital Signs: BP 132/64. Assessment: Hypertension.\n"
            "CONFIDENTIALITY NOTICE: This transmission is intended only "
            "for the intended recipient and may contain confidential "
            "medical records.\n"
        )
        code, reason = classify_text(text)
        assert code == CODE_MAIN
        assert reason == "clinical_content"
        assert CLASSIFICATION_LABELS[code] == "Main"

    def test_short_fax_cover_still_junk(self):
        import sys
        from pathlib import Path

        junk = Path("core-pipeline/stages/lib/blank_junk").resolve()
        if str(junk) not in sys.path:
            sys.path.insert(0, str(junk))
        from classify import CLASSIFICATION_LABELS, classify_text

        text = "Fax cover sheet\nThis fax is for the intended recipient only.\n"
        code, reason = classify_text(text)
        assert CLASSIFICATION_LABELS[code] == "Letter/Fax"
        assert reason == "letter_fax"

    def test_fax_footer_alone_is_letter_fax_not_record_request(self):
        """Bare confidentiality footers must not become Record Request junk."""
        import sys
        from pathlib import Path

        junk = Path("core-pipeline/stages/lib/blank_junk").resolve()
        if str(junk) not in sys.path:
            sys.path.insert(0, str(junk))
        from classify import CLASSIFICATION_LABELS, classify_text

        text = (
            "CONFIDENTIALITY NOTICE: This fax transmission is intended "
            "only for the intended recipient. If you are not the intended "
            "recipient please destroy this transmission.\n"
        )
        code, reason = classify_text(text)
        assert CLASSIFICATION_LABELS[code] == "Letter/Fax"
        assert reason == "letter_fax"


class TestModelBridge:
    """The blank/junk model decides the flag and the junk subtype."""

    @staticmethod
    def _bridge():
        import sys

        from conftest import LIB

        junk = str(LIB / "blank_junk")
        if junk not in sys.path:
            sys.path.insert(0, junk)
        import model_bridge

        return model_bridge

    def test_model_dir_comes_from_config(self, monkeypatch, tmp_path):
        import config

        bridge = self._bridge()
        monkeypatch.setattr(config, "BLANK_JUNK_MODEL_DIR", tmp_path)
        status = bridge.model_status()
        assert status["path"] == str(tmp_path / "tfidf_flat.joblib")
        assert status["ready"] is False
        assert "model file missing" in status["reason"]

    def test_model_keep_is_not_rewritten_by_a_blank_phrase(self):
        from classify import CODE_BLANK, CODE_MAIN

        code, reason, conf = self._bridge().classify_page(
            "This page intentionally left blank"
        )
        assert code in {CODE_BLANK, CODE_MAIN}
        assert reason.startswith("model:")
        assert "regex_fallback" not in reason
        assert conf is not None

    def test_clinical_note_is_main(self):
        from classify import CODE_MAIN

        text = (
            "Progress Note\n"
            "History of Present Illness: Patient presents with SOB.\n"
            "Assessment: Hypertension.\n"
            "Current Medications: Amlodipine.\n"
        )
        code, reason, conf = self._bridge().classify_page(text)
        assert code == CODE_MAIN
        assert reason.startswith("model:")
        assert conf is not None and conf >= 0.7

    def test_model_junk_subtype_comes_from_the_model_label(self):
        from classify import JUNK_CODES, CODE_BLANK

        text = (
            "MEDICAL RECORDS REQUEST\n"
            "Request for medical records\n"
            "Please send the following records for the patient listed.\n"
            "Records retrieval vendor: Copy service\n"
            "Fulfillment due within 10 business days"
        )
        code, reason, conf = self._bridge().classify_page(text)
        assert reason.startswith("model:")
        if code in JUNK_CODES - {CODE_BLANK}:
            assert "subtype:model:" in reason
        assert "subtype:regex:" not in reason
        assert "regex_fallback" not in reason
        assert conf is not None

    def test_junk_subtype_is_the_model_label(self):
        from classify import CODE_LETTER_FAX, CODE_OTHERS

        bridge = self._bridge()
        code, why = bridge._junk_subtype("JUNK_FAX_TRANSMISSION")
        assert (code, why) == (CODE_LETTER_FAX, "subtype:model:JUNK_FAX_TRANSMISSION")
        code, why = bridge._junk_subtype("JUNK")
        assert (code, why) == (CODE_OTHERS, "subtype:model:JUNK")

    def test_missing_model_is_stamped_as_a_fallback(self, monkeypatch):
        bridge = self._bridge()
        monkeypatch.setattr(bridge, "_load_service", lambda: None)
        _code, reason, conf = bridge.classify_page("Invoice Number 123 Amount Due")
        assert reason.startswith("regex_fallback:model_unavailable")
        assert conf is None

    def test_vendored_src_package_does_not_stay_on_sys_path(self):
        import sys

        bridge = self._bridge()
        assert bridge._load_service() is not None
        assert str(bridge._VENDOR) not in sys.path

    def test_health_reports_the_model_without_loading_it(self):
        status = self._bridge().model_status()
        assert status["ready"] is True
        assert status["model_version"]
        assert status["reason"] is None


class TestSubtypeMapping:
    def test_junk_always_gets_a_legal_subtype(self):
        """JUNK_CODES includes CODE_BLANK; blank is its own flag and takes
        precedence, so only the remaining codes become flag='junk'."""
        from classify import CODE_BLANK, JUNK_CODES

        for code in JUNK_CODES - {CODE_BLANK}:
            flag, subtype = _to_db_flag(code)
            assert flag == "junk"
            assert subtype in set(SUBTYPE_DB.values())

    def test_blank_wins_over_junk_even_though_it_is_in_junk_codes(self):
        from classify import CODE_BLANK, JUNK_CODES

        assert CODE_BLANK in JUNK_CODES
        flag, subtype = _to_db_flag(CODE_BLANK)
        assert flag == "blank"
        assert subtype is None

    def test_non_junk_has_no_subtype(self):
        from classify import CODE_BLANK, CODE_DUPLICATE, CODE_MAIN

        for code in (CODE_BLANK, CODE_DUPLICATE, CODE_MAIN):
            _flag, subtype = _to_db_flag(code)
            assert subtype is None
