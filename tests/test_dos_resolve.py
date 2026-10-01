"""Date of service: what the extraction staged, and how the chart is resolved from it."""
from __future__ import annotations

import pytest

from stages.lib.dos import resolve
from stages.lib.extraction.staging import Staged, StagedPage


def staged_page(*rows, name="1.jpg") -> StagedPage:
    return StagedPage(name, {"fields": {"dos": list(rows)}})


def dos_row(dos_from, dos_to=None, *, tier="encounter", key="Date of Service", score=0.9, **extra):
    return {
        "key": key, "tier": tier, "dos_from": dos_from, "dos_to": dos_to or dos_from,
        "score": score, "accepted": True, "selected": True, **extra,
    }


class TestPageDates:
    def test_single_date_is_iso(self):
        (found,) = resolve.page_dates(staged_page(dos_row("06/12/2024")))
        assert (found.dos_from, found.dos_to, found.is_pair) == ("2024-06-12", "2024-06-12", False)
        assert found.keyword == "Date of Service" and found.confidence == 0.9

    def test_month_name_dates_are_read(self):
        (found,) = resolve.page_dates(staged_page(dos_row("August 8, 2024")))
        assert found.dos_from == "2024-08-08"

    def test_admit_and_discharge_are_one_range(self):
        found = resolve.page_dates(staged_page(
            dos_row("05/01/2024", tier="admit", key="Admit Date", score=0.8),
            dos_row("05/04/2024", tier="discharge", key="Discharge Date", score=0.85),
        ))
        assert len(found) == 1
        assert (found[0].dos_from, found[0].dos_to, found[0].is_pair) == ("2024-05-01", "2024-05-04", True)
        assert found[0].keyword == "Admit Date+Discharge Date" and found[0].confidence == 0.85

    def test_rows_that_are_not_selected_are_ignored(self):
        assert resolve.page_dates(staged_page(dos_row("06/12/2024", selected=False))) == []

    def test_an_impossible_date_is_dropped(self):
        assert resolve.page_dates(staged_page(dos_row("02/30/2024"))) == []

    def test_unread_page_has_no_dates(self):
        assert resolve.page_dates(None) == []


def pd(iso, confidence=0.9):
    return resolve.PageDate(dos_from=iso, dos_to=iso, confidence=confidence, keyword="k", source="rules")


def chart(*pages):
    """pages: (text, [PageDate...])"""
    return resolve.resolve_chart([
        {"page_name": f"{i}.jpg", "page": i, "page_text": text, "dates": dates}
        for i, (text, dates) in enumerate(pages, 1)
    ])


@pytest.fixture
def page_types(monkeypatch):
    def install(mapping):
        monkeypatch.setattr(
            resolve, "page_type_name", lambda text, _n: next((v for k, v in mapping.items() if k in text), "")
        )
    return install


class TestResolve:
    def test_progress_note_span_beats_a_weak_date(self, page_types):
        page_types({"NOTE": "Progress Note"})
        rows = chart(
            ("NOTE", [pd("2024-03-14")]),
            ("body", [pd("2024-03-12", 0.6)]),
            ("body", [pd("2024-04-02", 0.9)]),
            ("body", []),
        )
        assert rows[1]["dos_from_iso"] == "2024-03-14" and rows[1]["match_type"] == "span"
        assert rows[1]["dates"][0]["dos_from"] == "2024-03-14"
        assert rows[2]["dos_from_iso"] == "2024-04-02"  # above the override score: keeps its own
        assert rows[3]["dos_from_iso"] == "2024-03-14"  # the span is still open

    def test_a_page_with_no_date_keeps_the_default(self, page_types):
        page_types({})
        (row,) = chart(("body", []))
        assert row["dos_from"] == "" and row["is_default"] is True
        assert row["doc_dos_from_iso"] == resolve.profile().default_date
        assert row["match_type"] == "no_date_found" and row["dates"] == []

    def test_demographics_and_injection_pages_keep_the_default(self, page_types):
        page_types({"DEMO": "Demographics", "INJ": "Injection Visit"})
        rows = chart(
            ("body", [pd("2024-03-14")]),
            ("DEMO", [pd("2024-05-01")]),
            ("INJ", [pd("2024-06-01")]),
        )
        default = resolve.profile().default_date
        for row in rows[1:]:
            assert row["dos_from"] == "" and row["doc_dos_from_iso"] == default and row["is_default"]
            assert row["match_type"] == "default_page"

    def test_non_encounter_page_never_replaces_an_encounter(self, page_types):
        page_types({"MEDS": "Medication List"})
        rows = chart(("body", [pd("2024-03-14")]), ("MEDS", [pd("2024-05-01")]))
        assert rows[1]["dos_from_iso"] == "2024-05-01"
        assert rows[1]["doc_dos_from_iso"] == "2024-03-14"
        assert rows[1]["match_type"] == "non_encounter_page"

    def test_columns_are_iso_and_mdy_twins(self, page_types):
        page_types({})
        (row,) = chart(("body", [pd("2024-03-14")]))
        assert row["dos_from"] == "03-14-2024" and row["dos_from_iso"] == "2024-03-14"


class TestStageWritesEachPageOnce:
    def test_found_dates_are_not_overwritten_and_pages_count_once(self, monkeypatch, tmp_path):
        from contextlib import contextmanager, nullcontext
        from types import SimpleNamespace

        import stages.lib.dos.stage as dos_stage

        pages = [{"id": i, "page_name": f"{i}.jpg", "page_number": i} for i in (1, 2, 3)]
        ctx = SimpleNamespace(chart_name="c", pages=pages, todo={1, 2, 3}, done=0, skipped=0)
        written: dict[int, list] = {}
        completed: list[int] = []

        @contextmanager
        def fake_stage_run(*_a, **_k):
            yield ctx

        def fake_upsert(_conn, *, page_id, date_of_service_from, extraction_method, **_k):
            written.setdefault(page_id, []).append((date_of_service_from, extraction_method))

        def fake_mark_completed(_conn, _ctx, page_id):
            completed.append(page_id)
            _ctx.done += 1

        staged = Staged({
            "version": 1, "model_version": "v002",
            "pages": {
                "1.jpg": {"fields": {"dos": [dos_row("03/14/2024")]}},
                "2.jpg": {"fields": {"dos": [dos_row("04/01/2024")]}},
            },
        })
        monkeypatch.setattr(dos_stage, "stage_run", fake_stage_run)
        monkeypatch.setattr(dos_stage, "connect", lambda: nullcontext(None))
        monkeypatch.setattr(dos_stage, "get_blank_junk_flags", lambda *a, **k: {})
        monkeypatch.setattr(dos_stage, "mark_skipped", lambda *a, **k: None)
        monkeypatch.setattr(dos_stage, "_page_texts", lambda *a, **k: {})
        monkeypatch.setattr(dos_stage, "ensure_staging", lambda *a, **k: staged)
        monkeypatch.setattr(dos_stage, "upsert_dos", fake_upsert)
        monkeypatch.setattr(dos_stage, "mark_completed", fake_mark_completed)
        monkeypatch.setattr(dos_stage, "imaging_csv", lambda *a: tmp_path / "dos.csv")
        monkeypatch.setattr(dos_stage, "write_csv", lambda path, *a: path)

        dos_stage.run(7)

        assert written == {
            1: [("2024-03-14", "rules")], 2: [("2024-04-01", "rules")], 3: [(None, "rules")],
        }
        assert sorted(completed) == [1, 2, 3]
        assert ctx.done == 3
