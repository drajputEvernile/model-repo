"""Key/value extraction plumbing: OCR input, staging, headings, sequencing markers, chain order.

None of these load a model — the model run is covered by scripts/simulate_extraction.py.
"""
from __future__ import annotations

import pytest

from stages.lib.extraction import ocr_input, staging
from stages.lib.extraction.stage import section_headers_of
from stages.lib.extraction.staging import Staged, StagedPage


def word(text, x, y=10):
    return {"content": text, "polygon": [x, y, x + 20, y, x + 20, y + 10, x, y + 10]}


class TestOcrInput:
    def test_azure_words_under_pages_meta_are_read(self, tmp_path):
        page = {"pagesMeta": [{"width": 800, "height": 1000, "words": [word("Name", 5)]}]}
        out = ocr_input.extraction_page(page, page_name="1.jpg", page_number=1, image_path=tmp_path / "1.jpg")
        assert out["width"] == 800 and out["words"][0]["content"] == "Name"
        assert out["fileName"] == "1.jpg" and out["imagePath"].endswith("1.jpg")

    def test_words_at_the_top_of_the_page_are_read(self, tmp_path):
        page = {"width": 800, "height": 1000, "words": [word("DOB", 5)]}
        assert ocr_input.has_word_boxes(page)

    def test_a_page_with_text_only_has_no_boxes(self, tmp_path):
        final1 = {"content": "Patient: Jane", "markdown": "Patient: Jane", "engine": "docling+rapidocr"}
        assert ocr_input.extraction_page(final1, page_name="1.jpg", page_number=1, image_path=tmp_path) is None
        assert ocr_input.extraction_page(None, page_name="1.jpg", page_number=1, image_path=tmp_path) is None

    def test_words_without_a_polygon_do_not_count(self):
        assert not ocr_input.has_word_boxes({"words": [{"content": "x", "polygon": []}]})


class FakeBox:
    pass


class TestStagingFile:
    @pytest.fixture(autouse=True)
    def workspace(self, tmp_path, monkeypatch):
        monkeypatch.setattr(staging, "staging_dir", lambda chart: tmp_path / chart / "staging")

    def payload(self):
        return {
            "version": staging.SCHEMA_VERSION, "chart_name": "c", "model_version": "v002",
            "pages": {"1.jpg": {"page_number": 1, "width": 800, "height": 1000, "fields": {
                "name": [{"value": "A B", "accepted": True, "selected": True},
                         {"value": "C D", "accepted": True, "selected": False},
                         {"value": "E F", "accepted": False, "selected": False}],
                "heading_heron": [{"text": "Plan", "accepted": True}, {"text": "x", "accepted": False}],
            }}},
        }

    def test_round_trip_and_drop(self):
        assert staging.read("c") is None
        staging.write("c", self.payload())
        staged = staging.read("c")
        assert isinstance(staged, Staged) and "1.jpg" in staged and staged.model_version == "v002"
        assert staging.drop("c") is True
        assert staging.read("c") is None and staging.drop("c") is False

    def test_accepted_and_selected(self):
        staging.write("c", self.payload())
        page = staging.read("c").page("1.jpg")
        assert [r["value"] for r in page.accepted("name")] == ["A B", "C D"]
        assert [r["value"] for r in page.selected("name")] == ["A B"]
        assert [r["text"] for r in page.selected("heading_heron")] == ["Plan"]  # a heading has no separate pick

    def test_another_version_is_ignored(self):
        data = self.payload()
        data["version"] = 99
        staging.write("c", data)
        assert staging.read("c") is None

    def test_unreadable_file_is_treated_as_missing(self, tmp_path):
        path = staging.staging_path("c")
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        assert staging.read("c") is None


class TestHitRecord:
    def test_boxes_are_lists_and_the_key_hit_is_left_out(self):
        from stages.lib.extraction.dos.extract import DosHit
        from stages.lib.extraction.util.geometry import Box
        from stages.lib.extraction.page_no.extract import PageHit

        hit = DosHit(key="DOS", region="mid", sentence="s", tier="encounter", value="v", score=0.123456, key_hit=object())
        record = staging.hit_record(hit)
        assert "key_hit" not in record and record["score"] == 0.1235
        page = staging.hit_record(PageHit(key="k", region="footer", sentence="s", box=Box(1, 2, 3, 4)))
        assert page["box"] == [1, 2, 3, 4]


class TestHeadings:
    def test_section_headers_have_the_shape_the_ocr_json_and_ui_read(self):
        page = StagedPage("1.jpg", {"width": 1000, "height": 2000, "fields": {"heading_heron": [
            {"text": "Plan", "level": "Heading", "box": [100, 200, 300, 240], "accepted": True},
            {"text": "Sub", "level": "Subheading", "box": [0, 0, 10, 10], "accepted": True},
            {"text": "No", "level": "Heading", "box": [0, 0, 10, 10], "accepted": False},
        ]}})
        first, second = section_headers_of(page)
        assert first["text"] == "Plan" and first["level"] == 1 and second["level"] == 2
        assert first["bbox"] == [100, 200, 300, 240]
        assert first["page_width"] == 1000 and first["coord_origin"] == "TOPLEFT"
        assert first["norm"] == {"left": 0.1, "top": 0.1, "width": 0.2, "height": 0.02}


BODY = "Office visit note for the patient. Assessment and plan follow. " * 8


class TestSequencingMarker:
    def marker(self, **row):
        from stages.lib.sequencing.stage import _staged_marker

        base = {"accepted": True, "selected": True}
        return _staged_marker(StagedPage("1.jpg", {"fields": {"page_no": [{**base, **row}]}}))

    def test_page_x_of_y(self):
        m = self.marker(page_no="2", page_total="45")
        assert (m.page_num, m.total_pages, m.pattern) == (2, 45, "page_x_of_y")

    def test_page_x_alone(self):
        m = self.marker(page_no="3", page_total="")
        assert (m.page_num, m.total_pages, m.pattern) == (3, None, "page_x")

    def test_impossible_numbers_are_not_markers(self):
        assert self.marker(page_no="9", page_total="3") is None
        assert self.marker(page_no="0", page_total="3") is None

    def test_unread_page_has_none(self):
        from stages.lib.sequencing.stage import _staged_marker

        assert _staged_marker(None) is None

    def test_extracted_markers_replace_the_text_patterns(self):
        from stages.lib.sequencing import compute_sequence_assignments
        from stages.lib.sequencing.engine.types import ExplicitMarker

        def page(pid, text, marker):
            return {"page_id": pid, "page_number": pid, "text": text,
                    "marker_extracted": True, "page_marker": marker}

        pages = [
            page(1, BODY + "Page 2 of 2", None),  # the text says page 2; the extraction found none
            page(2, BODY, ExplicitMarker(1, 2, "page_x_of_y", 1.0)),
        ]
        out = {a["page_id"]: a for a in compute_sequence_assignments(pages, cross_encoder_enabled=False)}
        assert out["2"]["marker_value"] == 1
        assert out["1"]["explicit_marker_found"] is False


class TestChain:
    def test_extraction_replaces_the_section_header_stage_and_runs_before_member(self):
        from orchestrator.runner import STAGE_NAMES

        assert "section_headers:1" not in STAGE_NAMES
        order = STAGE_NAMES
        assert order.index("ocr_final2:1") < order.index("kv_extract:1") < order.index("member_verify:1")
        assert order.index("kv_extract:1") < order.index("dos_extract:1") < order.index("page_sequencing:1")

    def test_memory_store_seed_matches(self):
        from db.memory_store import DEFAULT_PIPELINE_STAGES
        from orchestrator.runner import STAGE_CHAIN

        seeded = [(s["stage_name"], s["pass_no"]) for s in sorted(DEFAULT_PIPELINE_STAGES, key=lambda s: s["seq"])]
        assert seeded == [(name, no) for name, no, _ in STAGE_CHAIN]


class TestReadiness:
    def test_a_missing_weights_folder_is_named(self, monkeypatch, tmp_path):
        from stages.lib.extraction import engine
        from stages.lib.extraction.util import config

        gliner = tmp_path / "gliner_low"
        source = config.ModelSource("org/model", "rev", ("pytorch_model.bin",))
        monkeypatch.setattr(config, "Ner_Model_Path", gliner)
        monkeypatch.setattr(config, "Model_Sources", {gliner: [source]})
        monkeypatch.setattr(engine, "REQUIRED_PACKAGES", ())
        status = engine.readiness()
        assert status["ready"] is False and "gliner_low" in status["reason"]
        with pytest.raises(engine.ExtractionNotReady, match="gliner_low"):
            engine.require_ready()


class TestPinnedVersion:
    def test_the_pipeline_runs_v002_and_the_environment_cannot_change_it(self, monkeypatch):
        import importlib

        import config

        monkeypatch.setenv("EXTRACTION_MODEL_VERSION", "v001")
        importlib.reload(config)
        try:
            assert config.EXTRACTION_MODEL_VERSION == "v002"
        finally:
            monkeypatch.delenv("EXTRACTION_MODEL_VERSION")
            importlib.reload(config)

    def test_engine_reports_v002_and_it_is_the_trained_version_on_disk(self):
        import json

        from stages.lib.extraction import engine
        from stages.lib.extraction.util import config as kv_config

        assert engine.model_version() == "v002"
        manifest = json.loads((kv_config.Model_Registry / "v002" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["version"] == "v002"
        assert [p.name for p in kv_config.Model_Registry.iterdir()] == ["v002"]  # no other version ships
