"""Contracts between core-pipeline and review-ui.

The pipeline writes CSVs; the review UI reads them. Nothing type-checks that
seam at runtime, so it is pinned here: every column the UI looks for must be a
column the pipeline actually writes, and the values must be in the vocabulary
the UI knows how to render.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- blank/junk subtype vocabulary ------------------------------------------

# The label set the schema CHECK constrains junk_subtype to.
SCHEMA_SUBTYPES = {
    "Invoice",
    "Cover Page",
    "Record Request/Transmittal",
    "Instructions",
    "Letter/Fax",
    "Others",
}



# --- shared source-file walker ----------------------------------------------


# Directories that contain code we did not write. A virtualenv is the dangerous
# one: docs/API.md tells you to create core-pipeline/.venv, so on any machine
# that followed the setup, rglob("*.py") walks into site-packages and tries to
# parse third-party sources — several of which carry a UTF-8 BOM and are not
# valid input to ast.parse without it being stripped.
_SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", "site-packages", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".tox", ".eggs",
}


def _is_venv(directory):
    """A virtualenv is identified by pyvenv.cfg, whatever it is named."""
    return (directory / "pyvenv.cfg").is_file()


def repo_python_files(*roots):
    """Every .py file we actually own under `roots`.

    Walks manually rather than using rglob so an excluded directory is pruned
    instead of merely filtered — descending into site-packages is slow even
    when every result is discarded.
    """
    for root in roots:
        stack = [Path(root)]
        while stack:
            directory = stack.pop()
            try:
                entries = list(directory.iterdir())
            except (OSError, PermissionError):
                continue
            if any(e.name == "pyvenv.cfg" for e in entries):
                continue
            for entry in entries:
                if entry.is_dir():
                    if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                        continue
                    stack.append(entry)
                elif entry.suffix == ".py" and not entry.name.startswith("._"):
                    yield entry


class TestJunkSubtypeVocabulary:
    def test_schema_check_matches_the_ui_label_set(self):
        """The UI renders a fixed set of page types; the schema must allow
        exactly those and nothing else, or rows become unrenderable."""
        from app.services.imaging_overlays import index_junk_rows  # noqa: F401

        schema_sql = (REPO_ROOT / "schema" / "v1.sql").read_text(encoding="utf-8")
        for label in SCHEMA_SUBTYPES:
            assert f"'{label}'" in schema_sql, f"{label} missing from schema CHECK"

    def test_stage_maps_every_classifier_label_into_the_vocabulary(self):
        from stages.lib.blank_junk.stage import SUBTYPE_DB
        from classify import CLASSIFICATION_LABELS, JUNK_CODES

        for code in JUNK_CODES:
            label = CLASSIFICATION_LABELS.get(code, "")
            mapped = SUBTYPE_DB.get(label.strip().casefold(), "Others")
            assert mapped in SCHEMA_SUBTYPES, (
                f"classifier label {label!r} maps to {mapped!r}, "
                "which the schema CHECK rejects"
            )

    def test_unknown_labels_fall_back_to_others_not_null(self):
        from stages.lib.blank_junk.stage import SUBTYPE_DB

        assert SUBTYPE_DB.get("something the classifier invents", "Others") == "Others"


# --- CSV column contracts ---------------------------------------------------


class TestCsvColumns:
    def test_dos_csv_carries_what_the_overlay_reads(self):
        from stages.lib.dos.stage import DOS_COLS

        # imaging_overlays.dos_row_fields reads these first, then falls back.
        for column in ("dos_from", "dos_to", "dos_from_iso", "dos_to_iso",
                       "doc_dos_from", "doc_dos_to",
                       "doc_dos_from_iso", "doc_dos_to_iso"):
            assert column in DOS_COLS
        assert "chart_name" in DOS_COLS and "page_number" in DOS_COLS

    def test_hw_csv_column_is_one_the_overlay_recognises(self):
        from stages.lib.image_preprocess.stage import HW_COLS

        # index_hw_rows accepts handwritten / handwritten_or_printed / type.
        assert "handwritten_or_printed" in HW_COLS
        assert "confidence" in HW_COLS

    def test_junk_csv_carries_classification_and_group(self):
        from stages.lib.blank_junk.stage import JUNK_CSV_COLS

        # index_junk_rows reads page_classification (or classification/page_type).
        assert "page_classification" in JUNK_CSV_COLS
        assert "page_group" in JUNK_CSV_COLS
        # v7 additions that make the two passes legible in the file itself.
        assert "pass_no" in JUNK_CSV_COLS
        assert "is_final" in JUNK_CSV_COLS

    def test_member_csv_carries_the_reference_provenance_columns(self):
        from stages.lib.member.stage import MEMBER_EXTRACT_COLS

        for column in ("detection_source_name", "detection_source_dob",
                       "detection_source_member_id", "ner_key_source_name",
                       "page_status", "page_verified"):
            assert column in MEMBER_EXTRACT_COLS

    def test_member_summary_carries_the_reject_decision(self):
        from stages.lib.member.stage import MEMBER_SUMMARY_COLS

        for column in ("document_decision", "wrong_member_pages",
                       "reject_threshold", "ner_enabled"):
            assert column in MEMBER_SUMMARY_COLS

    def test_every_csv_writer_has_a_chart_name_column(self):
        """collect_rows filters combined packs on chart_name; a CSV without it
        would be attributed to every chart."""
        from stages.lib.blank_junk.stage import JUNK_CSV_COLS
        from stages.lib.dos.stage import DOS_COLS
        from stages.lib.member.stage import MEMBER_EXTRACT_COLS, MEMBER_SUMMARY_COLS
        from stages.lib.image_preprocess.stage import HW_COLS, ROTATION_COLS

        for cols in (JUNK_CSV_COLS, DOS_COLS, MEMBER_EXTRACT_COLS,
                     MEMBER_SUMMARY_COLS, HW_COLS, ROTATION_COLS):
            assert "chart_name" in cols


class TestPerChartCsvNamesAreRead:
    """The review UI matches per-chart files by suffix. A stage writing a
    suffix the UI does not branch on produces a file nothing reads — which is
    exactly how the junk stream went missing."""

    STAGE_SUFFIXES = {
        "_rotation.csv",
        "_hw_printed.csv",
        "_quality.csv",
        "_junk.csv",
        "_member_extraction.csv",
        "_member_verification.csv",
        "_dos.csv",
        "_codeable.csv",
        "_encounter.csv",
        "_sequencing.csv",
    }

    def test_folder_list_marks_every_stage_stream(self):
        source = (
            REPO_ROOT / "review-ui" / "backend" / "app" / "adapters" / "local"
            / "repository.py"
        ).read_text(encoding="utf-8")
        # The per-chart override loop in _pipeline_streams.
        start = source.index("Per-chart overrides under data/folders")
        block = source[start : start + 4500]
        for suffix in self.STAGE_SUFFIXES:
            assert suffix in block, (
                f"{suffix} is written by a stage but the folder list never "
                "marks its stream, so the badge stays blank"
            )

    def test_detail_view_reads_every_stage_stream(self):
        source = (
            REPO_ROOT / "review-ui" / "backend" / "app" / "adapters" / "local"
            / "repository.py"
        ).read_text(encoding="utf-8")
        for suffix in self.STAGE_SUFFIXES:
            token = suffix.replace("_", "").replace(".csv", "")
            assert f'_{suffix.lstrip("_")}' in source or token in source


# --- combined OCR text format ----------------------------------------------


class TestCombinedOcrFormat:
    def test_marker_round_trips(self):
        from db.paths import parse_combined_ocr_txt, write_combined_ocr_txt

        import config

        original_root = config.DATA_ROOT
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                config.DATA_ROOT = Path(tmp)
                pages = [("1.jpg", "first page text"), ("2.jpg", "second page text")]
                out = write_combined_ocr_txt("chartA", "prelim", pages)
                parsed = parse_combined_ocr_txt(out)
                assert parsed["1.jpg"] == "first page text"
                assert parsed["2.jpg"] == "second page text"
        finally:
            config.DATA_ROOT = original_root

    def test_final1_json_round_trips_like_final2(self):
        """Local Mode and load_ocr_text_for_page both read pages[].content."""
        from db.paths import load_ocr_text_for_page, parse_ocr_json, write_final1_json

        import config

        original_root = config.DATA_ROOT
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                config.DATA_ROOT = Path(tmp)
                pages = [
                    {
                        "pageNumber": 1,
                        "fileName": "1.jpg",
                        "content": "# Heading\nbody",
                        "markdown": "# Heading\nbody",
                        "document": {"body": [{"label": "section_header"}]},
                    }
                ]
                out = write_final1_json("chartA", pages, model="docling+rapidocr")
                assert out.name == "chartA_final1.json"
                parsed = parse_ocr_json(out)
                assert parsed["1.jpg"] == "# Heading\nbody"
                assert load_ocr_text_for_page("chartA", "1.jpg", prefer="final1") == (
                    "# Heading\nbody"
                )
        finally:
            config.DATA_ROOT = original_root


# --- stage registry consistency --------------------------------------------


class TestStageRegistry:
    def test_execution_chain_matches_the_schema_seed(self):
        """pipeline_stage drives progress reporting, STAGE_CHAIN drives
        execution. If they disagree, a chart can never reach 'completed'."""
        from orchestrator.runner import STAGE_CHAIN

        # v1.sql seeds the implemented stages; v2.sql registers the four
        # not-yet-orchestrated ones. The chain must match V1 exactly.
        schema_sql = (REPO_ROOT / "schema" / "v1.sql").read_text(encoding="utf-8")
        seed_start = schema_sql.index("INSERT INTO pipeline_stage")
        seed = schema_sql[seed_start : schema_sql.index(";", seed_start)]

        for name, pass_no, _fn in STAGE_CHAIN:
            assert f"('{name}',{' ' * (max(1, 16 - len(name)))}{pass_no}," in seed or \
                   f"'{name}'" in seed, f"{name} not seeded in pipeline_stage"

        # Every phase-1 seeded stage must have a callable in the chain.
        chain_keys = {(n, p) for n, p, _ in STAGE_CHAIN}
        for line in seed.splitlines():
            if not line.strip().startswith("('"):
                continue
            if "FALSE" in line:  # registered but not orchestrated yet
                continue
            parts = line.strip().strip("(),").split(",")
            name = parts[0].strip().strip("'")
            pass_no = int(parts[1].strip())
            assert (name, pass_no) in chain_keys, (
                f"pipeline_stage seeds {name}:{pass_no} as phase-1 but "
                "STAGE_CHAIN has no callable for it"
            )

    def test_stage_names_are_exposed_for_the_rerun_api(self):
        from orchestrator.runner import STAGE_NAMES

        assert "blank_junk:2" in STAGE_NAMES
        assert "member_verify:1" in STAGE_NAMES


# --- V1 / V2 schema split ----------------------------------------------------


class TestSchemaSplit:
    """schema/v1.sql is what is implemented; schema/v2.sql is the next phase.

    The split is only meaningful while it stays true, and nothing else enforces
    it — a stage that starts writing a V2 table would leave V1 silently wrong.
    """

    @staticmethod
    def _relations(path):
        import re

        text = (REPO_ROOT / "schema" / path).read_text(encoding="utf-8")
        code = "\n".join(re.sub(r"--.*$", "", ln) for ln in text.splitlines())
        return set(re.findall(r"CREATE TABLE (\w+)", code)) | set(
            re.findall(r"CREATE OR REPLACE VIEW (\w+)", code)
        )

    def test_no_relation_is_defined_in_both_files(self):
        overlap = self._relations("v1.sql") & self._relations("v2.sql")
        assert not overlap, f"defined twice: {sorted(overlap)}"

    def test_v1_never_references_a_v2_relation(self):
        """V1 must apply and run on its own — v2.sql is optional."""
        import re

        text = (REPO_ROOT / "schema" / "v1.sql").read_text(encoding="utf-8")
        code = "\n".join(re.sub(r"--.*$", "", ln) for ln in text.splitlines())
        v1, v2 = self._relations("v1.sql"), self._relations("v2.sql")
        referenced = set(re.findall(r"REFERENCES\s+(\w+)\s*\(", code)) | set(
            re.findall(r"(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)\b", code)
        )
        assert not (referenced & v2), f"V1 depends on V2: {sorted(referenced & v2)}"
        assert not (referenced - v1), f"V1 references undefined: {sorted(referenced - v1)}"

    def test_no_code_touches_a_v2_relation(self):
        """If this fails, the module is implemented — move its table to v1.sql."""
        import ast
        import re

        v2 = self._relations("v2.sql")
        offenders = []
        for path in repo_python_files(
            REPO_ROOT / "core-pipeline", REPO_ROOT / "review-ui" / "backend"
        ):
            if True:
                # utf-8-sig so a BOM is stripped rather than reaching ast.parse
                for node in ast.walk(
                    ast.parse(path.read_text(encoding="utf-8-sig"))
                ):
                    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                        continue
                    sql = node.value
                    if not re.search(r"\b(INSERT INTO|UPDATE|SELECT|DELETE FROM)\b", sql):
                        continue
                    hits = set(
                        re.findall(
                            r"(?:INSERT INTO|FROM|JOIN|UPDATE|DELETE FROM)\s+([a-z_][a-z0-9_]*)\b",
                            sql,
                        )
                    )
                    for rel in hits & v2:
                        offenders.append(f"{path.name}:{node.lineno} -> {rel}")
        assert not offenders, "code touches V2 relations:\n  " + "\n  ".join(sorted(offenders))


# --- cross-platform text IO --------------------------------------------------


class TestTextIoDeclaresEncoding:
    """Every text read/write must name its encoding.

    Python defaults to the *locale* encoding: UTF-8 on macOS/Linux, cp1252 on
    Windows. The repo's sources are UTF-8 and full of em-dashes, so a call that
    omits the encoding works everywhere the author tested and then dies on
    Windows with a UnicodeDecodeError about the 'charmap' codec.

    Checked via AST, not regex, so prose in docstrings and comments cannot
    trigger it — and neither can this docstring.
    """

    @staticmethod
    def _calls():
        """Yield (path, lineno, func_name, has_encoding_kwarg, args) per call."""
        import ast

        for path in repo_python_files(
            REPO_ROOT / "core-pipeline",
            REPO_ROOT / "review-ui" / "backend",
            REPO_ROOT / "tests",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute):
                    name = func.attr
                    # PIL's Image.open is binary and takes no encoding.
                    owner = getattr(func.value, "id", None) or getattr(
                        getattr(func.value, "attr", None), "__str__", lambda: ""
                    )()
                    if name == "open" and owner == "Image":
                        continue
                elif isinstance(func, ast.Name):
                    name = func.id
                else:
                    continue
                has_enc = any(k.arg == "encoding" for k in node.keywords)
                yield path, node.lineno, name, has_enc, node.args

    def test_read_text_and_write_text_declare_encoding(self):
        offenders = [
            f"{p.relative_to(REPO_ROOT)}:{ln} .{name}()"
            for p, ln, name, has_enc, _ in self._calls()
            if name in {"read_text", "write_text"} and not has_enc
        ]
        assert not offenders, (
            "text IO without encoding= (breaks on Windows cp1252):\n  "
            + "\n  ".join(offenders)
        )

    def test_open_in_text_mode_declares_encoding(self):
        import ast

        offenders = []
        for p, ln, name, has_enc, args in self._calls():
            if name != "open" or has_enc:
                continue
            # A binary mode carries no encoding, so those are fine.
            mode = next(
                (a.value for a in args if isinstance(a, ast.Constant)
                 and isinstance(a.value, str)),
                "r",
            )
            if "b" in mode:
                continue
            offenders.append(f"{p.relative_to(REPO_ROOT)}:{ln} open()")
        assert not offenders, (
            "open() in text mode without encoding= (breaks on Windows cp1252):\n  "
            + "\n  ".join(offenders)
        )


# --- database URL scheme -----------------------------------------------------


class TestDatabaseUrlScheme:
    """Both services must accept the same DATABASE_URL value.

    They share the variable name but talk to psycopg directly, which rejects
    SQLAlchemy's "+psycopg" dialect suffix with an error that never mentions the
    scheme:

        missing "=" after "postgresql+psycopg://..." in connection info string

    review-ui always normalised it; core-pipeline did not, so a value copied
    from one .env to the other silently failed to connect.
    """

    FORMS = [
        "postgresql://u:p@h:5432/d",
        "postgresql+psycopg://u:p@h:5432/d",
        "postgres+psycopg://u:p@h:5432/d",
        "postgresql+psycopg2://u:p@h:5432/d",
    ]

    def test_core_pipeline_accepts_every_form(self):
        from config import _psycopg_url

        for form in self.FORMS:
            assert _psycopg_url(form) == "postgresql://u:p@h:5432/d", form

    def test_review_ui_accepts_every_form(self):
        from app.adapters.postgres.repository import _psycopg_url

        for form in self.FORMS[:3]:  # review-ui does not claim psycopg2
            assert _psycopg_url(form) == "postgresql://u:p@h:5432/d", form

    def test_both_env_examples_use_the_same_scheme(self):
        """A value copied between the two .env files must work in both."""
        import re

        schemes = {}
        for name in ("core-pipeline/.env.example", "review-ui/.env.example"):
            text = (REPO_ROOT / name).read_text(encoding="utf-8")
            m = re.search(r"^DATABASE_URL=([a-z0-9+]+)://", text, re.M)
            assert m, f"{name} has no DATABASE_URL line"
            schemes[name] = m.group(1)
        assert len(set(schemes.values())) == 1, (
            "the two .env.example files disagree, so copying between them "
            f"breaks: {schemes}"
        )


# --- batch intake discovery --------------------------------------------------


class TestBatchFolderDiscovery:
    """What counts as a chart folder in a drop directory.

    Getting this wrong is expensive in both directions: a missed folder is a
    chart that silently never runs, and a spurious one is an ingest that fails
    on "no pages" halfway through an unattended batch.
    """

    @staticmethod
    def _drop(tmp_path):
        from PIL import Image

        def img(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), "white").save(path)

        img(tmp_path / "chart_a" / "p1.jpg")
        img(tmp_path / "chart_b" / "p1.png")
        img(tmp_path / ".hidden" / "p1.jpg")          # dotfolder: skipped
        (tmp_path / "no_images").mkdir()
        (tmp_path / "no_images" / "notes.txt").write_text("x", encoding="utf-8")
        (tmp_path / "loose.txt").write_text("x", encoding="utf-8")
        (tmp_path / "chart_a" / "._p1.jpg").write_text("", encoding="utf-8")
        return tmp_path

    def test_finds_only_subfolders_containing_images(self, tmp_path):
        from jobs.batch_intake import find_local_chart_folders

        names = [p.name for p in find_local_chart_folders(self._drop(tmp_path))]
        assert names == ["chart_a", "chart_b"], names

    def test_a_single_chart_folder_is_itself_the_chart(self, tmp_path):
        """Pointing at one chart must work, not return its subfolders."""
        from jobs.batch_intake import find_local_chart_folders

        drop = self._drop(tmp_path)
        found = find_local_chart_folders(drop / "chart_a")
        assert [p.name for p in found] == ["chart_a"]

    def test_appledouble_stubs_do_not_make_a_folder_a_chart(self, tmp_path):
        """A folder holding only ._ stubs has no real images."""
        from jobs.batch_intake import find_local_chart_folders

        (tmp_path / "stubs_only").mkdir()
        (tmp_path / "stubs_only" / "._p1.jpg").write_text("", encoding="utf-8")
        assert find_local_chart_folders(tmp_path) == []

    def test_missing_directory_is_an_error_not_an_empty_batch(self, tmp_path):
        from jobs.batch_intake import find_local_chart_folders

        with pytest.raises(RuntimeError, match="Not a directory"):
            find_local_chart_folders(tmp_path / "nope")

    def test_batch_rejects_both_sources_at_once(self):
        from jobs.batch_intake import run_batch

        with pytest.raises(ValueError, match="either local_read_path"):
            run_batch(local_read_path="/x", blob_container="c", blob_read_path="p")

    def test_batch_rejects_neither_source(self):
        from jobs.batch_intake import run_batch

        with pytest.raises(ValueError, match="either local_read_path"):
            run_batch()

    def test_batch_read_and_write_must_share_a_backend(self):
        """Same rule as /run: a write landing somewhere the caller did not mean
        is worse than an error."""
        from jobs.batch_intake import run_batch

        with pytest.raises(ValueError, match="local_write_path"):
            run_batch(local_read_path="/x", blob_write_path="out")
        with pytest.raises(ValueError, match="blob_write_path"):
            run_batch(
                blob_container="c", blob_read_path="p", local_write_path="/out"
            )

    SHARED = (
        "input_type", "container_name", "input_path", "output_path",
        "only", "run_through", "skip_ocr", "skip_completed", "skip_page_download",
    )
    REMOVED = (
        "force", "overwrite", "write_mode", "through", "redownload_pages",
        "blob_container", "blob_read_path", "blob_write_path",
        "local_read_path", "local_write_path", "run_id", "batch_id",
        "workers", "test_mode", "skip_db_write", "chart_id", "chart_names",
    )

    def test_batch_takes_the_same_options_as_run(self):
        """Batch is run-per-folder. An option that means one thing in run and
        another in batch is the bug this shape exists to prevent."""
        from api.main import BatchRequest, RunRequest

        for field in self.SHARED:
            assert field in RunRequest.model_fields, field
            assert field in BatchRequest.model_fields, field
        assert set(RunRequest.model_fields) - set(self.SHARED) == {"chart_name"}
        assert set(BatchRequest.model_fields) - set(self.SHARED) == {"chart_list", "sample"}

    def test_removed_options_are_gone_from_both(self):
        from api.main import BatchRequest, RunRequest

        for field in self.REMOVED:
            assert field not in RunRequest.model_fields, field
            assert field not in BatchRequest.model_fields, field

    def test_defaults_reprocess_everything_and_replace_outputs(self):
        """No force / overwrite knobs: unspecified means reprocess and replace."""
        from api.main import BatchRequest

        body = BatchRequest(input_type="local", input_path="/data/inbox")
        assert body.force is True
        assert body.overwrite is True
        assert body.redownload_pages is False, "skip_page_download defaults true"
        assert body.skip_completed is False
        assert body.charts_cap() is None and body.chart_list is None

    def test_skip_ocr_is_the_one_thing_that_turns_force_off(self):
        """The runner only reuses OCR on a non-force run (it re-runs every
        non-OCR stage itself), so skip_ocr must map to force=False."""
        from api.main import RunRequest

        body = RunRequest(input_type="local", input_path="/x", chart_name="c", skip_ocr=True)
        assert body.force is False

    def test_batch_sample_and_chart_list(self):
        from api.main import BatchRequest

        body = BatchRequest(
            input_type="local", input_path="/data/inbox", sample=2,
            chart_list="a, b;a",
        )
        assert body.charts_cap() == 2
        assert body.chart_list == ["a", "b"]

    def test_batch_workers_must_fit_the_db_pool(self, monkeypatch):
        from jobs import batch_intake as bi

        monkeypatch.setattr(bi, "STAGE_WORKERS", 4)
        monkeypatch.setattr(bi, "BATCH_POOL_HEADROOM", 2)
        import db as dbmod

        monkeypatch.setattr(dbmod, "DB_POOL_MAX", 10)
        with pytest.raises(ValueError, match="DB_POOL_MAX"):
            bi.resolve_batch_workers(4)
        assert bi.resolve_batch_workers(2) == 2

    def test_batch_has_no_folder_name(self):
        """Each sub-folder IS a chart and names itself, so one folder name
        could not mean anything across fifty of them."""
        from api.main import BatchRequest

        assert not [f for f in BatchRequest.model_fields if "folder_name" in f]

    def test_a_write_failure_is_recorded_not_raised(self, tmp_path, monkeypatch):
        """In a batch of fifty, one unwritable destination must be reported and
        stepped over, not abort the other forty-nine — and the chart itself
        still ran, so it must not be marked failed either."""
        import config
        from jobs.batch_intake import _write_one

        # A chart that does not exist on disk: write_chart raises, and
        # _write_one must turn that into a recorded outcome.
        monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "folders")
        result = _write_one(
            "no_such_chart",
            local_write_path=str(tmp_path / "out"),
            blob_container=None,
            blob_write_path=None,
            write_mode="skip_orig_pages",
            overwrite=False,
        )
        assert result["status"] == "failed"
        assert "error" in result

    def test_a_successful_write_is_recorded_with_its_destination(
        self, tmp_path, monkeypatch
    ):
        import config
        from jobs.batch_intake import _write_one

        root = tmp_path / "folders"
        (root / "c1" / "ocr").mkdir(parents=True)
        (root / "c1" / "ocr" / "t.txt").write_text("t", encoding="utf-8")
        monkeypatch.setattr(config, "DATA_ROOT", root)

        result = _write_one(
            "c1",
            local_write_path=str(tmp_path / "out"),
            blob_container=None,
            blob_write_path=None,
            write_mode="skip_orig_pages",
            overwrite=False,
        )
        assert result["status"] == "written"
        assert result["files_written"] == 1
        # Each chart lands under its own folder name, as in /run.
        assert result["destination"].endswith("c1")


# --- ingest request shape ----------------------------------------------------


class TestRunAcceptsBothModes:
    """input_type picks the backend; the rest of the shape is identical."""

    @staticmethod
    def _client(monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main

        # Pretend the database is fine, so only the shape checks can fail.
        monkeypatch.setattr(main, "_require_db", lambda: None)
        monkeypatch.setattr(main, "_bg_run", lambda payload: None)
        return TestClient(main.app, raise_server_exceptions=False), main

    BLOB = {
        "input_type": "blob",
        "container_name": "c",
        "input_path": "Raw_Input/Run1",
        "chart_name": "chart_x",
    }

    def test_input_type_is_required(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        r = client.post("/api/charts/run", json={"input_path": "/x", "chart_name": "x"})
        assert r.status_code == 422

    def test_blob_needs_a_container(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        body = {**self.BLOB}
        del body["container_name"]
        r = client.post("/api/charts/run", json=body)
        assert r.status_code == 400
        assert "container_name" in r.json()["detail"]

    def test_local_rejects_a_container(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        r = client.post(
            "/api/charts/run",
            json={"input_type": "local", "container_name": "c",
                  "input_path": "/x", "chart_name": "x"},
        )
        assert r.status_code == 400
        assert "container_name" in r.json()["detail"]

    def test_chart_name_is_required_for_run(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        body = {**self.BLOB}
        del body["chart_name"]
        assert client.post("/api/charts/run", json=body).status_code == 422

    def test_blob_mode_is_accepted_and_the_folder_names_the_chart(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        r = client.post("/api/charts/run", json=self.BLOB)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["input_type"] == "blob"
        assert body["chart_name"] == "chart_x"
        assert body["source"] == "c/Raw_Input/Run1/chart_x"
        assert body["write"] is None, "no output_path means no write"

    def test_the_write_destination_appends_the_folder_name(self, monkeypatch):
        """Read and write resolve the same way, so a chart keeps its identity
        on both sides and two charts cannot merge at the destination."""
        client, _ = self._client(monkeypatch)
        r = client.post(
            "/api/charts/run", json={**self.BLOB, "output_path": "/Processed/Run1/"}
        )
        assert r.status_code == 202, r.text
        write = r.json()["write"]
        assert write["destination"] == "c/Processed/Run1/chart_x"
        assert write["write_mode"] == "skip_orig_pages", "default omits pages/"
        assert write["overwrite"] is True, "reprocessed results replace old ones"

    def test_a_removed_option_is_rejected_not_ignored(self, monkeypatch):
        """Pydantic ignores unknown fields by default, so a caller still sending
        force=false would get a 202 and a full reprocess — the opposite of what
        they asked for."""
        client, _ = self._client(monkeypatch)
        for old in ({"force": False}, {"blob_read_path": "p"}, {"through": "ocr_final2"}):
            r = client.post("/api/charts/run", json={**self.BLOB, **old})
            assert r.status_code == 422, old
            assert next(iter(old)) in r.text

    def test_skip_completed_returns_200_for_a_finished_chart(self, monkeypatch):
        client, main = self._client(monkeypatch)
        monkeypatch.setattr(main, "_chart_finished", lambda name: "needs_review")
        r = client.post("/api/charts/run", json={**self.BLOB, "skip_completed": True})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "skipped"
        assert "needs_review" in r.json()["reason"]

    def test_skip_completed_runs_an_unfinished_chart(self, monkeypatch):
        client, main = self._client(monkeypatch)
        monkeypatch.setattr(main, "_chart_finished", lambda name: None)
        r = client.post("/api/charts/run", json={**self.BLOB, "skip_completed": True})
        assert r.status_code == 202

    def test_batch_skip_completed_drops_finished_charts(self, monkeypatch, tmp_path):
        client, main = self._client(monkeypatch)
        from jobs import batch_intake

        monkeypatch.setattr(main, "_bg_batch", lambda payload: None)
        monkeypatch.setattr(batch_intake, "assert_batch_workers_fit", lambda w: None)
        monkeypatch.setattr(
            batch_intake, "finished_chart_names", lambda names: {"done_chart"}
        )
        for name in ("done_chart", "new_chart"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "1.jpg").write_bytes(b"x")
        r = client.post(
            "/api/charts/batch-run",
            json={"input_type": "local", "input_path": str(tmp_path),
                  "skip_completed": True},
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["charts_found"] == 2
        assert body["charts_queued"] == 1
        assert body["charts_skipped_completed"] == ["done_chart"]

    @pytest.mark.parametrize(
        "path", ["/api/charts/import-local", "/api/charts/ingest", "/api/charts/register-local"]
    )
    def test_the_retired_intake_endpoints_are_gone(self, monkeypatch, path):
        """All folded into /run. Two endpoints for one job is how they drift."""
        client, _ = self._client(monkeypatch)
        r = client.post(path, json={"chart_name": "x", "source_path": "/x"})
        assert r.status_code in (404, 405)


class TestStageSelection:
    """`through` bounds the chain; `only` picks stages out of it."""

    @staticmethod
    def _client(monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main

        monkeypatch.setattr(main, "_require_db", lambda: None)
        monkeypatch.setattr(main, "_bg_run", lambda payload: None)
        return TestClient(main.app, raise_server_exceptions=False), main

    def test_resolve_stage_accepts_both_spellings(self):
        """Resolves to the right STAGE, not a fixed index — the chain order is
        allowed to change (rotation moved ahead of prelim OCR), and a test that
        pins positions fails for the wrong reason when it does."""
        from orchestrator.runner import STAGE_CHAIN, resolve_stage

        def at(token):
            name, pass_no, _fn = STAGE_CHAIN[resolve_stage(token)]
            return name, pass_no

        assert at("ocr_prelim") == ("ocr_prelim", 1)
        assert at("blank_junk:2") == ("blank_junk", 2)
        assert at("dos_extract") == ("dos_extract", 1)

    def test_a_bare_name_means_pass_1_not_whichever_pass_exists(self):
        """blank_junk runs twice. A bare `blank_junk` must be the pass-1 one,
        deterministically, or `--only blank_junk` means different things on
        different days."""
        from orchestrator.runner import STAGE_CHAIN, resolve_stage

        name, pass_no, _ = STAGE_CHAIN[resolve_stage("blank_junk")]
        assert (name, pass_no) == ("blank_junk", 1)

    def test_an_unknown_stage_raises_naming_the_known_ones(self):
        from orchestrator.runner import resolve_stage

        with pytest.raises(ValueError) as exc:
            resolve_stage("ocr_final3")
        assert "ocr_final3" in str(exc.value)
        assert "ocr_final1" in str(exc.value)

    def test_a_non_numeric_pass_is_rejected(self):
        from orchestrator.runner import resolve_stage

        with pytest.raises(ValueError):
            resolve_stage("blank_junk:two")

    BLOB = {"input_type": "blob", "container_name": "c",
            "input_path": "p", "chart_name": "x"}

    def test_a_bad_run_through_is_400_not_a_silent_no_op(self, monkeypatch):
        """Unvalidated, a typo reaches the background task and becomes a log
        line the caller never sees — the run simply does nothing."""
        client, _ = self._client(monkeypatch)
        r = client.post(
            "/api/charts/run", json={**self.BLOB, "run_through": "ocr_final3"}
        )
        assert r.status_code == 400
        assert "ocr_final3" in r.json()["detail"]

    def test_a_bad_only_is_400(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        r = client.post("/api/charts/run", json={**self.BLOB, "only": ["nope"]})
        assert r.status_code == 400

    def test_a_good_run_through_is_accepted_and_echoed(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        r = client.post(
            "/api/charts/run", json={**self.BLOB, "run_through": "ocr_final2"}
        )
        assert r.status_code == 202, r.text
        assert r.json()["run_through"] == "ocr_final2"

    def test_batch_takes_the_same_stage_options_as_run(self, monkeypatch):
        """Batch is run-per-folder; an option that means one thing in run and
        another in batch is the bug this shape exists to prevent."""
        from api.main import BatchRequest, RunRequest

        for field in ("run_through", "only"):
            assert field in RunRequest.model_fields
            assert field in BatchRequest.model_fields


# --- operator-facing behaviour -----------------------------------------------


class TestMemberStageReadsTheExtraction:
    """Member details are extracted once, right after OCR; verification only reads them."""

    def test_member_stage_takes_staged_pages_not_page_text(self):
        src = (
            REPO_ROOT / "core-pipeline" / "stages" / "lib" / "member" / "stage.py"
        ).read_text(encoding="utf-8")
        assert "ensure_staging(chart_id, chart_name)" in src
        assert '"staged": staged.page(p["page_name"])' in src
        assert "MEMBER_NER" not in src and "_ocr_text_map" not in src

    def test_no_stale_downloader_command_in_any_message(self):
        """The old path stopped existing when Reference/ was deleted."""
        offenders = []
        for path in repo_python_files(REPO_ROOT / "core-pipeline"):
            text = path.read_text(encoding="utf-8")
            if "Member_Verification.Models.model_downloader" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, f"stale downloader command in: {offenders}"


class TestStageLabels:
    """Every orchestrated stage has a display name, so logs read uniformly."""

    def test_every_stage_in_the_chain_has_a_label(self):
        from orchestrator.runner import STAGE_CHAIN
        from stages._support import STAGE_LABELS, stage_label

        for name, pass_no, _fn in STAGE_CHAIN:
            assert name in STAGE_LABELS, f"{name} has no display label"
            assert stage_label(name, pass_no)

    def test_a_second_pass_is_distinguishable(self):
        from stages._support import stage_label

        assert stage_label("blank_junk", 1) == "Blank/Junk"
        assert stage_label("blank_junk", 2) == "Blank/Junk pass 2"


class TestChartResetKeepsTheAuditTrail:
    """A re-run replaces results but must not erase the run log."""

    def test_pipeline_jobs_is_not_wiped(self):
        from db import CHART_RESULT_TABLES

        assert "pipeline_jobs" not in CHART_RESULT_TABLES, (
            "pipeline_jobs is the audit trail — the point of a re-run is being "
            "able to compare it against the previous attempt"
        )

    def test_the_client_roster_is_not_wiped(self):
        from db import CHART_RESULT_TABLES

        assert "manifest_member_list" not in CHART_RESULT_TABLES, (
            "the manifest is the client's data, not our output"
        )
        assert "page_ground_truth" not in CHART_RESULT_TABLES, (
            "page labels are the client's data, not pipeline output"
        )

    def test_clear_results_keep_manifest_preserves_roster_and_stages(self):
        """schema/clear_results_keep_manifest.sql must not wipe the roster."""
        text = (
            REPO_ROOT / "schema" / "clear_results_keep_manifest.sql"
        ).read_text(encoding="utf-8")
        # Strip SQL comments so a mention in a comment does not count.
        code = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("--")
        )
        assert "manifest_member_list" not in code.split("TRUNCATE")[1].split(";")[0], (
            "TRUNCATE must not include manifest_member_list"
        )
        assert "pipeline_stage" not in code.split("TRUNCATE")[1].split(";")[0], (
            "TRUNCATE must not include pipeline_stage"
        )
        for table in (
            "chart_list",
            "page_list",
            "ocr_results",
            "member_extraction_results",
            "pipeline_jobs",
        ):
            assert table in code, f"{table} should be truncated"

    def test_every_wiped_table_actually_has_a_chart_id(self):
        """A DELETE ... WHERE chart_id on a table without one is a runtime error."""
        import re

        from db import CHART_RESULT_TABLES

        schema = (REPO_ROOT / "schema" / "v1.sql").read_text(encoding="utf-8")
        code = "\n".join(re.sub(r"--.*$", "", ln) for ln in schema.splitlines())
        for table in CHART_RESULT_TABLES:
            m = re.search(rf"CREATE TABLE {table}\s*\((.*?)\n\);", code, re.S)
            assert m, f"{table} is not in v1.sql"
            assert "chart_id" in m.group(1), f"{table} has no chart_id column"


class TestWriteChartOut:
    """Write is folded into run/batch-run; /write returns 410. Sync is in write_chart."""

    def test_write_endpoint_is_gone(self, monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main

        monkeypatch.setattr(main, "_require_db", lambda: None)
        client = TestClient(main.app, raise_server_exceptions=False)
        r = client.post(
            "/api/charts/write",
            json={"chart_name": "c", "local_write_path": "/out"},
        )
        assert r.status_code == 410
        assert "run" in r.json()["detail"].lower()

    def test_rerun_endpoint_is_gone(self, monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main

        monkeypatch.setattr(main, "_require_db", lambda: None)
        client = TestClient(main.app, raise_server_exceptions=False)
        r = client.post("/api/charts/7/rerun", json={})
        assert r.status_code == 410
        assert "chart_id" in r.json()["detail"]

    def test_the_chart_name_is_appended_to_the_destination(self, tmp_path, monkeypatch):
        """Two charts written to one destination must not merge into it."""
        import config
        from jobs.export_chart import write_chart

        chart = tmp_path / "folders" / "chart_a"
        (chart / "ocr").mkdir(parents=True)
        (chart / "ocr" / "chart_a_prelim.txt").write_text("t", encoding="utf-8")
        monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "folders")
        out = write_chart("chart_a", local_path=str(tmp_path / "out" / "dir"))
        assert out["destination"].endswith("/chart_a")

    def test_writing_onto_the_workspace_itself_is_refused(self, tmp_path, monkeypatch):
        """Otherwise the source is its own destination and the copy truncates."""
        import config
        from jobs.export_chart import write_chart

        root = tmp_path / "folders"
        (root / "chart_b" / "ocr").mkdir(parents=True)
        (root / "chart_b" / "ocr" / "t.txt").write_text("t", encoding="utf-8")
        monkeypatch.setattr(config, "DATA_ROOT", root)

        with pytest.raises(RuntimeError, match="workspace itself"):
            write_chart("chart_b", local_path=str(root))

    def test_apple_double_stubs_are_not_written_out(self, tmp_path, monkeypatch):
        import config
        from jobs.export_chart import write_chart

        root = tmp_path / "folders"
        corrected = root / "chart_c" / "corrected-pages"
        corrected.mkdir(parents=True)
        (corrected / "1.jpg").write_bytes(b"x")
        (corrected / "._1.jpg").write_bytes(b"junk")
        monkeypatch.setattr(config, "DATA_ROOT", root)

        out = write_chart("chart_c", local_path=str(tmp_path / "out"))
        assert out["files_written"] == 1
        written = {p.name for p in (tmp_path / "out").rglob("*") if p.is_file()}
        assert written == {"1.jpg"}

    def test_sync_skips_existing_and_writes_missing(self, tmp_path, monkeypatch):
        """Default write checks what is already at the destination."""
        import config
        from jobs.export_chart import write_chart

        root = tmp_path / "folders"
        ocr = root / "chart_d" / "ocr"
        ocr.mkdir(parents=True)
        (ocr / "a.txt").write_text("a", encoding="utf-8")
        (ocr / "b.txt").write_text("b", encoding="utf-8")
        monkeypatch.setattr(config, "DATA_ROOT", root)

        dest = tmp_path / "out"
        first = write_chart("chart_d", local_path=str(dest))
        assert first["files_written"] == 2
        assert first["files_skipped"] == 0

        (ocr / "c.txt").write_text("c", encoding="utf-8")
        second = write_chart("chart_d", local_path=str(dest))
        assert second["files_written"] == 1
        assert second["files_skipped"] == 2
        assert (dest / "chart_d" / "ocr" / "c.txt").read_text(encoding="utf-8") == "c"

        third = write_chart("chart_d", local_path=str(dest), overwrite=True)
        assert third["files_written"] == 3
        assert third["files_skipped"] == 0

    def test_a_chart_with_no_workspace_raises(self, tmp_path, monkeypatch):
        import config
        from jobs.export_chart import write_chart

        monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "folders")
        with pytest.raises(RuntimeError, match="No chart workspace"):
            write_chart("ghost", local_path=str(tmp_path / "out"))


class TestManifestLookup:
    """GET /api/manifest/{record_id}: Postgres first, METADATA_ROOT fallback."""

    def test_lookup_on_disk_finds_matching_rows(self, tmp_path):
        from jobs.manifest_sweeper import lookup_manifest_members_on_disk

        csv_path = tmp_path / "metadata_R1_B1.csv"
        csv_path.write_text(
            "recordId,DummyFirstName,DummyLastName,DummyDOB,MemberID\n"
            "chart_a,Ada,Lovelace,1815-12-10,M1\n"
            "chart_b,Grace,Hopper,1906-12-09,M2\n",
            encoding="utf-8",
        )
        rows = lookup_manifest_members_on_disk("chart_a", metadata_root=tmp_path)
        assert len(rows) == 1
        assert rows[0]["member_name"] == "Ada Lovelace"
        assert rows[0]["external_member_id"] == "M1"
        assert rows[0]["run_id"] == "R1"
        assert rows[0]["batch_id"] == "B1"
        assert rows[0]["source_file"] == "metadata_R1_B1.csv"

    def test_demotest_metadata_loads_as_run_and_batch_test(self, tmp_path):
        from jobs.manifest_sweeper import lookup_manifest_members_on_disk
        from app.services.metadata_csv import clear_metadata_cache, run_batch_for_record

        csv_path = tmp_path / "metadata_demotest.csv"
        csv_path.write_text(
            "recordId,DummyFirstName,DummyLastName,DummyDOB,MemberID\n"
            "demotest,Ada,Lovelace,1815-12-10,M1\n",
            encoding="utf-8",
        )
        rows = lookup_manifest_members_on_disk("demotest", metadata_root=tmp_path)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "Test"
        assert rows[0]["batch_id"] == "Test"
        assert rows[0]["source_file"] == "metadata_demotest.csv"

        clear_metadata_cache()
        run_id, batch_id = run_batch_for_record(tmp_path, "demotest")
        assert (run_id, batch_id) == ("Test", "Test")

    def test_review_ui_local_mode_reads_run_batch_from_metadata_filename(self, tmp_path):
        """Landing page Run/Batch columns in DATA_MODE=local."""
        from app.services.metadata_csv import (
            clear_metadata_cache,
            run_batch_for_record,
        )

        clear_metadata_cache()
        (tmp_path / "metadata_R2_B3.csv").write_text(
            "recordId,DummyFirstName,DummyLastName\n"
            "52743839_44976074,Jane,Doe\n",
            encoding="utf-8",
        )
        run_id, batch_id = run_batch_for_record(tmp_path, "52743839_44976074")
        assert run_id == "R2"
        assert batch_id == "B3"
        assert run_batch_for_record(tmp_path, "missing") == (None, None)

    def test_local_mode_prefers_db_run_batch_over_metadata(self, tmp_path, monkeypatch):
        """Local Mode landing Run/Batch come from chart_list when DATABASE_URL is set."""
        from app.adapters.local import repository as local_repo
        from app.adapters.local.repository import LocalFolderRepository
        from app.services.metadata_csv import clear_metadata_cache

        clear_metadata_cache()
        folders = tmp_path / "folders"
        meta = tmp_path / "metadata"
        folders.mkdir()
        meta.mkdir()
        chart = folders / "52743839_44976074"
        chart.mkdir()
        (chart / "pages").mkdir()
        (chart / "pages" / "1.jpg").write_bytes(b"x")
        (meta / "metadata_R1_B1.csv").write_text(
            "recordId,DummyFirstName,DummyLastName\n"
            "52743839_44976074,Jane,Doe\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            local_repo,
            "fetch_chart_run_batch_map",
            lambda *a, **k: {"52743839_44976074": ("R2", "B4")},
        )
        repo = LocalFolderRepository(
            folders,
            metadata_root=meta,
            database_url="postgresql://aiuser:x@127.0.0.1:5432/imaging_outputs",
        )
        summaries = repo.list_folders()
        assert len(summaries) == 1
        assert summaries[0].run_id == "R2"
        assert summaries[0].batch_id == "B4"

        # No DB hit → metadata filename still works.
        monkeypatch.setattr(
            local_repo, "fetch_chart_run_batch_map", lambda *a, **k: {}
        )
        repo2 = LocalFolderRepository(
            folders,
            metadata_root=meta,
            database_url="postgresql://aiuser:x@127.0.0.1:5432/imaging_outputs",
        )
        summaries2 = repo2.list_folders()
        assert summaries2[0].run_id == "R1"
        assert summaries2[0].batch_id == "B1"

    def test_prefer_db_run_batch_fills_gaps(self):
        from app.services.chart_run_batch import prefer_db_run_batch

        assert prefer_db_run_batch(("R2", None), ("R1", "B1")) == ("R2", "B1")
        assert prefer_db_run_batch((None, "B4"), ("R9", "B1")) == ("R9", "B4")
        assert prefer_db_run_batch(None, ("R1", "B1")) == ("R1", "B1")
        assert prefer_db_run_batch(None, None) == ("Test", "Test")

    def test_prefer_db_manifest_fills_gaps(self):
        from app.services.chart_run_batch import prefer_db_manifest

        assert prefer_db_manifest(
            {"member": "From DB", "dob": None, "memberId": "M9"},
            {"member": "From CSV", "dob": "01/01/1950", "memberId": "M1"},
        ) == {"member": "From DB", "dob": "01/01/1950", "memberId": "M9"}

    def test_local_mode_prefers_db_manifest_over_metadata(self, tmp_path, monkeypatch):
        from app.adapters.local import repository as local_repo
        from app.adapters.local.repository import LocalFolderRepository
        from app.services.metadata_csv import clear_metadata_cache

        clear_metadata_cache()
        folders = tmp_path / "folders"
        meta = tmp_path / "metadata"
        folders.mkdir()
        meta.mkdir()
        chart = folders / "52743839_44976074"
        chart.mkdir()
        (meta / "metadata_R1_B1.csv").write_text(
            "recordId,DummyFirstName,DummyLastName,DummyDOB,MemberID\n"
            "52743839_44976074,Jane,Doe,01/01/1950,M-CSV\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            local_repo,
            "fetch_manifest_for_record",
            lambda *a, **k: {
                "member": "From DB",
                "dob": "02/02/1960",
                "memberId": "M-DB",
            },
        )
        repo = LocalFolderRepository(
            folders,
            metadata_root=meta,
            database_url="postgresql://aiuser:x@127.0.0.1:5432/imaging_outputs",
        )
        man = repo._manifest_for_folder("52743839_44976074")
        assert man.member == "From DB"
        assert man.memberId == "M-DB"
        assert man.dob == "02/02/1960"

        monkeypatch.setattr(
            local_repo, "fetch_manifest_for_record", lambda *a, **k: None
        )
        man2 = repo._manifest_for_folder("52743839_44976074")
        assert man2.member == "Jane Doe"
        assert man2.memberId == "M-CSV"

    def test_endpoint_falls_back_to_metadata_when_db_empty(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main
        from jobs import manifest_sweeper as sweeper

        meta = tmp_path / "metadata"
        meta.mkdir()
        (meta / "metadata_R2_B3.csv").write_text(
            "recordId,DummyFirstName,DummyLastName\n"
            "52743839_44976074,Jane,Doe\n",
            encoding="utf-8",
        )

        class _EmptyConn:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, *args, **kwargs):
                return self

            def fetchall(self):
                return []

        monkeypatch.setattr(main, "connect", lambda: _EmptyConn())
        monkeypatch.setattr(sweeper, "METADATA_ROOT", meta)
        client = TestClient(main.app, raise_server_exceptions=False)
        r = client.get("/api/manifest/52743839_44976074")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "metadata"
        assert body["record_id"] == "52743839_44976074"
        assert body["members"][0]["member_name"] == "Jane Doe"

    def test_endpoint_prefers_postgres(self, monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main

        db_row = {
            "id": 9,
            "record_id": "chart_x",
            "member_name": "From DB",
            "first_name": "From",
            "last_name": "DB",
        }

        class _DbConn:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, *args, **kwargs):
                return self

            def fetchall(self):
                return [db_row]

        monkeypatch.setattr(main, "connect", lambda: _DbConn())
        client = TestClient(main.app, raise_server_exceptions=False)
        r = client.get("/api/manifest/chart_x")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "postgres"
        assert body["members"][0]["member_name"] == "From DB"

    def test_endpoint_404_when_neither_has_rows(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main
        from jobs import manifest_sweeper as sweeper

        class _EmptyConn:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, *args, **kwargs):
                return self

            def fetchall(self):
                return []

        monkeypatch.setattr(main, "connect", lambda: _EmptyConn())
        monkeypatch.setattr(sweeper, "METADATA_ROOT", tmp_path / "empty")
        client = TestClient(main.app, raise_server_exceptions=False)
        r = client.get("/api/manifest/missing_chart")
        assert r.status_code == 404


class TestCapabilityReporting:
    """One source for the startup banner and /health.

    They disagreed before `capabilities` existed: the banner named the database
    and worker count, /health named NER and the DOS LLM, and neither mentioned
    blob at all — so "can this box read from a container?" was only answerable
    by submitting a chart and watching it fail.
    """

    @staticmethod
    def _reload(monkeypatch, **env):
        import importlib

        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        import config

        importlib.reload(config)
        import capabilities

        return importlib.reload(capabilities)

    def test_blob_is_not_ready_without_an_account_name(self, monkeypatch):
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_ACCOUNT_NAME=None,
            AZURE_STORAGE_CONNECTION_STRING=None,
        )
        blob = caps.blob_status()
        assert blob["ready"] is False
        assert "AZURE_STORAGE_ACCOUNT_NAME" in blob["reason"]

    def test_a_connection_string_alone_is_enough(self, monkeypatch):
        """It carries the account and the credential, so nothing else is needed."""
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=x;AccountKey=y;",
            AZURE_STORAGE_ACCOUNT_NAME=None,
        )
        blob = caps.blob_status()
        assert blob["ready"] is True
        assert blob["auth"] == "connection_string"

    def test_entra_needs_only_the_account_name(self, monkeypatch):
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_CONNECTION_STRING=None,
            AZURE_STORAGE_AUTH="entra",
            AZURE_STORAGE_ACCOUNT_NAME="acct",
            AZURE_STORAGE_ACCOUNT_KEY=None,
        )
        blob = caps.blob_status()
        assert blob["ready"] is True
        assert blob["auth"] == "entra"

    def test_key_auth_without_a_key_is_not_ready(self, monkeypatch):
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_CONNECTION_STRING=None,
            AZURE_STORAGE_AUTH="key",
            AZURE_STORAGE_ACCOUNT_NAME="acct",
            AZURE_STORAGE_ACCOUNT_KEY=None,
        )
        blob = caps.blob_status()
        assert blob["ready"] is False
        assert "AZURE_STORAGE_ACCOUNT_KEY" in blob["reason"]

    def test_the_precedence_matches_the_client_builder(self, monkeypatch):
        """A connection string wins over entra, exactly as
        get_blob_service_client does. If these drift, /health reports a
        credential the code would not use."""
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=x;AccountKey=y;",
            AZURE_STORAGE_AUTH="entra",
            AZURE_STORAGE_ACCOUNT_NAME="acct",
        )
        assert caps.blob_status()["auth"] == "connection_string"

    def test_every_not_ready_capability_names_a_reason(self, monkeypatch):
        """`ready: false` with no reason sends the reader guessing — the whole
        point of the endpoint is naming the one thing to fix."""
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_ACCOUNT_NAME=None,
            AZURE_STORAGE_CONNECTION_STRING=None,
            AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=None,
            AZURE_DOCUMENT_INTELLIGENCE_KEY=None,
            AZURE_OPENAI_ENDPOINT=None,
        )
        for name, status in caps.all_capabilities().items():
            if not status.get("ready"):
                assert status.get("reason"), f"{name} is not ready and says nothing"

    def test_all_capabilities_opens_no_socket(self, monkeypatch):
        """/health calls this on every request and must not hang on a network
        that is down, so the no-probe path must never connect."""
        import socket

        caps = self._reload(monkeypatch)

        def explode(*a, **k):
            raise AssertionError("all_capabilities() opened a socket")

        monkeypatch.setattr(socket.socket, "connect", explode)
        caps.all_capabilities()  # probe=False by default

    def test_health_and_the_banner_read_the_same_dict(self, monkeypatch):
        """The banner is a rendering of all_capabilities(), not a second
        opinion about the same environment."""
        caps = self._reload(monkeypatch)
        snapshot = caps.all_capabilities()
        labels = [label for label, _ in caps.startup_lines(snapshot)]
        assert labels == [
            "blob",
            "HW model",
            "RapidOCR",
            "blank/junk model",
            "final1 Docling",
            "final2 OCR",
            "key/value extraction",
            "skip OCR",
        ]
        assert "hw_model" in snapshot
        assert "rapidocr_models" in snapshot
        assert "blank_junk_model" in snapshot

    def test_health_exposes_extraction_at_the_top_level(self, monkeypatch):
        """The key/value extraction replaced the member NER and DOS LLM entries."""
        from fastapi.testclient import TestClient

        import api.main as main

        client = TestClient(main.app, raise_server_exceptions=False)
        body = client.get("/health").json()
        assert "extraction" in body and "ready" in body["extraction"]
        assert "blob" in body


class TestAzureSdkLogging:
    """The SDK's INFO logging buries the pipeline's own progress lines.

    `azure.core.pipeline.policies.http_logging_policy` logs every request and
    response header at INFO. Our per-page progress is also INFO, so raising the
    root logger to see ours turns on a few thousand lines of headers per chart.
    """

    def test_the_default_is_warning(self, monkeypatch):
        import logging

        monkeypatch.delenv("AZURE_LOG_LEVEL", raising=False)
        from logging_setup import azure_log_level

        assert azure_log_level() == logging.WARNING

    def test_an_unknown_value_falls_back_to_warning(self, monkeypatch):
        """logging.getLevelName returns the string 'Level banana' rather than
        raising, so an unvalidated value would set a level of a str."""
        import logging

        monkeypatch.setenv("AZURE_LOG_LEVEL", "banana")
        from logging_setup import azure_log_level

        assert azure_log_level() == logging.WARNING

    def test_it_can_be_turned_back_on_for_debugging(self, monkeypatch):
        import logging

        monkeypatch.setenv("AZURE_LOG_LEVEL", "debug")
        from logging_setup import azure_log_level

        assert azure_log_level() == logging.DEBUG

    def test_quieting_covers_every_azure_child_logger(self, monkeypatch):
        """Setting the parent `azure` logger is what makes one line cover
        azure.core, azure.identity, azure.storage and the DI client."""
        import logging

        monkeypatch.delenv("AZURE_LOG_LEVEL", raising=False)
        from logging_setup import quiet_noisy_loggers

        quiet_noisy_loggers()
        http_policy = logging.getLogger(
            "azure.core.pipeline.policies.http_logging_policy"
        )
        assert not http_policy.isEnabledFor(logging.INFO)
        assert not logging.getLogger("azure.identity").isEnabledFor(logging.INFO)

    def test_quieting_covers_docling_progress_spam(self, monkeypatch):
        """Docling logs every convert at INFO on ThreadPoolExecutor threads —
        that buries the pipeline's [batch#] [chart#] progress lines."""
        import logging

        monkeypatch.delenv("AZURE_LOG_LEVEL", raising=False)
        from logging_setup import quiet_noisy_loggers

        quiet_noisy_loggers()
        assert not logging.getLogger("docling.document_converter").isEnabledFor(
            logging.INFO
        )
        assert not logging.getLogger("docling.pipeline.base_pipeline").isEnabledFor(
            logging.INFO
        )
        assert logging.getLogger("docling").isEnabledFor(logging.WARNING)

    def test_warnings_and_errors_still_get_through(self, monkeypatch):
        """A 429 or a 403 must not be silenced along with the header dump."""
        import logging

        monkeypatch.delenv("AZURE_LOG_LEVEL", raising=False)
        from logging_setup import quiet_noisy_loggers

        quiet_noisy_loggers()
        policy = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        assert policy.isEnabledFor(logging.WARNING)
        assert policy.isEnabledFor(logging.ERROR)

    def test_our_own_loggers_are_untouched(self, monkeypatch):
        """Quieting must not reach the pipeline's progress lines."""
        import logging

        monkeypatch.delenv("AZURE_LOG_LEVEL", raising=False)
        from logging_setup import configure_logging

        root = logging.getLogger()
        previous = root.level
        try:
            configure_logging(logging.INFO)
            assert logging.getLogger("stages._support").isEnabledFor(logging.INFO)
            assert logging.getLogger("orchestrator.runner").isEnabledFor(logging.INFO)
        finally:
            root.setLevel(previous)

    def test_the_level_applies_even_when_logging_is_already_configured(self):
        """basicConfig is a no-op once a handler exists — pytest installs one,
        and so does anything that configures logging before we import. The
        requested level must still take effect, or progress lines vanish."""
        import logging

        from logging_setup import configure_logging

        root = logging.getLogger()
        previous = root.level
        try:
            root.setLevel(logging.WARNING)
            configure_logging(logging.INFO)
            assert root.level == logging.INFO
        finally:
            root.setLevel(previous)

    def test_log_lines_include_current_chart_name(self):
        """Every terminal line should tag ``[batch#] [chart#]``."""
        import io
        import logging
        import threading

        from logging_setup import (
            LOG_FORMAT,
            _ChartAwareFormatter,
            _ChartContextFilter,
            configure_logging,
            reset_current_chart,
            set_current_chart,
            set_worker_name,
        )

        configure_logging(logging.INFO)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_ChartAwareFormatter(LOG_FORMAT))
        handler.addFilter(_ChartContextFilter())
        log = logging.getLogger("test.chart_context")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
        prev_name = threading.current_thread().name
        set_worker_name("batch-3")
        token = set_current_chart("52743839_44976074")
        try:
            log.info("hello")
        finally:
            reset_current_chart(token)
            threading.current_thread().name = prev_name
            log.removeHandler(handler)
        line = stream.getvalue()
        assert "[batch-3]" in line
        assert "[52743839_44976074]" in line
        assert "hello" in line


class TestCorrectedPages:
    """Rotation correction writes corrected-pages/; every later stage reads it.

    The plumbing is on; the *writing* is off by default because the detector
    cannot recover a sideways page — see ROTATION_CORRECTION_ENABLED in
    config.py for the measurement.
    """

    def test_page_image_path_prefers_a_corrected_page(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
        (tmp_path / "c" / "pages").mkdir(parents=True)
        (tmp_path / "c" / "corrected-pages").mkdir(parents=True)
        (tmp_path / "c" / "pages" / "1.jpg").write_bytes(b"original")
        (tmp_path / "c" / "corrected-pages" / "1.jpg").write_bytes(b"corrected")

        assert config.page_image_path("c", "1.jpg").read_bytes() == b"corrected"
        assert config.page_image_source("c", "1.jpg") == (True, "corrected-pages/1.jpg")

    def test_it_falls_back_to_the_original(self, tmp_path, monkeypatch):
        """Correction is sparse: an upright page is never copied, and a chart
        processed before corrections existed has no folder at all."""
        import config

        monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
        (tmp_path / "c" / "pages").mkdir(parents=True)
        (tmp_path / "c" / "pages" / "1.jpg").write_bytes(b"original")

        assert config.page_image_path("c", "1.jpg").read_bytes() == b"original"
        assert config.page_image_source("c", "1.jpg") == (False, "pages/1.jpg")

    def test_page_list_records_which_image_to_use(self):
        """page_list.use_corrected + image_path mirror page_image_path for SQL."""
        import re
        from pathlib import Path

        sql = (Path(__file__).resolve().parents[1] / "schema" / "v1.sql").read_text(
            encoding="utf-8"
        )
        m = re.search(r"CREATE TABLE page_list\s*\((.*?)\n\);", sql, re.S)
        assert m, "page_list missing from v1.sql"
        body = m.group(1)
        assert "use_corrected" in body
        assert "image_path" in body

        patch = (
            Path(__file__).resolve().parents[1] / "schema" / "patch_output_path.sql"
        ).read_text(encoding="utf-8")
        assert "use_corrected" in patch
        assert "image_path" in patch

        import inspect
        from stages.lib.image_preprocess import stage as quality_rotation_hw

        src = inspect.getsource(quality_rotation_hw)
        assert "set_page_image_source" in src
        assert "page_image_source" in src

    def test_every_ocr_stage_goes_through_the_helper(self):
        """Three stages open page images. If one keeps using pages_dir directly
        it silently OCRs the uncorrected page, which is invisible in the data."""
        import inspect

        from stages.lib.ocr import stage_final1 as ocr_final1_docling
        from stages.lib.ocr import stage_final2 as ocr_final2_azure
        from stages.lib.ocr import stage_prelim as ocr_prelim_tesseract

        for module in (ocr_prelim_tesseract, ocr_final1_docling, ocr_final2_azure):
            source = inspect.getsource(module)
            assert "page_image_path" in source, f"{module.__name__} bypasses the helper"
            assert "pages_dir" not in source, f"{module.__name__} still uses pages_dir"

    def test_rotation_runs_before_every_ocr_stage(self):
        """The whole point of the reorder: a sideways page must be corrected
        before anything reads it."""
        from orchestrator.runner import STAGE_CHAIN

        order = [name for name, _pass, _fn in STAGE_CHAIN]
        quality = order.index("ocr_quality")
        for ocr_stage in ("ocr_prelim", "ocr_final1", "ocr_final2"):
            assert quality < order.index(ocr_stage), f"{ocr_stage} runs before rotation"

    def test_the_chain_and_the_schema_seed_agree_on_order(self):
        """pipeline_stage.seq drives progress reporting, STAGE_CHAIN drives
        execution. Reordering one and not the other makes /api/charts report a
        different stage from the one running."""
        import re
        from pathlib import Path

        from orchestrator.runner import STAGE_CHAIN

        sql = (Path(__file__).resolve().parents[1] / "schema" / "v1.sql").read_text(
            encoding="utf-8"
        )
        block = sql[sql.index("INSERT INTO pipeline_stage") :]
        block = block[: block.index(";")]
        seeded = [
            (m.group(1), int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"\('(\w+)',\s*(\d+),\s*(\d+),", block)
        ]
        seeded.sort(key=lambda r: r[2])
        assert [(n, p) for n, p, _ in seeded] == [
            (n, p) for n, p, _ in STAGE_CHAIN
        ], "STAGE_CHAIN and pipeline_stage.seq disagree"

    def test_correction_is_on_by_default(self):
        """Earned by measurement, not assumed: with OSD driving the coarse
        angle the round trip recovers every page with readable text, and a
        270-degree page goes from 0.011 OCR similarity to 1.000. It was off
        while the geometric detector was the only option."""
        import importlib

        import config

        assert importlib.reload(config).ROTATION_CORRECTION_ENABLED is True

    def test_the_flag_can_turn_it_off(self, monkeypatch):
        """Records orientation without rewriting any image — for a chart set
        where the scans must stay byte-identical to what arrived."""
        import importlib

        monkeypatch.setenv("ROTATION_CORRECTION_ENABLED", "false")
        import config

        assert importlib.reload(config).ROTATION_CORRECTION_ENABLED is False
        monkeypatch.delenv("ROTATION_CORRECTION_ENABLED")
        importlib.reload(config)

    def test_the_detectors_coarse_rotation_is_not_used(self):
        """The whole point of the OSD route. If the stage ever falls back to
        the geometric detector's coarse angle it will confidently rotate pages
        the wrong way again — it scored 0 of 6 with confidence 1.000."""
        import inspect

        from stages.lib.image_preprocess import stage as quality_rotation_hw

        source = inspect.getsource(quality_rotation_hw._detect_rotation)
        assert "osd_rotation" in source
        # The detector's own result must not drive the correction.
        correct_src = inspect.getsource(quality_rotation_hw._write_corrected)
        assert "_get_detector().correct" not in correct_src
        assert '"mirror": False' in correct_src

    def test_a_reimport_clears_stale_corrections(self):
        """A left-behind corrected-pages/1.jpg would be preferred by
        page_image_path, so a new scan would be OCR'd as the old one —
        but only when the operator asks to redownload pages."""
        import inspect

        from db.paths import clear_page_image_dirs
        from stages.utilities import download_blob

        source = inspect.getsource(download_blob.import_local_folder)
        assert "clear_page_image_dirs" in source
        assert "redownload_pages" in source
        # force alone must not wipe pages (prefer workspace).
        assert "clear_chart_workspace" not in source
        clear_src = inspect.getsource(clear_page_image_dirs)
        assert "corrected-pages" in clear_src

    def test_ensure_chart_images_prefers_workspace(self, tmp_path, monkeypatch):
        """Existing pages/ are reused; no Raw_Input fetch when local is present."""
        import config
        from stages.utilities.download_blob import ensure_chart_images

        monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
        pages = tmp_path / "chart_a" / "pages"
        pages.mkdir(parents=True)
        (pages / "1.jpg").write_bytes(b"local-page")

        calls: list[str] = []

        def _boom(*_a, **_k):
            calls.append("download")
            raise AssertionError("must not download when workspace has pages")

        monkeypatch.setattr(
            "stages.utilities.download_blob._download_missing_from_raw_input", _boom
        )

        out = ensure_chart_images(
            "chart_a",
            register=False,
            blob_container="c",
            blob_path="Raw_Input/chart_a",
            output_path="Processed/chart_a",
        )
        assert out["image_source"] == "workspace"
        assert out["page_count"] == 1
        assert calls == []

    def test_ensure_chart_images_downloads_pages_from_raw_input(
        self, tmp_path, monkeypatch
    ):
        """Empty workspace → Raw_Input only (not Processed pages)."""
        import config
        from stages.utilities.download_blob import ensure_chart_images

        monkeypatch.setattr(config, "DATA_ROOT", tmp_path)

        def _fake_download(*, chart_name, blob_container, blob_path):
            dest = tmp_path / chart_name / "pages"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "1.jpg").write_bytes(b"from-raw")
            return 1, 0

        monkeypatch.setattr(
            "stages.utilities.download_blob._download_missing_from_raw_input",
            _fake_download,
        )

        out = ensure_chart_images(
            "chart_c",
            register=False,
            blob_container="c",
            blob_path="Raw_Input/Run1/chart_c",
            output_path="Processed/chart_c",
        )
        assert out["image_source"].startswith("raw_input:")
        assert (tmp_path / "chart_c" / "pages" / "1.jpg").read_bytes() == b"from-raw"

    def test_write_exports_the_corrected_pages(self):
        from jobs.export_chart import CHART_SUBDIRS

        assert "corrected-pages" in CHART_SUBDIRS


def _tesseract_osd_available() -> bool:
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ImportError:
        return False
    try:
        pytesseract.image_to_osd(np.full((100, 100, 3), 255, dtype=np.uint8))
    except Exception as exc:
        # "Too few characters" proves Tesseract AND the osd traineddata are
        # present — it got far enough to judge the (blank) page.
        return "Too few characters" in str(exc)
    return True


class TestOsdOrientation:
    """Coarse rotation comes from Tesseract OSD, not the geometric detector.

    The detector recovered 0 of 6 sideways pages while reporting confidence
    1.000 on the wrong answers, so there was no way to tell its good answers
    from its bad ones. These tests pin the guards that keep a bad answer from
    reaching the image.
    """

    def test_no_pytesseract_returns_none_rather_than_raising(self, monkeypatch):
        """Orientation is an optimisation, not a precondition: a missing
        dependency must degrade to 'leave the page alone'."""
        import builtins

        from stages.lib.image_preprocess import osd

        real_import = builtins.__import__

        def no_pytesseract(name, *args, **kwargs):
            if name.startswith("pytesseract"):
                raise ImportError("no pytesseract")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pytesseract)
        assert osd.detect_rotation(object()) is None

    def test_a_non_quadrant_rotation_is_rejected(self, monkeypatch):
        """Only 0/90/180/270 are meaningful here; anything else is a parse
        problem, and acting on it would skew the page."""
        from stages.lib.image_preprocess import osd

        monkeypatch.setattr(
            osd, "detect_rotation",
            osd.detect_rotation,  # keep the real one; patch its input instead
        )
        import pytesseract
        from pytesseract import Output  # noqa: F401

        monkeypatch.setattr(
            pytesseract, "image_to_osd",
            lambda *a, **k: {"rotate": 45, "orientation_conf": 9.0, "script": "Latin"},
        )
        assert osd.detect_rotation(object()) is None

    def test_low_confidence_is_rejected(self, monkeypatch):
        """A wrongly rotated page is worse than an uncorrected one."""
        import pytesseract

        from stages.lib.image_preprocess import osd

        monkeypatch.setattr(
            pytesseract, "image_to_osd",
            lambda *a, **k: {"rotate": 90, "orientation_conf": 0.1, "script": "Latin"},
        )
        assert osd.detect_rotation(object()) is None

    def test_a_confident_quadrant_answer_is_accepted(self, monkeypatch):
        import pytesseract

        from stages.lib.image_preprocess import osd

        monkeypatch.setattr(
            pytesseract, "image_to_osd",
            lambda *a, **k: {"rotate": 270, "orientation_conf": 6.5, "script": "Latin"},
        )
        assert osd.detect_rotation(object())["rotation"] == 270

    @pytest.mark.skipif(
        not _tesseract_osd_available(), reason="tesseract + osd traineddata not here"
    )
    def test_the_round_trip_recovers_a_rotated_page(self):
        """The acceptance test the geometric detector could not pass: rotate a
        page, detect, correct, and get the original back pixel for pixel."""
        import cv2
        import numpy as np

        from stages.lib.image_preprocess.osd import detect_rotation
        from stages.lib.image_preprocess.rotation import correct_image

        page = (
            Path(__file__).resolve().parents[1]
            / "review-ui" / "data" / "folders"
            / "demo_chart_240315_1012" / "pages" / "3.jpg"
        )
        if not page.is_file():
            pytest.skip("demo chart not present")
        img = cv2.imread(str(page))

        for code in (
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ):
            rotated = cv2.rotate(img, code)
            found = detect_rotation(rotated)
            assert found is not None, "OSD declined on a page full of text"
            fixed = correct_image(
                rotated,
                {"rotation": found["rotation"], "tilt": 0.0, "mirror": False},
            )
            assert np.array_equal(fixed, img), "correction did not recover the original"


class TestStartupProbeCannotBlock:
    """The blob probe runs at startup and must never hold the server up.

    Observed in the field: `DefaultAzureCredential` on a machine with no
    managed identity and no `az login` tried nine credential sources with
    backoff and took 96 seconds — with startup blocked behind it. The `timeout`
    passed to the SDK call bounds only the HTTP request; credential acquisition
    happens first and ignores it. Same failure the database probe was rewritten
    to avoid.
    """

    def test_a_slow_probe_returns_at_the_deadline(self, monkeypatch):
        import time

        import capabilities

        def slow(status, timeout):
            time.sleep(30)
            status["reachable"] = True

        monkeypatch.setattr(capabilities, "_blob_round_trip", slow)
        monkeypatch.setattr(
            capabilities, "blob_status",
            lambda: {"container": "c", "account": "a", "auth": "entra", "ready": True},
        )
        started = time.monotonic()
        result = capabilities.probe_blob(timeout=1)
        elapsed = time.monotonic() - started

        assert elapsed < 5, f"probe blocked for {elapsed:.1f}s"
        assert result["reachable"] is None, "a timeout is not a verdict"
        assert "unverified" in result["reason"]

    def test_a_timeout_is_not_reported_as_a_failure(self, monkeypatch):
        """`reachable: None` means we did not find out. Rendering that as
        UNREACHABLE would send someone debugging credentials that are fine."""
        import capabilities

        line = capabilities._one_line(
            {"ready": True, "reachable": None, "reason": "probe exceeded 5s"}, "OK"
        )
        assert "UNREACHABLE" not in line
        assert line.startswith("OK")

    def test_a_real_failure_still_reads_as_unreachable(self, monkeypatch):
        import capabilities

        line = capabilities._one_line(
            {"ready": True, "reachable": False, "reason": "AuthorizationFailure"}, "OK"
        )
        assert "UNREACHABLE" in line

    def test_the_probe_does_not_repeat_the_sdk_credential_dump(self, monkeypatch):
        """DefaultAzureCredential logs ~20 lines naming all nine sources it
        tried. The banner line this probe feeds says the same thing in one line
        with the reason attached, so during the probe the dump is redundant —
        and it lands on every single start."""
        import logging

        import capabilities

        identity = logging.getLogger("azure.identity")
        seen = []

        class Capture(logging.Handler):
            def emit(self, record):
                seen.append(record.getMessage())

        handler = Capture()
        identity.addHandler(handler)
        identity.setLevel(logging.WARNING)

        def fail(*a, **k):
            identity.warning("DefaultAzureCredential failed ... nine sources ...")
            raise RuntimeError("ClientAuthenticationError")

        import db.blob_store

        monkeypatch.setattr(db.blob_store, "get_container_client", fail)
        try:
            status = {"ready": True}
            capabilities._blob_round_trip(status, 1)
            assert seen == [], "the SDK dump leaked through the probe"
            assert status["reachable"] is False
            # Restored, so a failure during real chart processing still warns.
            assert identity.isEnabledFor(logging.WARNING)
        finally:
            identity.removeHandler(handler)

    def test_an_unconfigured_blob_never_starts_a_thread(self, monkeypatch):
        """No credentials means nothing to probe — it must not cost a thread
        or a second on every start."""
        import capabilities

        monkeypatch.setattr(
            capabilities, "blob_status",
            lambda: {"ready": False, "reason": "not configured"},
        )

        def explode(*a, **k):
            raise AssertionError("probed an unconfigured blob")

        monkeypatch.setattr(capabilities, "_blob_round_trip", explode)
        assert capabilities.probe_blob()["ready"] is False


class TestCredentialFailureReporting:
    """One failed blob call produced four copies of an eighty-line dump.

    The SDK's chained-credential logger emits ~20 lines naming all nine
    sources, then raises a ClientAuthenticationError whose message is that same
    text — and the retry policy repeats the whole thing once per attempt. Our
    callers log the exception, so the warning is a verbatim duplicate.
    """

    def test_the_chained_credential_logger_is_always_quiet(self, monkeypatch):
        import logging

        monkeypatch.delenv("AZURE_LOG_LEVEL", raising=False)
        from logging_setup import quiet_noisy_loggers

        quiet_noisy_loggers()
        chained = logging.getLogger("azure.identity._credentials.chained")
        assert not chained.isEnabledFor(logging.WARNING)
        assert chained.isEnabledFor(logging.ERROR), "a real error must still pass"

    def test_debug_still_wins(self, monkeypatch):
        """Someone deliberately debugging credentials must be able to see it."""
        import logging

        monkeypatch.setenv("AZURE_LOG_LEVEL", "DEBUG")
        from logging_setup import quiet_noisy_loggers

        quiet_noisy_loggers()
        chained = logging.getLogger("azure.identity._credentials.chained")
        assert chained.isEnabledFor(logging.DEBUG)
        monkeypatch.delenv("AZURE_LOG_LEVEL")
        quiet_noisy_loggers()

    def test_a_credential_failure_is_recognised(self):
        from api.main import _is_credential_failure

        class ClientAuthenticationError(Exception):
            pass

        assert _is_credential_failure(ClientAuthenticationError("no token"))

    def test_it_is_recognised_through_a_wrapping_exception(self):
        """run_download may wrap it; the operator still needs the credential
        advice rather than fifteen frames of SDK plumbing."""
        from api.main import _is_credential_failure

        class ClientAuthenticationError(Exception):
            pass

        try:
            try:
                raise ClientAuthenticationError("no token")
            except Exception as inner:
                raise RuntimeError("download failed") from inner
        except Exception as exc:
            assert _is_credential_failure(exc)

    def test_an_unrelated_failure_keeps_its_traceback(self):
        """Only auth failures get the short treatment — anything else still
        needs the stack, and swallowing it would hide real bugs."""
        from api.main import _is_credential_failure

        assert not _is_credential_failure(ValueError("disk full"))
        assert not _is_credential_failure(RuntimeError("No images in ..."))


class TestInteractiveBlobAuth:
    """`AZURE_STORAGE_AUTH=entra_interactive` adds a browser prompt as a last
    resort, for a developer machine with no managed identity and no `az`.

    `DefaultAzureCredential` deliberately excludes InteractiveBrowserCredential,
    which is why such a machine fails with a twenty-line list of things it
    tried. The V1 prototype chained the browser in explicitly and worked there.
    """

    @staticmethod
    def _reload(monkeypatch, auth):
        import importlib

        monkeypatch.setenv("AZURE_STORAGE_AUTH", auth)
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        import config

        importlib.reload(config)
        import capabilities

        return importlib.reload(capabilities)

    def test_interactive_is_not_reached_by_plain_entra(self, monkeypatch):
        """A service must never open a browser because someone wrote `entra`.
        The interactive chain is opt-in by name, not a fallback."""
        from db.blob_store import ENTRA_INTERACTIVE_MODES, ENTRA_MODES

        assert not (ENTRA_MODES & ENTRA_INTERACTIVE_MODES)
        assert "entra" not in ENTRA_INTERACTIVE_MODES

    def test_the_mode_is_reported_distinctly(self, monkeypatch):
        caps = self._reload(monkeypatch, "entra_interactive")
        assert caps.blob_status()["auth"] == "entra_interactive"
        assert caps.blob_status()["ready"] is True

    def test_the_startup_probe_never_prompts(self, monkeypatch):
        """The probe runs at startup with nobody necessarily watching. A browser
        prompt there would block the server start on a human."""
        caps = self._reload(monkeypatch, "entra_interactive")

        def explode(*a, **k):
            raise AssertionError("the startup probe tried to authenticate")

        monkeypatch.setattr(caps, "_blob_round_trip", explode)
        result = caps.probe_blob(timeout=1)
        assert result["probe"].startswith("skipped")

    def test_plain_entra_still_probes(self, monkeypatch):
        """Non-interactive auth has nothing to prompt, so it is still checked."""
        caps = self._reload(monkeypatch, "entra")
        called = []
        monkeypatch.setattr(
            caps, "_blob_round_trip",
            lambda status, timeout: called.append(True),
        )
        caps.probe_blob(timeout=1)
        assert called, "plain entra should still be probed"

    def test_the_chain_tries_silent_credentials_first(self):
        """V1 put the browser first and prompted even where a silent credential
        existed. On the Azure VM the managed identity must answer and nobody
        should ever see a browser."""
        import inspect

        from db import blob_store

        source = inspect.getsource(blob_store._interactive_credential)
        order = [
            source.index("ManagedIdentityCredential("),
            source.index("AzureCliCredential("),
            source.index("InteractiveBrowserCredential("),
        ]
        assert order == sorted(order), "browser must be the last resort"

    def test_the_credential_is_built_once(self, monkeypatch):
        """A prompt per call would be unusable; the cache and the singleton are
        what make it one login for the process."""
        import inspect

        from db import blob_store

        source = inspect.getsource(blob_store._interactive_credential)
        assert "cache_persistence_options" in source
        assert "_INTERACTIVE_CREDENTIAL" in source


class TestManagedIdentityBlobAuth:
    """VM / App Service managed identity — explicit mode + client_id."""

    def _reload(self, monkeypatch, **env):
        import importlib

        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        import config

        importlib.reload(config)
        import capabilities

        return importlib.reload(capabilities)

    def test_managed_identity_is_reported(self, monkeypatch):
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_CONNECTION_STRING=None,
            AZURE_STORAGE_AUTH="managed_identity",
            AZURE_STORAGE_ACCOUNT_NAME="acct",
            AZURE_CLIENT_ID="11111111-2222-3333-4444-555555555555",
            AZURE_PRINCIPAL_ID="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            AZURE_STORAGE_ACCOUNT_KEY=None,
        )
        blob = caps.blob_status()
        assert blob["ready"] is True
        assert blob["auth"] == "managed_identity"
        assert blob["client_id"] == "11111111-2222-3333-4444-555555555555"
        assert blob["principal_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_mi_alias_is_accepted(self, monkeypatch):
        from db.blob_store import MANAGED_IDENTITY_MODES

        assert "mi" in MANAGED_IDENTITY_MODES
        assert "msi" in MANAGED_IDENTITY_MODES
        caps = self._reload(
            monkeypatch,
            AZURE_STORAGE_CONNECTION_STRING=None,
            AZURE_STORAGE_AUTH="mi",
            AZURE_STORAGE_ACCOUNT_NAME="acct",
            AZURE_STORAGE_ACCOUNT_KEY=None,
        )
        assert caps.blob_status()["auth"] == "managed_identity"

    def test_managed_identity_is_not_interactive(self):
        from db.blob_store import (
            ENTRA_INTERACTIVE_MODES,
            MANAGED_IDENTITY_MODES,
        )

        assert not (MANAGED_IDENTITY_MODES & ENTRA_INTERACTIVE_MODES)

    def test_interactive_chain_passes_client_id_when_set(self, monkeypatch):
        import inspect

        from db import blob_store

        source = inspect.getsource(blob_store._interactive_credential)
        assert "client_id" in source
        assert "_mi_client_id" in inspect.getsource(blob_store)


class TestEnsureBlobReady:
    def test_ensure_blob_ready_is_idempotent(self, monkeypatch):
        from db import blob_store

        monkeypatch.setattr(blob_store, "_BLOB_READY", False)
        calls = {"n": 0}

        class FakeClient:
            def get_container_properties(self):
                calls["n"] += 1

            credential = None

        monkeypatch.setattr(
            blob_store, "get_container_client", lambda container=None: FakeClient()
        )
        blob_store.ensure_blob_ready("c")
        blob_store.ensure_blob_ready("c")
        assert calls["n"] == 1


class TestWriteModes:
    """`skip_orig_pages` (default) vs `all_files`.

    The originals came FROM the source you are usually writing back to, so
    re-sending them doubles storage and transfer for bytes already there.
    corrected-pages/ is still sent, because the pipeline produced those and
    the source does not have them.
    """

    @staticmethod
    def _chart(tmp_path, monkeypatch):
        import config

        root = tmp_path / "folders"
        chart = root / "c1"
        for sub in ("pages", "corrected-pages", "ocr", "imaging"):
            (chart / sub).mkdir(parents=True)
        (chart / "pages" / "1.jpg").write_bytes(b"orig1")
        (chart / "pages" / "2.jpg").write_bytes(b"orig2")
        (chart / "corrected-pages" / "2.jpg").write_bytes(b"fixed2")
        (chart / "ocr" / "c1_prelim.txt").write_text("text", encoding="utf-8")
        (chart / "imaging" / "c1_dos.csv").write_text("a,b", encoding="utf-8")
        monkeypatch.setattr(config, "DATA_ROOT", root)
        return root

    def test_the_default_omits_the_original_pages(self, tmp_path, monkeypatch):
        from jobs.export_chart import write_chart

        self._chart(tmp_path, monkeypatch)
        out = write_chart("c1", local_path=str(tmp_path / "out"))
        written = {
            str(p.relative_to(tmp_path / "out" / "c1")).replace("\\", "/")
            for p in (tmp_path / "out" / "c1").rglob("*")
            if p.is_file()
        }
        assert written == {
            "corrected-pages/2.jpg", "ocr/c1_prelim.txt", "imaging/c1_dos.csv",
        }
        assert out["write_mode"] == "skip_orig_pages"

    def test_all_files_sends_the_originals_too(self, tmp_path, monkeypatch):
        from jobs.export_chart import write_chart

        self._chart(tmp_path, monkeypatch)
        write_chart("c1", local_path=str(tmp_path / "out"), write_mode="all_files")
        written = {
            str(p.relative_to(tmp_path / "out" / "c1")).replace("\\", "/")
            for p in (tmp_path / "out" / "c1").rglob("*")
            if p.is_file()
        }
        assert "pages/1.jpg" in written and "pages/2.jpg" in written
        assert "corrected-pages/2.jpg" in written

    def test_a_corrected_page_is_sent_even_by_default(self, tmp_path, monkeypatch):
        """The point of the default: skip what the source already has, keep
        what the pipeline made."""
        from jobs.export_chart import subdirs_for

        assert "corrected-pages" in subdirs_for("skip_orig_pages")
        assert "pages" not in subdirs_for("skip_orig_pages")
        assert "pages" in subdirs_for("all_files")

    def test_an_unknown_mode_raises_rather_than_guessing(self):
        from jobs.export_chart import subdirs_for

        with pytest.raises(ValueError, match="write_mode"):
            subdirs_for("everything")

    def test_a_chart_with_only_originals_says_why_nothing_was_written(
        self, tmp_path, monkeypatch
    ):
        """Default mode on a chart that produced no output would silently write
        an empty folder; the message names all_files as the way out."""
        import config

        from jobs.export_chart import write_chart

        root = tmp_path / "folders"
        (root / "c2" / "pages").mkdir(parents=True)
        (root / "c2" / "pages" / "1.jpg").write_bytes(b"x")
        monkeypatch.setattr(config, "DATA_ROOT", root)

        with pytest.raises(RuntimeError, match="all_files"):
            write_chart("c2", local_path=str(tmp_path / "out"))
