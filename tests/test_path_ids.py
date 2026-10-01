"""Tests for run/batch inference from path segments."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core-pipeline"
for path in (str(CORE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from db.path_ids import infer_run_batch_from_path, resolve_run_batch  # noqa: E402


def test_run_batch_either_order():
    assert infer_run_batch_from_path("Raw_Input/Run1/Batch1/DEID_PNGs") == ("R1", "B1")
    assert infer_run_batch_from_path("Batch1/Run1") == ("R1", "B1")
    assert infer_run_batch_from_path("run_2", "batch_3") == ("R2", "B3")


def test_short_forms():
    assert infer_run_batch_from_path("R12/B4") == ("R12", "B4")


def test_explicit_wins():
    assert resolve_run_batch("R9", None, "Run1/Batch1") == ("R9", "B1")
    assert resolve_run_batch(None, "B8", "Run1/Batch1") == ("R1", "B8")


def test_missing_segments():
    assert infer_run_batch_from_path("Raw_Input/DEID_PNGs") == (None, None)
    assert resolve_run_batch(None, None, "Raw_Input/Demotest/DEID_Images") == (
        "Test",
        "Test",
    )
    assert resolve_run_batch(None, None, "Raw_Input/Run1/DEID_Images") == ("R1", "Test")


def test_derive_output_path_raw_input_deid():
    from db.path_ids import derive_output_path, output_write_prefix, resolve_output_path

    assert (
        derive_output_path(
            "Raw_Input/Run1/Batch1/DEID_Images",
            "52743839_44976074",
        )
        == "Processed/Run1/Batch1/52743839_44976074"
    )
    assert (
        derive_output_path(
            "Raw_Input/Run1/Batch1/DEID_PNGs/52743839_44976074",
            "52743839_44976074",
        )
        == "Processed/Run1/Batch1/52743839_44976074"
    )
    assert (
        output_write_prefix("Processed/Run1/Batch1/52743839_44976074")
        == "Processed/Run1/Batch1"
    )
    # Explicit write path is used as-is (no Raw_Input→Processed remap).
    assert (
        resolve_output_path(
            "FolderName",
            write_path="MyOut/Custom/Run9",
            read_path="Raw_Input/Run1/Batch1/DEID_Images",
        )
        == "MyOut/Custom/Run9/FolderName"
    )
    assert (
        resolve_output_path(
            "FolderName",
            write_path="MyOut/Custom/Run9/FolderName",
        )
        == "MyOut/Custom/Run9/FolderName"
    )
    assert (
        resolve_output_path(
            "FolderName",
            read_path="Raw_Input/Run1/Batch1/DEID_Images",
        )
        == "Processed/Run1/Batch1/FolderName"
    )


def test_windows_separators_in_run_batch_inference():
    assert infer_run_batch_from_path(r"Raw_Input\Run1\Batch1\DEID_PNGs") == ("R1", "B1")


def test_normalize_fs_and_blob_paths():
    from db.paths import (
        normalize_blob_path,
        normalize_folder_name,
        normalize_fs_path,
    )

    assert normalize_fs_path(r"C:\data\inbox") == "C:/data/inbox"
    assert normalize_fs_path(r"C:\data\inbox\\") == "C:/data/inbox"
    assert normalize_fs_path("/data/inbox/") == "/data/inbox"
    assert normalize_fs_path(r"\\server\share\charts") == "//server/share/charts"
    assert normalize_fs_path("  ") is None
    assert normalize_fs_path(None) is None

    assert normalize_blob_path(r"Raw_Input\Run1\Batch1") == "Raw_Input/Run1/Batch1"
    assert normalize_blob_path("/Processed/Run1/") == "Processed/Run1"
    assert normalize_blob_path("") is None

    assert normalize_folder_name(r"inbox\52743839_44976074") == "52743839_44976074"
    assert normalize_folder_name("52743839_44976074/") == "52743839_44976074"


def test_run_request_force_defaults_true():
    from api.main import RunRequest

    body = RunRequest(input_type="local", input_path="/x", chart_name="c")
    assert body.force is True


def test_dotenv_override_wins_for_ner_model_id(tmp_path, monkeypatch):
    """Local uvicorn: .env beats a leftover shell export of gliner_low."""
    import os

    from dotenv import load_dotenv

    env = tmp_path / ".env"
    env.write_text("MEMBER_NER_MODEL_ID=gliner_medium\n", encoding="utf-8")
    monkeypatch.setenv("MEMBER_NER_MODEL_ID", "gliner_low")
    load_dotenv(env, override=True)
    assert os.environ["MEMBER_NER_MODEL_ID"] == "gliner_medium"
