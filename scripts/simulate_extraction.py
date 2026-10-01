"""Run the integrated extraction through every stage that uses it, on a prepared test document.

No OCR runs and nothing is written to Postgres: the document's OCR JSON (Azure Document
Intelligence "read" output, one entry per page) is loaded as the chart's Final2 result into the
in-memory store (the ``--skip-db-write`` backend), then the real stages run in chain order:

    kv_extract -> member_verify -> dos_extract -> page_subtype -> encounter_type -> page_sequencing

and the staging is dropped, as the orchestrator does when a chart completes. The workspace (page copies,
CSVs, the staging file while it exists) goes to a temporary folder.

    cd core-pipeline
    python ../scripts/simulate_extraction.py <document folder> \
        --member "David Thomas" --dob 03/20/1981 --member-id E2694028

The document folder holds ``OCR/*.json`` and ``RAW/<images>``; OCR pages are matched to the
images in order. Models are read from EXTRACTION_MODELS_ROOT (default core-pipeline/models).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core-pipeline"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("document", type=Path, help="folder with OCR/*.json and RAW/<page images>")
    parser.add_argument("--chart-name", default=None, help="record id (default: the OCR file's recordId)")
    parser.add_argument("--member", required=True, help='manifest name, "First [Middle] Last"')
    parser.add_argument("--dob", default="", help="manifest DOB, MM/DD/YYYY")
    parser.add_argument("--member-id", default="", help="manifest member ID")
    parser.add_argument("--keep", action="store_true", help="keep the temporary workspace and print its path")
    return parser.parse_args()


def _final2_page(ocr_page: dict, page_name: str, number: int) -> dict:
    """An Azure page in the shape stage 5 stores: words and lines live under pagesMeta."""
    meta = {
        "pageNumber": number,
        "angle": ocr_page.get("angle"),
        "width": ocr_page.get("width"),
        "height": ocr_page.get("height"),
        "unit": ocr_page.get("unit"),
        "lines": [{"content": ln.get("content", ""), "polygon": ln.get("polygon", [])} for ln in ocr_page.get("lines") or []],
        "words": [
            {"content": w.get("content", ""), "confidence": w.get("confidence"), "polygon": w.get("polygon", [])}
            for w in ocr_page.get("words") or []
        ],
        "barcodes": [],
    }
    return {
        "pageNumber": number,
        "fileName": page_name,
        "content": ocr_page.get("content") or "",
        "pagesMeta": [meta],
        "section_header_candidates": [],
        "section_headers": [],
        "languages": ocr_page.get("languages") or [],
    }


def main() -> int:
    args = _args()
    work = Path(tempfile.mkdtemp(prefix="kv_sim_"))
    os.environ["DATA_ROOT"] = str(work / "folders")
    os.environ["METADATA_ROOT"] = str(work / "metadata")
    sys.path.insert(0, str(CORE))

    ocr_files = sorted((args.document / "OCR").glob("*.json"))
    images = sorted((args.document / "RAW").iterdir())
    if not ocr_files or not images:
        raise SystemExit(f"{args.document}: needs OCR/*.json and RAW/<images>")
    ocr_doc = json.loads(ocr_files[0].read_text(encoding="utf-8"))
    ocr_pages = ocr_doc["pages"]
    if len(ocr_pages) != len(images):
        raise SystemExit(f"{len(ocr_pages)} OCR pages but {len(images)} images")
    chart = args.chart_name or ocr_doc.get("recordId") or args.document.name

    from db import (
        connect,
        enable_skip_db_write,
        get_member_summary,
        init_page_stages,
        list_pages,
        upsert_manifest_member,
        upsert_ocr_result,
    )
    from db.memory_store import is_skip_db_write
    from db.paths import write_final2_json
    from orchestrator.runner import STAGE_CHAIN
    from stages.lib.extraction import engine, staging
    from stages.utilities.download_blob import import_local_folder

    store = enable_skip_db_write(reset=True)
    assert is_skip_db_write()

    status = engine.readiness()
    print(f"extraction readiness: {status}")
    if not status["ready"]:
        raise SystemExit(f"cannot simulate: {status['reason']}")

    intake = import_local_folder(args.document / "RAW", chart_name=chart, force=True)
    chart_id, chart_name = intake["chart_id"], intake["chart_name"]
    names = args.member.split()
    first, last = names[0], names[-1]
    middle = " ".join(names[1:-1]) or None
    with connect() as conn:
        pages = list_pages(conn, chart_id)
        final2 = [_final2_page(op, pg["page_name"], pg["page_number"]) for op, pg in zip(ocr_pages, pages)]
        for entry, pg in zip(final2, pages):
            upsert_ocr_result(conn, chart_id=chart_id, page_id=pg["id"], ocr_type="azuredocintel", raw_text=json.dumps(entry))
        upsert_manifest_member(
            conn, record_id=chart, member_name=args.member, first_name=first, middle_name=middle,
            last_name=last, member_dob=args.dob or None, external_member_id=args.member_id or None,
        )
        init_page_stages(conn, chart_id)
    write_final2_json(chart_name, final2)
    print(f"chart {chart_name}: {len(pages)} page(s) in {work}")

    chain = {f"{name}:{no}": fn for name, no, fn in STAGE_CHAIN}
    results = {}
    for key in ("kv_extract:1", "member_verify:1", "dos_extract:1", "page_subtype:1", "encounter_type:1", "page_sequencing:1"):
        print(f"=== {key}")
        results[key] = chain[key](chart_id, force=True)

    extraction = results["kv_extract:1"]
    print("\n--- kv_extract:", {k: v for k, v in extraction.items() if k != "skipped"})
    staged = staging.read(chart_name)
    for page in staged or []:
        found = {
            field: [row.get("value") or row.get("text") or row.get("dos_from") or row.get("provider_name") or "(signature)"
                    for row in page.selected(field)]
            for field in page.fields
        }
        print(f"  {page.name}: " + "; ".join(f"{f}={v}" for f, v in found.items() if v))

    print("\n--- member_verify")
    with connect() as conn:
        summary = get_member_summary(conn, chart_id)
    print("  summary:", {k: summary[k] for k in ("final_status", "document_decision", "pages_checked", "pages_matched", "wrong_member_pages", "reject_threshold", "decision_reason")})
    for row in store.member_extractions.values():
        print(f"  page {row.get('page_id')}: {row.get('page_status'):<13} name={row.get('extracted_name')!r} "
              f"dob={row.get('extracted_dob')} id={row.get('extracted_member_id')} "
              f"src={row.get('detection_source_name')}/{row.get('detection_source_dob')}/{row.get('detection_source_member_id')}")

    print("\n--- dos_extract")
    for page_id, row in sorted(store.dos.items()):
        print(f"  page {page_id}: page={row.get('date_of_service_from')}..{row.get('date_of_service_to')} "
              f"doc={row.get('date_of_service_from_doclevel')} conf={row.get('confidence')} "
              f"method={row.get('extraction_method')} dates={len(row.get('dates') or [])}")

    print("\n--- page_sequencing")
    for page_id, row in sorted(store.sequencing.items()):
        print(f"  page {page_id}: {row}")

    final2_json = json.loads((Path(os.environ["DATA_ROOT"]) / chart_name / "ocr" / f"{chart_name}_final2.json").read_text(encoding="utf-8"))
    print("\n--- section_headers written to the final2 JSON:",
          [len(p.get("section_headers") or []) for p in final2_json["pages"]])

    print("\n--- no database was touched:",
          "psycopg" not in sys.modules and is_skip_db_write())
    staging_before_drop = staging.staging_path(chart_name).is_file()
    print(f"--- staging present until the chart completes: {staging_before_drop}")
    print(f"--- staging dropped on completion: {staging.drop(chart_name)}")

    if args.keep:
        print(f"workspace kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
