"""Member verification, ported from V1 Member_Verification.

Takes the member name / DOB / ID the extraction layer staged (stages/lib/extraction) and
checks them against the manifest. Public surface used by stages/lib/member/stage.py.
"""
from .engine import (
    DETECTION_SOURCE_DB,
    PAGE_STATUS_DB,
    PageResult,
    RecordResult,
    classify_page,
    detect_name_mode,
    document_verified,
    expected_from_manifest,
    page_result_to_v1_row,
    staged_page_fields,
    summary_status,
    verify_page,
    verify_record,
)

__all__ = [
    "DETECTION_SOURCE_DB",
    "PAGE_STATUS_DB",
    "PageResult",
    "RecordResult",
    "classify_page",
    "detect_name_mode",
    "document_verified",
    "expected_from_manifest",
    "page_result_to_v1_row",
    "staged_page_fields",
    "summary_status",
    "verify_page",
    "verify_record",
]
