-- =====================================================================
-- AI Imaging & Coding Pipeline — PostgreSQL Schema  ·  V1 (IMPLEMENTED)
-- =====================================================================
-- Everything in this file is built, wired and running today. Every table
-- here is written or read by core-pipeline or review-ui.
--
-- FIRST-TIME SETUP (empty database)
-- ---------------------------------------------------------------------
--     psql "$DATABASE_URL" -f schema/v1.sql        # required — all of this
--     psql "$DATABASE_URL" -f schema/v2.sql        # optional — proposals only
--
-- EXISTING DATABASE (already applied an older v1 / early v2)
-- ---------------------------------------------------------------------
--     psql "$DATABASE_URL" -f schema/patch_output_path.sql
-- Do not re-apply v1.sql on a live DB — CREATE TABLE will fail. The patch
-- is additive and idempotent (output_path, encounter/sequencing tables,
-- stage registry, page_stage_status seeds).
--
-- Apply V1 first: V2 depends on it. V1 stands alone — nothing in V1
-- references a V2 table.
--
-- WHAT IS IN HERE — tables + the 12 phase-1 stages that write them
-- ---------------------------------------------------------------------
--   pipeline_stage               stage registry (12 phase-1 stages)
--   chart_list / page_list       identity + lifecycle (+ output_path)
--   page_stage_status            per-page per-stage per-pass progress
--   manifest_member_list         the client roster
--   ocr_results                  ocr_prelim / ocr_final1 / ocr_final2
--   ocr_quality_results          ocr_quality
--   blank_junk_classification    blank_junk pass 1+2 (+ v_page_blank_junk_final)
--   member_extraction_results    member_verify
--   member_verification_summary  member_verify
--   dos_extraction_results       dos_extract
--   page_classification          page_subtype (codeable / non-codeable / discharge)
--   encounter_type_results       encounter_type  (was a V2 proposal)
--   page_sequencing_results      page_sequencing (was a V2 proposal)
--   pipeline_jobs                run log
--   v_chart_stage_progress       chart status rollup
--
-- Stage page_subtype (codeable TF) writes page_classification + CSV.
-- Anything else not listed above lives in v2.sql.
--
-- ---------------------------------------------------------------------
-- NAMING CONVENTIONS — every table in BOTH files obeys these. A new table
-- that breaks one of them is a bug, not a style preference.
-- ---------------------------------------------------------------------
--  1. snake_case everywhere. No camelCase, no abbreviations that are not
--     already domain words (dos = date of service, ocr, ner, hw).
--  2. Primary key is  id BIGSERIAL PRIMARY KEY. The one exception is
--     pipeline_stage, a static registry keyed by its natural key
--     (stage_name, pass_no) — it is referenced by name, never by id.
--  3. A foreign key is named <referenced_table_singular>_id and nothing
--     else:  chart_id, page_id, user_id, model_id, matched_member_list_id.
--     Never reviewed_by / imported_by / review_id for the same idea.
--  4. Every table has  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
--     and                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
--     with a  trg_<table>_updated_at  trigger. No bespoke row-birth names
--     (no imported_at / compared_at / evaluated_at / decided_at).
--     Timestamps that mean something *other* than row lifecycle keep their
--     own name: started_at, completed_at, queued_at, trained_at.
--  5. A model/rule score in [0,1] is  confidence NUMERIC(5,4). When a table
--     carries more than one, each is prefixed: hw_confidence, quality_score.
--     Never confidence_score / similarity_score for the same idea.
--  6. An ordinal within a parent row is  seq INT.  Not chunk_index, not
--     sequence_no, not position.
--  7. Stage identity is always the pair  stage_name VARCHAR(50) +
--     pass_no SMALLINT.  A retry counter is  attempt INT.
--  8. Blob location is  blob_container VARCHAR(150) + blob_path TEXT.
--  9. Booleans read as assertions: is_final, is_active, mirrored,
--     rotation_applied, signature_present, use_corrected.
-- 10. Indexes:  idx_<table>_<columns>.  Views:  v_<name> for helper views;
--     report views keep their business name (consolidated_chart_results).
--
-- Naming/shape changes from v7 are listed at the top of v2.sql.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
SET search_path TO public;

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------
-- STAGE REGISTRY
-- ---------------------------------------------------------------------
-- The pipeline's shape as data. Orchestrator reads execution order from
-- here; chart status is derived by joining page_stage_status against it.

CREATE TABLE pipeline_stage (
    stage_name   VARCHAR(50)  NOT NULL,
    pass_no      SMALLINT     NOT NULL DEFAULT 1,
    seq          INT          NOT NULL,
    label        VARCHAR(80)  NOT NULL,
    is_phase1    BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (stage_name, pass_no),
    UNIQUE (seq)
);

CREATE TRIGGER trg_pipeline_stage_updated_at
    BEFORE UPDATE ON pipeline_stage
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO pipeline_stage (stage_name, pass_no, seq, label, is_phase1) VALUES
    -- Rotation is first: it corrects the page image, and every OCR pass below
    -- reads the corrected copy when one exists. It needs no OCR of its own.
    ('ocr_quality',      1, 10, 'Rotation + Quality + Handwriting', TRUE),
    ('ocr_prelim',       1, 20, 'Preliminary OCR (Tesseract)',   TRUE),
    ('blank_junk',       1, 30, 'Blank/Junk/Duplicate — pass 1',  TRUE),
    ('ocr_final1',       1, 40, 'Final OCR 1 (RapidOCR)',        TRUE),
    ('ocr_final2',       1, 50, 'Final OCR 2 (Azure DocIntel)',  TRUE),
    ('kv_extract',       1, 55, 'Key/Value Extraction',          TRUE),
    ('blank_junk',       2, 60, 'Blank/Junk/Duplicate — pass 2',  TRUE),
    ('member_verify',    1, 70, 'Member Extraction + Verify',    TRUE),
    ('dos_extract',      1, 80, 'Date-of-Service Extraction',    TRUE),
    ('page_subtype',     1, 85, 'Codeable / Non Codeable (TF)', TRUE),
    ('encounter_type',   1, 90, 'Encounter Type (TF)',           TRUE),
    ('page_sequencing',  1, 95, 'Page Sequencing',               TRUE);
-- rejection_logic (seq 120, is_phase1 = FALSE) remains in v2.sql.

-- ---------------------------------------------------------------------
-- MASTER + REFERENCE TABLES
-- ---------------------------------------------------------------------

CREATE TABLE chart_list (
    id                  BIGSERIAL PRIMARY KEY,
    chart_name          VARCHAR(150) NOT NULL,
    page_count          INT,

    -- Lifecycle only. Where the chart *is* lives in current_stage.
    status              VARCHAR(20) NOT NULL DEFAULT 'received'
                        CHECK (status IN (
                            'received','downloading','processing',
                            'completed','failed','needs_review','rejected'
                        )),
    current_stage       VARCHAR(50),
    current_pass        SMALLINT,

    -- How this row came to exist. 'manifest' rows are placeholders created by
    -- the manifest sweeper for charts that may never be ingested; the UI
    -- filters them out until pages arrive.
    source              VARCHAR(20) NOT NULL DEFAULT 'blob'
                        CHECK (source IN ('blob','local','manifest')),

    blob_container      VARCHAR(150),
    blob_path           TEXT,
    -- Derived Processed/… destination for this chart (see derive_output_path).
    -- Example: Raw_Input/Run1/Batch1/DEID_PNGs/<chart>
    --       →  Processed/Run1/Batch1/<chart>
    output_path         TEXT,
    run_id              VARCHAR(50),
    batch_id            VARCHAR(50),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One row per chart name. v6 relied on SELECT ... ORDER BY id DESC LIMIT 1,
    -- which let two concurrent ingests create two charts with the same name.
    UNIQUE (chart_name)
);
CREATE INDEX idx_chart_list_status ON chart_list(status);
CREATE INDEX idx_chart_list_run_batch ON chart_list(run_id, batch_id);
CREATE INDEX idx_chart_list_source ON chart_list(source);

CREATE TRIGGER trg_chart_list_updated_at
    BEFORE UPDATE ON chart_list
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE TABLE page_list (
    id               BIGSERIAL PRIMARY KEY,
    chart_id         BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_name        VARCHAR(150) NOT NULL,
    page_number      INT,

    -- Image identity: image-level duplicate detection and download idempotency.
    image_sha256     CHAR(64),
    file_size_bytes  BIGINT,

    -- Which workspace image OCR/stages should read. Mirrors page_image_path():
    --   use_corrected=false → image_path = pages/<page_name>     (Raw_Input ingest)
    --   use_corrected=true  → image_path = corrected-pages/<file> (stage 1 wrote one)
    -- Not the same as ocr_quality_results.rotation_applied: a TIFF→JPG copy can
    -- set use_corrected without a geometric rotation having been applied.
    use_corrected    BOOLEAN NOT NULL DEFAULT FALSE,
    image_path       TEXT,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (chart_id, page_name)
);
CREATE INDEX idx_page_list_chart_id ON page_list(chart_id);
CREATE INDEX idx_page_list_chart_page_number ON page_list(chart_id, page_number);
CREATE INDEX idx_page_list_image_sha256 ON page_list(image_sha256)
    WHERE image_sha256 IS NOT NULL;

CREATE TRIGGER trg_page_list_updated_at
    BEFORE UPDATE ON page_list
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Per-page, per-stage, per-pass progress. Replaces v6's 11 status columns.

-- Per-page, per-stage, per-pass progress. Replaces v6's 11 status columns.
CREATE TABLE page_stage_status (
    id             BIGSERIAL PRIMARY KEY,
    chart_id       BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id        BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,
    stage_name     VARCHAR(50) NOT NULL,
    pass_no        SMALLINT NOT NULL DEFAULT 1,
    status         VARCHAR(20) NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','processing','completed','failed','skipped')),
    attempt        INT NOT NULL DEFAULT 0,
    skip_reason    VARCHAR(50),
    error_message  TEXT,
    started_at     TIMESTAMPTZ,
    completed_at   TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, stage_name, pass_no)
);
CREATE INDEX idx_page_stage_status_chart_stage_pass ON page_stage_status(chart_id, stage_name, pass_no);
CREATE INDEX idx_page_stage_status_chart_status ON page_stage_status(chart_id, status);

CREATE TRIGGER trg_page_stage_status_updated_at
    BEFORE UPDATE ON page_stage_status
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- MANIFEST (client-provided member roster)
-- ---------------------------------------------------------------------
-- A manifest batch (metadata_R1_B1.csv) lists the members expected in a set of
-- charts, keyed by the client's RecordId.
--
-- record_id IS the chart name: chart_list.chart_name holds the same RecordId,
-- and the chart folder on disk is named after it. There is deliberately no
-- chart_id column — it would be a second spelling of the same fact, NULL until
-- ingest and needing a backfill. Join instead:
--
--     JOIN chart_list c ON c.chart_name = m.record_id
--
-- That also keeps a manifest loadable before its charts are ingested, which is
-- the normal case: the roster usually arrives first.
--
-- Name parts are stored separately because the verification rules match
-- first / middle / last independently — a single "member_name" string cannot
-- drive classify_two_word_name / classify_three_word_name.

CREATE TABLE manifest_member_list (
    id                  BIGSERIAL PRIMARY KEY,
    -- The client's RecordId, which is also chart_list.chart_name and the chart
    -- folder name. This is the only identity the table needs.
    record_id           VARCHAR(150) NOT NULL,

    member_name         VARCHAR(255) NOT NULL,
    first_name          VARCHAR(100),
    middle_name         VARCHAR(100),
    last_name           VARCHAR(100),
    member_dob          DATE,
    external_member_id  VARCHAR(100),

    run_id              VARCHAR(50),
    batch_id            VARCHAR(50),
    source_file         VARCHAR(255),
    -- Where this row came from: a blob path, or a local filesystem path when
    -- the manifest was loaded with --local or picked up by import-folder.
    -- Named source_blob_path in v7, which was wrong for the local case.
    source_path         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_manifest_member_list_record_id ON manifest_member_list(record_id);
CREATE INDEX idx_manifest_member_list_run_batch ON manifest_member_list(run_id, batch_id);

-- Upsert key 1: MemberID present.
CREATE UNIQUE INDEX idx_manifest_member_list_upsert_id
    ON manifest_member_list (record_id, external_member_id)
    WHERE external_member_id IS NOT NULL AND external_member_id <> '';

-- Upsert key 2: no MemberID — fall back to name + DOB. v6 documented this key
-- but never indexed it, so the fallback path was a seq scan and racy.
CREATE UNIQUE INDEX idx_manifest_member_list_upsert_name
    ON manifest_member_list (record_id, lower(member_name), COALESCE(member_dob, 'epoch'::date))
    WHERE external_member_id IS NULL OR external_member_id = '';

CREATE TRIGGER trg_manifest_member_list_updated_at
    BEFORE UPDATE ON manifest_member_list
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- Client page-level labels (the imaging spreadsheet). Keyed on chart folder
-- name + page file stem, not chart_id, so a load can precede ingest.
-- Id 1 matches 1.jpg / 1.png / 1.tif. Values stay as the client wrote them
-- ("Yes", "07/17/2025", "Accept") — the review UI compares them.
CREATE TABLE page_ground_truth (
    id                  BIGSERIAL PRIMARY KEY,
    chart_name          VARCHAR(150) NOT NULL,
    page_number         INT NOT NULL,
    source_page_id      VARCHAR(50),
    member_name         TEXT,
    member_dob          TEXT,
    dos_from            TEXT,
    dos_to              TEXT,
    encounter_type      TEXT,
    page_type           TEXT,
    codeable            TEXT,
    blank_page          TEXT,
    junk_page           TEXT,
    is_invoice          TEXT,
    page_sequence       TEXT,
    rotation            TEXT,
    is_visible          TEXT,
    rendering_provider  TEXT,
    provider_specialty  TEXT,
    provider_signature  TEXT,
    deleted_level       TEXT,
    source_path         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_page_ground_truth_chart_page UNIQUE (chart_name, page_number)
);
CREATE INDEX idx_page_ground_truth_chart_name ON page_ground_truth(chart_name);

CREATE TRIGGER trg_page_ground_truth_updated_at
    BEFORE UPDATE ON page_ground_truth
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ---------------------------------------------------------------------
-- PAGE-LEVEL RESULT TABLES
-- ---------------------------------------------------------------------

CREATE TABLE ocr_results (
    id           BIGSERIAL PRIMARY KEY,
    chart_id     BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id      BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,
    ocr_type     VARCHAR(30) NOT NULL CHECK (ocr_type IN ('tesseract','docling','azuredocintel')),
    raw_text     TEXT,
    char_count   INT,
    text_sha256  CHAR(64),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, ocr_type)
);
CREATE INDEX idx_ocr_results_chart_id ON ocr_results(chart_id);
CREATE INDEX idx_ocr_results_page_id ON ocr_results(page_id);

CREATE TRIGGER trg_ocr_results_updated_at
    BEFORE UPDATE ON ocr_results
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE TABLE ocr_quality_results (
    id                      BIGSERIAL PRIMARY KEY,
    chart_id                BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id                 BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,
    -- PAGE QUALITY GRADE — measured by quality_analyzer (0–10 → stored [0,1]).
    -- Tag thresholds: high ≥ 0.70, medium ≥ 0.40, else low.
    quality_tag             VARCHAR(10) CHECK (quality_tag IS NULL OR
                                               quality_tag IN ('high','medium','low')),
    quality_score           NUMERIC(5,4),
    -- Submetrics (0–10 scale) + warnings from the analyzer. Empty object when
    -- the analyzer did not run.
    quality_detail          JSONB NOT NULL DEFAULT '{}'::jsonb
                            CHECK (jsonb_typeof(quality_detail) = 'object'),
    input_dpi               NUMERIC(8,2),

    -- ConvNeXt / RF handwriting classifier. 'uncertain' = model abstained.
    printed_or_handwritten  VARCHAR(20) CHECK (printed_or_handwritten IN
                            ('printed','handwritten','mixed','uncertain')),
    hw_method               VARCHAR(30),
    hw_confidence           NUMERIC(5,4),

    orientation_angle       NUMERIC(6,2),
    tilt_angle              NUMERIC(6,2),
    mirrored                BOOLEAN,
    rotation_applied        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id)
);
CREATE INDEX idx_ocr_quality_results_chart_id ON ocr_quality_results(chart_id);
CREATE INDEX idx_ocr_quality_results_page_id ON ocr_quality_results(page_id);

CREATE TRIGGER trg_ocr_quality_results_updated_at
    BEFORE UPDATE ON ocr_quality_results
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- One row per page per pass. Exactly one row per page carries is_final, so
-- downstream stages never have to guess which pass won.

-- One row per page per pass. Exactly one row per page carries is_final, so
-- downstream stages never have to guess which pass won.
CREATE TABLE blank_junk_classification (
    id                BIGSERIAL PRIMARY KEY,
    chart_id          BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id           BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,
    pass_no           SMALLINT NOT NULL DEFAULT 1,
    blank_junk_flag   VARCHAR(20) CHECK (blank_junk_flag IN (
                          'blank','junk','duplicate','not_blank_junk'
                      )),
    -- Constrained to the label set review-ui renders (imaging_overlays.py).
    junk_subtype      VARCHAR(50) CHECK (junk_subtype IS NULL OR junk_subtype IN (
                          'Invoice','Cover Page','Record Request/Transmittal',
                          'Instructions','Letter/Fax','Others'
                      )),
    duplicate_of_page_id BIGINT REFERENCES page_list(id) ON DELETE SET NULL,
    ocr_source        VARCHAR(30) NOT NULL DEFAULT 'tesseract'
                      CHECK (ocr_source IN ('tesseract','docling','azuredocintel')),
    is_final          BOOLEAN NOT NULL DEFAULT FALSE,
    confidence        NUMERIC(5,4),
    reason            TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, pass_no)
);
CREATE INDEX idx_blank_junk_classification_chart_id ON blank_junk_classification(chart_id);
CREATE INDEX idx_blank_junk_classification_page_id ON blank_junk_classification(page_id);
CREATE UNIQUE INDEX idx_blank_junk_classification_one_final
    ON blank_junk_classification (page_id) WHERE is_final;

CREATE TRIGGER trg_blank_junk_classification_updated_at
    BEFORE UPDATE ON blank_junk_classification
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE OR REPLACE VIEW v_page_blank_junk_final AS
SELECT chart_id, page_id, blank_junk_flag, junk_subtype, duplicate_of_page_id,
       ocr_source, pass_no, confidence, reason, updated_at
FROM blank_junk_classification
WHERE is_final;


-- ONE table for date of service. The page-level and document-level values are
-- single-valued columns; every date the page carries lives in the multi-valued
-- `dates` field. v7 split this across dos_extraction_results +
-- dos_extraction_dates, which cost a DELETE plus one INSERT per date on every
-- page, and nothing ever read the child table back.
CREATE TABLE dos_extraction_results (
    id                              BIGSERIAL PRIMARY KEY,
    chart_id                        BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id                         BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,

    -- PAGE LEVEL — single value. The best/primary DOS for this page.
    date_of_service_from            DATE,
    date_of_service_to              DATE,

    -- DOCUMENT LEVEL — single value. The chart-wide DOS this page rolled into.
    date_of_service_from_doclevel   DATE,
    date_of_service_to_doclevel     DATE,

    -- MULTI-VALUED — every date found on the page, in the order found.
    -- A JSON array of objects whose keys mirror the column names above:
    --   [{"seq": 1,
    --     "date_of_service_from": "2024-03-15",
    --     "date_of_service_to":   "2024-03-15",
    --     "source_keyword":       "Date of Service",
    --     "confidence":           0.95}, ...]
    -- Empty array when the page carries none — never NULL.
    dates                           JSONB NOT NULL DEFAULT '[]'::jsonb
                                    CHECK (jsonb_typeof(dates) = 'array'),
    -- Derived, so it can never drift from `dates`.
    date_count                      INT GENERATED ALWAYS AS
                                    (jsonb_array_length(dates)) STORED,

    extraction_method               VARCHAR(20)
                                    CHECK (extraction_method IS NULL OR extraction_method IN
                                          ('rules','llm','rules+llm')),
    confidence                      NUMERIC(5,4),
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id)
);
CREATE INDEX idx_dos_extraction_results_chart_id ON dos_extraction_results(chart_id);
CREATE INDEX idx_dos_extraction_results_page_id ON dos_extraction_results(page_id);

CREATE TRIGGER trg_dos_extraction_results_updated_at
    BEFORE UPDATE ON dos_extraction_results
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- Codeable / non-codeable / discharge (page_subtype stage).
-- classification_category uses discharge_summary (schema name); UI shows
-- "Discharge Frequency". page_subtype holds the matched keyword page type.
CREATE TABLE page_classification (
    id                        BIGSERIAL PRIMARY KEY,
    chart_id                  BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id                   BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,
    page_subtype              VARCHAR(200),
    classification_category   VARCHAR(20) NOT NULL CHECK (classification_category IN (
                                  'codeable','non_codeable','discharge_summary'
                              )),
    duplicate_flag            BOOLEAN NOT NULL DEFAULT FALSE,
    confidence                NUMERIC(5,4),
    confidence_level          VARCHAR(10) CHECK (confidence_level IS NULL OR confidence_level IN (
                                  'high','medium','low'
                              )),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id)
);
CREATE INDEX idx_page_classification_chart_id ON page_classification(chart_id);
CREATE INDEX idx_page_classification_page_id ON page_classification(page_id);

CREATE TRIGGER trg_page_classification_updated_at
    BEFORE UPDATE ON page_classification
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- Encounter type: one label for all pages that share a page-level DOS.
CREATE TABLE encounter_type_results (
    id                BIGSERIAL PRIMARY KEY,
    chart_id          BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id           BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,
    encounter_type    VARCHAR(30) NOT NULL CHECK (encounter_type IN (
                          'outpatient_f2f',
                          'outpatient_tele',
                          'inpatient',
                          'home'
                      )),
    confidence        NUMERIC(5,4),
    matched_keyword   VARCHAR(200),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id)
);
CREATE INDEX idx_encounter_type_results_chart_id ON encounter_type_results(chart_id);
CREATE INDEX idx_encounter_type_results_page_id ON encounter_type_results(page_id);

CREATE TRIGGER trg_encounter_type_results_updated_at
    BEFORE UPDATE ON encounter_type_results
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- Suggested page order (does not reorder files on disk). Ported from
-- advantmed-document-processing sequencing (markers → streams → headers → CE).
CREATE TABLE page_sequencing_results (
    id                     BIGSERIAL PRIMARY KEY,
    chart_id               BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id                BIGINT NOT NULL REFERENCES page_list(id) ON DELETE CASCADE,
    original_page_number   INT,
    seq                    INT,
    confidence             NUMERIC(5,4),
    sequence_method        VARCHAR(64),
    review_flag            BOOLEAN NOT NULL DEFAULT FALSE,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id)
);
CREATE INDEX idx_page_sequencing_results_chart_id ON page_sequencing_results(chart_id);
CREATE INDEX idx_page_sequencing_results_page_id ON page_sequencing_results(page_id);

CREATE TRIGGER trg_page_sequencing_results_updated_at
    BEFORE UPDATE ON page_sequencing_results
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ---------------------------------------------------------------------
-- VERIFICATION & DECISION
-- ---------------------------------------------------------------------
-- Columns mirror the V1 Member_Verification contract so a ported run is
-- diffable against the reference implementation.

CREATE TABLE member_extraction_results (
    id                             BIGSERIAL PRIMARY KEY,
    chart_id                       BIGINT NOT NULL REFERENCES chart_list(id) ON DELETE CASCADE,
    page_id                        BIGINT REFERENCES page_list(id) ON DELETE CASCADE,

    extracted_name                 VARCHAR(255),
    extracted_dob                  DATE,
    extracted_member_id            VARCHAR(100),

    -- How each field was found: rule pass, NER pass, or not at all.
    detection_source_name          VARCHAR(20)
                                   CHECK (detection_source_name IS NULL OR
                                          detection_source_name IN ('rule_based','ner','')),
    detection_source_dob           VARCHAR(20)
                                   CHECK (detection_source_dob IS NULL OR
                                          detection_source_dob IN ('rule_based','ner','')),
    detection_source_member_id     VARCHAR(20)
                                   CHECK (detection_source_member_id IS NULL OR
                                          detection_source_member_id IN ('rule_based','ner','')),
    -- Which key sentence the NER pass read the value out of ("Patient Name:").
    ner_key_source_name            VARCHAR(100),
    ner_key_source_dob             VARCHAR(100),
    ner_key_source_member_id       VARCHAR(100),

    -- The V1 per-page verdict. Without wrong_member there is no reject path.
    page_status                    VARCHAR(20)
                                   CHECK (page_status IS NULL OR page_status IN
                                          ('verified','wrong_member','not_verified')),
    page_verified                  BOOLEAN,

    confidence                     NUMERIC(5,4),
    provided_name                  VARCHAR(255),
    provided_dob                   DATE,
    provided_external_member_id    VARCHAR(100),
    matched_member_list_id         BIGINT REFERENCES manifest_member_list(id) ON DELETE SET NULL,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id)
);
CREATE INDEX idx_member_extraction_results_chart_id ON member_extraction_results(chart_id);
CREATE INDEX idx_member_extraction_results_page_id ON member_extraction_results(page_id);
CREATE INDEX idx_member_extraction_results_matched ON member_extraction_results(matched_member_list_id);
CREATE INDEX idx_member_extraction_results_page_status ON member_extraction_results(chart_id, page_status);

CREATE TRIGGER trg_member_extraction_results_updated_at
    BEFORE UPDATE ON member_extraction_results
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE TABLE member_verification_summary (
    id                  BIGSERIAL PRIMARY KEY,
    chart_id            BIGINT NOT NULL UNIQUE REFERENCES chart_list(id) ON DELETE CASCADE,

    final_status        VARCHAR(20) NOT NULL
                        CHECK (final_status IN (
                            'verified','failed','needs_review','skipped'
                        )),
    -- Accept / Reject from what_if_rules: wrong-member pages >= reject_threshold.
    document_decision   VARCHAR(10)
                        CHECK (document_decision IS NULL OR document_decision IN ('accept','reject')),

    matched_member_list_id BIGINT REFERENCES manifest_member_list(id) ON DELETE SET NULL,
    matched_name        VARCHAR(255),
    name_mode           VARCHAR(2) CHECK (name_mode IS NULL OR name_mode IN ('2','3','')),

    confidence          NUMERIC(5,4),
    pages_checked       INT,
    pages_matched       INT,
    wrong_member_pages  INT NOT NULL DEFAULT 0,
    reject_threshold    INT,

    decision_reason     VARCHAR(50),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_member_verification_summary_updated_at
    BEFORE UPDATE ON member_verification_summary
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- Audit log today; a work queue when a worker process lands. lease_expires_at
-- and heartbeat_at are what a claim/lease loop needs.
CREATE TABLE pipeline_jobs (
    id                BIGSERIAL PRIMARY KEY,
    chart_id          BIGINT REFERENCES chart_list(id) ON DELETE CASCADE,
    stage_name        VARCHAR(50) NOT NULL,
    pass_no           SMALLINT NOT NULL DEFAULT 1,
    status            VARCHAR(20) NOT NULL DEFAULT 'queued'
                      CHECK (status IN ('queued','running','completed','failed','retrying')),
    queue_name        VARCHAR(100),
    attempt           INT NOT NULL DEFAULT 1,
    worker_id         VARCHAR(100),
    lease_expires_at  TIMESTAMPTZ,
    heartbeat_at      TIMESTAMPTZ,
    pages_total       INT,
    pages_done        INT,
    pages_failed      INT,
    pages_skipped     INT,
    queued_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    duration_seconds  NUMERIC GENERATED ALWAYS AS (
                          EXTRACT(EPOCH FROM (completed_at - started_at))
                      ) STORED,
    error_message     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_pipeline_jobs_updated_at
    BEFORE UPDATE ON pipeline_jobs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE INDEX idx_pipeline_jobs_chart_id ON pipeline_jobs(chart_id);
CREATE INDEX idx_pipeline_jobs_stage_status ON pipeline_jobs(stage_name, status);
CREATE INDEX idx_pipeline_jobs_claim ON pipeline_jobs(status, queue_name, queued_at)
    WHERE status IN ('queued','retrying');


-- ---------------------------------------------------------------------
-- DERIVED VIEWS
-- ---------------------------------------------------------------------

-- Per-chart, per-stage page rollup. The orchestrator's chart-status rule
-- ("earliest stage where not every page is completed|skipped") reads this.
CREATE OR REPLACE VIEW v_chart_stage_progress AS
SELECT
    c.id                                                       AS chart_id,
    s.stage_name,
    s.pass_no,
    s.seq,
    s.label,
    s.is_phase1,
    COUNT(pss.id)                                              AS rows_present,
    (SELECT COUNT(*) FROM page_list p WHERE p.chart_id = c.id) AS pages_total,
    COUNT(*) FILTER (WHERE pss.status = 'pending')             AS pending,
    COUNT(*) FILTER (WHERE pss.status = 'processing')          AS processing,
    COUNT(*) FILTER (WHERE pss.status = 'completed')           AS completed,
    COUNT(*) FILTER (WHERE pss.status = 'failed')              AS failed,
    COUNT(*) FILTER (WHERE pss.status = 'skipped')             AS skipped
FROM chart_list c
CROSS JOIN pipeline_stage s
LEFT JOIN page_stage_status pss
       ON pss.chart_id = c.id
      AND pss.stage_name = s.stage_name
      AND pss.pass_no = s.pass_no
GROUP BY c.id, s.stage_name, s.pass_no, s.seq, s.label, s.is_phase1;
