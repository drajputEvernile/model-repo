-- Key/value extraction replaces the section-header stage.
--
-- One stage right after OCR now finds the headings (and the member, DOS, page-number,
-- provider and e-signature fields the later stages read), so `section_headers` is renamed
-- in place: same position (seq 55), new name. No table changes — nothing the pipeline
-- stores has moved.
--
-- Safe to run twice, and on a database that never had `section_headers`.

UPDATE pipeline_stage
   SET stage_name = 'kv_extract',
       label      = 'Key/Value Extraction',
       updated_at = now()
 WHERE stage_name = 'section_headers'
   AND pass_no = 1
   AND NOT EXISTS (SELECT 1 FROM pipeline_stage WHERE stage_name = 'kv_extract' AND pass_no = 1);

-- Charts already in the database keep their per-page progress under the new name. A page
-- that has both rows (rename applied by hand earlier) keeps the new one.
UPDATE page_stage_status s
   SET stage_name = 'kv_extract'
 WHERE s.stage_name = 'section_headers'
   AND NOT EXISTS (
       SELECT 1 FROM page_stage_status n
        WHERE n.page_id = s.page_id AND n.stage_name = 'kv_extract' AND n.pass_no = s.pass_no
   );
DELETE FROM page_stage_status WHERE stage_name = 'section_headers';
