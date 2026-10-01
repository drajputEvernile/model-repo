# Modular API design

Design rules and proposed endpoint split for selling / toggling pipeline
capabilities as independent modules (standalone or composed per project).

This is a **target shape**. Today the product still runs via monolithic
`POST /api/charts/run` and `batch-run` over the stage chain. New module
routes should follow these rules so projects can enable a subset without
rewriting orchestration.

---

## Design rules

1. **One module = one capability** with clear inputs and outputs (OCR text,
   blank/junk verdict, member fields, DOS dates, …).
2. **`model` (or `engine`) selects the implementation** inside that module
   (`tesseract`, `docling`, `ner`, `azure_openai`, `rules`, …).
3. **POST = run** — prefer async `202` + job id for chart-scale work; sync
   is acceptable for small single-page calls.
4. **GET = results** — fetch stored output by `chart_id` / `page_id` /
   `job_id` (and `model` when multiple engines wrote rows).
5. **Modules never call each other.** Orchestration is a separate thin
   layer (today’s stage chain / a future `POST /pipeline/run`).
6. **Project config** = which modules are enabled + default `models` map.
   The same module can ship alone (`POST /ocr` only) or as part of a chain.
7. **Degraded runs stay visible** — missing optional deps stamp what
   actually ran (`pages_no_word_boxes`, `extraction.ready=false`)
   rather than failing silently.
8. **One bad page must not sink a chart** — per-page errors are recorded;
   the module continues.

---

## Major modules / endpoints

### 1. `POST` / `GET` `/ocr`

Page text extraction only.

| `model` | Today’s engine |
|---|---|
| `tesseract` | Preliminary OCR |
| `docling` / `rapidocr` | Final OCR 1 |
| `azure_docintel` | Final OCR 2 (billed per page) |

- **POST** — pages or chart + `model` + options  
- **GET** — `?chart_id=&page_id=&model=` → text + structured JSON  
- **Toggle** — which OCR engines a project may use  

### 2. `POST` / `GET` `/quality`

Rotation + handwriting (imaging prep, not OCR).

- Future `model`s: `hw_classifier`, `heuristic`, …  
- **Out** — orientation, printed/handwritten, quality placeholders  
- **Today** — pipeline stage 1  

### 3. `POST` / `GET` `/classification/blank-junk`

Blank / junk / duplicate.

- Optional `pass=1|2`  
- **Out** — flag, junk subtype, confidence  
- **Toggle** — on/off; keyword packs per project  

### 4. `POST` / `GET` `/classification/page-type`

Codeable / non-codeable page type.

- **Model today** — keyword families (`codeable_canon.json`): word-boundary hits, header-weighted, margin confidence, family spans  
- **Out** — `page_type`, tag (codeable / non-codeable / discharge)  
- **Toggle** — project-specific canon  

### 5. `POST` / `GET` `/classification/encounter`

Encounter type (IP / OP / F2F / …).

- **Model** — tiered evidence per visit (page type → setting text → hints), no carry-forward  
- **Toggle** — encounter canon per client  

### 6. `POST` / `GET` `/member`

Identity extract; verify as a sibling resource.

| `model` | Role |
|---|---|
| `kv_extract` | The key/value extraction (rules find candidates, the trained version picks; GLiNER reads the sentences) |

Suggested shape:

- `POST /member/extract` — fields from page word boxes  
- `POST /member/verify` — compare to manifest (needs `/manifest`)  
- `GET /member?chart_id=` — extraction + summary  

**Toggle** — none; the extraction model version is pinned to v002.

### 7. `POST` / `GET` `/dos`

Dates of service.

| `model` | Role |
|---|---|
| `kv_extract` | The extraction's chosen dates, resolved across the chart |

- **Out** — page-level + doc-level dates (JSONB)  
- **Toggle** — the extraction model version  

### 8. `POST` / `GET` `/headers`

Section-header match over existing OCR JSON (cheap re-run; no OCR).

- **Model** — MiniLM / embedding matcher  
- **Toggle** — header canon per project  

### 9. `POST` / `GET` `/sequencing`

Page order / visit grouping.

- **Models** — rules; optional cross-encoder  
- **Toggle** — on/off  

### 10. Supporting (not ML modules, but required)

| Endpoint | Purpose |
|---|---|
| `POST` / `GET` `/charts` | Chart registry, status, run/batch ids |
| `POST` / `GET` `/manifest` | Expected member identity |
| `POST` `/pipeline` or `/jobs` | Compose selected modules for a chart |
| `GET` `/health` | Which modules / models are actually ready |

---

## Orchestration (project toggles)

```http
POST /pipeline/run
Content-Type: application/json

{
  "chart_id": "...",
  "modules": ["quality", "ocr", "blank-junk", "member", "dos"],
  "models": {
    "ocr": ["tesseract", "docling"],
    "member": "ner",
    "dos": "rules"
  }
}
```

- **`modules`** — ordered (or dependency-resolved) list to run.  
- **`models`** — per-module engine choice(s).  
- Omitting a module = that capability is off for the project/run.

---

## Short client-facing list

If the API surface should stay small for sales / integration docs:

1. **`/ocr`** — engines as `model`  
2. **`/quality`** — rotation / handwriting  
3. **`/page-classification`** — blank-junk + page-type (+ encounter)  
4. **`/member`** — extract + verify; `model=ner|azure_openai|rules`  
5. **`/dos`** — `model=rules|azure_openai`  
6. **`/pipeline`** — compose the above  

That maps to the existing stage chain without forcing every client through a
monolithic `/run`.

---

## Mapping to today’s stages

| Module | Current stage(s) |
|---|---|
| `/quality` | `quality_rotation_hw` |
| `/ocr` (`tesseract`) | `ocr_prelim` |
| `/classification/blank-junk` | `blank_junk` pass 1 & 2 |
| `/ocr` (`docling`) | `ocr_final1` |
| `/ocr` (`azure_docintel`) | `ocr_final2` |
| `/headers` | `kv_extract` |
| `/member` | `member_extract_verify` |
| `/dos` | `dos_extract` |
| `/classification/page-type` | `page_subtype` |
| `/classification/encounter` | `encounter_type` |
| `/sequencing` | `page_sequencing` |

See also [`FLOW.md`](FLOW.md), [`LOGIC.md`](LOGIC.md), [`API.md`](API.md).
