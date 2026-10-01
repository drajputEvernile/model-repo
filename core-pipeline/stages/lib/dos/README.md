# DOS

The dates are found by the key/value extraction (`../extraction`), staged, and resolved
here: `resolve.py` decides which encounter each page belongs to (progress-note spans,
default date for demographics/injection pages); `stage.py` writes one DB row and one CSV
row per page (`imaging/<chart>_dos.csv`). Page types, the default date and the span
override score live in `../keyword-canon/dos_canon.json` and reload on change.

```csv
chart_name,page_name,page_number,dos_from,dos_to,dos_from_iso,dos_to_iso,doc_dos_from,doc_dos_to,doc_dos_from_iso,doc_dos_to_iso,match_type,keyword,confidence,extraction_method
```
