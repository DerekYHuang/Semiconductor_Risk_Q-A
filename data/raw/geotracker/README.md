# RWQCB GeoTracker monitoring records

**What to get:** Groundwater monitoring reports — these are the most technical documents in
the corpus (contaminant concentrations over time, well data) and the best test of whether
your chunking/retrieval setup can handle dense tabular text.

**Where:**
- GeoTracker public search: https://geotracker.waterboards.ca.gov/
  — search by county (Santa Clara) and case type ("Cleanup Program Site" is the relevant filter
  for former industrial/semiconductor sites)
- Each site has a document archive tab — download a handful of recent monitoring reports (PDF)
  per site. You don't need the full history, 2-3 recent reports per site is enough.

**How to save:** Keep as PDF, one subfolder per site if you're pulling multiple reports, e.g.
`geotracker/sitename/report_2024.pdf`.

**Note on quality:** These PDFs are often scanned tables or have inconsistent formatting.
`src/ingest/parse_documents.py` uses `pypdf` for text extraction, which will sometimes mangle
tables — this is expected and worth noting honestly in README section 4 rather than hiding it.

**Roughly how many:** 5-8 site report sets is plenty.
