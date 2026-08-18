# Superfund NPL site records

**What to get:** Site narratives and status summaries for Santa Clara County NPL sites
(e.g. the Middlefield-Ellis-Whisman area, Fairchild Semiconductor sites, Intel sites,
National Semiconductor, Signetics, etc.).

**Where:**
- EPA Superfund Site Search: https://cumulis.epa.gov/supercpad/CurSites/srchsites.cfm
  — filter by state (CA) and county (Santa Clara)
- EPA Envirofacts Superfund data: https://www.epa.gov/enviro/envirofacts-data-service-api
  (you already have experience pulling from EPA/TRI data via the footprint tracker project —
  same general API family)
- Each site's EPA cleanup page has a downloadable/copyable narrative + site progress profile —
  save these as `.html` or copy the text into a `.txt` file per site.

**How to save:** One file per site, named like `sitename_superfund.txt` or `.html`.
`src/ingest/parse_documents.py` will pick up both formats from this folder.

**Roughly how many:** 8-15 site records is plenty for a first pass — you don't need every
site in the county, just enough for the eval set to have real variety.
