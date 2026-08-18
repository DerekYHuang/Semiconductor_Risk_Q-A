# Building the evaluation set

`eval_set.jsonl` has 5 placeholder rows to show the format. **Expand this to 20-30 rows before
you start tuning anything** — this file is what turns the project from a demo into a measured
system, and it needs to exist before you optimize, not after.

## How to write good rows

- Write each question **after actually reading the source document**, not from memory of the
  general topic. The point is to test whether the system retrieves and grounds correctly, so
  the answer needs to be verifiably "in there."
- Prefer specific, checkable facts over vague ones: "What contaminant was detected at
  [site] in the [year] monitoring report?" is checkable. "Is [site] dangerous?" is not.
- Spread questions across categories so a weak spot doesn't hide in the average:
  - `site_status` — regulatory status, cleanup phase
  - `contaminant_type` — which chemical(s), concentrations
  - `vapor_mitigation` — whether/how vapor intrusion is being addressed
  - `timeline` — dates, duration of monitoring/cleanup
  - `regulatory_agency` — which agency (EPA/DTSC/RWQCB) is responsible
- Include a few questions the system *should* fail or decline to answer — e.g. asking about a
  site or detail that isn't in your corpus at all. A system that confidently makes something up
  when it shouldn't know is a finding worth reporting, not a bug to hide.

## Fields

- `id` — short unique identifier
- `question` — the question text
- `answer` — the correct answer, in your own words, with enough specificity to grade against
- `source_doc` — which raw document (filename) the answer comes from, for retrieval-precision scoring
- `category` — one of the categories above
