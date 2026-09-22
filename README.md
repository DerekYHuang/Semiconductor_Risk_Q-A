# Silicon Valley Semiconductor Contamination & Vapor-Intrusion Risk Q&A (Local RAG + Fine-Tuning)

A locally-run retrieval-augmented generation (RAG) system, with an optional LoRA fine-tuning
step, that answers questions about semiconductor-linked groundwater contamination and
vapor-intrusion risk in Santa Clara County — grounded in real EPA/DTSC/RWQCB regulatory
documents rather than a model's general knowledge.

Everything runs locally: no cloud, no paid API keys. Embeddings via a small Hugging Face
model, vector search via Chroma, generation via a small open-weight LLM served through Ollama.

---

## 1. Why I'm looking at this problem

Santa Clara County has more EPA Superfund sites than any other county in the United States —
most of them legacy semiconductor and electronics manufacturing facilities where solvents like
trichloroethylene (TCE) were used to clean chips and leaked into groundwater from the 1960s
through the 1980s. Decades later, many of these plumes are still being monitored, and TCE vapor
can migrate up through soil into homes, schools, and offices built on or near old fab sites —
this is "vapor intrusion."

I'm building this because:
- I'm targeting data scientist / data engineer roles specifically in the Bay Area
  semiconductor and data center sectors, and this is the industry's own environmental legacy —
  understanding it makes me a more informed candidate, not just a portfolio checkbox.
- The underlying documents (Superfund records, DTSC EnviroStor case files, RWQCB GeoTracker
  monitoring reports) are long, jargon-heavy, and scattered across three different government
  systems. That's a genuine information-access problem — the kind RAG is actually built for,
  not a toy use case.
- It connects directly to my Data Center Energy & Water Footprint Tracker project: both are
  about the physical/environmental footprint of the same industry in the same county, just at
  different points in that industry's lifecycle (today's data center water/power draw vs.
  yesterday's fab-site groundwater contamination).

## 2. Why I chose this data

| Source | What it is | Why it's useful here |
|---|---|---|
| EPA Superfund NPL (Envirofacts / CIMC) | Official National Priorities List site records, including site narratives and cleanup status | Authoritative, public, well-structured — same EPA data family I already used in the footprint tracker |
| DTSC EnviroStor | California's state-level contaminated-site tracker, includes non-NPL sites the federal list misses | Fills in the "smaller" former fab sites that never made the federal Superfund list but still matter locally |
| RWQCB GeoTracker | Regional Water Quality Control Board's groundwater monitoring and cleanup database | Has the actual technical monitoring reports (contaminant concentrations over time) — the most detailed and hardest-to-read documents, which is exactly where RAG earns its keep over a static table |
| Vapor intrusion guidance / mitigation reports (EPA & DTSC) | Public guidance docs and site-specific vapor mitigation status reports | Lets the system answer the human-impact question ("is this site's risk being addressed?") not just the regulatory-status question |

All four sources are public records, text-heavy (PDFs, HTML case pages), and none require an
API key. The final corpus: 7 Superfund site records, 6 EnviroStor case summaries, 3 GeoTracker
monitoring report sets, and 4 vapor intrusion guidance documents (23 source files total).

## 3. Approach

1. **Baseline RAG** — chunk source documents, embed with a small open-source embedding model,
   store in a local vector DB (Chroma), retrieve top-k chunks, generate an answer with a small
   local LLM (via Ollama).
2. **Build a hand-written evaluation set** (27 Q&A pairs sourced directly from the documents,
   14 currently filled in) *before* trying to improve anything — this is what turns the project
   from "I built a chatbot" into "I built and measured a system."
3. **Score retrieval and generation separately.** A question can fail two different ways: the
   wrong document gets retrieved (`retrieval_hit = False`), or the right document is retrieved
   but the model still answers wrong or refuses (`retrieval_hit = True`, `judged_correct =
   False`). Logging both, plus the full retrieved chunk text per question, is what makes it
   possible to tell which stage to fix.
4. **Iterate against the eval set** — chunk-size variations, embedding model choice, top_k,
   hybrid retrieval. Track before/after numbers for each change.
5. **(Stretch) LoRA fine-tune** the local LLM on domain Q&A pairs using Hugging Face PEFT, and
   compare fine-tuned vs. base-model accuracy on the same eval set.

Full local stack:
- Embeddings: `BAAI/bge-small-en-v1.5` (Hugging Face, CPU-friendly)
- Vector store: Chroma (local, file-based)
- LLM: `llama3.2:3b` via Ollama, run on CPU (the machine's GPU driver has a CUDA
  toolchain mismatch with Ollama's bundled runtime — documented in "Known issues" below
  rather than worked around silently)
- Fine-tuning: Hugging Face PEFT (LoRA) — not yet attempted
- Eval: custom retrieval + answer-correctness metrics with refusal detection
  (`src/eval/metrics.py`)

## 4. How I actually cleaned and ran everything

**Document counts and formats actually pulled:** 23 source documents across four categories —
7 Superfund site narratives (HTML/text, saved from individual EPA `csitinfo.cfm` pages), 6
DTSC EnviroStor case summaries, 3 GeoTracker monitoring report sets (PDF), and 4 vapor
intrusion guidance documents (a mix of PDF technical guidance and an HTML overview page).

**Chunking:** started at 400 tokens / 60 overlap (`config.yaml`). Most source documents are
short — 1 to 4 KB of parsed text, meaning many documents produce only 1-3 chunks at this
setting. This is a known limitation worth testing against smaller chunk sizes; not yet done.

**Retrieval configuration:** `top_k = 5`, embedding model `BAAI/bge-small-en-v1.5`. Not yet
tuned against alternatives.

**A metrics bug found and fixed during the first baseline run:** the original correctness
metric was pure keyword overlap between the generated answer and the reference answer. On the
first run, a question where the model explicitly said *"I couldn't find the information..."*
still scored 0.6 overlap and was marked correct — because the refusal happened to restate
several keywords from the question itself (site/company names) while providing zero actual
answer. Fixed by adding an explicit refusal-phrase check (`is_refusal()` in `metrics.py`) that
overrides the keyword score: any answer containing a refusal phrase is now always scored
incorrect, regardless of overlap.

**A hallucination found during evaluation, not fixed:** EPA's own public sources disagree with
each other on two facts in this corpus (Intel's site size is given as both 4-acre and 1-acre
across two different EPA documents; Synertek's site is described as both 1.5-acre/one-building
and 3.5-acre/five-buildings). When asked about the Synertek site size, the model responded with
a third figure — "1-acre, one building" — matching neither EPA source, while explicitly
claiming *"I verified the information in the saved file."* This is logged as a known finding
(see eval row `q012`'s `note` field) rather than removed from the eval set, since a model
confidently fabricating a number while claiming verification is exactly the kind of failure
this project is supposed to surface.

**Eval tooling:** `run_eval.py` writes both a summary CSV and a `_chunks.jsonl` file containing
the full retrieved chunk text for every question, so a failure can be diagnosed as a retrieval
problem (wrong/missing chunk) vs. a generation problem (right chunk, wrong answer) rather than
guessed at.

[TODO: once the remaining eval rows (q015-q027) are filled in and a chunk-size/top_k tuning
pass is run, document the final configuration and what changed, e.g. "increasing top_k from 5
to 8 improved retrieval hit rate on X category from Y% to Z%."]

## 5. Results — answering back to the problem

**Baseline run (7 questions, original scoring):** 85.7% retrieval hit rate, 57.1% answer
correctness.

**Baseline run, rescored after the refusal-detection fix (same 7 questions):** answer
correctness dropped to 42.9% — this drop is a *correction*, not a regression. One previously
"correct" answer (Applied Materials site size) was actually a refusal that had gamed the old
keyword-overlap metric.

**Second run (14 questions, expanded eval set):** retrieval hit rate and correctness numbers
tracked per-category in `outputs/eval_results/`. [TODO: paste the summary table from the most
recent `eval_<run_name>.csv` run here once q015-q027 are filled in and a full run is done.]

**What the system gets right:** questions with a single, cleanly-stated fact in a short
document — dates (Intel's NPL removal date, TRW's listing date), named entities (the Triple
Site's three companies), site-specific figures with no cross-document ambiguity (JASCO's
acreage and operating years, Spectra-Physics's acreage) — are answered correctly and cite the
source document. Example: asked what two companies join AMD in the Triple Site, the system
correctly answered "TRW Microwave, Inc. and Signetics Inc.," citing both source documents.

**Where it breaks down:**
1. **Refusals on facts that may not be in the saved document at all** (National Semiconductor's
   contaminant list, the Los Paseos health study) — likely means those specific facts weren't
   present in the *EnviroStor/Superfund page actually saved*, since EnviroStor case pages tend
   to be shorter status summaries rather than full narratives. Not yet confirmed by checking the
   raw parsed text directly against the retrieved chunks.
2. **A retrieval miss on a well-known fact** (National Semiconductor's 300,000-person drinking
   water estimate) — the model listed the *other* sites it *did* retrieve, none of which was
   National Semiconductor, suggesting either an embedding/chunking issue or that the fact isn't
   in the saved document.
3. **A hallucinated figure presented as verified** (Synertek site size — see section 4). This is
   the most concerning failure mode found so far, since it's confident and wrong rather than an
   honest refusal.

**Honest limitations:**
- Coverage gaps in EnviroStor, GeoTracker, and vapor-intrusion guidance documents — 13 of 27
  eval questions are still unfilled TODOs, so current metrics are not yet a full picture of the
  system.
- Two facts in the corpus have genuine source conflicts in EPA's own public data (not a system
  bug, but something the eval set has to account for — see `note` fields on q008 and q012).
- No retrieval/chunking tuning has been done yet — current numbers are an unoptimized baseline.
- GPU acceleration is currently disabled (CUDA toolchain mismatch between this machine's driver
  and Ollama's bundled runtime); all runs use CPU inference, which is slower but does not affect
  correctness.

---

## Project structure

```
semiconductor-contamination-rag/
├── data/
│   ├── raw/                  # YOU populate this — see data/raw/README.md in each subfolder
│   │   ├── superfund/
│   │   ├── envirostor/
│   │   ├── geotracker/
│   │   └── vapor_intrusion/
│   ├── processed/
│   │   ├── parsed/           # output of src/ingest
│   │   ├── chunks/           # output of src/chunk
│   │   └── vector_store/     # Chroma DB lives here
│   └── eval/
│       ├── eval_set.jsonl    # hand-written Q&A eval set — question/answer/source_doc/category,
│       │                     # plus an optional "note" field for known caveats (source
│       │                     # conflicts, documented failure modes) — never put caveats in
│       │                     # the "question" field itself, since that text gets sent to the LLM
│       └── README.md         # guidance on writing good eval rows
├── src/
│   ├── ingest/                # parse raw PDFs/HTML into clean text
│   ├── chunk/                 # split cleaned text into chunks
│   ├── embed/                 # build the Chroma vector store
│   ├── retrieval/             # RAG query pipeline + CLI
│   ├── eval/                  # eval harness + metrics (retrieval hit, refusal-gated
│   │                           correctness, per-question chunk-level debug logging)
│   ├── finetune/               # LoRA fine-tuning (stretch goal, not yet attempted)
│   └── app/                   # Streamlit demo UI
├── outputs/
│   ├── eval_results/          # eval run outputs: eval_<run>.csv (summary) and
│   │                           # eval_<run>_chunks.jsonl (full retrieved-chunk detail per
│   │                           # question, for diagnosing retrieval vs. generation failures)
│   └── figures/                # any charts for your portfolio writeup
├── requirements.txt
├── config.yaml
└── README.md
```

## Setup

```powershell
# from your project root
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# install Ollama separately: https://ollama.com/download
# then, in a dedicated terminal window (leave it running):
ollama serve
ollama pull llama3.2:3b
```

## Getting the data (you do this step)

See `data/raw/*/README.md` in each subfolder for exact source links and what to download.
None of these require an account or API key. Save raw PDFs/HTML as-is into the matching
subfolder — `src/ingest/` handles the parsing.

## Run order

```powershell
python src\ingest\parse_documents.py
python src\chunk\chunk_documents.py
python src\embed\build_vector_store.py
python src\retrieval\query_cli.py --question "your question here"
python src\eval\run_eval.py --run_name baseline
streamlit run src\app\streamlit_app.py
```

## Known issues

- **GPU/CUDA:** on this machine, running Ollama with GPU acceleration enabled crashes with
  `CUDA error: the provided PTX was compiled with an unsupported toolchain` — a driver/runtime
  version mismatch, not a project bug. Workaround: run `ollama serve` with `CUDA_VISIBLE_DEVICES`
  set to `-1` (forces CPU-only inference) in whatever terminal launches it, before starting the
  pipeline scripts. A permanent NVIDIA driver update would likely resolve this if GPU speed
  becomes worth the effort later.