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

[TODO after Phase 1: 2-3 sentences on the specific question(s) I ended up focusing the eval set
on, once I've actually read through the source documents — e.g. "which neighborhoods have
active vapor mitigation systems," "which sites have the longest-running unresolved plumes," etc.]

## 2. Why I chose this data

| Source | What it is | Why it's useful here |
|---|---|---|
| EPA Superfund NPL (Envirofacts / CIMC) | Official National Priorities List site records, including site narratives and cleanup status | Authoritative, public, well-structured — same EPA data family I already used in the footprint tracker |
| DTSC EnviroStor | California's state-level contaminated-site tracker, includes non-NPL sites the federal list misses | Fills in the "smaller" former fab sites that never made the federal Superfund list but still matter locally |
| RWQCB GeoTracker | Regional Water Quality Control Board's groundwater monitoring and cleanup database | Has the actual technical monitoring reports (contaminant concentrations over time) — the most detailed and hardest-to-read documents, which is exactly where RAG earns its keep over a static table |
| Vapor intrusion guidance / mitigation reports (EPA & DTSC) | Public guidance docs and site-specific vapor mitigation status reports | Lets the system answer the human-impact question ("is this address at risk?") not just the regulatory-status question |

All four sources are public records, text-heavy (PDFs, HTML case pages), and none require an
API key — good fit for a from-scratch RAG pipeline where I control the whole path from raw
document to answer. This also reuses the data-sourcing muscle from the footprint tracker
project instead of starting from zero.

## 3. Approach

1. **Baseline RAG** — chunk source documents, embed with a small open-source embedding model,
   store in a local vector DB (Chroma), retrieve top-k chunks, generate an answer with a small
   local LLM (via Ollama).
2. **Build a hand-written evaluation set** (20-30 Q&A pairs sourced directly from the documents)
   *before* trying to improve anything — this is what turns the project from "I built a chatbot"
   into "I built and measured a system," which is the actual skill employers are checking for.
3. **Iterate against the eval set** — try chunk-size variations, a different embedding model,
   hybrid keyword+vector retrieval, and/or a reranker. Track before/after numbers for each change.
4. **(Stretch) LoRA fine-tune** the local LLM on domain Q&A pairs using Hugging Face PEFT, and
   compare fine-tuned vs. base-model accuracy on the same eval set.
5. Ship a small local CLI/Streamlit interface — no deployment infrastructure needed, this is a
   portfolio project, not a production service.

Full local stack:
- Embeddings: `BAAI/bge-small-en-v1.5` (Hugging Face, CPU-friendly)
- Vector store: Chroma (local, file-based)
- LLM: `llama3.2:3b` or `phi3:mini` via Ollama (runs on CPU)
- Fine-tuning: Hugging Face PEFT (LoRA)
- Eval: custom retrieval + answer-correctness metrics (see `src/eval/metrics.py`)

## 4. How I actually cleaned and ran everything

[TODO: fill this in as you go — this section should read like a methods section, not marketing
copy. Suggested things to capture once you've done them:]

- [TODO: document counts and formats actually pulled from each source — e.g. "14 Superfund
  site narratives (HTML), 9 GeoTracker monitoring reports (PDF), 3 DTSC EnviroStor case
  summaries"]
- [TODO: any cleaning quirks specific to these sources — GeoTracker PDFs in particular tend to
  have table-heavy monitoring data that doesn't extract cleanly; note whatever workaround you
  used]
- [TODO: final chunk size/overlap settings you landed on, and why — link back to the eval
  numbers that justified the choice]
- [TODO: which retrieval configuration won, and by how much, vs. the naive baseline]
- [TODO: if you did the fine-tuning stretch step, note training set size, LoRA rank/config, and
  training time on your hardware]

## 5. Results — answering back to the problem

[TODO: fill in after Phase 3 evaluation. Suggested structure:]

**Retrieval quality:** [TODO: precision/recall or hit-rate @k, baseline vs. best config]

**Answer correctness on the hand-written eval set:** [TODO: e.g. "X/30 correct with baseline
config, Y/30 after tuning chunking/retrieval"]

**Fine-tuning delta (if attempted):** [TODO: base model vs. fine-tuned accuracy on the same
eval set]

**What the system actually surfaces:** [TODO: 2-3 concrete example Q&A pairs from the eval set
that show the system correctly grounding an answer in a specific source document — this is the
most compelling thing to show in an interview, more than the aggregate metric]

**Honest limitations:** [TODO: e.g. document coverage gaps, sites the system can't answer about,
known failure modes you found during eval]

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
│   │   ├── chunks/           # output of src/chunk
│   │   └── vector_store/     # Chroma DB lives here
│   └── eval/
│       └── eval_set.jsonl    # your hand-written Q&A eval set — template included
├── src/
│   ├── ingest/                # parse raw PDFs/HTML into clean text
│   ├── chunk/                 # split cleaned text into chunks
│   ├── embed/                 # build the Chroma vector store
│   ├── retrieval/             # RAG query pipeline + CLI
│   ├── eval/                  # eval harness + metrics
│   ├── finetune/              # LoRA fine-tuning (stretch goal)
│   └── app/                   # Streamlit demo UI
├── outputs/
│   ├── eval_results/          # eval run outputs (json/csv), used to fill README section 5
│   └── figures/                # any charts for your portfolio writeup
├── requirements.txt
├── config.yaml
└── README.md
```

## Setup

```powershell
# from G:\semiconductor-contamination-rag (or wherever you clone this)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# install Ollama separately: https://ollama.com/download
ollama pull llama3.2:3b
```

## Getting the data (you do this step)

See `data/raw/*/README.md` in each subfolder for exact source links and what to download.
None of these require an account or API key. Save raw PDFs/HTML as-is into the matching
subfolder — `src/ingest/` handles the parsing.

## Run order

```powershell
python src/ingest/parse_documents.py
python src/chunk/chunk_documents.py
python src/embed/build_vector_store.py
python src/retrieval/query_cli.py --question "your question here"
python src/eval/run_eval.py
streamlit run src/app/streamlit_app.py
```
