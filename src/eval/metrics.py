"""
Metrics for scoring the RAG pipeline against the hand-written eval set.

Two things are measured separately, on purpose:
  1. Retrieval quality — did we pull back the right source document at all?
  2. Answer correctness — given what was retrieved, was the generated answer right?

Keeping these separate matters: if answers are wrong, this tells you whether to fix
retrieval (chunking/embedding/top_k) or generation (prompt/model), instead of guessing.
"""

# Phrases the model uses when it's declining to answer. If any of these appear,
# the answer is scored incorrect regardless of keyword overlap — otherwise a refusal
# that happens to echo the question's own keywords back can score as a false positive.
# (Found during the first baseline run: a refusal scored 0.6 overlap and was marked
# correct because it repeated site/company names from the question itself while saying
# it had no information.)
REFUSAL_PHRASES = [
    "couldn't find",
    "could not find",
    "i don't have",
    "i do not have",
    "no information",
    "not mentioned in",
    "cannot find",
    "can't find",
    "unable to find",
]


def is_refusal(generated_answer: str) -> bool:
    lower = generated_answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def retrieval_hit(retrieved_chunks: list[dict], expected_source_doc: str) -> bool:
    """Did the expected source document appear anywhere in the retrieved chunks?"""
    retrieved_sources = {c["source_doc"] for c in retrieved_chunks}
    # expected_source_doc from the eval set may not have the ingest-time prefix
    # (e.g. "superfund__") — match on substring so eval authoring stays simple.
    return any(expected_source_doc in src for src in retrieved_sources)


def keyword_overlap_score(generated_answer: str, reference_answer: str) -> float:
    """
    Simple, transparent correctness proxy: fraction of reference-answer keywords
    (words longer than 3 chars) that appear in the generated answer.

    This is intentionally simple and inspectable rather than another model grading
    another model — good enough for a portfolio-scale eval set, and you can always
    manually spot-check disagreements since the eval set is only ~20-30 rows.

    Note: this alone is not sufficient — see is_refusal() and score_result() below.
    A refusal can still score nonzero overlap if it happens to restate keywords
    from the question. Always gate on is_refusal() before trusting this score.
    """
    reference_words = {w.lower().strip(".,;:") for w in reference_answer.split() if len(w) > 3}
    if not reference_words:
        return 0.0
    generated_lower = generated_answer.lower()
    hits = sum(1 for w in reference_words if w in generated_lower)
    return hits / len(reference_words)


def score_result(result: dict, eval_row: dict, correctness_threshold: float = 0.4) -> dict:
    """Score a single pipeline result against its eval-set row."""
    hit = retrieval_hit(result["retrieved_chunks"], eval_row["source_doc"])
    overlap = keyword_overlap_score(result["answer"], eval_row["answer"])
    refusal = is_refusal(result["answer"])
    correct = (overlap >= correctness_threshold) and not refusal

    return {
        "id": eval_row["id"],
        "category": eval_row.get("category", "uncategorized"),
        "retrieval_hit": hit,
        "keyword_overlap": round(overlap, 3),
        "is_refusal": refusal,
        "judged_correct": correct,
    }