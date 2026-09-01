"""
Run the full eval set through the RAG pipeline and report retrieval + answer metrics.

Run:
    python src/eval/run_eval.py --run_name baseline

Each run writes two files to outputs/eval_results/:
  - eval_<run_name>.csv        summary metrics per question (for tracking runs over time)
  - eval_<run_name>_chunks.jsonl   full retrieved chunk text per question (for debugging
                                    *why* a question failed -- was the right chunk retrieved
                                    at all, and if so, did it actually contain the fact?)

Read the _chunks.jsonl file whenever a question fails retrieval_hit or judged_correct --
it shows you exactly what the model saw, which tells you whether to fix retrieval
(chunking/embedding/top_k) or fix the eval question itself (fact not actually in the
saved document).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "retrieval"))
from rag_pipeline import RAGPipeline  # noqa: E402

from metrics import score_result  # noqa: E402


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        # skip blank lines and comment lines -- JSONL has no native comment syntax,
        # but it's easy to accidentally leave these in when hand-editing the file
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        rows.append(json.loads(line))

    real_rows = [r for r in rows if "[TODO" not in r["question"]]
    if len(real_rows) < len(rows):
        print(f"[warn] {len(rows) - len(real_rows)} placeholder rows skipped — fill in eval_set.jsonl for real results.")
    return real_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Label for this eval run, e.g. 'baseline' or 'chunk400_top8'")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / "config.yaml") as f:
        import yaml
        config = yaml.safe_load(f)

    eval_path = root / config["paths"]["eval_set"]
    eval_rows = load_eval_set(eval_path)

    if not eval_rows:
        print("No real eval rows found. Fill in data/eval/eval_set.jsonl before running eval.")
        return

    print(f"Loading RAG pipeline and running {len(eval_rows)} eval questions...")
    pipeline = RAGPipeline(config)

    scored_rows = []
    chunk_debug_rows = []
    for row in eval_rows:
        result = pipeline.query(row["question"])
        score = score_result(result, row)
        score["question"] = row["question"]
        score["generated_answer"] = result["answer"]
        score["reference_answer"] = row["answer"]
        scored_rows.append(score)

        # Full retrieved-chunk record for debugging -- this is what lets you tell the
        # difference between "the right chunk was retrieved but the model still got it
        # wrong" (a generation problem) vs. "the fact was never in the retrieved chunks
        # at all" (a retrieval/chunking problem, or the fact isn't in the saved doc).
        chunk_debug_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected_source_doc": row["source_doc"],
                "retrieval_hit": score["retrieval_hit"],
                "judged_correct": score["judged_correct"],
                "retrieved_chunks": [
                    {
                        "source_doc": c["source_doc"],
                        "distance": round(c["distance"], 4),
                        "text": c["text"],
                    }
                    for c in result["retrieved_chunks"]
                ],
            }
        )

        flag = ""
        if score["is_refusal"]:
            flag = " [REFUSAL]"
        elif not score["judged_correct"] and score["retrieval_hit"]:
            flag = " [RETRIEVED BUT WRONG -- check chunks file]"
        elif not score["retrieval_hit"]:
            flag = " [RETRIEVAL MISS]"
        print(f"  [{row['id']}] retrieval_hit={score['retrieval_hit']} judged_correct={score['judged_correct']}{flag}")

    df = pd.DataFrame(scored_rows)

    out_dir = root / config["paths"]["eval_results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = out_dir / f"eval_{run_name}.csv"
    df.to_csv(out_path, index=False)

    chunks_path = out_dir / f"eval_{run_name}_chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for row in chunk_debug_rows:
            f.write(json.dumps(row) + "\n")

    print("\n" + "=" * 50)
    print(f"Run: {run_name}")
    print(f"Retrieval hit rate:   {df['retrieval_hit'].mean():.1%}")
    print(f"Answer correctness:   {df['judged_correct'].mean():.1%}")
    print(f"Refusal rate:         {df['is_refusal'].mean():.1%}")
    print("By category:")
    print(df.groupby("category")[["retrieval_hit", "judged_correct"]].mean().to_string())
    print("=" * 50)
    print(f"\nSummary metrics saved to {out_path}")
    print(f"Full retrieved-chunk detail saved to {chunks_path}")
    print("Open the _chunks.jsonl file for any failing question to see exactly what the")
    print("model was looking at -- that tells you whether it's a retrieval problem or a")
    print("generation problem.")
    print("Copy the summary numbers above into README.md section 5 once you're happy with a run.")


if __name__ == "__main__":
    main()