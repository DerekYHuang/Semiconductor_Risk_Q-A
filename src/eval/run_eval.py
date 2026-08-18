"""
Run the full eval set through the RAG pipeline and report retrieval + answer metrics.

Run:
    python src/eval/run_eval.py --run_name baseline

Each run writes a timestamped-or-named CSV to outputs/eval_results/, so you can keep
every iteration's results side by side and pull the before/after numbers directly into
README.md section 5 instead of re-describing them from memory.
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
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    # Filter out un-filled template rows so a fresh clone doesn't error out —
    # but warn loudly, since an eval run on 0 real rows is not a real eval run.
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
    for row in eval_rows:
        result = pipeline.query(row["question"])
        score = score_result(result, row)
        score["question"] = row["question"]
        score["generated_answer"] = result["answer"]
        score["reference_answer"] = row["answer"]
        scored_rows.append(score)
        print(f"  [{row['id']}] retrieval_hit={score['retrieval_hit']} judged_correct={score['judged_correct']}")

    df = pd.DataFrame(scored_rows)

    out_dir = root / config["paths"]["eval_results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{run_name}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 50)
    print(f"Run: {run_name}")
    print(f"Retrieval hit rate:   {df['retrieval_hit'].mean():.1%}")
    print(f"Answer correctness:   {df['judged_correct'].mean():.1%}")
    print("By category:")
    print(df.groupby("category")[["retrieval_hit", "judged_correct"]].mean().to_string())
    print("=" * 50)
    print(f"\nFull results saved to {out_path}")
    print("Copy the summary numbers above into README.md section 5 once you're happy with a run.")


if __name__ == "__main__":
    main()
