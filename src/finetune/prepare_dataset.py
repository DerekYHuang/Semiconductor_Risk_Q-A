"""
STRETCH GOAL — only do this after the base RAG pipeline is working and evaluated.

Converts your eval_set.jsonl (plus any additional Q&A pairs you write specifically
for training, since 20-30 eval rows alone is too small to train on — keep eval and
training questions separate so you're not evaluating on data the model was tuned on)
into the instruction-tuning format Hugging Face's `datasets` + `trl`/`peft` expect.

Run:
    python src/finetune/prepare_dataset.py

Output:
    data/processed/finetune_dataset.jsonl
"""

import json
from pathlib import Path


def format_example(question: str, answer: str, context: str = "") -> dict:
    """
    Instruction-tuning format: a single training example is a full prompt->completion
    pair. Keep the system framing consistent with SYSTEM_PROMPT in rag_pipeline.py so
    the fine-tuned model's behavior matches what the RAG pipeline expects at inference.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are answering questions about semiconductor-linked groundwater "
                    "contamination and vapor intrusion risk in Santa Clara County."
                ),
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    root = Path(__file__).resolve().parents[2]
    eval_path = root / "data" / "eval" / "eval_set.jsonl"
    # IMPORTANT: for real fine-tuning you want a training set separate from eval_set.jsonl.
    # Create data/finetune/train_qa.jsonl with the same {question, answer, source_doc,
    # category} schema and additional rows before running this for real — using the
    # eval set itself here is only a placeholder so the script runs end to end.
    train_qa_path = root / "data" / "eval" / "eval_set.jsonl"
    out_path = root / "data" / "processed" / "finetune_dataset.jsonl"

    rows = [json.loads(line) for line in open(train_qa_path, encoding="utf-8")]
    rows = [r for r in rows if "[TODO" not in r["question"]]

    if not rows:
        print("No filled-in Q&A rows found. Fill in a training Q&A set before running this.")
        return

    examples = [format_example(r["question"], r["answer"]) for r in rows]

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} training examples to {out_path}")
    print("Reminder: keep this training set separate from data/eval/eval_set.jsonl so your")
    print("fine-tuned-vs-base comparison in README section 5 is measuring generalization,")
    print("not memorization of the eval questions themselves.")


if __name__ == "__main__":
    main()
