"""
STRETCH GOAL — compare the LoRA fine-tuned model against the base model on the
same eval_set.jsonl used for the RAG pipeline, so the two "improve accuracy" stories
(better retrieval vs. better generation) are measured on an apples-to-apples basis.

Run:
    python src/finetune/evaluate_finetuned.py

Output:
    outputs/eval_results/finetune_comparison.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "eval"))
from metrics import keyword_overlap_score  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def generate_answer(model, tokenizer, question: str) -> str:
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    config = load_config()
    ft_config = config["finetune"]
    root = Path(__file__).resolve().parents[2]

    eval_path = root / config["paths"]["eval_set"]
    eval_rows = [json.loads(line) for line in open(eval_path, encoding="utf-8")]
    eval_rows = [r for r in eval_rows if "[TODO" not in r["question"]]

    if not eval_rows:
        print("No real eval rows found — fill in data/eval/eval_set.jsonl first.")
        return

    print(f"Loading base model: {ft_config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(ft_config["base_model"])
    base_model = AutoModelForCausalLM.from_pretrained(ft_config["base_model"])

    adapter_path = root / ft_config["output_dir"]
    if not adapter_path.exists():
        print(f"No fine-tuned adapter found at {adapter_path}. Run lora_finetune.py first.")
        return

    print(f"Loading LoRA adapter from {adapter_path}")
    finetuned_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    results = []
    for row in eval_rows:
        base_answer = generate_answer(base_model, tokenizer, row["question"])
        ft_answer = generate_answer(finetuned_model, tokenizer, row["question"])

        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "base_score": round(keyword_overlap_score(base_answer, row["answer"]), 3),
                "finetuned_score": round(keyword_overlap_score(ft_answer, row["answer"]), 3),
            }
        )
        print(f"[{row['id']}] base={results[-1]['base_score']} finetuned={results[-1]['finetuned_score']}")

    df = pd.DataFrame(results)
    out_dir = root / config["paths"]["eval_results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "finetune_comparison.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 50)
    print(f"Base model avg score:      {df['base_score'].mean():.3f}")
    print(f"Fine-tuned model avg score: {df['finetuned_score'].mean():.3f}")
    print("=" * 50)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
