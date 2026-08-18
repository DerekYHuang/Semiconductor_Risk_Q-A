"""
STRETCH GOAL — LoRA fine-tune a small local LLM on the domain Q&A dataset.

This trains a LoRA adapter (not a full model) so it's realistic to run on a laptop
CPU/single consumer GPU. Config values (rank, alpha, learning rate, epochs) come from
config.yaml under `finetune:` — record what you actually used in README section 4.

Run:
    python src/finetune/lora_finetune.py

Requires data/processed/finetune_dataset.jsonl — run prepare_dataset.py first.

Note: this uses a Hugging Face model directly (not through Ollama) since PEFT/LoRA
training needs direct access to model weights and gradients. After training, you'd
merge the adapter and either run it via `transformers` directly or convert it for
Ollama — that conversion step is a good "what I'd do with more time" line for your
README if you don't get to it.
"""

from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def format_for_training(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}


def main():
    config = load_config()
    ft_config = config["finetune"]
    root = Path(__file__).resolve().parents[2]
    dataset_path = root / "data" / "processed" / "finetune_dataset.jsonl"

    if not dataset_path.exists():
        print(f"No dataset found at {dataset_path}. Run prepare_dataset.py first.")
        return

    print(f"Loading base model: {ft_config['base_model']}")
    print("(This downloads model weights the first time — expect a wait and disk usage.)")

    tokenizer = AutoTokenizer.from_pretrained(ft_config["base_model"])
    model = AutoModelForCausalLM.from_pretrained(ft_config["base_model"])

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda ex: format_for_training(ex, tokenizer))
    dataset = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=1024),
        batched=False,
    )

    lora_config = LoraConfig(
        r=ft_config["lora_rank"],
        lora_alpha=ft_config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],  # standard for Llama-family attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check: should be a tiny % of total params

    training_args = TrainingArguments(
        output_dir=ft_config["output_dir"],
        num_train_epochs=ft_config["epochs"],
        learning_rate=ft_config["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_strategy="epoch",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(ft_config["output_dir"])
    tokenizer.save_pretrained(ft_config["output_dir"])
    print(f"LoRA adapter saved to {ft_config['output_dir']}")
    print("Next: run evaluate_finetuned.py to compare against the base model on eval_set.jsonl")


if __name__ == "__main__":
    main()
