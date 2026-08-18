"""
Split parsed documents (data/processed/parsed/*.txt) into overlapping token chunks
and write them to a single JSONL file for embedding.

Run:
    python src/chunk/chunk_documents.py

Chunk size/overlap are read from config.yaml — change them there and re-run this
script when you're iterating on retrieval quality (see README section 4 for the
comparison you should record).
"""

import json
from pathlib import Path

import tiktoken
import yaml
from tqdm import tqdm

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
ENCODING = tiktoken.get_encoding("cl100k_base")  # good general-purpose tokenizer for chunk sizing


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODING.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def main():
    config = load_config()
    root = Path(__file__).resolve().parents[2]
    parsed_dir = root / "data" / "processed" / "parsed"
    out_path = root / config["paths"]["processed_chunks"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = config["chunking"]["chunk_size_tokens"]
    overlap = config["chunking"]["chunk_overlap_tokens"]

    parsed_files = list(parsed_dir.glob("*.txt"))
    if not parsed_files:
        print(f"No parsed documents found in {parsed_dir}.")
        print("Run src/ingest/parse_documents.py first.")
        return

    all_chunks = []
    for path in tqdm(parsed_files, desc="Chunking documents"):
        text = path.read_text(encoding="utf-8")
        pieces = chunk_text(text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": f"{path.stem}__chunk{i}",
                    "source_doc": path.name,
                    "text": piece,
                }
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Wrote {len(all_chunks)} chunks from {len(parsed_files)} documents to {out_path}")
    print(f"(chunk_size={chunk_size} tokens, overlap={overlap} tokens)")


if __name__ == "__main__":
    main()
