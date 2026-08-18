"""
Embed chunks (data/processed/chunks/chunks.jsonl) with a local Hugging Face model
and store them in a local Chroma vector database.

Run:
    python src/embed/build_vector_store.py

This rebuilds the vector store from scratch each run, so it's safe to re-run after
changing the embedding model or chunk settings in config.yaml.
"""

import json
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    root = Path(__file__).resolve().parents[2]
    chunks_path = root / config["paths"]["processed_chunks"]
    store_path = root / config["paths"]["vector_store"]
    store_path.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        print(f"No chunks file found at {chunks_path}. Run src/chunk/chunk_documents.py first.")
        return

    chunks = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]
    if not chunks:
        print("Chunks file is empty — nothing to embed.")
        return

    print(f"Loading embedding model: {config['embedding']['model_name']}")
    model = SentenceTransformer(config["embedding"]["model_name"])

    client = chromadb.PersistentClient(path=str(store_path))
    # Fresh collection each run so stale chunks from a previous chunk_size don't linger
    try:
        client.delete_collection("contamination_docs")
    except Exception:
        pass
    collection = client.create_collection("contamination_docs")

    batch_size = 32
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + storing"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source_doc": c["source_doc"]} for c in batch],
        )

    print(f"Stored {len(chunks)} chunks in vector store at {store_path}")


if __name__ == "__main__":
    main()
