"""
Core RAG pipeline: retrieve relevant chunks from the vector store, then generate an
answer with a local LLM via Ollama. Imported by query_cli.py, run_eval.py, and the
Streamlit app — keep all retrieval/generation logic here so those three stay thin.
"""

from pathlib import Path

import chromadb
import ollama
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

SYSTEM_PROMPT = """You are answering questions about semiconductor-linked groundwater \
contamination and vapor intrusion risk in Santa Clara County, using only the provided \
source excerpts. Answer only from the given context. If the context does not contain \
the answer, say so explicitly rather than guessing. Cite which source document(s) you \
used."""


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        root = Path(__file__).resolve().parents[2]
        store_path = root / self.config["paths"]["vector_store"]

        self.embed_model = SentenceTransformer(self.config["embedding"]["model_name"])
        self.client = chromadb.PersistentClient(path=str(store_path))
        self.collection = self.client.get_collection("contamination_docs")
        self.llm_model = self.config["llm"]["model_name"]
        self.temperature = self.config["llm"]["temperature"]

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.config["retrieval"]["top_k"]
        query_embedding = self.embed_model.encode([question]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved.append({"text": doc, "source_doc": meta["source_doc"], "distance": dist})
        return retrieved

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source_doc']}]\n{c['text']}" for c in retrieved_chunks
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def query(self, question: str, top_k: int | None = None) -> dict:
        retrieved = self.retrieve(question, top_k)
        answer = self.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
        }
