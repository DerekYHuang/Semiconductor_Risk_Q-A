"""
Quick command-line query interface for the RAG pipeline.

Usage:
    python src/retrieval/query_cli.py --question "What contaminant was found at [site]?"

    Or run with no --question for an interactive loop.
"""

import argparse

from rag_pipeline import RAGPipeline


def print_result(result: dict):
    print("\n" + "=" * 70)
    print(f"Q: {result['question']}")
    print("-" * 70)
    print(f"A: {result['answer']}")
    print("-" * 70)
    print("Retrieved from:")
    for chunk in result["retrieved_chunks"]:
        print(f"  - {chunk['source_doc']}  (distance={chunk['distance']:.3f})")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    print("Loading RAG pipeline (embedding model + vector store)...")
    pipeline = RAGPipeline()

    if args.question:
        result = pipeline.query(args.question, top_k=args.top_k)
        print_result(result)
    else:
        print("Interactive mode — type a question, or 'quit' to exit.")
        while True:
            question = input("\n> ")
            if question.strip().lower() in {"quit", "exit"}:
                break
            result = pipeline.query(question, top_k=args.top_k)
            print_result(result)


if __name__ == "__main__":
    main()
