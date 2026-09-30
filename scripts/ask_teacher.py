"""
scripts/ask_teacher.py
======================
CLI runner for the RAG AI Teacher Assistant (Module 9).

Allows querying the indexed lecture knowledge base from the terminal,
with options for Top-K chunks, similarity threshold, and source lecture filtering.

Usage:
    python scripts/ask_teacher.py "What is machine learning?"
    python scripts/ask_teacher.py "Explain supervised learning" --top-k 3
    python scripts/ask_teacher.py "What is gradient descent?" --threshold 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.rag import RAGAnswer, RAGTeacherPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the RAG AI Teacher Assistant from the command line."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default="What is machine learning?",
        help="The question to ask the AI Teacher Assistant.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold (optional).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter query to a specific lecture video/transcript filename.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display corpus statistics before querying.",
    )

    args = parser.parse_args()

    pipeline = RAGTeacherPipeline()

    if args.stats:
        stats = pipeline.get_corpus_stats()
        print("\n=== Corpus Knowledge Base Statistics ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("=========================================\n")

    print(f"\nStudent Question: \"{args.question}\"")
    print("Searching indexed lecture material and synthesizing answer...\n")

    result: RAGAnswer = pipeline.answer_question(
        question=args.question,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        source_filter=args.source,
    )

    print("=" * 60)
    print("AI TEACHER ASSISTANT ANSWER:")
    print("=" * 60)
    print(result.answer)
    print("=" * 60)

    if result.sources:
        print("\nSOURCES & TIMESTAMPS:")
        for src in result.sources:
            print(
                f"  [{src['rank']}] Video: {src['source_file']} | "
                f"Time: {src['timestamp_range']} | "
                f"Relevance: {src['similarity_score']:.1%}"
            )
            print(f"      Excerpt: \"{src['text']}\"")
    else:
        print("\nNo source excerpts retrieved.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
