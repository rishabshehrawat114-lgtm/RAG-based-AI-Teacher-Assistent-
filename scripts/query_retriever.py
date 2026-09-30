#!/usr/bin/env python3
"""
scripts/query_retriever.py
==========================
CLI interface for semantic retrieval over lecture chunk embeddings (Module 7).

Allows students and educators to enter a question and retrieve the most
semantically relevant lecture chunks, complete with video sources, timestamps,
and similarity scores.

Usage:
------
# 1. Interactive multi-query session:
python scripts/query_retriever.py

# 2. Single-shot query:
python scripts/query_retriever.py --query "What is machine learning?"

# 3. Custom Top-K and Similarity Threshold:
python scripts/query_retriever.py --query "What is machine learning?" --top-k 3 --threshold 0.35

# 4. Scope query to a specific lecture source file:
python scripts/query_retriever.py --query "functions" --source "python_lecture.mp3"
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_DIR,
)
from src.retrieval import (
    LectureRetriever,
    RetrievalError,
    format_retrieval_results,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the retrieval CLI."""
    parser = argparse.ArgumentParser(
        description="Search lecture embeddings and retrieve semantically relevant chunks (Module 7)."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Question or query string. If omitted, starts an interactive prompt session.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top matching chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Optional minimum cosine similarity threshold (e.g. 0.30).",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Optional source lecture filename filter (e.g. 'python_lecture.mp3').",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDING_DIR,
        help=f"Directory containing embedding archives (default: {EMBEDDING_DIR}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence Transformers model name (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    return parser.parse_args()


def run_single_query(
    retriever: LectureRetriever,
    query: str,
    top_k: int,
    threshold: float | None,
    source_file: str | None,
) -> None:
    """Executes a single search query and displays the formatted results."""
    try:
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=threshold,
            source_file=source_file,
        )
        output_display = format_retrieval_results(results=results, query=query, top_k=top_k)
        print(output_display)
    except RetrievalError as exc:
        print(f"\nRetrieval Error: {exc}\n", file=sys.stderr)


def interactive_session(
    retriever: LectureRetriever,
    top_k: int,
    threshold: float | None,
    source_file: str | None,
) -> None:
    """Runs a continuous interactive REPL loop answering student questions."""
    print("==================================================")
    print("RAG AI Teacher Assistant — Interactive Retrieval")
    print("==================================================")
    print(f"Top-K: {top_k} | Threshold: {threshold or 'None'} | Source: {source_file or 'All lectures'}")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Enter your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting retrieval session. Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Exiting retrieval session. Goodbye!")
            break

        run_single_query(
            retriever=retriever,
            query=query,
            top_k=top_k,
            threshold=threshold,
            source_file=source_file,
        )
        print()


def main() -> int:
    """Main CLI entrypoint."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    args = parse_arguments()

    print("Initializing semantic retriever...")
    try:
        retriever = LectureRetriever(
            embeddings_dir=args.embeddings_dir,
            model_name=args.model,
            auto_load=True,
        )
        total_chunks = len(retriever.chunks_metadata)
        print(f"Retriever ready. Loaded {total_chunks} chunk(s) from '{args.embeddings_dir}'.\n")
    except RetrievalError as exc:
        print(f"Error initializing retriever: {exc}", file=sys.stderr)
        return 1

    if args.query:
        run_single_query(
            retriever=retriever,
            query=args.query,
            top_k=args.top_k,
            threshold=args.threshold,
            source_file=args.source,
        )
        return 0

    interactive_session(
        retriever=retriever,
        top_k=args.top_k,
        threshold=args.threshold,
        source_file=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
