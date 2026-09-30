"""
scripts/run_evaluation.py
=========================
Reproducible CLI evaluation script for Module 13 RAG benchmarking.

Executes the benchmark dataset against the live course retriever and pipeline,
prints an evaluation report, and saves JSON/Markdown artifacts to data/evaluation/results/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K
from src.evaluation import (
    DatasetError,
    RAGEvaluator,
    format_terminal_report,
    load_benchmark_dataset,
    save_evaluation_results,
)
from src.rag import RAGTeacherPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RAG Evaluation and Quality Benchmarking for the AI Teaching Assistant."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "evaluation" / "rag_benchmark.json"),
        help="Path to the JSON evaluation benchmark dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "evaluation" / "results"),
        help="Directory to save JSON and Markdown evaluation results.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_RELEVANCE_THRESHOLD,
        help=f"Relevance cutoff threshold (default: {DEFAULT_RELEVANCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top retrieved passages to evaluate (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--skip-answers",
        action="store_true",
        help="Skip end-to-end answer generation to run retrieval-only evaluation faster.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    print(f"Loading benchmark dataset from: {dataset_path}")
    try:
        questions = load_benchmark_dataset(dataset_path)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not questions:
        print("WARNING: Benchmark dataset is empty. Evaluation aborted.", file=sys.stderr)
        return 1

    print(f"Initializing RAG Teacher Pipeline & Evaluator (threshold={args.threshold:.4f}, top_k={args.top_k})...")
    pipeline = RAGTeacherPipeline()

    # Guard: check if course knowledge base has indexed chunks
    corpus_stats = pipeline.get_corpus_stats()
    if corpus_stats.get("total_chunks", 0) == 0:
        print(
            "Evaluation notice: Course Knowledge Base has 0 indexed chunks.\n"
            "Please index course lectures before evaluating retrieval quality.",
            file=sys.stderr,
        )
        return 1

    evaluator = RAGEvaluator(
        pipeline=pipeline,
        relevance_threshold=args.threshold,
        top_k=args.top_k,
    )

    print(f"Executing evaluation across {len(questions)} benchmark questions...")
    summary = evaluator.evaluate_dataset(
        questions=questions,
        dataset_path=str(dataset_path),
        evaluate_answers=not args.skip_answers,
    )

    # Print terminal report
    report_text = format_terminal_report(summary)
    print("\n" + report_text + "\n")

    # Save results
    json_path, md_path = save_evaluation_results(summary, output_dir=output_dir)
    print(f"Evaluation results successfully saved to:")
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
