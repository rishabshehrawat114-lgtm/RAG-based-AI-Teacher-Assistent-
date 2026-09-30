"""
src/evaluation
==============
RAG Quality Evaluation and Benchmarking framework for the AI Teaching Assistant.
"""

from src.evaluation.citation_metrics import (
    evaluate_answer_citations,
    validate_citation_record,
)
from src.evaluation.dataset import DatasetError, load_benchmark_dataset, validate_benchmark_record
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.models import (
    BenchmarkQuestion,
    EvaluationSummary,
    QuestionEvaluationResult,
)
from src.evaluation.relevance_metrics import (
    calculate_relevance_metrics,
    calculate_similarity_score_analysis,
)
from src.evaluation.report import (
    format_terminal_report,
    generate_markdown_report,
    save_evaluation_results,
)
from src.evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_lecture_accuracy,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)

__all__ = [
    "BenchmarkQuestion",
    "QuestionEvaluationResult",
    "EvaluationSummary",
    "DatasetError",
    "load_benchmark_dataset",
    "validate_benchmark_record",
    "RAGEvaluator",
    "calculate_recall_at_k",
    "calculate_hit_at_k",
    "calculate_reciprocal_rank",
    "calculate_lecture_accuracy",
    "calculate_relevance_metrics",
    "calculate_similarity_score_analysis",
    "validate_citation_record",
    "evaluate_answer_citations",
    "format_terminal_report",
    "generate_markdown_report",
    "save_evaluation_results",
]
