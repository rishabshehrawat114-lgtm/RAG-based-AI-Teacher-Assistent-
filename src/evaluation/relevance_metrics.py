"""
src/evaluation/relevance_metrics.py
==================================
Calculation functions for relevance gate quality, abstention behavior, and similarity score distributions.
"""

from __future__ import annotations

from typing import Sequence

from src.evaluation.models import QuestionEvaluationResult


def calculate_relevance_metrics(
    results: Sequence[QuestionEvaluationResult],
) -> dict[str, float | int]:
    """
    Computes confusion matrix and relevance decision rates across benchmark results.

    Args:
        results: Sequence of evaluated question results.

    Returns:
        Dictionary containing counts and percentages:
        - decision_accuracy: (TP + TN) / total
        - false_acceptance_rate (FAR): FP / total_irrelevant
        - false_rejection_rate (FRR): FN / total_relevant
        - acceptance_rate: total_accepted / total
        - true_positives, true_negatives, false_positives, false_negatives
        - total_relevant, total_irrelevant
    """
    total = len(results)
    if total == 0:
        return {
            "total_questions": 0,
            "total_relevant": 0,
            "total_irrelevant": 0,
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "decision_accuracy": 0.0,
            "false_acceptance_rate": 0.0,
            "false_rejection_rate": 0.0,
            "acceptance_rate": 0.0,
        }

    tp = 0  # Expected True, Actual True
    tn = 0  # Expected False, Actual False
    fp = 0  # Expected False, Actual True (False Acceptance)
    fn = 0  # Expected True, Actual False (False Rejection)

    total_relevant = 0
    total_irrelevant = 0

    for r in results:
        if r.expected_decision:
            total_relevant += 1
            if r.actual_decision:
                tp += 1
            else:
                fn += 1
        else:
            total_irrelevant += 1
            if r.actual_decision:
                fp += 1
            else:
                tn += 1

    decision_accuracy = float((tp + tn) / total) if total > 0 else 0.0
    far = float(fp / total_irrelevant) if total_irrelevant > 0 else 0.0
    frr = float(fn / total_relevant) if total_relevant > 0 else 0.0
    acceptance_rate = float((tp + fp) / total) if total > 0 else 0.0

    return {
        "total_questions": total,
        "total_relevant": total_relevant,
        "total_irrelevant": total_irrelevant,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "decision_accuracy": decision_accuracy,
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "acceptance_rate": acceptance_rate,
    }


def calculate_similarity_score_analysis(
    results: Sequence[QuestionEvaluationResult],
) -> dict[str, float | None]:
    """
    Analyzes cosine similarity score distributions across relevant vs irrelevant questions.

    Args:
        results: Sequence of evaluated question results.

    Returns:
        Dictionary with score statistics:
        - avg_top_score_relevant
        - avg_top_score_rejected
        - min_relevant_top_score
        - max_irrelevant_top_score
    """
    relevant_scores: list[float] = []
    rejected_scores: list[float] = []
    irrelevant_scores: list[float] = []

    for r in results:
        score = float(r.top_score)
        if r.expected_decision:
            relevant_scores.append(score)
        else:
            irrelevant_scores.append(score)

        if not r.actual_decision:
            rejected_scores.append(score)

    avg_rel = float(sum(relevant_scores) / len(relevant_scores)) if relevant_scores else None
    avg_rej = float(sum(rejected_scores) / len(rejected_scores)) if rejected_scores else None
    min_rel = float(min(relevant_scores)) if relevant_scores else None
    max_irrel = float(max(irrelevant_scores)) if irrelevant_scores else None

    return {
        "avg_top_score_relevant": avg_rel,
        "avg_top_score_rejected": avg_rej,
        "min_relevant_top_score": min_rel,
        "max_irrelevant_top_score": max_irrel,
    }
