"""
src/evaluation/retrieval_metrics.py
==================================
Calculation functions for retrieval quality: Recall@K, Hit@K, MRR, and Lecture Retrieval Accuracy.
"""

from __future__ import annotations

from typing import Sequence


def calculate_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """
    Computes Recall@K: fraction of expected relevant chunks present in the top-K retrieved chunks.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        expected_chunk_ids: Ground truth list of relevant chunk IDs.
        k: Cutoff rank (e.g. 1, 3, 5).

    Returns:
        Recall@K as a float between 0.0 and 1.0. If expected_chunk_ids is empty, returns 0.0.
    """
    if not expected_chunk_ids:
        return 0.0
    if k <= 0 or not retrieved_chunk_ids:
        return 0.0

    top_k = set(retrieved_chunk_ids[:k])
    expected = set(expected_chunk_ids)
    hits = len(top_k.intersection(expected))
    return float(hits / len(expected))


def calculate_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int,
) -> bool:
    """
    Computes Hit@K: returns True if at least one expected relevant chunk is present in top-K.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        expected_chunk_ids: Ground truth list of relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        True if hit occurs, False otherwise.
    """
    if not expected_chunk_ids or k <= 0 or not retrieved_chunk_ids:
        return False

    top_k = set(retrieved_chunk_ids[:k])
    expected = set(expected_chunk_ids)
    return len(top_k.intersection(expected)) > 0


def calculate_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
) -> float:
    """
    Computes the Reciprocal Rank (RR): 1 / rank of the first relevant chunk found.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        expected_chunk_ids: Ground truth list of relevant chunk IDs.

    Returns:
        1 / rank (1-indexed) if a match is found; 0.0 if no match exists in retrieved results.
    """
    if not expected_chunk_ids or not retrieved_chunk_ids:
        return 0.0

    expected = set(expected_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return 1.0 / float(rank)

    return 0.0


def calculate_lecture_accuracy(
    retrieved_lecture_ids: Sequence[str],
    expected_lecture_id: str | None,
) -> bool | None:
    """
    Determines whether the top retrieved result belongs to the expected lecture.

    Args:
        retrieved_lecture_ids: Ordered list of lecture IDs from retrieved chunks.
        expected_lecture_id: Expected target lecture ID (or None if cross-lecture/out-of-scope).

    Returns:
        True if top retrieved chunk matches expected lecture ID.
        False if top retrieved chunk does not match.
        None if expected_lecture_id is not specified (e.g. cross-lecture/out-of-scope).
    """
    if expected_lecture_id is None or str(expected_lecture_id).strip() == "":
        return None
    if not retrieved_lecture_ids:
        return False

    top_retrieved = str(retrieved_lecture_ids[0]).strip().lower()
    expected = str(expected_lecture_id).strip().lower()
    return top_retrieved == expected
