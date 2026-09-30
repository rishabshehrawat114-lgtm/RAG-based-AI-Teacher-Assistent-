"""
src/evaluation/citation_metrics.py
=================================
Validation rules and quality metrics for source citations produced by the RAG system.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

TIMESTAMP_REGEX = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?$")


def validate_citation_record(
    citation: dict[str, Any],
    known_lectures: set[str],
    retrieved_chunk_ids: set[str] | None = None,
) -> tuple[bool, str]:
    """
    Validates a single source citation record against indexed course knowledge and retrieved evidence.

    Checks:
    1. Citation contains a non-empty lecture or source reference.
    2. Citation lecture exists in known lectures catalog / corpus.
    3. Timestamp format is valid (MM:SS or HH:MM:SS or numeric seconds) and start <= end.
    4. Citation references a chunk that was actually retrieved in the candidate pool.

    Args:
        citation: Source citation dictionary.
        known_lectures: Set of known normalized lecture titles and IDs in the course catalog.
        retrieved_chunk_ids: Optional set of chunk IDs that were returned by retrieval.

    Returns:
        Tuple of (is_valid: bool, error_reason: str).
    """
    if not isinstance(citation, dict):
        return False, f"Citation must be a dict, got {type(citation).__name__}."

    # 1. Source reference check
    lec_id = str(citation.get("lecture_id") or "").strip().lower()
    lec_title = str(citation.get("lecture_title") or "").strip().lower()
    source_file = str(citation.get("source_filename") or citation.get("source_file") or "").strip().lower()

    if not (lec_id or lec_title or source_file):
        return False, "Citation is missing lecture/source identifier."

    # 2. Known lecture / source check
    if known_lectures:
        has_match = (
            lec_id in known_lectures
            or lec_title in known_lectures
            or source_file in known_lectures
            or any(k in lec_title or lec_title in k for k in known_lectures if k)
        )
        if not has_match:
            return False, f"Citation references unknown lecture '{lec_title or lec_id or source_file}'."

    # 3. Timestamp sanity check
    ts_start = citation.get("timestamp_start") or citation.get("start_time")
    ts_end = citation.get("timestamp_end") or citation.get("end_time")

    if ts_start is not None and ts_end is not None:
        if isinstance(ts_start, (int, float)) and isinstance(ts_end, (int, float)):
            if ts_start < 0 or ts_end < 0 or ts_start > ts_end:
                return False, f"Invalid numeric timestamp span: [{ts_start}, {ts_end}]."
        elif isinstance(ts_start, str) and isinstance(ts_end, str):
            clean_s = ts_start.strip()
            clean_e = ts_end.strip()
            if clean_s and not TIMESTAMP_REGEX.match(clean_s):
                return False, f"Malformed start timestamp format: '{clean_s}'."
            if clean_e and not TIMESTAMP_REGEX.match(clean_e):
                return False, f"Malformed end timestamp format: '{clean_e}'."

    # 4. Correspondence to retrieved chunks
    chunk_id = str(citation.get("chunk_id", "")).strip()
    if retrieved_chunk_ids is not None and chunk_id:
        if chunk_id not in retrieved_chunk_ids:
            return False, f"Citation chunk '{chunk_id}' was not in retrieved candidate chunks."

    return True, ""


def evaluate_answer_citations(
    citations: Sequence[dict[str, Any]],
    known_lectures: set[str],
    retrieved_chunk_ids: set[str] | None = None,
    is_abstention: bool = False,
) -> tuple[int, int, bool | None, list[str]]:
    """
    Evaluates all citations for an answer turn.

    Args:
        citations: List of source citations attached to the answer.
        known_lectures: Set of valid normalized course lecture IDs and titles.
        retrieved_chunk_ids: Set of chunk IDs returned by retrieval.
        is_abstention: Whether the answer was an abstention / out-of-scope fallback.

    Returns:
        Tuple of (total_evaluated, valid_count, is_valid_bool_or_none, errors_list).
    """
    if is_abstention:
        # Abstentions are not expected to produce citations
        return 0, 0, None, []

    if not citations:
        return 0, 0, False, ["Accepted answer produced zero citations."]

    total = len(citations)
    valid_count = 0
    errors: list[str] = []

    for idx, c in enumerate(citations, start=1):
        ok, reason = validate_citation_record(c, known_lectures, retrieved_chunk_ids)
        if ok:
            valid_count += 1
        else:
            errors.append(f"Citation {idx}: {reason}")

    all_valid = (valid_count == total)
    return total, valid_count, all_valid, errors
