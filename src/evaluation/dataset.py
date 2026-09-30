"""
src/evaluation/dataset.py
=========================
Benchmark dataset loading, schema verification, and validation helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.models import BenchmarkQuestion

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Custom exception raised when evaluation benchmark dataset cannot be loaded or validated."""
    pass


def validate_benchmark_record(record: dict[str, Any], index: int = 0) -> tuple[bool, str]:
    """
    Validates the structure and required fields of a raw benchmark record.

    Args:
        record: Raw dictionary representing a question.
        index: Index of the question in the dataset for reporting.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    if not isinstance(record, dict):
        return False, f"Record {index} must be a dictionary, got {type(record).__name__}."

    q_id = str(record.get("id", "")).strip()
    if not q_id:
        return False, f"Record {index} missing required non-empty 'id' field."

    question = str(record.get("question", "")).strip()
    if not question:
        return False, f"Record {index} ({q_id}) missing required non-empty 'question' text."

    if "should_answer" not in record:
        return False, f"Record {index} ({q_id}) missing required boolean 'should_answer' flag."

    if not isinstance(record.get("should_answer"), bool):
        return False, f"Record {index} ({q_id}) 'should_answer' must be a boolean."

    chunk_ids = record.get("expected_relevant_chunk_ids")
    if chunk_ids is not None and not isinstance(chunk_ids, list):
        return False, f"Record {index} ({q_id}) 'expected_relevant_chunk_ids' must be a list."

    history = record.get("conversation_history")
    if history is not None and not isinstance(history, list):
        return False, f"Record {index} ({q_id}) 'conversation_history' must be a list."

    return True, ""


def load_benchmark_dataset(dataset_path: Path | str) -> list[BenchmarkQuestion]:
    """
    Loads and validates the benchmark questions from a JSON file.

    Args:
        dataset_path: Path to the JSON benchmark file.

    Returns:
        List of validated BenchmarkQuestion instances.

    Raises:
        DatasetError: If the file does not exist, contains invalid JSON, or violates schema.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise DatasetError(f"Benchmark dataset file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Benchmark file contains invalid JSON ({path}): {exc}") from exc
    except Exception as exc:
        raise DatasetError(f"Failed to read benchmark dataset ({path}): {exc}") from exc

    if not isinstance(raw_data, list):
        raise DatasetError(f"Benchmark dataset root must be a JSON array, got {type(raw_data).__name__}.")

    if len(raw_data) == 0:
        logger.warning("Benchmark dataset at %s is empty (0 questions).", path)
        return []

    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_data):
        is_valid, err_msg = validate_benchmark_record(item, index=idx)
        if not is_valid:
            raise DatasetError(f"Validation failed for benchmark dataset: {err_msg}")

        q_id = str(item.get("id")).strip()
        if q_id in seen_ids:
            raise DatasetError(f"Duplicate question ID '{q_id}' found at index {idx}.")
        seen_ids.add(q_id)

        questions.append(BenchmarkQuestion.from_dict(item))

    logger.info("Successfully loaded %d benchmark questions from %s", len(questions), path)
    return questions
