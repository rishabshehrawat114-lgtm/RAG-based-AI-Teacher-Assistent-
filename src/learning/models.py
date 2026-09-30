"""
src/learning/models.py
======================
Data structures and models for Module 12 Learning Features.

Includes:
- LearningResult: Encapsulates outputs from Summary, Key Concepts, and Explain Lecture.
- QuizQuestion: Represents a single validated multiple-choice question.
- QuizResult: Encapsulates a complete validated quiz session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LearningResult:
    """
    Structured outcome of a learning activity (Summary, Key Concepts, Explain Lecture).
    """

    activity_type: str  # "summary", "concepts", "explain"
    title: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    lecture_scope: str = "All Lectures"
    is_fallback: bool = False
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts to a standard dictionary representation."""
        return asdict(self)


@dataclass
class QuizQuestion:
    """
    A single grounded multiple-choice question for student practice.
    """

    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts to a standard dictionary representation."""
        return asdict(self)


@dataclass
class QuizResult:
    """
    A collection of validated quiz questions grounded in course material.
    """

    lecture_scope: str
    questions: list[QuizQuestion] = field(default_factory=list)
    num_questions: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    is_fallback: bool = False
    error_message: str | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts to a standard dictionary representation."""
        return asdict(self)
