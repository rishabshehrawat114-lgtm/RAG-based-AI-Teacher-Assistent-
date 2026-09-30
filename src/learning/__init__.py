"""
src/learning
============
Module 12: Learning Features for RAG AI Teacher Assistant.

Provides:
- Lecture Summary (grounded overview, main topics, takeaways)
- Key Concepts (traceable concept extraction)
- Explain This Lecture (pedagogical, student-friendly explanation)
- Quiz Generation & MCQ Practice (grounded assessment with strict JSON validation)
"""

from src.learning.concepts import generate_key_concepts
from src.learning.explain import explain_lecture
from src.learning.models import LearningResult, QuizQuestion, QuizResult
from src.learning.prompts import (
    CONCEPTS_SYSTEM_PROMPT,
    EXPLAIN_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_concepts_messages,
    build_explain_messages,
    build_quiz_messages,
    build_summary_messages,
)
from src.learning.quiz import generate_quiz, parse_quiz_json, validate_quiz_data
from src.learning.summary import generate_summary

__all__ = [
    "LearningResult",
    "QuizQuestion",
    "QuizResult",
    "generate_summary",
    "generate_key_concepts",
    "explain_lecture",
    "generate_quiz",
    "validate_quiz_data",
    "parse_quiz_json",
    "SUMMARY_SYSTEM_PROMPT",
    "CONCEPTS_SYSTEM_PROMPT",
    "EXPLAIN_SYSTEM_PROMPT",
    "QUIZ_SYSTEM_PROMPT",
    "build_summary_messages",
    "build_concepts_messages",
    "build_explain_messages",
    "build_quiz_messages",
]
