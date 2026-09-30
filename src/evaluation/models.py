"""
src/evaluation/models.py
========================
Data models for RAG evaluation, benchmarking, metrics, and failure diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchmarkQuestion:
    """Represents a single benchmark question with expected retrieval and relevance ground truth."""

    id: str
    question: str
    question_type: str = "direct_lecture"
    expected_lecture_id: str | None = None
    expected_topic: str = ""
    expected_relevant_chunk_ids: list[str] = field(default_factory=list)
    should_answer: bool = True
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    lecture_scope: str = "All Lectures"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkQuestion:
        return cls(
            id=str(data.get("id", "")).strip(),
            question=str(data.get("question", "")).strip(),
            question_type=str(data.get("question_type", "direct_lecture")).strip(),
            expected_lecture_id=data.get("expected_lecture_id"),
            expected_topic=str(data.get("expected_topic", "")).strip(),
            expected_relevant_chunk_ids=[
                str(c).strip() for c in data.get("expected_relevant_chunk_ids", []) if str(c).strip()
            ],
            should_answer=bool(data.get("should_answer", True)),
            conversation_history=data.get("conversation_history") or [],
            lecture_scope=str(data.get("lecture_scope", "All Lectures")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionEvaluationResult:
    """Detailed evaluation outcomes for a single benchmark question."""

    question_id: str
    question: str
    question_type: str
    expected_decision: bool
    actual_decision: bool
    decision_match: bool
    top_score: float
    top_k_scores: list[float]
    retrieved_chunk_ids: list[str]
    retrieved_lecture_ids: list[str]
    expected_chunk_ids: list[str]
    expected_lecture_id: str | None
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    lecture_match: bool | None
    citations_evaluated: int
    valid_citations: int
    citation_valid: bool | None
    answer_available: bool
    is_abstention: bool
    deterministic_checks_passed: bool
    failure_reasons: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics across the full benchmark suite."""

    timestamp: str
    dataset_path: str
    relevance_threshold: float
    total_questions: int
    questions_by_type: dict[str, int]
    retrieval_metrics: dict[str, float]
    relevance_metrics: dict[str, float | int]
    score_analysis: dict[str, float | None]
    citation_metrics: dict[str, float | int]
    answer_metrics: dict[str, Any]
    question_results: list[QuestionEvaluationResult] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "relevance_threshold": self.relevance_threshold,
            "total_questions": self.total_questions,
            "questions_by_type": self.questions_by_type,
            "retrieval_metrics": self.retrieval_metrics,
            "relevance_metrics": self.relevance_metrics,
            "score_analysis": self.score_analysis,
            "citation_metrics": self.citation_metrics,
            "answer_metrics": self.answer_metrics,
            "failures": self.failures,
            "question_results": [q.to_dict() for q in self.question_results],
        }
