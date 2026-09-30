"""
tests/test_evaluation.py
========================
Automated test suite for Module 13: RAG Evaluation & Quality Benchmarking.

Validates:
- Benchmark dataset loading and schema validation
- Retrieval metrics: Recall@K, Hit@K, MRR, Lecture Accuracy
- Relevance / abstention metrics: Decision accuracy, FAR, FRR, score analysis
- Citation / provenance validation
- RAGEvaluator engine integration and report generation
- Offline evaluation safety (no API key required)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    BenchmarkQuestion,
    DatasetError,
    EvaluationSummary,
    QuestionEvaluationResult,
    RAGEvaluator,
    calculate_hit_at_k,
    calculate_lecture_accuracy,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_relevance_metrics,
    calculate_similarity_score_analysis,
    evaluate_answer_citations,
    format_terminal_report,
    generate_markdown_report,
    load_benchmark_dataset,
    save_evaluation_results,
    validate_benchmark_record,
    validate_citation_record,
)
from src.rag import RAGTeacherPipeline


class TestRAGEvaluationModule(unittest.TestCase):
    """Test suite covering all components of the RAG Evaluation framework."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Dataset Loading and Schema Validation Tests
    # -------------------------------------------------------------------------
    def test_01_dataset_loading_and_validation(self) -> None:
        """Verifies loading a valid benchmark JSON dataset into BenchmarkQuestion models."""
        sample_data = [
            {
                "id": "q1",
                "question": "What is Python?",
                "question_type": "direct_lecture",
                "expected_lecture_id": "lec_01",
                "expected_topic": "Intro",
                "expected_relevant_chunk_ids": ["c1", "c2"],
                "should_answer": True,
            },
            {
                "id": "q2",
                "question": "What is the capital of Mars?",
                "question_type": "out_of_scope",
                "expected_lecture_id": None,
                "expected_topic": "Astronomy",
                "expected_relevant_chunk_ids": [],
                "should_answer": False,
            },
        ]
        file_path = self.temp_path / "valid_benchmark.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        questions = load_benchmark_dataset(file_path)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].id, "q1")
        self.assertTrue(questions[0].should_answer)
        self.assertEqual(questions[0].expected_relevant_chunk_ids, ["c1", "c2"])
        self.assertEqual(questions[1].id, "q2")
        self.assertFalse(questions[1].should_answer)

    def test_02_dataset_schema_validation_rejects_malformed_records(self) -> None:
        """Verifies that records missing required fields or with bad types are rejected."""
        # Missing question text
        is_ok, err = validate_benchmark_record({"id": "q1", "should_answer": True})
        self.assertFalse(is_ok)
        self.assertIn("question", err)

        # Missing should_answer
        is_ok, err = validate_benchmark_record({"id": "q1", "question": "valid?"})
        self.assertFalse(is_ok)
        self.assertIn("should_answer", err)

        # Non-boolean should_answer
        is_ok, err = validate_benchmark_record({"id": "q1", "question": "valid?", "should_answer": "yes"})
        self.assertFalse(is_ok)
        self.assertIn("boolean", err)

    def test_03_duplicate_question_ids_rejected(self) -> None:
        """Verifies duplicate question IDs raise DatasetError."""
        sample_data = [
            {"id": "dup_01", "question": "Question A", "should_answer": True},
            {"id": "dup_01", "question": "Question B", "should_answer": False},
        ]
        file_path = self.temp_path / "dup_benchmark.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        with self.assertRaises(DatasetError) as ctx:
            load_benchmark_dataset(file_path)
        self.assertIn("Duplicate question ID", str(ctx.exception))

    def test_04_missing_dataset_file_raises_dataset_error(self) -> None:
        """Verifies attempting to load a nonexistent file raises DatasetError."""
        with self.assertRaises(DatasetError):
            load_benchmark_dataset(self.temp_path / "nonexistent.json")

    def test_05_empty_dataset_returns_empty_list(self) -> None:
        """Verifies an empty JSON array returns an empty list without error."""
        file_path = self.temp_path / "empty_benchmark.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        res = load_benchmark_dataset(file_path)
        self.assertEqual(res, [])

    # -------------------------------------------------------------------------
    # 2. Retrieval Metrics Unit Tests
    # -------------------------------------------------------------------------
    def test_06_recall_at_k(self) -> None:
        """Tests Recall@K calculation with single and multiple expected chunks."""
        retrieved = ["c1", "c2", "c3", "c4", "c5"]

        # 1 expected chunk: c2 at rank 2
        self.assertEqual(calculate_recall_at_k(retrieved, ["c2"], k=1), 0.0)
        self.assertEqual(calculate_recall_at_k(retrieved, ["c2"], k=2), 1.0)
        self.assertEqual(calculate_recall_at_k(retrieved, ["c2"], k=5), 1.0)

        # 2 expected chunks: c2 (rank 2) and c5 (rank 5)
        self.assertEqual(calculate_recall_at_k(retrieved, ["c2", "c5"], k=1), 0.0)
        self.assertEqual(calculate_recall_at_k(retrieved, ["c2", "c5"], k=3), 0.5)
        self.assertEqual(calculate_recall_at_k(retrieved, ["c2", "c5"], k=5), 1.0)

    def test_07_recall_at_k_empty_or_zero(self) -> None:
        """Tests Recall@K edge cases: empty expected, empty retrieved, or invalid k."""
        self.assertEqual(calculate_recall_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(calculate_recall_at_k(["c1"], [], k=5), 0.0)
        self.assertEqual(calculate_recall_at_k(["c1"], ["c1"], k=0), 0.0)

    def test_08_hit_at_k(self) -> None:
        """Tests Hit@K binary metric."""
        retrieved = ["c1", "c2", "c3", "c4", "c5"]

        # Hit@1
        self.assertTrue(calculate_hit_at_k(retrieved, ["c1"], k=1))
        self.assertFalse(calculate_hit_at_k(retrieved, ["c2"], k=1))

        # Hit@3
        self.assertTrue(calculate_hit_at_k(retrieved, ["c3"], k=3))
        self.assertFalse(calculate_hit_at_k(retrieved, ["c4"], k=3))

        # Hit@5
        self.assertTrue(calculate_hit_at_k(retrieved, ["c5"], k=5))
        self.assertFalse(calculate_hit_at_k(retrieved, ["c99"], k=5))

    def test_09_mean_reciprocal_rank(self) -> None:
        """Tests Reciprocal Rank calculation."""
        retrieved = ["c1", "c2", "c3", "c4", "c5"]

        # Rank 1 -> RR = 1.0
        self.assertEqual(calculate_reciprocal_rank(retrieved, ["c1"]), 1.0)
        # Rank 2 -> RR = 0.5
        self.assertEqual(calculate_reciprocal_rank(retrieved, ["c2"]), 0.5)
        # Rank 4 -> RR = 0.25
        self.assertEqual(calculate_reciprocal_rank(retrieved, ["c4"]), 0.25)
        # No match -> RR = 0.0
        self.assertEqual(calculate_reciprocal_rank(retrieved, ["c99"]), 0.0)
        # Empty
        self.assertEqual(calculate_reciprocal_rank([], ["c1"]), 0.0)

    def test_10_lecture_retrieval_accuracy(self) -> None:
        """Tests top retrieved lecture accuracy calculation."""
        retrieved_lectures = ["lec_01", "lec_01", "lec_02"]

        # Correct lecture
        self.assertTrue(calculate_lecture_accuracy(retrieved_lectures, "lec_01"))
        # Incorrect lecture
        self.assertFalse(calculate_lecture_accuracy(retrieved_lectures, "lec_02"))
        # Expected is None (e.g. cross-lecture / out of scope)
        self.assertIsNone(calculate_lecture_accuracy(retrieved_lectures, None))
        # Empty retrieved
        self.assertFalse(calculate_lecture_accuracy([], "lec_01"))

    # -------------------------------------------------------------------------
    # 3. Relevance Metrics Unit Tests
    # -------------------------------------------------------------------------
    def test_11_relevance_metrics_confusion_matrix(self) -> None:
        """Tests relevance decision accuracy, FAR, and FRR calculation."""
        results = [
            # True Positive
            QuestionEvaluationResult(
                question_id="1", question="q", question_type="direct",
                expected_decision=True, actual_decision=True, decision_match=True,
                top_score=0.6, top_k_scores=[0.6], retrieved_chunk_ids=["c1"],
                retrieved_lecture_ids=["l1"], expected_chunk_ids=["c1"], expected_lecture_id="l1",
                recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0,
                hit_at_1=True, hit_at_3=True, hit_at_5=True, reciprocal_rank=1.0,
                lecture_match=True, citations_evaluated=1, valid_citations=1, citation_valid=True,
                answer_available=True, is_abstention=False, deterministic_checks_passed=True,
            ),
            # True Negative
            QuestionEvaluationResult(
                question_id="2", question="q", question_type="out_of_scope",
                expected_decision=False, actual_decision=False, decision_match=True,
                top_score=0.1, top_k_scores=[0.1], retrieved_chunk_ids=["c2"],
                retrieved_lecture_ids=["l1"], expected_chunk_ids=[], expected_lecture_id=None,
                recall_at_1=0.0, recall_at_3=0.0, recall_at_5=0.0,
                hit_at_1=False, hit_at_3=False, hit_at_5=False, reciprocal_rank=0.0,
                lecture_match=None, citations_evaluated=0, valid_citations=0, citation_valid=None,
                answer_available=True, is_abstention=True, deterministic_checks_passed=True,
            ),
            # False Positive (False Acceptance)
            QuestionEvaluationResult(
                question_id="3", question="q", question_type="out_of_scope",
                expected_decision=False, actual_decision=True, decision_match=False,
                top_score=0.45, top_k_scores=[0.45], retrieved_chunk_ids=["c3"],
                retrieved_lecture_ids=["l1"], expected_chunk_ids=[], expected_lecture_id=None,
                recall_at_1=0.0, recall_at_3=0.0, recall_at_5=0.0,
                hit_at_1=False, hit_at_3=False, hit_at_5=False, reciprocal_rank=0.0,
                lecture_match=None, citations_evaluated=1, valid_citations=1, citation_valid=True,
                answer_available=True, is_abstention=False, deterministic_checks_passed=False,
            ),
            # False Negative (False Rejection)
            QuestionEvaluationResult(
                question_id="4", question="q", question_type="direct",
                expected_decision=True, actual_decision=False, decision_match=False,
                top_score=0.25, top_k_scores=[0.25], retrieved_chunk_ids=["c4"],
                retrieved_lecture_ids=["l1"], expected_chunk_ids=["c4"], expected_lecture_id="l1",
                recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0,
                hit_at_1=True, hit_at_3=True, hit_at_5=True, reciprocal_rank=1.0,
                lecture_match=True, citations_evaluated=0, valid_citations=0, citation_valid=None,
                answer_available=True, is_abstention=True, deterministic_checks_passed=False,
            ),
        ]

        metrics = calculate_relevance_metrics(results)
        self.assertEqual(metrics["total_questions"], 4)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["decision_accuracy"], 0.5)
        self.assertEqual(metrics["false_acceptance_rate"], 0.5)  # FP / (FP + TN) = 1 / 2
        self.assertEqual(metrics["false_rejection_rate"], 0.5)   # FN / (TP + FN) = 1 / 2

    def test_12_relevance_metrics_empty_results(self) -> None:
        """Verifies calculate_relevance_metrics handles empty lists safely."""
        metrics = calculate_relevance_metrics([])
        self.assertEqual(metrics["total_questions"], 0)
        self.assertEqual(metrics["decision_accuracy"], 0.0)

    def test_13_similarity_score_analysis(self) -> None:
        """Tests score distribution metrics calculation."""
        results = [
            MagicMock(expected_decision=True, actual_decision=True, top_score=0.60),
            MagicMock(expected_decision=True, actual_decision=True, top_score=0.40),
            MagicMock(expected_decision=False, actual_decision=False, top_score=0.20),
            MagicMock(expected_decision=False, actual_decision=False, top_score=0.10),
        ]
        sa = calculate_similarity_score_analysis(results)
        self.assertAlmostEqual(sa["avg_top_score_relevant"], 0.50)
        self.assertAlmostEqual(sa["min_relevant_top_score"], 0.40)
        self.assertAlmostEqual(sa["avg_top_score_rejected"], 0.15)
        self.assertAlmostEqual(sa["max_irrelevant_top_score"], 0.20)

    # -------------------------------------------------------------------------
    # 4. Citation Validation Tests
    # -------------------------------------------------------------------------
    def test_14_valid_citation_record(self) -> None:
        """Verifies a fully valid citation record passes validation."""
        known = {"lecture 1", "lec_01", "lecture1.mp3"}
        retrieved_ids = {"c1", "c2"}
        citation = {
            "lecture_id": "lec_01",
            "lecture_title": "Lecture 1",
            "source_filename": "lecture1.mp3",
            "timestamp_start": "02:15",
            "timestamp_end": "03:45",
            "chunk_id": "c1",
        }
        is_ok, err = validate_citation_record(citation, known, retrieved_ids)
        self.assertTrue(is_ok)
        self.assertEqual(err, "")

    def test_15_invalid_lecture_in_citation(self) -> None:
        """Verifies citation with unknown lecture fails validation."""
        known = {"lec_01"}
        citation = {
            "lecture_id": "unknown_lecture_99",
            "lecture_title": "Unknown",
            "chunk_id": "c1",
        }
        is_ok, err = validate_citation_record(citation, known)
        self.assertFalse(is_ok)
        self.assertIn("unknown lecture", err)

    def test_16_invalid_timestamp_in_citation(self) -> None:
        """Verifies invalid timestamp span fails validation."""
        known = {"lec_01"}
        citation = {
            "lecture_id": "lec_01",
            "start_time": 100.0,
            "end_time": 50.0,  # start > end
            "chunk_id": "c1",
        }
        is_ok, err = validate_citation_record(citation, known)
        self.assertFalse(is_ok)
        self.assertIn("Invalid numeric timestamp", err)

    def test_17_citation_chunk_not_in_retrieved_pool(self) -> None:
        """Verifies citation referencing unretrieved chunk fails."""
        known = {"lec_01"}
        citation = {
            "lecture_id": "lec_01",
            "chunk_id": "unretrieved_chunk_999",
        }
        is_ok, err = validate_citation_record(citation, known, retrieved_chunk_ids={"c1", "c2"})
        self.assertFalse(is_ok)
        self.assertIn("not in retrieved candidate", err)

    def test_18_abstention_does_not_fail_citations(self) -> None:
        """Verifies that an abstention response is not flagged as a citation error."""
        total, valid, is_valid, errors = evaluate_answer_citations(
            citations=[],
            known_lectures={"lec_01"},
            is_abstention=True,
        )
        self.assertEqual(total, 0)
        self.assertEqual(valid, 0)
        self.assertIsNone(is_valid)
        self.assertEqual(len(errors), 0)

    # -------------------------------------------------------------------------
    # 5. Evaluator Integration & Reporting Tests
    # -------------------------------------------------------------------------
    def test_19_evaluator_runs_on_mock_retriever(self) -> None:
        """Tests that RAGEvaluator runs end-to-end on mock retriever and pipeline."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_id": "c1",
                "lecture_id": "lec_01",
                "similarity_score": 0.55,
                "text": "Variables in Python store data values.",
                "source_filename": "lec01.mp3",
                "timestamp_start": "01:00",
                "timestamp_end": "02:00",
            }
        ]
        mock_retriever.chunks_metadata = [{"lecture_id": "lec_01"}]

        evaluator = RAGEvaluator(retriever=mock_retriever, relevance_threshold=0.32)
        question = BenchmarkQuestion(
            id="test_01",
            question="What is a variable?",
            expected_lecture_id="lec_01",
            expected_relevant_chunk_ids=["c1"],
            should_answer=True,
        )

        res = evaluator.evaluate_question(question, evaluate_answer_generation=False)
        self.assertTrue(res.decision_match)
        self.assertTrue(res.hit_at_1)
        self.assertEqual(res.reciprocal_rank, 1.0)
        self.assertTrue(res.lecture_match)
        self.assertEqual(res.top_score, 0.55)

    def test_20_evaluator_evaluates_dataset_and_generates_reports(self) -> None:
        """Verifies full summary aggregation and JSON/Markdown file export."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"chunk_id": "c1", "lecture_id": "lec_01", "similarity_score": 0.60}
        ]
        mock_retriever.chunks_metadata = [{"lecture_id": "lec_01"}]

        evaluator = RAGEvaluator(retriever=mock_retriever, relevance_threshold=0.32)
        questions = [
            BenchmarkQuestion(id="q1", question="What is Python?", should_answer=True, expected_relevant_chunk_ids=["c1"]),
            BenchmarkQuestion(id="q2", question="Out of scope?", should_answer=False, expected_relevant_chunk_ids=[]),
        ]

        summary = evaluator.evaluate_dataset(questions, dataset_path="test_bench.json", evaluate_answers=False)
        self.assertIsInstance(summary, EvaluationSummary)
        self.assertEqual(summary.total_questions, 2)

        # Test terminal formatting
        term_text = format_terminal_report(summary)
        self.assertIn("RAG QUALITY EVALUATION REPORT", term_text)
        self.assertIn("Recall@1", term_text)

        # Test markdown generation
        md_text = generate_markdown_report(summary)
        self.assertIn("# 📊 RAG Quality & Retrieval Benchmark Report", md_text)

        # Test saving
        json_file, md_file = save_evaluation_results(summary, output_dir=self.temp_path / "results")
        self.assertTrue(json_file.is_file())
        self.assertTrue(md_file.is_file())


if __name__ == "__main__":
    unittest.main()
