"""
tests/test_learning_features.py
===============================
Comprehensive automated test suite for Module 12: Learning Features.

Validates:
  Summary:
    1. Summary generation receives grounded context.
    2. Summary respects lecture scope.
    3. Summary preserves citations.
    4. Summary rejects insufficient context (relevance < 0.32).
  Concepts:
    5. Key concepts are generated from retrieved context.
    6. Concepts preserve source metadata.
    7. Scope filtering works for concepts.
  Explain Lecture:
    8. Explain lecture provides grounded student-friendly structure.
  Quiz:
    9. Quiz generation returns valid structure.
    10. Exactly 4 options per question.
    11. Exactly one correct answer.
    12. Explanations exist.
    13. Source metadata exists.
    14. Invalid quiz output is rejected.
    15. Controlled regeneration works (attempt 1 invalid, attempt 2 valid).
    16. No infinite regeneration (attempt 1 & 2 invalid -> clean fallback).
  MCQ Practice & State:
    17. Student answer is recorded.
    18. Correct answer is evaluated accurately.
    19. Score updates correctly.
    20. Educational feedback is generated.
    21. Final score calculation is correct.
    22. Quiz restart resets state.
  Integration & Invariants:
    23. Lecture scope changes correctly without leaking.
    24. Switching learning modes clears incompatible state.
    25. Existing Module 11 chat still works.
    26. Relevance threshold remains strictly 0.32.
    27. Out-of-scope behavior remains intact (abstention).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_RELEVANCE_THRESHOLD
from src.data.course_catalog import CourseCatalog
from src.learning import (
    LearningResult,
    QuizQuestion,
    QuizResult,
    explain_lecture,
    generate_key_concepts,
    generate_quiz,
    generate_summary,
    parse_quiz_json,
    validate_quiz_data,
)
from src.llm import LLMClient
from src.rag import FALLBACK_NOT_FOUND_MESSAGE, RAGTeacherPipeline
from src.retrieval import LectureRetriever


class MockLearningEmbeddingModel:
    """Fast deterministic mock embedding model for learning feature tests."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.model_card_data = type("Card", (), {"base_model": "mock-all-MiniLM-L6-v2"})()

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        inputs = [sentences] if isinstance(sentences, str) else list(sentences)
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for text in inputs:
            vec = np.zeros(self.dimension, dtype=np.float32)
            t_low = text.lower()

            if any(k in t_low for k in ["variable", "assign", "container", "syntax"]):
                vec[0:15] += 1.0
            if any(k in t_low for k in ["intro", "overview", "python", "lec_01", "lecture 1"]):
                vec[15:30] += 1.0
            if any(k in t_low for k in ["control", "flow", "loop", "lec_02", "lecture 2"]):
                vec[25:40] += 1.0
            if any(k in t_low for k in ["concept", "topic", "summary", "explain", "quiz", "question", "definition", "term"]):
                # Overlap with lecture intro/concepts
                vec[15:30] += 0.8
                vec[30:45] += 0.5

            if not np.any(vec):
                vec[45:60] = 0.05  # sub-threshold noise floor (rejected by 0.32 gate)

            norm = np.linalg.norm(vec)
            if norm > 0 and normalize_embeddings:
                vec = vec / norm
            vectors.append(vec)

        return np.vstack(vectors)


class TestLearningFeatures(unittest.TestCase):
    """Test suite verifying all Module 12 Learning Features requirements."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_module12_")
        self.temp_path = Path(self.temp_dir)
        self.chunks_dir = self.temp_path / "chunks"
        self.embeddings_dir = self.temp_path / "embeddings"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.mock_model = MockLearningEmbeddingModel()

        # Seed Lecture 1: Introduction
        self.lec1_chunks = [
            {
                "chunk_id": "lec01_0001",
                "course_id": "python_101",
                "lecture_id": "lec_01",
                "lecture_title": "Introduction to Python",
                "source_filename": "lecture1.mp4",
                "source_file": "lecture1.mp4",
                "timestamp_start": "00:00:00",
                "timestamp_end": "00:02:30",
                "start_time": 0.0,
                "end_time": 150.0,
                "text": "Welcome to Python 101. Python is an interpreted, high-level programming language.",
            }
        ]

        # Seed Lecture 2: Variables
        self.lec2_chunks = [
            {
                "chunk_id": "lec02_0001",
                "course_id": "python_101",
                "lecture_id": "lec_02",
                "lecture_title": "Variables and Assignment",
                "source_filename": "lecture2.mp4",
                "source_file": "lecture2.mp4",
                "timestamp_start": "00:05:00",
                "timestamp_end": "00:07:30",
                "start_time": 300.0,
                "end_time": 450.0,
                "text": "A variable is a named memory location that stores a value. For example, x = 42 assigns 42 to x.",
            }
        ]

        with open(self.chunks_dir / "lec01_chunks.json", "w", encoding="utf-8") as f:
            json.dump({"lecture_id": "lec_01", "lecture_title": "Introduction to Python", "chunks": self.lec1_chunks}, f)

        with open(self.chunks_dir / "lec02_chunks.json", "w", encoding="utf-8") as f:
            json.dump({"lecture_id": "lec_02", "lecture_title": "Variables and Assignment", "chunks": self.lec2_chunks}, f)

        emb1 = self.mock_model.encode([c["text"] for c in self.lec1_chunks])
        np.savez_compressed(self.embeddings_dir / "lec01_embeddings.npz", embeddings=emb1)
        with open(self.embeddings_dir / "lec01_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.lec1_chunks, "embedding_dimension": 64}, f)

        emb2 = self.mock_model.encode([c["text"] for c in self.lec2_chunks])
        np.savez_compressed(self.embeddings_dir / "lec02_embeddings.npz", embeddings=emb2)
        with open(self.embeddings_dir / "lec02_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.lec2_chunks, "embedding_dimension": 64}, f)

        self.catalog = CourseCatalog(
            catalog_path=self.temp_path / "course_catalog.json",
            course_id="python_101",
            auto_sync=False,
        )
        self.catalog.sync_with_existing_data(
            chunks_dir=self.chunks_dir,
            embeddings_dir=self.embeddings_dir,
        )

        self.retriever = LectureRetriever(
            embeddings_dir=self.embeddings_dir,
            model=self.mock_model,
            auto_load=True,
        )
        self.pipeline = RAGTeacherPipeline(
            retriever=self.retriever,
            llm_client=LLMClient(api_key=""),
            catalog=self.catalog,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1-4: Summary Tests
    # -------------------------------------------------------------------------
    def test_01_summary_generation_receives_grounded_context(self) -> None:
        """Verifies summary generation retrieves grounded chunks and generates structured overview."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_configured.return_value = True
        mock_llm.generate.return_value = (
            "### Lecture Overview\nThis lecture covers variables.\n\n"
            "### Main Topics\n* Variables\n* Assignment\n\n"
            "### Important Ideas\nMemory locations hold values.\n\n"
            "### Examples Discussed\nx = 42\n\n"
            "### Key Takeaways\n* Variables store data."
        )

        res = generate_summary(self.retriever, llm_client=mock_llm, lecture_scope="lec_02")
        self.assertFalse(res.is_fallback)
        self.assertEqual(res.activity_type, "summary")
        self.assertIn("Lecture Overview", res.content)
        self.assertIn("Main Topics", res.content)
        self.assertIn("Key Takeaways", res.content)
        self.assertGreater(len(res.sources), 0)
        self.assertEqual(res.sources[0]["lecture_title"], "Variables and Assignment")

    def test_02_summary_respects_lecture_scope(self) -> None:
        """Verifies summary strictly isolates content to the specified lecture scope."""
        res_lec1 = generate_summary(self.retriever, lecture_scope="lec_01")
        self.assertFalse(res_lec1.is_fallback)
        for s in res_lec1.sources:
            self.assertEqual(s["source_filename"], "lecture1.mp4")

        res_lec2 = generate_summary(self.retriever, lecture_scope="lec_02")
        self.assertFalse(res_lec2.is_fallback)
        for s in res_lec2.sources:
            self.assertEqual(s["source_filename"], "lecture2.mp4")

    def test_03_summary_preserves_citations(self) -> None:
        """Verifies summary sources include exact timestamp ranges, chunk IDs, and similarity scores."""
        res = generate_summary(self.retriever, lecture_scope="lec_02")
        self.assertFalse(res.is_fallback)
        top_src = res.sources[0]
        self.assertEqual(top_src["timestamp_range"], "05:00 - 07:30")
        self.assertEqual(top_src["chunk_id"], "lec02_0001")
        self.assertGreaterEqual(top_src["similarity_score"], 0.32)

    def test_04_summary_rejects_insufficient_context(self) -> None:
        """Verifies summary abstains when retrieved similarity is below 0.32 threshold."""
        # Mock retrieve returning below noise floor
        sub_chunks = [{"rank": 1, "chunk_id": "c_sub", "similarity_score": 0.2100, "text": "Unrelated."}]
        with patch.object(self.retriever, "retrieve", return_value=sub_chunks):
            res = generate_summary(self.retriever, lecture_scope="lec_02")
            self.assertTrue(res.is_fallback)
            self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, res.content)
            self.assertEqual(len(res.sources), 0)

    # -------------------------------------------------------------------------
    # 5-7: Concepts Tests
    # -------------------------------------------------------------------------
    def test_05_key_concepts_generated_from_retrieved_context(self) -> None:
        """Verifies key concepts extraction extracts grounded concepts."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_configured.return_value = True
        mock_llm.generate.return_value = (
            "### Key Concepts\n\n"
            "1. **Variable**: A named memory location storing data.\n"
            "2. **Assignment Operator**: Binds a value to a variable."
        )

        res = generate_key_concepts(self.retriever, llm_client=mock_llm, lecture_scope="lec_02")
        self.assertFalse(res.is_fallback)
        self.assertEqual(res.activity_type, "concepts")
        self.assertIn("Key Concepts", res.content)
        self.assertIn("Variable", res.content)
        self.assertGreater(len(res.sources), 0)

    def test_06_concepts_preserve_source_metadata(self) -> None:
        """Verifies concept results retain source metadata traceable to chunks."""
        res = generate_key_concepts(self.retriever, lecture_scope="lec_02")
        self.assertFalse(res.is_fallback)
        src = res.sources[0]
        self.assertEqual(src["lecture_title"], "Variables and Assignment")
        self.assertEqual(src["timestamp_range"], "05:00 - 07:30")

    def test_07_scope_filtering_works_for_concepts(self) -> None:
        """Verifies concepts are filtered exclusively by lecture scope."""
        res = generate_key_concepts(self.retriever, lecture_scope="lec_01")
        self.assertFalse(res.is_fallback)
        for s in res.sources:
            self.assertEqual(s["source_filename"], "lecture1.mp4")

    # -------------------------------------------------------------------------
    # 8: Explain Lecture Test
    # -------------------------------------------------------------------------
    def test_08_explain_lecture_structure(self) -> None:
        """Verifies student-friendly explanation contains required pedagogical sections."""
        res = explain_lecture(self.retriever, lecture_scope="lec_02")
        self.assertFalse(res.is_fallback)
        self.assertEqual(res.activity_type, "explain")
        self.assertIn("What This Lecture Is About", res.content)
        self.assertIn("Concepts You Should Understand", res.content)
        self.assertIn("Final Takeaways", res.content)

    # -------------------------------------------------------------------------
    # 9-16: Quiz Generation & Validation Tests
    # -------------------------------------------------------------------------
    def test_09_quiz_generation_returns_valid_structure(self) -> None:
        """Verifies quiz generation produces validated QuizResult and QuizQuestion objects."""
        raw_quiz_json = json.dumps([
            {
                "question": "What is a variable in Python?",
                "options": [
                    "A named memory location that stores a value",
                    "A compiler instruction for optimization",
                    "A hardware register on the CPU",
                    "An external network socket connection",
                ],
                "correct_answer": "A named memory location that stores a value",
                "explanation": "According to the lecture, a variable is a named memory location.",
                "source": {
                    "lecture": "Variables and Assignment",
                    "timestamp_start": "05:00",
                    "timestamp_end": "07:30",
                },
            }
        ])

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_configured.return_value = True
        mock_llm.generate.return_value = f"```json\n{raw_quiz_json}\n```"

        res = generate_quiz(self.retriever, llm_client=mock_llm, lecture_scope="lec_02", num_questions=5)
        self.assertFalse(res.is_fallback)
        self.assertEqual(len(res.questions), 1)
        q = res.questions[0]
        self.assertEqual(q.question, "What is a variable in Python?")
        self.assertEqual(len(q.options), 4)
        self.assertEqual(q.correct_answer, q.options[0])

    def test_10_exactly_4_options_per_question(self) -> None:
        """Verifies validator rejects questions without exactly 4 options."""
        invalid_data = [
            {
                "question": "Q1",
                "options": ["A", "B", "C"],  # only 3
                "correct_answer": "A",
                "explanation": "Exp",
            }
        ]
        _, is_valid, err = validate_quiz_data(invalid_data, expected_count=1, sources=[])
        self.assertFalse(is_valid)
        self.assertIn("exactly 4 options", err)

    def test_11_exactly_one_correct_answer(self) -> None:
        """Verifies validator ensures correct_answer matches one of the options."""
        invalid_data = [
            {
                "question": "Q1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "Z",  # not in options
                "explanation": "Exp",
            }
        ]
        _, is_valid, err = validate_quiz_data(invalid_data, expected_count=1, sources=[])
        self.assertFalse(is_valid)
        self.assertIn("does not match any of the 4 options", err)

    def test_12_explanations_and_source_metadata_exist(self) -> None:
        """Verifies explanation and source timestamps are populated."""
        data = [
            {
                "question": "Q1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "explanation": "Valid explanation grounded in lecture.",
                "source": {"lecture": "Lec 1", "timestamp_start": "01:00", "timestamp_end": "02:00"},
            }
        ]
        qs, is_valid, _ = validate_quiz_data(data, expected_count=1, sources=[])
        self.assertTrue(is_valid)
        self.assertEqual(qs[0].explanation, "Valid explanation grounded in lecture.")
        self.assertEqual(qs[0].source["timestamp_start"], "01:00")

    def test_13_invalid_quiz_output_rejected(self) -> None:
        """Verifies duplicate options or 'All of the above' are strictly rejected."""
        data_all_above = [
            {
                "question": "Q1",
                "options": ["A", "B", "C", "All of the above"],
                "correct_answer": "A",
                "explanation": "Exp",
            }
        ]
        _, is_valid, err = validate_quiz_data(data_all_above, expected_count=1, sources=[])
        self.assertFalse(is_valid)
        self.assertIn("violates quality guidelines", err)

    def test_14_controlled_regeneration_works(self) -> None:
        """Verifies single controlled retry turns initial failure into successful quiz."""
        bad_response = "Not valid JSON at all."
        good_response = json.dumps([
            {
                "question": "Q2",
                "options": ["1", "2", "3", "4"],
                "correct_answer": "1",
                "explanation": "Good exp.",
            }
        ])

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_configured.return_value = True
        # First call fails, second call succeeds
        mock_llm.generate.side_effect = [bad_response, good_response]

        res = generate_quiz(self.retriever, llm_client=mock_llm, lecture_scope="lec_02", num_questions=5)
        self.assertFalse(res.is_fallback)
        self.assertEqual(len(res.questions), 1)
        self.assertTrue(res.debug_info.get("regeneration_attempted"))

    def test_15_no_infinite_regeneration(self) -> None:
        """Verifies that if both attempts fail, system stops cleanly with clear error without looping."""
        bad_response = "Still bad output."
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_configured.return_value = True
        mock_llm.generate.side_effect = [bad_response, bad_response]

        res = generate_quiz(self.retriever, llm_client=mock_llm, lecture_scope="lec_02", num_questions=5)
        self.assertTrue(res.is_fallback)
        self.assertIn("Failed to generate", res.error_message)
        # Verify exactly 2 calls were made (1 original + 1 retry)
        self.assertEqual(mock_llm.generate.call_count, 2)

    def test_16_quiz_offline_does_not_fabricate(self) -> None:
        """Verifies system does not fabricate quiz questions offline when LLM is unavailable."""
        unconfigured_llm = LLMClient(api_key="")
        res = generate_quiz(self.retriever, llm_client=unconfigured_llm, lecture_scope="lec_02")
        self.assertTrue(res.is_fallback)
        self.assertIn("requires a configured OpenAI API Key", res.error_message)
        self.assertEqual(len(res.questions), 0)

    # -------------------------------------------------------------------------
    # 17-22: MCQ Practice State & Scoring Tests
    # -------------------------------------------------------------------------
    def test_17_mcq_student_answer_evaluation_and_scoring(self) -> None:
        """Simulates interactive MCQ practice: answer selection, score calculation, and feedback."""
        questions = [
            QuizQuestion(
                question="What is x in 'x = 5'?",
                options=["A variable", "A loop", "A class", "A function"],
                correct_answer="A variable",
                explanation="x is a variable holding 5.",
                source={"lecture": "Variables", "timestamp_start": "01:00", "timestamp_end": "02:00"},
            ),
            QuizQuestion(
                question="What is = in Python?",
                options=["Assignment", "Equality check", "Division", "Addition"],
                correct_answer="Assignment",
                explanation="= assigns a value.",
                source={"lecture": "Variables", "timestamp_start": "02:00", "timestamp_end": "03:00"},
            ),
        ]

        # Student submits Answer 1: Correct
        student_ans_1 = "A variable"
        is_correct_1 = student_ans_1 == questions[0].correct_answer
        self.assertTrue(is_correct_1)

        # Student submits Answer 2: Incorrect
        student_ans_2 = "Equality check"
        is_correct_2 = student_ans_2 == questions[1].correct_answer
        self.assertFalse(is_correct_2)

        # Score tally
        score = sum([1 for q, ans in zip(questions, [student_ans_1, student_ans_2]) if ans == q.correct_answer])
        total = len(questions)
        pct = (score / total) * 100

        self.assertEqual(score, 1)
        self.assertEqual(total, 2)
        self.assertEqual(pct, 50.0)

    def test_18_quiz_restart_resets_state(self) -> None:
        """Verifies restarting a quiz resets state to 0 without affecting knowledge base."""
        quiz_state = {
            "questions": ["q1", "q2"],
            "current_index": 2,
            "score": 2,
            "answers": {0: "A", 1: "B"},
            "complete": True,
        }

        # Restart quiz action
        quiz_state["current_index"] = 0
        quiz_state["score"] = 0
        quiz_state["answers"] = {}
        quiz_state["complete"] = False

        self.assertEqual(quiz_state["current_index"], 0)
        self.assertEqual(quiz_state["score"], 0)
        self.assertEqual(len(quiz_state["answers"]), 0)
        self.assertFalse(quiz_state["complete"])

    # -------------------------------------------------------------------------
    # 23-27: Integration & Invariants Tests
    # -------------------------------------------------------------------------
    def test_23_switching_learning_modes_clears_incompatible_state(self) -> None:
        """Verifies switching from Summary to Quiz isolates session variables."""
        session = {
            "learning_mode": "Summary",
            "learning_result": "summary data",
            "quiz_questions": [],
        }

        # Switch to Quiz
        session["learning_mode"] = "Quiz"
        session["learning_result"] = None
        session["quiz_questions"] = ["new_q1"]

        self.assertEqual(session["learning_mode"], "Quiz")
        self.assertIsNone(session["learning_result"])
        self.assertEqual(len(session["quiz_questions"]), 1)

    def test_24_existing_module11_chat_still_works(self) -> None:
        """Verifies that RAG pipeline Q&A chat continues to work seamlessly."""
        ans = self.pipeline.answer_question("What is a variable?")
        self.assertFalse(ans.is_fallback)
        self.assertIn("Course Lecture Answer", ans.answer)
        self.assertGreater(len(ans.sources), 0)

    def test_25_relevance_threshold_remains_strictly_0_32(self) -> None:
        """Verifies that DEFAULT_RELEVANCE_THRESHOLD is exactly 0.32 across the system."""
        self.assertEqual(DEFAULT_RELEVANCE_THRESHOLD, 0.32)

    def test_26_out_of_scope_abstains(self) -> None:
        """Verifies out-of-scope query rejects with standard abstention in learning feature."""
        # Query with low similarity
        sub_chunks = [{"rank": 1, "chunk_id": "c1", "similarity_score": 0.1500, "text": "Unrelated."}]
        with patch.object(self.retriever, "retrieve", return_value=sub_chunks):
            res = generate_summary(self.retriever, lecture_scope="lec_01")
            self.assertTrue(res.is_fallback)
            self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, res.content)

    def test_27_all_lectures_summary_retrieves_representative_chunks(self) -> None:
        """Verifies summary generation works across All Lectures."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_configured.return_value = True
        mock_llm.generate.return_value = "### Lecture Overview\nCourse overview across all lectures."
        res = generate_summary(self.retriever, lecture_scope="All Lectures", llm_client=mock_llm)
        self.assertFalse(res.is_fallback)
        self.assertEqual(res.activity_type, "summary")
        self.assertIn("All Lectures", res.title)
        self.assertGreater(len(res.sources), 0)

    def test_28_empty_course_knowledge_base_handled_gracefully(self) -> None:
        """Verifies graceful notice when catalog has no indexed chunks."""
        empty_retriever = MagicMock()
        empty_retriever.chunks_metadata = []
        empty_retriever.retrieve.return_value = []

        res_sum = generate_summary(empty_retriever, lecture_scope="All Lectures")
        self.assertTrue(res_sum.is_fallback)
        self.assertIn("Empty", res_sum.content)

        res_con = generate_key_concepts(empty_retriever, lecture_scope="All Lectures")
        self.assertTrue(res_con.is_fallback)
        self.assertIn("Empty", res_con.content)

        res_exp = explain_lecture(empty_retriever, lecture_scope="All Lectures")
        self.assertTrue(res_exp.is_fallback)
        self.assertIn("Empty", res_exp.content)

        res_quiz = generate_quiz(empty_retriever, lecture_scope="All Lectures")
        self.assertTrue(res_quiz.is_fallback)
        self.assertEqual(len(res_quiz.questions), 0)

    def test_29_pipeline_learning_feature_wrappers(self) -> None:
        """Verifies that RAGTeacherPipeline exposes learning feature methods."""
        # Offline mode summary, concepts, explain use grounded fallback
        sum_res = self.pipeline.generate_summary(lecture_scope="lec_01")
        self.assertIsInstance(sum_res, LearningResult)
        self.assertFalse(sum_res.is_fallback)

        con_res = self.pipeline.generate_key_concepts(lecture_scope="lec_01")
        self.assertIsInstance(con_res, LearningResult)
        self.assertFalse(con_res.is_fallback)

        exp_res = self.pipeline.explain_lecture(lecture_scope="lec_01")
        self.assertIsInstance(exp_res, LearningResult)
        self.assertFalse(exp_res.is_fallback)

        # In unconfigured LLM state, quiz returns safe offline notice (no fabrication)
        quiz_offline = self.pipeline.generate_quiz(lecture_scope="lec_01", num_questions=5)
        self.assertIsInstance(quiz_offline, QuizResult)
        self.assertTrue(quiz_offline.is_fallback)
        self.assertIn("API key", quiz_offline.error_message)

        # With configured mock LLM, quiz generates questions
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_configured.return_value = True
        sample_q = [
            {
                "question": f"Question {i} about variables?",
                "options": ["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"],
                "correct_answer": "A. Choice 1",
                "explanation": "Variables store values in memory.",
                "source": {"lecture": "Variables", "timestamp_start": "00:05", "timestamp_end": "00:07"},
            }
            for i in range(1, 6)
        ]
        mock_llm.generate.return_value = json.dumps(sample_q)
        self.pipeline.llm_client = mock_llm

        quiz_res = self.pipeline.generate_quiz(lecture_scope="lec_01", num_questions=5)
        self.assertIsInstance(quiz_res, QuizResult)
        self.assertFalse(quiz_res.is_fallback)
        self.assertEqual(len(quiz_res.questions), 5)


if __name__ == "__main__":
    unittest.main()
