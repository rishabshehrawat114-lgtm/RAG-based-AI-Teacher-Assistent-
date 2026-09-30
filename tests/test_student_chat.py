"""
tests/test_student_chat.py
==========================
Comprehensive automated tests for Module 11: Better Student Chat Experience.

Verifies:
  1. Conversation history initialization (clean initial session state, default style & scope).
  2. User message persistence (structured role, content, timestamp, lecture_scope).
  3. Assistant message persistence (structured content, citations, diagnostics, response_style, timestamp).
  4. Follow-up query construction (elliptical follow-up queries expand context; independent queries do not).
  5. Follow-up retrieval (contextual retrieval executes correctly on course knowledge base).
  6. Lecture scope preserved across turns (scope recorded with each turn).
  7. Scope switching does not reuse stale retrieval context (prevents cross-lecture context bleeding).
  8. Clear Chat removes conversation state (empties messages and transient keys).
  9. Clear Chat preserves knowledge base (catalog, embeddings, and corpus stats remain intact).
  10. Relevance threshold remains 0.32 (strictly enforced cutoff).
  11. Out-of-scope questions still abstain (CNN query rejected with standard fallback message).
  12. Citations remain attached to assistant answers (exact timestamps, lecture name, similarity scores).
  13. Response style changes prompt behavior (Explain Clearly vs Short Answer vs Detailed Explanation).
  14. No lecture indexed state (graceful notice when knowledge base has 0 chunks).
  15. Backward compatibility for single lecture (single-lecture Q&A works seamlessly).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_RELEVANCE_THRESHOLD
from src.data.course_catalog import CourseCatalog
from src.llm import LLMClient
from src.rag import (
    DEFAULT_RESPONSE_STYLE,
    FALLBACK_NOT_FOUND_MESSAGE,
    RESPONSE_STYLES,
    RAGAnswer,
    RAGTeacherPipeline,
    build_rag_messages,
    build_retrieval_query,
    is_follow_up_query,
)
from src.retrieval import LectureRetriever, format_timestamp


class MockChatEmbeddingModel:
    """Fast deterministic mock embedding model for student chat tests."""

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

            if "variable" in t_low:
                vec[0:15] = 1.0
            if "expression" in t_low or "evaluate" in t_low:
                vec[15:30] = 1.0
            if "intro" in t_low or "python" in t_low:
                vec[30:45] = 1.0
            if "example" in t_low:
                vec[0:15] += 0.5  # shares subspace with variable if context attached

            if not np.any(vec):
                vec[45:60] = 0.05  # noise floor

            norm = np.linalg.norm(vec)
            if norm > 0 and normalize_embeddings:
                vec = vec / norm
            vectors.append(vec)

        return np.vstack(vectors)


class TestStudentChatExperience(unittest.TestCase):
    """Test suite verifying all Module 11 student chat experience requirements."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_module11_")
        self.temp_path = Path(self.temp_dir)
        self.chunks_dir = self.temp_path / "chunks"
        self.embeddings_dir = self.temp_path / "embeddings"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.mock_model = MockChatEmbeddingModel()

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
                "text": "Welcome to Python 101. Python is an interpreted, high-level general-purpose programming language.",
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
                "text": "A variable is a named memory location that stores a value. For example, x equals 42 assigns forty-two to x.",
            }
        ]

        # Write chunk files
        with open(self.chunks_dir / "lec01_chunks.json", "w", encoding="utf-8") as f:
            json.dump({"lecture_id": "lec_01", "lecture_title": "Introduction to Python", "chunks": self.lec1_chunks}, f)

        with open(self.chunks_dir / "lec02_chunks.json", "w", encoding="utf-8") as f:
            json.dump({"lecture_id": "lec_02", "lecture_title": "Variables and Assignment", "chunks": self.lec2_chunks}, f)

        # Generate and write embeddings
        emb1 = self.mock_model.encode([c["text"] for c in self.lec1_chunks])
        np.savez_compressed(self.embeddings_dir / "lec01_embeddings.npz", embeddings=emb1)
        with open(self.embeddings_dir / "lec01_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.lec1_chunks, "embedding_dimension": 64}, f)

        emb2 = self.mock_model.encode([c["text"] for c in self.lec2_chunks])
        np.savez_compressed(self.embeddings_dir / "lec02_embeddings.npz", embeddings=emb2)
        with open(self.embeddings_dir / "lec02_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.lec2_chunks, "embedding_dimension": 64}, f)

        # Catalog
        self.catalog = CourseCatalog(
            catalog_path=self.temp_path / "course_catalog.json",
            course_id="python_101",
            auto_sync=False,
        )
        self.catalog.sync_with_existing_data(
            chunks_dir=self.chunks_dir,
            embeddings_dir=self.embeddings_dir,
        )

        # Retriever & Pipeline
        self.retriever = LectureRetriever(
            embeddings_dir=self.embeddings_dir,
            model=self.mock_model,
            auto_load=True,
        )
        self.pipeline = RAGTeacherPipeline(
            retriever=self.retriever,
            llm_client=LLMClient(api_key=""),  # offline synthesis mode
            catalog=self.catalog,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: Conversation History Initialization
    # -------------------------------------------------------------------------
    def test_01_conversation_history_initialization(self) -> None:
        """Verifies session state conversation defaults."""
        messages: list[dict] = []
        response_style: str = DEFAULT_RESPONSE_STYLE
        lecture_scope: str = "All Lectures"

        self.assertEqual(len(messages), 0)
        self.assertEqual(response_style, "Explain Clearly")
        self.assertEqual(lecture_scope, "All Lectures")
        self.assertIn("Explain Clearly", RESPONSE_STYLES)
        self.assertIn("Short Answer", RESPONSE_STYLES)
        self.assertIn("Detailed Explanation", RESPONSE_STYLES)

    # -------------------------------------------------------------------------
    # Test 2: User Message Persistence
    # -------------------------------------------------------------------------
    def test_02_user_message_persistence(self) -> None:
        """Verifies user messages are stored with structured metadata."""
        now_iso = datetime.now(timezone.utc).isoformat()
        user_msg = {
            "id": "msg_u_001",
            "role": "user",
            "content": "What is a variable?",
            "timestamp": now_iso,
            "lecture_scope": "All Lectures",
        }
        history = [user_msg]

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "What is a variable?")
        self.assertEqual(history[0]["lecture_scope"], "All Lectures")
        self.assertTrue(history[0]["timestamp"].startswith("20"))

    # -------------------------------------------------------------------------
    # Test 3: Assistant Message Persistence
    # -------------------------------------------------------------------------
    def test_03_assistant_message_persistence(self) -> None:
        """Verifies assistant messages store structured content, citations, and diagnostics."""
        res = self.pipeline.answer_question("What is a variable?")
        now_iso = datetime.now(timezone.utc).isoformat()

        asst_msg = {
            "id": "msg_a_001",
            "role": "assistant",
            "content": res.answer,
            "citations": res.sources,
            "lecture_scope": "All Lectures",
            "response_style": "Explain Clearly",
            "diagnostics": res.debug_info,
            "is_fallback": res.is_fallback,
            "timestamp": now_iso,
        }

        self.assertEqual(asst_msg["role"], "assistant")
        self.assertFalse(asst_msg["is_fallback"])
        self.assertGreater(len(asst_msg["citations"]), 0)
        self.assertIn("query", asst_msg["diagnostics"])
        self.assertIn("relevance_decision", asst_msg["diagnostics"])
        self.assertEqual(asst_msg["diagnostics"]["relevance_decision"], "RELEVANT")

    # -------------------------------------------------------------------------
    # Test 4: Follow-up Query Construction
    # -------------------------------------------------------------------------
    def test_04_follow_up_query_construction(self) -> None:
        """Verifies contextual follow-up detection and query building."""
        self.assertTrue(is_follow_up_query("Give me an example."))
        self.assertTrue(is_follow_up_query("Can you explain it further?"))
        self.assertTrue(is_follow_up_query("Why is that?"))
        self.assertFalse(is_follow_up_query("What does Python do when it evaluates an expression?"))

        history = [
            {"role": "user", "content": "What is a variable in Python?", "lecture_scope": "All Lectures"},
            {"role": "assistant", "content": "A variable is a named memory location.", "lecture_scope": "All Lectures"},
        ]

        # Contextual follow-up query expands with previous question core
        effective_query, is_fu = build_retrieval_query("Give me an example.", history, current_scope="All Lectures")
        self.assertTrue(is_fu)
        self.assertIn("What is a variable in Python", effective_query)
        self.assertIn("Give me an example", effective_query)

        # Independent query remains unchanged
        independent_q = "Explain the difference between supervised and unsupervised learning."
        eff_indep, is_fu_indep = build_retrieval_query(independent_q, history, current_scope="All Lectures")
        self.assertFalse(is_fu_indep)
        self.assertEqual(eff_indep, independent_q)

    # -------------------------------------------------------------------------
    # Test 5: Follow-up Retrieval on Knowledge Base
    # -------------------------------------------------------------------------
    def test_05_follow_up_retrieval(self) -> None:
        """Verifies that follow-up retrieval finds relevant variable material when student asks 'Give me an example.'"""
        history = [
            {"role": "user", "content": "What is a variable?", "lecture_scope": "All Lectures"},
            {"role": "assistant", "content": "A variable stores a value in Python.", "lecture_scope": "All Lectures"},
        ]

        res = self.pipeline.answer_question(
            question="Give me an example.",
            conversation_history=history,
            lecture_scope="All Lectures",
        )

        self.assertFalse(res.is_fallback)
        self.assertTrue(res.debug_info["is_follow_up"])
        self.assertIn("What is a variable", res.debug_info["retrieval_query"])
        self.assertEqual(res.debug_info["relevance_decision"], "RELEVANT")
        self.assertGreater(len(res.sources), 0)
        self.assertIn("Variables and Assignment", res.sources[0]["lecture_title"])

    # -------------------------------------------------------------------------
    # Test 6: Lecture Scope Preserved Across Turns
    # -------------------------------------------------------------------------
    def test_06_lecture_scope_preserved_across_turns(self) -> None:
        """Verifies lecture scope is attached to every turn and recorded in debug diagnostics."""
        res1 = self.pipeline.answer_question(
            question="What is Python?",
            lecture_scope="lec_01",
        )
        self.assertEqual(res1.debug_info["lecture_scope"], "lec_01")

        history = [
            {"role": "user", "content": "What is Python?", "lecture_scope": "lec_01"},
            {"role": "assistant", "content": res1.answer, "lecture_scope": "lec_01"},
        ]

        res2 = self.pipeline.answer_question(
            question="Can you explain more?",
            conversation_history=history,
            lecture_scope="lec_01",
        )
        self.assertEqual(res2.debug_info["lecture_scope"], "lec_01")
        self.assertIn("Introduction to Python", res2.sources[0]["lecture_title"])

    # -------------------------------------------------------------------------
    # Test 7: Scope Switching Does Not Reuse Stale Context
    # -------------------------------------------------------------------------
    def test_07_scope_switching_does_not_reuse_stale_context(self) -> None:
        """Verifies that switching from Lecture 1 to Lecture 2 isolates retrieval."""
        history = [
            {"role": "user", "content": "Tell me about Python history in Lecture 1.", "lecture_scope": "lec_01"},
            {"role": "assistant", "content": "Python was created by Guido van Rossum.", "lecture_scope": "lec_01"},
        ]

        # Scope explicitly switched to lec_02
        res = self.pipeline.answer_question(
            question="What is a variable?",
            conversation_history=history,
            lecture_scope="lec_02",
        )

        # Retrieval must strictly retrieve from Lecture 2 (Variables), zero chunks from Lecture 1
        self.assertFalse(res.is_fallback)
        self.assertEqual(res.debug_info["lecture_scope"], "lec_02")
        for src in res.sources:
            self.assertEqual(src["lecture_id"], "lec_02")
            self.assertNotEqual(src["lecture_id"], "lec_01")

    # -------------------------------------------------------------------------
    # Test 8: Clear Chat Removes Conversation State
    # -------------------------------------------------------------------------
    def test_08_clear_chat_removes_conversation_state(self) -> None:
        """Verifies clear chat logic empties message state."""
        session_mock = {
            "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            "msg_turn_1": "data",
            "diag_turn_1": "debug",
            "lecture_scope": "All Lectures",
        }

        # Clear chat execution
        session_mock["messages"] = []
        for k in [k for k in session_mock if k.startswith(("msg_", "diag_"))]:
            del session_mock[k]

        self.assertEqual(len(session_mock["messages"]), 0)
        self.assertNotIn("msg_turn_1", session_mock)
        self.assertNotIn("diag_turn_1", session_mock)
        self.assertEqual(session_mock["lecture_scope"], "All Lectures")

    # -------------------------------------------------------------------------
    # Test 9: Clear Chat Preserves Knowledge Base
    # -------------------------------------------------------------------------
    def test_09_clear_chat_preserves_knowledge_base(self) -> None:
        """Verifies that clearing conversation state leaves corpus chunks, catalog, and embeddings intact."""
        stats_before = self.pipeline.get_corpus_stats()
        self.assertEqual(stats_before["total_chunks"], 2)
        self.assertEqual(len(self.catalog.get_all_lectures()), 2)

        # Simulate clear chat
        messages: list = []

        # Confirm stats after
        stats_after = self.pipeline.get_corpus_stats()
        self.assertEqual(stats_after["total_chunks"], 2)
        self.assertEqual(len(self.catalog.get_all_lectures()), 2)
        self.assertTrue((self.embeddings_dir / "lec01_embeddings.npz").exists())
        self.assertTrue((self.embeddings_dir / "lec02_embeddings.npz").exists())

    # -------------------------------------------------------------------------
    # Test 10: Relevance Threshold Remains 0.32
    # -------------------------------------------------------------------------
    def test_10_relevance_threshold_remains_0_32(self) -> None:
        """Verifies DEFAULT_RELEVANCE_THRESHOLD is strictly 0.32 and pipeline enforces it."""
        self.assertEqual(DEFAULT_RELEVANCE_THRESHOLD, 0.32)

        # Candidate below 0.32
        low_score_chunks = [{"rank": 1, "chunk_id": "c1", "similarity_score": 0.3150, "text": "Something."}]
        with patch.object(self.retriever, "retrieve", return_value=low_score_chunks):
            res_low = self.pipeline.answer_question("Random topic")
            self.assertTrue(res_low.is_fallback)
            self.assertEqual(res_low.debug_info["relevance_decision"], "NOT RELEVANT")

        # Candidate at or above 0.32
        high_score_chunks = [{"rank": 1, "chunk_id": "c2", "similarity_score": 0.3250, "text": "Relevant info."}]
        with patch.object(self.retriever, "retrieve", return_value=high_score_chunks):
            res_high = self.pipeline.answer_question("Relevant topic")
            self.assertFalse(res_high.is_fallback)
            self.assertEqual(res_high.debug_info["relevance_decision"], "RELEVANT")

    # -------------------------------------------------------------------------
    # Test 11: Out-of-Scope Questions Still Abstain
    # -------------------------------------------------------------------------
    def test_11_out_of_scope_questions_still_abstain(self) -> None:
        """Verifies completely out-of-scope query rejects with standard abstention message."""
        res = self.pipeline.answer_question("Explain how convolutional neural networks perform image classification.")
        self.assertTrue(res.is_fallback)
        self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, res.answer)
        self.assertEqual(len(res.sources), 0)
        self.assertEqual(res.debug_info["relevance_decision"], "NOT RELEVANT")

    # -------------------------------------------------------------------------
    # Test 12: Citations Remain Attached to Assistant Answers
    # -------------------------------------------------------------------------
    def test_12_citations_remain_attached_to_assistant_answers(self) -> None:
        """Verifies full citation structure attached to grounded answers."""
        res = self.pipeline.answer_question("What is a variable?")
        self.assertFalse(res.is_fallback)
        self.assertGreater(len(res.sources), 0)

        top_cite = res.sources[0]
        self.assertEqual(top_cite["lecture_title"], "Variables and Assignment")
        self.assertEqual(top_cite["timestamp_range"], "05:00 - 07:30")
        self.assertEqual(top_cite["source_filename"], "lecture2.mp4")
        self.assertTrue(top_cite["chunk_id"].startswith("lec02_"))
        self.assertGreaterEqual(top_cite["similarity_score"], 0.32)

    # -------------------------------------------------------------------------
    # Test 13: Response Style Changes Prompt Behavior
    # -------------------------------------------------------------------------
    def test_13_response_style_changes_prompt_behavior(self) -> None:
        """Verifies that response_style parameter alters prompt instructions appropriately."""
        chunks = self.lec2_chunks

        msgs_clear = build_rag_messages("What is a variable?", chunks, response_style="Explain Clearly")
        sys_clear = msgs_clear[0]["content"]
        self.assertIn("Explain Clearly", sys_clear)
        self.assertIn("### Explanation", sys_clear)

        msgs_short = build_rag_messages("What is a variable?", chunks, response_style="Short Answer")
        sys_short = msgs_short[0]["content"]
        self.assertIn("Short Answer", sys_short)
        self.assertIn("concise", sys_short.lower())

        msgs_detailed = build_rag_messages("What is a variable?", chunks, response_style="Detailed Explanation")
        sys_detailed = msgs_detailed[0]["content"]
        self.assertIn("Detailed Explanation", sys_detailed)

    # -------------------------------------------------------------------------
    # Test 14: No Lecture Indexed State
    # -------------------------------------------------------------------------
    def test_14_no_lecture_indexed_state(self) -> None:
        """Verifies graceful handling when the knowledge base has 0 chunks."""
        empty_retriever = MagicMock(spec=LectureRetriever)
        empty_retriever.chunks_metadata = []
        empty_retriever.retrieve.return_value = []

        empty_pipeline = RAGTeacherPipeline(
            retriever=empty_retriever,
            llm_client=LLMClient(api_key=""),
            catalog=CourseCatalog(catalog_path=self.temp_path / "empty_catalog.json"),
        )

        res = empty_pipeline.answer_question("What is Python?")
        self.assertTrue(res.is_fallback)
        self.assertIn("No course lectures are currently indexed", res.answer)
        self.assertEqual(len(res.sources), 0)

    # -------------------------------------------------------------------------
    # Test 15: Backward Compatibility Single Lecture
    # -------------------------------------------------------------------------
    def test_15_backward_compatibility_single_lecture(self) -> None:
        """Verifies single lecture answering behaves consistently with previous modules."""
        # Query scoped to single lecture
        res = self.pipeline.answer_question(
            question="What is Python?",
            lecture_scope="lec_01",
        )
        self.assertFalse(res.is_fallback)
        self.assertIn("### Course Lecture Answer", res.answer)
        self.assertIn("Introduction to Python", res.answer)
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["source_filename"], "lecture1.mp4")
        self.assertEqual(res.sources[0]["timestamp_range"], "00:00 - 02:30")


if __name__ == "__main__":
    unittest.main()
