"""
tests/test_rag_pipeline.py
==========================
Comprehensive tests for Module 9: RAG Teacher Assistant pipeline,
grounded prompt engineering, LLM client integration, conversation history,
retrieval validation, source attribution, and out-of-context fallback.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import (
    DEFAULT_CONVERSATION_TURNS,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)
from src.llm import LLMClient, LLMError
from src.rag import (
    FALLBACK_NOT_FOUND_MESSAGE,
    RAGAnswer,
    RAGTeacherPipeline,
    RESPONSE_STYLE_PROMPTS,
    SYSTEM_TEACHER_PROMPT,
    build_rag_messages,
    format_rag_context,
    synthesize_grounded_answer,
)
from src.retrieval import LectureRetriever


class TestRAGPrompt(unittest.TestCase):
    """Tests for prompt formatting and chat message construction."""

    def setUp(self) -> None:
        self.sample_chunks = [
            {
                "rank": 1,
                "chunk_id": "lecture01_0001",
                "source_file": "lecture01.mp4",
                "start_time": 65.0,  # 01:05
                "end_time": 130.0,  # 02:10
                "similarity_score": 0.885,
                "text": "Supervised learning uses labeled training datasets to train algorithms.",
            },
            {
                "rank": 2,
                "chunk_id": "lecture01_0002",
                "source_file": "lecture01.mp4",
                "start_time": 135.0,  # 02:15
                "end_time": 190.0,  # 03:10
                "similarity_score": 0.742,
                "text": "Unsupervised learning discovers hidden patterns in unlabeled data.",
            },
        ]

    def test_format_rag_context_empty(self) -> None:
        """Verifies that an empty chunk list outputs a clear notice."""
        result = format_rag_context([])
        self.assertIn("No matching lecture excerpts", result)

    def test_format_rag_context_with_chunks(self) -> None:
        """Verifies that chunks are formatted with timestamps, source, chunk ID, and score."""
        result = format_rag_context(self.sample_chunks)
        self.assertIn("--- EXCERPT 1 ---", result)
        self.assertIn("Source Video: lecture01.mp4", result)
        self.assertIn("Timestamp: 01:05 - 02:10", result)
        self.assertIn("Chunk ID: lecture01_0001", result)
        self.assertIn("0.885", result)
        self.assertIn("Supervised learning uses labeled", result)

        self.assertIn("--- EXCERPT 2 ---", result)
        self.assertIn("Timestamp: 02:15 - 03:10", result)
        self.assertIn("Unsupervised learning discovers", result)

    def test_build_rag_messages_structure(self) -> None:
        """Verifies message construction with system prompt, context, and student question."""
        messages = build_rag_messages(
            question="What is supervised learning?",
            retrieved_chunks=self.sample_chunks,
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("AI Teacher Assistant", messages[0]["content"])
        self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, messages[1]["content"])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("STUDENT QUESTION:", messages[-1]["content"])
        self.assertIn("What is supervised learning?", messages[-1]["content"])
        self.assertIn("01:05 - 02:10", messages[-1]["content"])

    def test_build_rag_messages_with_conversation_history(self) -> None:
        """Verifies that recent conversation turns are included for follow-up continuity."""
        history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a field of AI..."},
            {"role": "user", "content": "Give me an example."},
            {"role": "assistant", "content": "For example, spam filtering..."},
        ]
        messages = build_rag_messages(
            question="Can you give another one?",
            retrieved_chunks=self.sample_chunks,
            conversation_history=history,
            max_history_turns=2,
        )

        # System message (1) + 4 history messages + Current prompt (1) = 6 total
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is machine learning?")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Can you give another one?", messages[-1]["content"])


class TestLLMClient(unittest.TestCase):
    """Tests for LLMClient API handling, error handling, and mock responses."""

    def test_unconfigured_client_raises_error_on_generate(self) -> None:
        """Verifies that an unconfigured client raises LLMError with clear instructions."""
        client = LLMClient(api_key="")
        self.assertFalse(client.is_configured())
        with self.assertRaises(LLMError) as ctx:
            client.generate([{"role": "user", "content": "Hello"}])
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_empty_messages_raises_error(self) -> None:
        """Verifies that passing empty messages raises an error."""
        client = LLMClient(api_key="sk-test-fake-key-for-unit-testing")
        with self.assertRaises(LLMError):
            client.generate([])

    @patch("openai.OpenAI")
    def test_client_generate_success(self, mock_openai_cls: MagicMock) -> None:
        """Verifies that client calls OpenAI chat completions correctly and returns answer."""
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        # Mock completion response structure
        mock_choice = MagicMock()
        mock_choice.message.content = "Supervised learning maps inputs to outputs using labels."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="sk-test-valid-dummy-key", model_name="gpt-4o-mini")
        client._client = mock_instance

        answer = client.generate([{"role": "user", "content": "Explain supervised learning"}])
        self.assertEqual(answer, "Supervised learning maps inputs to outputs using labels.")
        mock_instance.chat.completions.create.assert_called_once()


class TestRAGTeacherPipeline(unittest.TestCase):
    """Tests for unified RAGTeacherPipeline end-to-end integration."""

    def setUp(self) -> None:
        # Mock LectureRetriever
        self.mock_retriever = MagicMock(spec=LectureRetriever)
        self.mock_retriever.model_name = "all-MiniLM-L6-v2"
        self.mock_retriever.chunks_metadata = [
            {
                "chunk_id": "test_chunk_0001",
                "source_file": "python_lecture.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "word_count": 15,
                "text": "Machine learning is a subset of artificial intelligence.",
            }
        ]

        # Mock LLMClient
        self.mock_llm = MagicMock(spec=LLMClient)
        self.mock_llm.model_name = "gpt-4o-mini"
        self.mock_llm.is_configured.return_value = True
        self.mock_llm.generate.return_value = (
            "Machine learning is a subset of AI as mentioned in python_lecture.mp4 (00:00 - 00:10)."
        )

        self.pipeline = RAGTeacherPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm,
        )

    def test_get_corpus_stats(self) -> None:
        """Verifies corpus statistics reporting."""
        stats = self.pipeline.get_corpus_stats()
        self.assertEqual(stats["total_chunks"], 1)
        self.assertEqual(stats["unique_sources"], ["python_lecture.mp4"])
        self.assertEqual(stats["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(stats["llm_model"], "gpt-4o-mini")
        self.assertTrue(stats["llm_configured"])

    def test_answer_question_with_retrieval_and_llm(self) -> None:
        """Verifies full question answering flow with source attribution."""
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "test_chunk_0001",
                "source_file": "python_lecture.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "similarity_score": 0.85,
                "text": "Machine learning is a subset of artificial intelligence.",
            }
        ]

        res = self.pipeline.answer_question(
            question="What is machine learning?",
            top_k=3,
        )

        self.assertIsInstance(res, RAGAnswer)
        self.assertEqual(res.question, "What is machine learning?")
        self.assertIn("subset of AI", res.answer)
        self.assertFalse(res.is_fallback)
        self.assertEqual(len(res.sources), 1)

        src = res.sources[0]
        self.assertEqual(src["source_file"], "python_lecture.mp4")
        self.assertEqual(src["start_formatted"], "00:00")
        self.assertEqual(src["end_formatted"], "00:10")
        self.assertEqual(src["timestamp_range"], "00:00 - 00:10")
        self.assertEqual(src["similarity_score"], 0.85)

        # Check debug diagnostics
        self.assertEqual(res.debug_info["retrieved_count"], 1)
        self.assertEqual(res.debug_info["chunk_ids"], ["test_chunk_0001"])
        self.assertTrue(res.debug_info["llm_called"])

    def test_answer_question_out_of_context_fallback(self) -> None:
        """
        Step 11 requirement: Test a question whose answer does NOT exist in the content.
        Verifies that system outputs the exact fallback message without hallucination.
        """
        # When no chunks match (e.g. out of domain question)
        self.mock_retriever.retrieve.return_value = []

        res = self.pipeline.answer_question(
            question="How do you build a nuclear reactor in Minecraft?",
        )

        self.assertIsInstance(res, RAGAnswer)
        self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, res.answer)
        self.assertIn("### Course Lecture Answer", res.answer)
        self.assertTrue(res.is_fallback)
        self.assertEqual(len(res.sources), 0)
        self.assertEqual(res.debug_info["retrieved_count"], 0)
        self.assertFalse(res.debug_info["llm_called"])

    def test_answer_question_empty_query(self) -> None:
        """Verifies handling of blank/whitespace question."""
        res = self.pipeline.answer_question("   ")
        self.assertTrue(res.is_fallback)
        self.assertIn("valid educational question", res.answer)

    def test_answer_question_offline_synthesis_when_unconfigured(self) -> None:
        """Verifies that an unconfigured pipeline provides grounded local context synthesis."""
        self.mock_llm.is_configured.return_value = False
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "test_chunk_0001",
                "source_file": "python_lecture.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "similarity_score": 0.75,
                "text": "Machine learning is a subset of artificial intelligence.",
            }
        ]

        res = self.pipeline.answer_question("What is machine learning?")
        self.assertFalse(res.is_fallback)
        self.assertIn("python_lecture.mp4", res.answer)
        self.assertIn("00:00 - 00:10", res.answer)
        self.assertIn("Machine learning is a subset", res.answer)
        self.assertTrue(res.debug_info["offline_synthesis"])

    def test_answer_question_follow_up_awareness(self) -> None:
        """Verifies that conversation history is passed into prompt assembly."""
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "test_chunk_0001",
                "source_file": "python_lecture.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "similarity_score": 0.85,
                "text": "Supervised learning maps input features to target labels.",
            }
        ]

        history = [
            {"role": "user", "content": "What is supervised learning?"},
            {"role": "assistant", "content": "It maps inputs to outputs."},
        ]

        res = self.pipeline.answer_question(
            question="Can you give an example of it?",
            conversation_history=history,
        )

        self.assertFalse(res.is_fallback)
        self.mock_llm.generate.assert_called_once()
        # Check that messages passed to LLM included conversation history
        call_args = self.mock_llm.generate.call_args[0][0]
        roles = [m["role"] for m in call_args]
        self.assertIn("system", roles)
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_relevance_test1_assignment_vs_equality_is_relevant(self) -> None:
        """
        TEST 1 (Step 10): Relevant question regarding assignment vs equality.
        Expected: RELEVANT, sources include assignment vs equality section, answer generated.
        """
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "lecture_chunk_0012",
                "source_file": "python_lecture.mp3",
                "start_time": 2637.0,  # 43:57
                "end_time": 2905.0,    # 48:25
                "similarity_score": 0.5542,
                "text": (
                    "In Python, the equal sign is an assignment operator rather than mathematical equality. "
                    "An assignment binds a value to a variable name."
                ),
            }
        ]

        res = self.pipeline.answer_question(
            question="What is the difference between assignment and equality in computer science?",
        )

        self.assertFalse(res.is_fallback)
        self.assertEqual(res.debug_info["relevance_decision"], "RELEVANT")
        self.assertGreaterEqual(res.debug_info["top_score"], DEFAULT_RELEVANCE_THRESHOLD)
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["chunk_id"], "lecture_chunk_0012")
        self.assertIn("43:57 - 48:25", res.sources[0]["timestamp_range"])

    def test_relevance_test2_expression_evaluation_is_relevant(self) -> None:
        """
        TEST 2 (Step 10): Relevant question regarding Python expression evaluation.
        Expected: RELEVANT, sources discuss expressions/evaluation from lecture.
        """
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "lecture_chunk_0008",
                "source_file": "python_lecture.mp3",
                "start_time": 1884.0,  # 31:24
                "end_time": 2178.0,    # 36:18
                "similarity_score": 0.6407,
                "text": "When Python evaluates an expression like 3 plus 2, it reduces it to a value of type integer.",
            }
        ]

        res = self.pipeline.answer_question(
            question="What does Python do when it evaluates an expression?",
        )

        self.assertFalse(res.is_fallback)
        self.assertEqual(res.debug_info["relevance_decision"], "RELEVANT")
        self.assertGreaterEqual(res.debug_info["top_score"], DEFAULT_RELEVANCE_THRESHOLD)
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["chunk_id"], "lecture_chunk_0008")
        self.assertIn("31:24 - 36:18", res.sources[0]["timestamp_range"])

    def test_relevance_test3_out_of_scope_cnn_abstains(self) -> None:
        """
        TEST 3 (Step 10): Out-of-scope question regarding convolutional neural networks.
        Expected: NOT RELEVANT, abstains with FALLBACK_NOT_FOUND_MESSAGE, LLM not called, no sources.
        """
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "lecture_chunk_0001",
                "source_file": "python_lecture.mp3",
                "start_time": 0.0,
                "end_time": 7.0,
                "similarity_score": 0.3121,
                "text": "Today we are going to study machine learning. Machine learning is a subset of artificial intelligence.",
            }
        ]

        # Reset mock LLM to verify it is NOT called
        self.mock_llm.generate.reset_mock()

        res = self.pipeline.answer_question(
            question="Explain how convolutional neural networks perform image classification.",
        )

        self.assertTrue(res.is_fallback)
        self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, res.answer)
        self.assertEqual(res.debug_info["relevance_decision"], "NOT RELEVANT")
        self.assertFalse(res.debug_info["llm_called"])
        self.assertEqual(len(res.sources), 0)
        self.mock_llm.generate.assert_not_called()

    def test_adaptive_retrieval_handles_broad_beginning_question(self) -> None:
        """
        Regression test: Verifies that broad natural language questions (e.g. score ~0.39)
        are accepted as relevant when passing the threshold.
        """
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "test_chunk_0001",
                "source_file": "python_lecture.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "similarity_score": 0.3973,
                "text": "All right, welcome to the lecture. Today we cover Python fundamentals.",
            }
        ]

        res = self.pipeline.answer_question(
            question="What topics are introduced at the beginning of the lecture?",
            similarity_threshold=None,
        )

        self.assertFalse(res.is_fallback)
        self.assertEqual(res.debug_info["relevance_decision"], "RELEVANT")
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["chunk_id"], "test_chunk_0001")

    def test_adaptive_retrieval_rejects_sub_noise_floor_queries(self) -> None:
        """
        Verifies that out-of-domain queries whose scores fall below threshold (< 0.32)
        are rejected with the abstention fallback message.
        """
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "test_chunk_0001",
                "source_file": "python_lecture.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "similarity_score": 0.1227,
                "text": "Today we are going to study machine learning.",
            }
        ]

        res = self.pipeline.answer_question(
            question="What is the capital city of France?",
            similarity_threshold=None,
        )

        self.assertTrue(res.is_fallback)
        self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, res.answer)
        self.assertEqual(res.debug_info["relevance_decision"], "NOT RELEVANT")
        self.assertEqual(len(res.sources), 0)

    def test_generalization_relevant_variable_creation(self) -> None:
        """Generalization: Relevant question regarding variable binding (score ~0.49)."""
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "chunk_0012",
                "source_file": "python_lecture.mp3",
                "start_time": 2637.0,
                "end_time": 2905.0,
                "similarity_score": 0.4891,
                "text": "Variables in Python are created and bound to values using assignment statements.",
            }
        ]
        res = self.pipeline.answer_question("How are variables created in Python?")
        self.assertFalse(res.is_fallback)
        self.assertEqual(res.debug_info["relevance_decision"], "RELEVANT")
        self.assertEqual(len(res.sources), 1)

    def test_generalization_unrelated_baking_cake(self) -> None:
        """Generalization: Unrelated question regarding baking cake (score ~0.19) is rejected."""
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "chunk_0003",
                "source_file": "python_lecture.mp3",
                "start_time": 469.0,
                "end_time": 738.0,
                "similarity_score": 0.1963,
                "text": "Code optimization and memory.",
            }
        ]
        res = self.pipeline.answer_question("How do you bake a chocolate cake in the oven?")
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.debug_info["relevance_decision"], "NOT RELEVANT")
        self.assertEqual(len(res.sources), 0)

    def test_generalization_unrelated_gradient_descent(self) -> None:
        """Generalization: Unrelated advanced ML question not in lecture (score ~0.25) is rejected."""
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "chunk_0002",
                "source_file": "python_lecture.mp3",
                "start_time": 244.0,
                "end_time": 495.0,
                "similarity_score": 0.2482,
                "text": "We are going to start to organize our code.",
            }
        ]
        res = self.pipeline.answer_question("How does gradient descent optimize neural network weights?")
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.debug_info["relevance_decision"], "NOT RELEVANT")
        self.assertEqual(len(res.sources), 0)

    @patch("src.rag.pipeline.save_dataset")
    @patch("src.rag.pipeline.build_dataset")
    @patch("src.rag.pipeline.embed_chunk_file")
    @patch("src.rag.pipeline.save_chunks")
    @patch("src.rag.pipeline.chunk_transcript")
    @patch("src.rag.pipeline.save_processed_transcript")
    @patch("src.rag.pipeline.process_transcript")
    @patch("src.rag.pipeline.transcribe_audio")
    def test_index_audio_successful_with_overwrite(
        self,
        mock_transcribe: MagicMock,
        mock_process: MagicMock,
        mock_save_proc: MagicMock,
        mock_chunk: MagicMock,
        mock_save_chunks: MagicMock,
        mock_embed: MagicMock,
        mock_build_df: MagicMock,
        mock_save_df: MagicMock,
    ) -> None:
        """Verifies that index_audio passes overwrite parameter cleanly to save_chunks and embed_chunk_file."""
        import pandas as pd
        import tempfile
        from pathlib import Path

        mock_transcribe.return_value = {"segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "Hello"}]}
        mock_process.return_value = ({"segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "Hello"}]}, {})
        mock_chunk.return_value = {"chunks": [{"chunk_id": "c1", "text": "Hello"}]}
        mock_save_chunks.return_value = Path("dummy_chunk_file.json")
        mock_build_df.return_value = pd.DataFrame([{"chunk_id": "c1", "text": "Hello"}])

        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)
            result = self.pipeline.index_audio(tmp_audio_path, overwrite=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["chunk_count"], 1)
        mock_save_chunks.assert_called_once()
        _, kwargs = mock_save_chunks.call_args
        self.assertTrue(kwargs.get("overwrite"))

    def test_concise_prompt_instructions(self) -> None:
        """Verifies that prompt instructions explicitly forbid transcript dumping and enforce conciseness."""
        self.assertIn("Answer the student's question directly", SYSTEM_TEACHER_PROMPT)
        self.assertIn("not everything mentioned in the retrieved lecture transcript", SYSTEM_TEACHER_PROMPT)
        self.assertIn("Source context may contain unrelated surrounding lecture material", SYSTEM_TEACHER_PROMPT)
        self.assertIn("Do not copy, reproduce, or summarize the entire context", SYSTEM_TEACHER_PROMPT)

        messages = build_rag_messages(
            question="What does Python do when it evaluates an expression?",
            retrieved_chunks=[
                {
                    "rank": 1,
                    "chunk_id": "c1",
                    "source_file": "lecture.mp3",
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "similarity_score": 0.9,
                    "text": "When Python evaluates an expression, it calculates it to a single value.",
                }
            ],
            response_style="Short Answer",
        )
        user_msg = messages[-1]["content"]
        self.assertIn("Answer the question that was asked", user_msg)
        self.assertIn("Do NOT dump the transcript", user_msg)
        self.assertIn("REFERENCE EXAMPLES FOR CONCISENESS", user_msg)

    def test_response_styles_target_lengths(self) -> None:
        """Verifies that response styles define targeted word counts."""
        short_prompt = RESPONSE_STYLE_PROMPTS["Short Answer"]
        clear_prompt = RESPONSE_STYLE_PROMPTS["Explain Clearly"]
        detailed_prompt = RESPONSE_STYLE_PROMPTS["Detailed Explanation"]

        self.assertIn("30–70 words", short_prompt)
        self.assertIn("60–120 words", clear_prompt)
        self.assertIn("120–220 words", detailed_prompt)

    def test_synthesize_grounded_answer_avoids_transcript_dumping(self) -> None:
        """Verifies that local synthesis extracts relevant sentences and avoids dumping a long transcript."""
        long_transcript = (
            "And the type of 3 plus 2 is an integer. Right? So basically what I've done here, I've put an expression within this type command. "
            "And that's okay. Right? That's, in fact, encouraged in Python. You don't just want to calculate and then stick in. "
            "So the important thing to remember when we're doing expressions is Python reads the expression, but it does not store the expression in memory. "
            "What it does is it reads the expression, evaluates it to one single value, and then it stores the result value in memory. "
            "So it never stores the expression, it evaluates the expression, and then stores the value. "
            "Let's have you try these in our console. What are the values of the following expressions? 13 minus 4 divided by 12 times 12."
        )

        # Short Answer
        ans_short = synthesize_grounded_answer(
            question="What does Python do when it evaluates an expression?",
            source_text=long_transcript,
            lecture_title="Python Basics",
            timestamp_range="31:24 - 36:18",
            response_style="Short Answer",
        )
        words_short = len(ans_short.split())
        self.assertLessEqual(words_short, 70)
        self.assertNotIn("13 minus 4 divided by 12", ans_short)
        self.assertNotIn("console", ans_short)
        self.assertIn("evaluates", ans_short.lower())

        # Explain Clearly
        ans_clear = synthesize_grounded_answer(
            question="What does Python do when it evaluates an expression?",
            source_text=long_transcript,
            lecture_title="Python Basics",
            timestamp_range="31:24 - 36:18",
            response_style="Explain Clearly",
        )
        words_clear = len(ans_clear.split())
        self.assertLessEqual(words_clear, 120)
        self.assertIn("### Explanation", ans_clear)
        self.assertNotIn("console", ans_clear)

    def test_llm_error_fallback_avoids_transcript_dumping(self) -> None:
        """Verifies that when LLM encounters an error, fallback answer does not dump raw transcript."""
        self.mock_llm.generate.side_effect = LLMError("Rate limit exceeded")
        self.mock_retriever.retrieve.return_value = [
            {
                "rank": 1,
                "chunk_id": "chunk_long",
                "source_file": "lecture.mp3",
                "start_time": 100.0,
                "end_time": 200.0,
                "similarity_score": 0.82,
                "text": (
                    "The equal sign binds this value to this variable name in memory. "
                    "So in memory, there is going to be a space called x that points to that value. "
                    "Let's have you try this in console. What are the values of these? I don't know off the top of my head."
                ),
            }
        ]

        res = self.pipeline.answer_question(
            question="What is a variable in Python?",
            response_style="Short Answer",
        )
        self.assertFalse(res.is_fallback)
        self.assertIn("binds this value to this variable name", res.answer)
        self.assertNotIn("console", res.answer)
        self.assertNotIn("off the top of my head", res.answer)
        self.assertTrue(res.debug_info["offline_synthesis"])

    def test_example_query_expansion_for_expression(self) -> None:
        """Verifies controlled query expansion for short example queries."""
        from src.rag import build_retrieval_query, extract_example_topic

        topic = extract_example_topic("Give me an example of an expression.")
        self.assertIsNotNone(topic)
        self.assertIn("expression", topic)

        eff_q, is_fu = build_retrieval_query("Give me an example of an expression.")
        self.assertTrue(is_fu)
        self.assertIn("expression", eff_q.lower())
        self.assertIn("3 + 2", eff_q)

        # Non-example query should remain unexpanded
        eff_q2, is_fu2 = build_retrieval_query("How do quantum computers calculate prime factors?")
        self.assertFalse(is_fu2)
        self.assertEqual(eff_q2, "How do quantum computers calculate prime factors?")

    def test_answers_omit_review_timestamp_ui_text(self) -> None:
        """Verifies that generated answers do not duplicate UI timestamp citation phrases."""
        sample_text = (
            "Python reads the expression, evaluates it to one single value, and then it stores the result value in memory. "
            "It never stores the expression, it evaluates the expression, and then stores the value. "
            "For example, 3 plus 2 evaluates to 5."
        )
        for style in ["Short Answer", "Explain Clearly", "Detailed Explanation"]:
            ans = synthesize_grounded_answer(
                question="What does Python do when it evaluates an expression?",
                source_text=sample_text,
                lecture_title="Python Lecture",
                timestamp_range="31:24 - 36:18",
                response_style=style,
            )
            self.assertNotIn("Review the timestamp interval", ans)
            self.assertNotIn("Review the timestamp", ans)

    def test_retriever_warmup_interface(self) -> None:
        """Verifies that LectureRetriever.warmup() pre-loads embedding model without errors."""
        mock_retriever = LectureRetriever(
            embeddings_dir=self.pipeline.embeddings_dir,
            model=MagicMock(),
            auto_load=False,
        )
        # Should execute cleanly without raising
        mock_retriever.warmup()
        self.assertIsNotNone(mock_retriever.model)


if __name__ == "__main__":
    unittest.main()


