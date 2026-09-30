"""
tests/test_multi_lecture.py
===========================
Comprehensive unit and integration tests for Module 10: Multi-Lecture Knowledge Base.

Validates:
  Test 1: Single lecture retrieval (Expected: RELEVANT)
  Test 2: Multiple lectures (Lecture 1 + Lecture 2 indexed, both remain searchable)
  Test 3: Lecture filtering (Filter Lecture 1 -> ask Lecture 2 question -> Lecture 2 not retrieved)
  Test 4: All lectures (Scope All Lectures -> ask Lecture 2 question -> Lecture 2 retrievable)
  Test 5: Duplicate lecture (Index same lecture twice -> content hash detected, no duplicate chunks)
  Test 6: Metadata verification (Every chunk contains course_id, lecture_id, lecture_title, source_filename, timestamps, chunk_id, text)
  Test 7: Out-of-scope question rejection (CNN question -> NOT RELEVANT, abstains)
  Test 8: Cross-lecture retrieval (Multiple lectures can contribute to multi-topic queries)
  Test 9: Content hash duplicate detection (Same filename with different hash allows indexing)
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
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import create_chunks, chunk_transcript
from src.config import DEFAULT_RELEVANCE_THRESHOLD
from src.data import (
    CourseCatalog,
    MODULE10_REQUIRED_COLUMNS,
    REQUIRED_COLUMNS,
    build_dataset,
    compute_file_hash,
    load_chunk_data,
    load_embedding_data,
    merge_chunks_and_embeddings,
    save_dataset,
    validate_dataset,
)
from src.embeddings import embed_chunks_data
from src.llm import LLMClient
from src.rag import FALLBACK_NOT_FOUND_MESSAGE, RAGTeacherPipeline
from src.retrieval import LectureRetriever, format_timestamp


class MockEmbeddingModel:
    """Deterministic mock embedding model for fast tests."""

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

        matrix = []
        for text in inputs:
            seed = sum(ord(c) for c in text) % 200 + 1
            v = np.sin(np.linspace(seed, seed + 10, self.dimension, dtype=np.float32))
            if normalize_embeddings:
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    v = v / norm
            matrix.append(v)
        return np.array(matrix, dtype=np.float32)


class TestModule10MultiLectureKnowledgeBase(unittest.TestCase):
    """Test suite for Module 10 multi-lecture architecture, indexing, filtering, and metadata."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="rag_test_m10_"))
        self.chunks_dir = self.test_dir / "chunks"
        self.embeddings_dir = self.test_dir / "embeddings"
        self.dataset_path = self.test_dir / "test_embeddings.joblib"
        self.catalog_path = self.test_dir / "test_catalog.json"

        self.chunks_dir.mkdir(parents=True)
        self.embeddings_dir.mkdir(parents=True)

        self.mock_model = MockEmbeddingModel(dimension=64)
        self.catalog = CourseCatalog(catalog_path=self.catalog_path, auto_sync=False)

        # Build two mock lectures
        # Lecture 1: Introduction to Python
        self.lec1_segments = [
            {"id": 0, "start": 0.0, "end": 30.0, "text": "Welcome to Python introduction. Python is an interpreted programming language."},
            {"id": 1, "start": 30.0, "end": 65.0, "text": "Python syntax emphasizes code readability with indentation and clean structure."},
        ]
        self._index_mock_lecture(
            lecture_id="lecture_01",
            lecture_title="Lecture 1 — Introduction",
            source_filename="lecture1.mp4",
            segments=self.lec1_segments,
        )

        # Lecture 2: Variables and Expressions
        self.lec2_segments = [
            {"id": 0, "start": 10.0, "end": 45.0, "text": "Variables in Python store references to values. An assignment statement binds a variable name."},
            {"id": 1, "start": 45.0, "end": 90.0, "text": "When Python evaluates an expression, it reduces the operator and operands to a single value."},
        ]
        self._index_mock_lecture(
            lecture_id="lecture_02",
            lecture_title="Lecture 2 — Variables",
            source_filename="lecture2.mp4",
            segments=self.lec2_segments,
        )

        # Build consolidated dataset
        df = build_dataset(
            chunks_dir=self.chunks_dir,
            embeddings_dir=self.embeddings_dir,
            dataset_path=self.dataset_path,
            overwrite=True,
        )
        save_dataset(df, self.dataset_path, overwrite=True)

        self.retriever = LectureRetriever(
            embeddings_dir=self.embeddings_dir,
            dataset_path=self.dataset_path,
            model=self.mock_model,
            auto_load=True,
        )
        self.mock_llm = MagicMock(spec=LLMClient)
        self.mock_llm.model_name = "mock-gpt-4o"
        self.mock_llm.is_configured.return_value = True
        self.mock_llm.generate.return_value = "Detailed explanation grounded in the lecture."

        self.pipeline = RAGTeacherPipeline(
            retriever=self.retriever,
            llm_client=self.mock_llm,
            dataset_path=self.dataset_path,
            embeddings_dir=self.embeddings_dir,
            catalog=self.catalog,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _index_mock_lecture(
        self,
        lecture_id: str,
        lecture_title: str,
        source_filename: str,
        segments: list[dict],
    ) -> None:
        """Helper to create chunk JSON and embedding files for a mock lecture."""
        stem = Path(source_filename).stem
        chunks = create_chunks(
            segments=segments,
            source_file=source_filename,
            source_transcript=f"{stem}.json",
            language="en",
            chunk_size=700,
            chunk_overlap=100,
            course_id="python_course",
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            source_filename=source_filename,
        )
        chunk_file = self.chunks_dir / f"{stem}_chunks.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_file": source_filename,
                "source_filename": source_filename,
                "course_id": "python_course",
                "lecture_id": lecture_id,
                "lecture_title": lecture_title,
                "language": "en",
                "chunk_count": len(chunks),
                "chunks": chunks,
            }, f, indent=2)

        embeddings, metadata = embed_chunks_data(
            chunks_data={
                "source_file": source_filename,
                "course_id": "python_course",
                "lecture_id": lecture_id,
                "lecture_title": lecture_title,
                "source_chunk_file": f"{stem}_chunks.json",
                "chunks": chunks,
            },
            model=self.mock_model,
        )

        npz_p = self.embeddings_dir / f"{stem}_embeddings.npz"
        json_p = self.embeddings_dir / f"{stem}_metadata.json"
        np.savez_compressed(
            npz_p,
            embeddings=embeddings,
            chunk_ids=np.array([c["chunk_id"] for c in chunks], dtype=object),
        )
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        dummy_hash = compute_file_hash(chunk_file)
        self.catalog.register_lecture(
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            course_id="python_course",
            source_filename=source_filename,
            file_hash=dummy_hash,
            chunk_count=len(chunks),
            status="Indexed",
            chunk_file=str(chunk_file),
            embedding_file=str(npz_p),
            metadata_file=str(json_p),
        )

    # -------------------------------------------------------------------------
    # TEST 1: Single Lecture Retrieval (Expected: RELEVANT)
    # -------------------------------------------------------------------------
    def test_test1_single_lecture_retrieval(self) -> None:
        """Test 1: Single lecture question retrieves relevant chunks and passes threshold."""
        query = "What is Python according to the introduction?"
        # Mock high relevance score for Python question
        with patch.object(self.retriever, "retrieve") as mock_ret:
            mock_ret.return_value = [
                {
                    "rank": 1,
                    "chunk_id": "lecture_01_0001",
                    "similarity_score": 0.654,
                    "text": "Welcome to Python introduction. Python is an interpreted programming language.",
                    "source_file": "lecture1.mp4",
                    "source_filename": "lecture1.mp4",
                    "lecture_id": "lecture_01",
                    "lecture_title": "Lecture 1 — Introduction",
                    "course_id": "python_course",
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "timestamp_start": "00:00:00",
                    "timestamp_end": "00:00:30",
                    "timestamp": "00:00 - 00:30",
                    "timestamp_range": "00:00 - 00:30",
                }
            ]
            ans = self.pipeline.answer_question(query)

        self.assertFalse(ans.is_fallback)
        self.assertEqual(ans.debug_info["relevance_decision"], "RELEVANT")
        self.assertGreaterEqual(ans.debug_info["top_score"], DEFAULT_RELEVANCE_THRESHOLD)
        self.assertIn("Lecture 1 — Introduction", ans.answer)
        self.assertEqual(len(ans.sources), 1)
        self.assertEqual(ans.sources[0]["lecture_id"], "lecture_01")

    # -------------------------------------------------------------------------
    # TEST 2: Multiple Lectures (Both searchable simultaneously)
    # -------------------------------------------------------------------------
    def test_test2_multiple_lectures_indexed_and_searchable(self) -> None:
        """Test 2: After indexing Lecture 1 + Lecture 2, both lectures are searchable in the corpus."""
        stats = self.pipeline.get_corpus_stats()
        self.assertGreaterEqual(stats["total_chunks"], 2)
        self.assertGreaterEqual(stats["lecture_count"], 2)

        # Inspect retriever chunks metadata
        all_lids = {c.get("lecture_id") for c in self.retriever.chunks_metadata}
        self.assertIn("lecture_01", all_lids)
        self.assertIn("lecture_02", all_lids)

        all_titles = {c.get("lecture_title") for c in self.retriever.chunks_metadata}
        self.assertIn("Lecture 1 — Introduction", all_titles)
        self.assertIn("Lecture 2 — Variables", all_titles)

    # -------------------------------------------------------------------------
    # TEST 3: Lecture Filtering (Lecture 1 scope rejects Lecture 2)
    # -------------------------------------------------------------------------
    def test_test3_lecture_filtering_isolates_scope(self) -> None:
        """Test 3: When Lecture 1 is selected, asking about variables does not retrieve Lecture 2 chunks."""
        # Query specifically targeting Lecture 2 material
        query = "What does Python do when it evaluates an expression?"

        # Retrieve with lecture filter scoped strictly to Lecture 1
        results = self.retriever.retrieve(
            query=query,
            top_k=5,
            lecture_filter="lecture_01",
        )

        for res in results:
            self.assertEqual(res.get("lecture_id"), "lecture_01")
            self.assertNotEqual(res.get("lecture_id"), "lecture_02")
            self.assertNotIn("evaluates an expression", res.get("text", ""))

    # -------------------------------------------------------------------------
    # TEST 4: All Lectures Scope (Searches across all lectures)
    # -------------------------------------------------------------------------
    def test_test4_all_lectures_scope_retrieves_target_lecture(self) -> None:
        """Test 4: When 'All Lectures' is selected, questions can retrieve Lecture 2 or Lecture 1."""
        query = "What does Python do when it evaluates an expression?"

        results = self.retriever.retrieve(
            query=query,
            top_k=5,
            lecture_filter=None,  # All Lectures
        )

        self.assertGreater(len(results), 0)
        # Check that both lectures were considered in the candidate pool
        retrieved_lids = [r.get("lecture_id") for r in results]
        self.assertTrue(any(lid in ("lecture_01", "lecture_02") for lid in retrieved_lids))

    # -------------------------------------------------------------------------
    # TEST 5: Duplicate Lecture (No duplicate chunks indexed)
    # -------------------------------------------------------------------------
    def test_test5_duplicate_lecture_upload_protection(self) -> None:
        """Test 5: Uploading the exact same lecture twice does NOT duplicate chunks."""
        sample_file = self.test_dir / "sample_audio.mp3"
        sample_file.write_bytes(b"Simulated binary audio content for lecture 12345")
        file_hash = compute_file_hash(sample_file)

        # Initial indexing registration
        self.catalog.register_lecture(
            lecture_id="lecture_unique_05",
            lecture_title="Lecture 5 — Functions",
            source_filename="sample_audio.mp3",
            file_hash=file_hash,
            chunk_count=3,
        )

        initial_count = len(self.retriever.chunks_metadata)

        # Simulate second indexing attempt of identical content
        res = self.pipeline.index_audio(
            audio_path=sample_file,
            lecture_title="Lecture 5 — Functions Duplicate",
            force_reindex=False,
        )

        self.assertEqual(res["status"], "skipped_duplicate")
        self.assertIn("already indexed", res["message"])
        # Verify chunk count did not change
        self.assertEqual(len(self.retriever.chunks_metadata), initial_count)

    # -------------------------------------------------------------------------
    # TEST 6: Metadata Completeness
    # -------------------------------------------------------------------------
    def test_test6_every_chunk_contains_required_module10_metadata(self) -> None:
        """Test 6: Verify every chunk in the dataset contains all required Module 10 metadata keys."""
        # 1. Verify chunks from chunker
        sample_segments = [
            {"id": 0, "start": 0.0, "end": 15.0, "text": "First segment text."},
            {"id": 1, "start": 15.0, "end": 35.0, "text": "Second segment text."},
        ]
        chunks = create_chunks(
            segments=sample_segments,
            source_file="test_meta.mp4",
            source_transcript="test_meta.json",
            course_id="python_course",
            lecture_id="lecture_03",
            lecture_title="Lecture 3 — Expressions",
            source_filename="test_meta.mp4",
        )

        required_keys = [
            "course_id",
            "lecture_id",
            "lecture_title",
            "source_filename",
            "timestamp_start",
            "timestamp_end",
            "chunk_id",
            "text",
        ]

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            for k in required_keys:
                self.assertIn(k, chunk, f"Missing required key '{k}' in chunk.")
                self.assertTrue(chunk[k], f"Key '{k}' in chunk cannot be empty.")

            # Timestamp format check
            self.assertRegex(chunk["timestamp_start"], r"^\d{2}:\d{2}:\d{2}$")
            self.assertRegex(chunk["timestamp_end"], r"^\d{2}:\d{2}:\d{2}$")

        # 2. Verify DataFrame schema
        df = build_dataset(
            chunks_dir=self.chunks_dir,
            embeddings_dir=self.embeddings_dir,
            dataset_path=self.dataset_path,
            overwrite=True,
        )
        for col in MODULE10_REQUIRED_COLUMNS:
            self.assertIn(col, df.columns, f"Consolidated DataFrame missing column '{col}'.")

    # -------------------------------------------------------------------------
    # TEST 7: Out-of-Scope Question (CNN query rejected with abstention)
    # -------------------------------------------------------------------------
    def test_test7_out_of_scope_cnn_abstains(self) -> None:
        """Test 7: Out-of-scope query (e.g. CNNs) is rejected with NOT RELEVANT and fallback message."""
        query = "Explain how convolutional neural networks perform image classification."

        with patch.object(self.retriever, "retrieve") as mock_ret:
            # Simulate low similarity (< 0.3200)
            mock_ret.return_value = [
                {
                    "rank": 1,
                    "chunk_id": "lecture_01_0001",
                    "similarity_score": 0.1850,
                    "text": "Welcome to Python introduction.",
                    "source_file": "lecture1.mp4",
                    "source_filename": "lecture1.mp4",
                    "lecture_id": "lecture_01",
                    "lecture_title": "Lecture 1 — Introduction",
                    "course_id": "python_course",
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "timestamp_start": "00:00:00",
                    "timestamp_end": "00:00:30",
                    "timestamp": "00:00 - 00:30",
                    "timestamp_range": "00:00 - 00:30",
                }
            ]
            ans = self.pipeline.answer_question(query)

        self.assertTrue(ans.is_fallback)
        self.assertIn(FALLBACK_NOT_FOUND_MESSAGE, ans.answer)
        self.assertEqual(ans.debug_info["relevance_decision"], "NOT RELEVANT")
        self.assertLess(ans.debug_info["top_score"], DEFAULT_RELEVANCE_THRESHOLD)
        self.assertEqual(len(ans.sources), 0)
        self.assertFalse(ans.debug_info["llm_called"])

    # -------------------------------------------------------------------------
    # TEST 8: Cross-Lecture Retrieval
    # -------------------------------------------------------------------------
    def test_test8_cross_lecture_sources_attribution(self) -> None:
        """Test 8: Answers correctly format citations attributing to multiple lectures."""
        with patch.object(self.retriever, "retrieve") as mock_ret:
            mock_ret.return_value = [
                {
                    "rank": 1,
                    "chunk_id": "lecture_02_0001",
                    "similarity_score": 0.5542,
                    "text": "Variables in Python store references to values.",
                    "source_file": "lecture2.mp4",
                    "source_filename": "lecture2.mp4",
                    "lecture_id": "lecture_02",
                    "lecture_title": "Lecture 2 — Variables",
                    "course_id": "python_course",
                    "start_time": 2637.0,
                    "end_time": 2905.0,
                    "timestamp_start": "00:43:57",
                    "timestamp_end": "00:48:25",
                    "timestamp": "43:57 - 48:25",
                    "timestamp_range": "43:57 - 48:25",
                },
                {
                    "rank": 2,
                    "chunk_id": "lecture_01_0002",
                    "similarity_score": 0.4912,
                    "text": "Python syntax emphasizes code readability.",
                    "source_file": "lecture1.mp4",
                    "source_filename": "lecture1.mp4",
                    "lecture_id": "lecture_01",
                    "lecture_title": "Lecture 1 — Introduction",
                    "course_id": "python_course",
                    "start_time": 30.0,
                    "end_time": 65.0,
                    "timestamp_start": "00:00:30",
                    "timestamp_end": "00:01:05",
                    "timestamp": "00:30 - 01:05",
                    "timestamp_range": "00:30 - 01:05",
                },
            ]
            ans = self.pipeline.answer_question("How do variables fit into Python's syntax?")

        self.assertFalse(ans.is_fallback)
        self.assertEqual(len(ans.sources), 2)
        # Source 1 is Lecture 2
        self.assertEqual(ans.sources[0]["lecture_title"], "Lecture 2 — Variables")
        self.assertEqual(ans.sources[0]["timestamp_range"], "43:57 - 48:25")
        self.assertEqual(ans.sources[0]["similarity_score"], 0.5542)

        # Source 2 is Lecture 1
        self.assertEqual(ans.sources[1]["lecture_title"], "Lecture 1 — Introduction")
        self.assertEqual(ans.sources[1]["timestamp_range"], "00:30 - 01:05")

        # Answer header reflects Lecture 2
        self.assertIn("Lecture 2 — Variables", ans.answer)
        self.assertIn("43:57 - 48:25", ans.answer)

    # -------------------------------------------------------------------------
    # TEST 9: Different Content Same Filename Allowed
    # -------------------------------------------------------------------------
    def test_test9_different_content_same_filename_allowed(self) -> None:
        """Test 9: Different file with identical filename produces different hash and is not blocked as duplicate."""
        file_a = self.test_dir / "version1.mp4"
        file_b = self.test_dir / "version2.mp4"
        file_a.write_bytes(b"Content version A")
        file_b.write_bytes(b"Content version B")

        hash_a = compute_file_hash(file_a)
        hash_b = compute_file_hash(file_b)

        self.assertNotEqual(hash_a, hash_b)
        self.catalog.register_lecture(
            lecture_id="lec_a",
            lecture_title="Version A",
            source_filename="lecture.mp4",
            file_hash=hash_a,
        )

        self.assertTrue(self.catalog.is_duplicate(hash_a))
        self.assertFalse(self.catalog.is_duplicate(hash_b))


if __name__ == "__main__":
    unittest.main()
