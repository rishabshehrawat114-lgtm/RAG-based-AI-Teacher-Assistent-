"""
tests/test_retrieval.py
=======================
Unit tests for Module 7: Top Matching Chunks / Semantic Retrieval.

Tests cover:
  1. Query embedding generation
  2. Cosine similarity calculation (identity, orthogonal, negative)
  3. Correct descending ranking order (Rank 1 -> highest score)
  4. Top-K behavior (limiting returned result count)
  5. Similarity threshold behavior (filtering chunks below threshold)
  6. Metadata remains strictly aligned with vectors throughout retrieval
  7. Source lecture filtering (restricting query to specific lecture file)
  8. Empty query handling (validation and error raising)
  9. Query against multiple lectures simultaneously
  10. No-result behavior (empty list returned when no chunk meets threshold)
  11. Timestamp formatting helper (format_timestamp)
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBEDDING_DIR
from src.retrieval import (
    LectureRetriever,
    RetrievalError,
    compute_cosine_similarity,
    format_retrieval_results,
    format_timestamp,
    load_all_lecture_embeddings,
    retrieve,
)


class MockEmbeddingModel:
    """Fast deterministic embedding model for unit testing without network overhead."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            input_list = [sentences]
        else:
            input_list = list(sentences)

        matrix = []
        for text in input_list:
            # Deterministic vector based on character sum
            seed = sum(ord(c) for c in text) % 250 + 1
            raw = np.sin(np.linspace(seed, seed + 10, self.dimension, dtype=np.float32))
            if normalize_embeddings:
                norm = float(np.linalg.norm(raw))
                if norm > 0:
                    raw = raw / norm
            matrix.append(raw)

        return np.array(matrix, dtype=np.float32)


class TestSemanticRetrieval(unittest.TestCase):
    """Test suite for semantic similarity retrieval and multi-lecture ranking."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="rag_test_retrieval_"))
        self.dim = 64
        self.mock_model = MockEmbeddingModel(dimension=self.dim)

        # Create two mock lecture archives in test_dir:
        # Lecture 1: python_lecture (2 chunks)
        # Lecture 2: ml_lecture (2 chunks)
        self._create_mock_archive(
            stem="python_lecture",
            source_file="python_lecture.mp3",
            source_transcript="python_lecture.json",
            chunks_data=[
                {
                    "chunk_id": "python_0001",
                    "text": "Introduction to Python programming variables and basic types.",
                    "start_time": 0.0,
                    "end_time": 60.0,
                },
                {
                    "chunk_id": "python_0002",
                    "text": "Python functions, loops, and object-oriented programming principles.",
                    "start_time": 60.0,
                    "end_time": 120.0,
                },
            ],
        )

        self._create_mock_archive(
            stem="ml_lecture",
            source_file="ml_lecture.mp3",
            source_transcript="ml_lecture.json",
            chunks_data=[
                {
                    "chunk_id": "ml_0001",
                    "text": "Supervised machine learning algorithms including linear regression.",
                    "start_time": 10.0,
                    "end_time": 75.0,
                },
                {
                    "chunk_id": "ml_0002",
                    "text": "Overfitting and regularization techniques like Lasso and Ridge.",
                    "start_time": 75.0,
                    "end_time": 150.0,
                },
            ],
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_archive(
        self,
        stem: str,
        source_file: str,
        source_transcript: str,
        chunks_data: list[dict],
    ):
        """Helper to create matching .npz and .json files in test directory."""
        texts = [c["text"] for c in chunks_data]
        embeddings = self.mock_model.encode(texts)

        # Build metadata
        meta_chunks = []
        for idx, (c, text) in enumerate(zip(chunks_data, texts)):
            meta_chunks.append({
                "chunk_id": c["chunk_id"],
                "embedding_index": idx,
                "source_file": source_file,
                "source_transcript": source_transcript,
                "language": "en",
                "start_time": float(c["start_time"]),
                "end_time": float(c["end_time"]),
                "word_count": len(text.split()),
                "text": text,
            })

        metadata = {
            "source_file": source_file,
            "language": "en",
            "embedding_model": "mock-model",
            "embedding_dimension": self.dim,
            "chunk_count": len(meta_chunks),
            "normalized": True,
            "chunks": meta_chunks,
        }

        # Save files
        npz_p = self.test_dir / f"{stem}_embeddings.npz"
        json_p = self.test_dir / f"{stem}_metadata.json"

        np.savez_compressed(
            npz_p,
            embeddings=embeddings,
            chunk_ids=np.array([c["chunk_id"] for c in meta_chunks], dtype=object),
        )
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def test_format_timestamp(self):
        """Test 11: Timestamp formatting helper correctly converts seconds."""
        self.assertEqual(format_timestamp(0.0), "00:00")
        self.assertEqual(format_timestamp(5.42), "00:05")
        self.assertEqual(format_timestamp(125.0), "02:05")
        self.assertEqual(format_timestamp(3665.0), "01:01:05")
        self.assertEqual(format_timestamp(125.5, include_ms=True), "02:05.5")
        self.assertEqual(format_timestamp(-10), "00:00")
        self.assertEqual(format_timestamp(None), "00:00")

    def test_cosine_similarity_calculation(self):
        """Test 2: Cosine similarity calculation is mathematically sound and bounded."""
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([
            [1.0, 0.0, 0.0],   # Identical -> sim = 1.0
            [0.0, 1.0, 0.0],   # Orthogonal -> sim = 0.0
            [-1.0, 0.0, 0.0],  # Opposite -> sim = -1.0
            [0.5, 0.5, 0.0],   # Partial -> sim = 0.5 / sqrt(0.5) = 0.7071
        ], dtype=np.float32)

        sims = compute_cosine_similarity(q, c)
        self.assertEqual(len(sims), 4)
        self.assertAlmostEqual(float(sims[0]), 1.0, places=5)
        self.assertAlmostEqual(float(sims[1]), 0.0, places=5)
        self.assertAlmostEqual(float(sims[2]), -1.0, places=5)
        self.assertAlmostEqual(float(sims[3]), 0.707106, places=4)

    def test_query_embedding_generation(self):
        """Test 1: Query embedding generation produces normalized 1D vector."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        vec = retriever.embed_query("What is machine learning?")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (self.dim,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_correct_descending_ranking(self):
        """Test 3: Retrieved results are sorted strictly in descending order of similarity."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        results = retriever.retrieve("Overfitting and regularization", top_k=4)
        self.assertTrue(len(results) > 1)

        scores = [r["similarity_score"] for r in results]
        ranks = [r["rank"] for r in results]

        # Verify ranks are 1, 2, 3, 4
        self.assertEqual(ranks, list(range(1, len(results) + 1)))
        # Verify scores are strictly non-increasing
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_top_k_behavior(self):
        """Test 4: Top-K limits return results to exactly K items."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        # Total chunks in corpus = 4
        res_k1 = retriever.retrieve("Python functions", top_k=1)
        self.assertEqual(len(res_k1), 1)

        res_k2 = retriever.retrieve("Python functions", top_k=2)
        self.assertEqual(len(res_k2), 2)

        res_k10 = retriever.retrieve("Python functions", top_k=10)
        self.assertEqual(len(res_k10), 4)  # Capped at total available

    def test_similarity_threshold_behavior(self):
        """Test 5: Similarity threshold excludes chunks below the cutoff."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        # Without threshold: returns all 4
        all_results = retriever.retrieve("algorithms", top_k=4, similarity_threshold=None)
        self.assertEqual(len(all_results), 4)

        top_score = all_results[0]["similarity_score"]
        cutoff = top_score - 0.05

        filtered_results = retriever.retrieve("algorithms", top_k=4, similarity_threshold=cutoff)
        for r in filtered_results:
            self.assertGreaterEqual(r["similarity_score"], cutoff)

    def test_metadata_remains_aligned(self):
        """Test 6: Result records retain complete source, timestamps, and chunk IDs."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        results = retriever.retrieve("Supervised learning", top_k=1)
        self.assertEqual(len(results), 1)
        top = results[0]

        # Verify required keys
        expected_keys = {
            "rank", "chunk_id", "similarity_score", "text",
            "source_file", "source_transcript", "start_time", "end_time", "language"
        }
        self.assertTrue(expected_keys.issubset(top.keys()))
        self.assertIsInstance(top["rank"], int)
        self.assertIsInstance(top["similarity_score"], float)
        self.assertIsInstance(top["start_time"], float)
        self.assertIsInstance(top["end_time"], float)
        self.assertTrue(top["end_time"] >= top["start_time"])

    def test_source_filtering(self):
        """Test 7: Source filter restricts search exclusively to specified lecture."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )

        # Restrict to ml_lecture.mp3 only
        ml_results = retriever.retrieve("variables and loops", top_k=4, source_file="ml_lecture.mp3")
        for r in ml_results:
            self.assertEqual(r["source_file"], "ml_lecture.mp3")

        # Restrict to python_lecture.mp3 only
        py_results = retriever.retrieve("variables and loops", top_k=4, source_file="python_lecture.mp3")
        for r in py_results:
            self.assertEqual(r["source_file"], "python_lecture.mp3")

    def test_empty_query_handling(self):
        """Test 8: Empty or whitespace query raises RetrievalError."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        with self.assertRaises(RetrievalError):
            retriever.retrieve("")

        with self.assertRaises(RetrievalError):
            retriever.retrieve("    \n\t   ")

        with self.assertRaises(RetrievalError):
            retriever.retrieve(None)

    def test_query_against_multiple_lectures(self):
        """Test 9: Retriever searches across all lectures in the directory without cross-pollution."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        results = retriever.retrieve("programming concepts and machine learning", top_k=4)
        self.assertEqual(len(results), 4)

        sources = {r["source_file"] for r in results}
        # Both lecture files must be represented
        self.assertIn("python_lecture.mp3", sources)
        self.assertIn("ml_lecture.mp3", sources)

    def test_no_result_behavior(self):
        """Test 10: Returns empty list when threshold is impossibly high."""
        retriever = LectureRetriever(
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        # Threshold 0.999 exceeds any score for dissimilar text
        results = retriever.retrieve("completely unrelated concept", top_k=5, similarity_threshold=0.999)
        self.assertEqual(results, [])

        # Verify formatting helper displays no-result message
        display = format_retrieval_results(results, query="unrelated")
        self.assertIn("No sufficiently relevant lecture chunks were found.", display)

    def test_convenience_retrieve_function(self):
        """Verify the module-level retrieve() function works directly."""
        results = retrieve(
            query="Python variables",
            top_k=2,
            embeddings_dir=self.test_dir,
            model=self.mock_model,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
