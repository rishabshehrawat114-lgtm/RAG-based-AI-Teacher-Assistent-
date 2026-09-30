"""
tests/test_embeddings.py
========================
Unit tests for Module 6: Dense Vector Embeddings for Lecture Chunks.

Tests cover:
  1. Embedding model loading / mock compatibility
  2. Single text produces an embedding vector
  3. Embedding dimension is detected automatically (not hardcoded)
  4. Multiple chunks produce an embedding matrix
  5. Number of embeddings equals number of valid chunks
  6. Metadata count equals embedding row count
  7. Chunk IDs remain strictly aligned with embedding rows (row i == chunk i)
  8. Empty / whitespace chunks are safely skipped without crash
  9. Saved embeddings (.npz + .json) can be loaded and verified
  10. Embedding values are valid numeric float32 values
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
from src.embeddings import (
    EmbeddingError,
    embed_chunk_file,
    embed_chunks_data,
    generate_embeddings,
    get_embedding_dimension,
    load_embeddings,
    save_embeddings,
)


class MockSentenceTransformer:
    """
    Lightweight mock of a SentenceTransformer model for fast, deterministic unit testing
    without requiring internet downloads or heavy GPU/CPU overhead.
    """

    def __init__(self, dimension: int = 128, model_name: str = "mock-mini-model"):
        self.dimension = dimension
        self.model_card_data = type("Card", (), {"base_model": model_name})()

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
        if isinstance(sentences, str):
            input_list = [sentences]
        else:
            input_list = list(sentences)

        if not input_list:
            return np.empty((0, self.dimension), dtype=np.float32)

        matrix = []
        for text in input_list:
            # Deterministic vector based on character values
            seed = sum(ord(c) for c in text) % 500 + 1
            raw = np.sin(np.linspace(seed, seed + 10, self.dimension, dtype=np.float32))
            if normalize_embeddings:
                norm = float(np.linalg.norm(raw))
                if norm > 0:
                    raw = raw / norm
            matrix.append(raw)

        return np.array(matrix, dtype=np.float32)


class TestEmbeddingsModule(unittest.TestCase):
    """Comprehensive test suite for dense vector embedding generation and metadata alignment."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="rag_test_embeddings_"))
        self.mock_model = MockSentenceTransformer(dimension=128)

        # Standard sample chunk dataset
        self.sample_chunks_data = {
            "source_file": "python_lecture.mp3",
            "language": "en",
            "chunk_count": 3,
            "chunking_config": {"chunk_size": 700, "chunk_overlap": 100},
            "chunks": [
                {
                    "chunk_id": "python_lecture_0001",
                    "source_file": "python_lecture.mp3",
                    "source_transcript": "python_lecture.json",
                    "language": "en",
                    "start_time": 0.0,
                    "end_time": 45.2,
                    "word_count": 120,
                    "segment_start_id": 0,
                    "segment_end_id": 5,
                    "text": "Welcome to Python programming. In this course we cover data structures.",
                },
                {
                    "chunk_id": "python_lecture_0002",
                    "source_file": "python_lecture.mp3",
                    "source_transcript": "python_lecture.json",
                    "language": "en",
                    "start_time": 40.1,
                    "end_time": 95.8,
                    "word_count": 145,
                    "segment_start_id": 4,
                    "segment_end_id": 11,
                    "text": "Data structures like lists, sets, and dictionaries allow efficient storage.",
                },
                {
                    "chunk_id": "python_lecture_0003",
                    "source_file": "python_lecture.mp3",
                    "source_transcript": "python_lecture.json",
                    "language": "en",
                    "start_time": 90.0,
                    "end_time": 150.0,
                    "word_count": 130,
                    "segment_start_id": 10,
                    "segment_end_id": 16,
                    "text": "Next we will implement functions and modules for clean software architecture.",
                },
            ],
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_embedding_dimension_detected_automatically(self):
        """Test 3: Automatically detects dimension from model without hardcoding 384."""
        model_128 = MockSentenceTransformer(dimension=128)
        model_256 = MockSentenceTransformer(dimension=256)
        model_512 = MockSentenceTransformer(dimension=512)

        self.assertEqual(get_embedding_dimension(model_128), 128)
        self.assertEqual(get_embedding_dimension(model_256), 256)
        self.assertEqual(get_embedding_dimension(model_512), 512)

    def test_single_text_produces_embedding(self):
        """Test 2: A single text input produces a 2D matrix of shape (1, dimension)."""
        text = "Machine learning is a method of learning patterns from data."
        vec = generate_embeddings(
            texts=[text],
            model=self.mock_model,
            normalize=True,
        )

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (1, 128))
        self.assertEqual(vec.dtype, np.float32)

    def test_embedding_values_are_numeric(self):
        """Test 10: Embedding values are finite real numbers with valid L2 norm."""
        texts = ["Artificial intelligence", "Deep learning neural networks"]
        embeddings = generate_embeddings(texts=texts, model=self.mock_model, normalize=True)

        self.assertTrue(np.all(np.isfinite(embeddings)))
        self.assertFalse(np.any(np.isnan(embeddings)))
        # Verify L2 normalization (unit length norm ≈ 1.0)
        for row in embeddings:
            norm = np.linalg.norm(row)
            self.assertAlmostEqual(float(norm), 1.0, places=4)

    def test_multiple_chunks_produce_embedding_matrix(self):
        """Test 4: Multiple input chunks produce an embedding matrix matching chunk count."""
        embeddings, metadata = embed_chunks_data(
            chunks_data=self.sample_chunks_data,
            model=self.mock_model,
        )

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (3, 128))
        self.assertEqual(metadata["chunk_count"], 3)
        self.assertEqual(metadata["embedding_dimension"], 128)

    def test_number_of_embeddings_equals_valid_chunks(self):
        """Test 5: Number of embedding rows strictly matches number of valid chunks."""
        embeddings, metadata = embed_chunks_data(
            chunks_data=self.sample_chunks_data,
            model=self.mock_model,
        )
        self.assertEqual(embeddings.shape[0], len(self.sample_chunks_data["chunks"]))
        self.assertEqual(len(metadata["chunks"]), embeddings.shape[0])

    def test_metadata_count_equals_embedding_row_count(self):
        """Test 6: Metadata chunk count equals embedding matrix row count."""
        embeddings, metadata = embed_chunks_data(
            chunks_data=self.sample_chunks_data,
            model=self.mock_model,
        )
        self.assertEqual(len(metadata["chunks"]), embeddings.shape[0])
        self.assertEqual(metadata["chunk_count"], embeddings.shape[0])

    def test_chunk_ids_remain_aligned_with_embedding_rows(self):
        """Test 7: Chunk IDs strictly maintain deterministic row-to-metadata alignment."""
        embeddings, metadata = embed_chunks_data(
            chunks_data=self.sample_chunks_data,
            model=self.mock_model,
        )

        expected_ids = [
            "python_lecture_0001",
            "python_lecture_0002",
            "python_lecture_0003",
        ]

        for idx, (expected_id, chunk_meta) in enumerate(zip(expected_ids, metadata["chunks"])):
            self.assertEqual(chunk_meta["chunk_id"], expected_id)
            self.assertEqual(chunk_meta["embedding_index"], idx)
            # Embedding vector row idx strictly aligns with this chunk
            self.assertEqual(embeddings[idx].shape, (128,))

    def test_empty_and_whitespace_chunks_are_skipped(self):
        """Test 8: Empty or whitespace-only chunks are safely skipped without crashing."""
        chunks_with_empties = {
            "source_file": "edge_cases.mp3",
            "language": "en",
            "chunks": [
                {"chunk_id": "chunk_0001", "text": "Valid first chunk."},
                {"chunk_id": "chunk_0002", "text": ""},  # empty string
                {"chunk_id": "chunk_0003", "text": "   \n\t   "},  # whitespace only
                {"chunk_id": "chunk_0004", "text": "Valid fourth chunk."},
            ],
        }

        embeddings, metadata = embed_chunks_data(
            chunks_data=chunks_with_empties,
            model=self.mock_model,
        )

        # Only chunk 1 and 4 are valid
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(metadata["chunk_count"], 2)
        self.assertEqual(metadata["chunks"][0]["chunk_id"], "chunk_0001")
        self.assertEqual(metadata["chunks"][0]["embedding_index"], 0)
        self.assertEqual(metadata["chunks"][1]["chunk_id"], "chunk_0004")
        self.assertEqual(metadata["chunks"][1]["embedding_index"], 1)

    def test_saved_embeddings_can_be_loaded_again(self):
        """Test 9: Saved .npz and .json files can be loaded and retain exact values."""
        embeddings, metadata = embed_chunks_data(
            chunks_data=self.sample_chunks_data,
            model=self.mock_model,
        )

        npz_p, json_p = save_embeddings(
            embeddings=embeddings,
            metadata=metadata,
            output_stem="python_lecture",
            output_dir=self.test_dir,
            overwrite=True,
        )

        self.assertTrue(npz_p.exists())
        self.assertTrue(json_p.exists())

        loaded_matrix, loaded_meta = load_embeddings(
            input_stem_or_path="python_lecture",
            embeddings_dir=self.test_dir,
        )

        # Check matrix shape and numerical equivalence
        self.assertEqual(loaded_matrix.shape, embeddings.shape)
        np.testing.assert_allclose(loaded_matrix, embeddings, rtol=1e-5, atol=1e-5)

        # Check metadata alignment
        self.assertEqual(loaded_meta["chunk_count"], 3)
        self.assertEqual(loaded_meta["embedding_dimension"], 128)
        self.assertEqual(len(loaded_meta["chunks"]), 3)
        self.assertEqual(loaded_meta["chunks"][0]["chunk_id"], "python_lecture_0001")

    def test_embed_chunk_file_end_to_end(self):
        """End-to-end test of embed_chunk_file creating .npz and .json."""
        chunk_file = self.test_dir / "sample_lecture_chunks.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_chunks_data, f)

        res = embed_chunk_file(
            chunk_file=chunk_file,
            output_dir=self.test_dir,
            model=self.mock_model,
            overwrite=True,
        )

        self.assertIsNotNone(res)
        npz_p, json_p = res
        self.assertEqual(npz_p.name, "sample_lecture_embeddings.npz")
        self.assertEqual(json_p.name, "sample_lecture_metadata.json")

        # Test skip when overwrite=False
        res_skip = embed_chunk_file(
            chunk_file=chunk_file,
            output_dir=self.test_dir,
            model=self.mock_model,
            overwrite=False,
        )
        self.assertIsNone(res_skip)

    def test_invalid_chunk_structure_raises_error(self):
        """Verify malformed input dictionary raises EmbeddingError."""
        with self.assertRaises(EmbeddingError):
            embed_chunks_data({"no_chunks_key": 123}, model=self.mock_model)

        with self.assertRaises(EmbeddingError):
            embed_chunks_data("not_a_dict", model=self.mock_model)

    def test_load_embedding_model_and_inference(self):
        """Test 1: Embedding model loads and generates 384-dim normalized vector."""
        from src.embeddings import load_embedding_model
        model = load_embedding_model("all-MiniLM-L6-v2")
        self.assertIsNotNone(model)
        dim = get_embedding_dimension(model)
        self.assertEqual(dim, 384)

        # Single inference test with real model
        text = "Machine learning is a subset of artificial intelligence."
        vec = generate_embeddings(texts=[text], model=model, normalize=True)
        self.assertEqual(vec.shape, (1, 384))
        self.assertAlmostEqual(float(np.linalg.norm(vec[0])), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
