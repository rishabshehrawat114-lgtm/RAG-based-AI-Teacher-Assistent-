"""
tests/test_dataset.py
=====================
Unit tests for Module 8: Bulk Embedding Data Loading with Pandas and Joblib Persistence.

Tests cover:
  1. DataFrame creation from chunks and embeddings
  2. Correct row count
  3. Required columns presence
  4. Unique chunk_id enforcement
  5. Correct embedding dimension validation
  6. Chunk-to-embedding strict alignment (row i vector == chunk i)
  7. Multi-lecture merging and source isolation
  8. Joblib dataset save (save_dataset)
  9. Joblib dataset load (load_dataset)
  10. Loaded DataFrame equality with saved DataFrame (numerical and metadata parity)
  11. Invalid data rejection (duplicate IDs, NaN in vectors, dimension mismatch, empty text)
  12. Retrieval compatibility (LectureRetriever loading from Joblib dataset)
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    DatasetBuilderError,
    REQUIRED_COLUMNS,
    build_dataset,
    load_chunk_data,
    load_dataset,
    load_embedding_data,
    merge_chunks_and_embeddings,
    save_dataset,
    validate_dataset,
)
from src.retrieval import LectureRetriever


class TestDatasetBuilder(unittest.TestCase):
    """Test suite for Module 8 consolidated dataset construction and Joblib persistence."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="rag_test_dataset_"))
        self.chunks_dir = self.test_dir / "chunks"
        self.embeddings_dir = self.test_dir / "embeddings"
        self.processed_dir = self.test_dir / "processed"
        self.chunks_dir.mkdir(parents=True)
        self.embeddings_dir.mkdir(parents=True)
        self.processed_dir.mkdir(parents=True)

        self.dim = 32

        # Create two sample lecture datasets:
        # Lecture 1: python_lecture (2 chunks)
        # Lecture 2: ml_lecture (3 chunks)
        self._create_mock_files(
            stem="python_lecture",
            source_file="python_lecture.mp3",
            chunks=[
                {
                    "chunk_id": "python_0001",
                    "text": "Introduction to Python syntax and control flow.",
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "word_count": 7,
                    "segment_start_id": 0,
                    "segment_end_id": 2,
                },
                {
                    "chunk_id": "python_0002",
                    "text": "Data structures including lists, dictionaries, and sets.",
                    "start_time": 30.0,
                    "end_time": 75.0,
                    "word_count": 7,
                    "segment_start_id": 3,
                    "segment_end_id": 6,
                },
            ],
        )

        self._create_mock_files(
            stem="ml_lecture",
            source_file="ml_lecture.mp3",
            chunks=[
                {
                    "chunk_id": "ml_0001",
                    "text": "Supervised learning overview and cost function minimization.",
                    "start_time": 10.0,
                    "end_time": 60.0,
                    "word_count": 7,
                    "segment_start_id": 0,
                    "segment_end_id": 3,
                },
                {
                    "chunk_id": "ml_0002",
                    "text": "Gradient descent optimization algorithm step by step.",
                    "start_time": 60.0,
                    "end_time": 115.0,
                    "word_count": 7,
                    "segment_start_id": 4,
                    "segment_end_id": 7,
                },
                {
                    "chunk_id": "ml_0003",
                    "text": "Regularization to combat overfitting in deep neural networks.",
                    "start_time": 115.0,
                    "end_time": 180.0,
                    "word_count": 8,
                    "segment_start_id": 8,
                    "segment_end_id": 12,
                },
            ],
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_files(self, stem: str, source_file: str, chunks: list[dict]):
        """Helper to create matching chunk JSON, embeddings NPZ, and metadata JSON."""
        # 1. Chunk file
        chunk_file = self.chunks_dir / f"{stem}_chunks.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_file": source_file,
                "language": "en",
                "chunk_count": len(chunks),
                "chunks": chunks,
            }, f, indent=2)

        # 2. Embedding files (.npz + _metadata.json)
        num_chunks = len(chunks)
        # Deterministic vectors based on chunk text
        vectors = []
        for c in chunks:
            seed = sum(ord(ch) for ch in c["text"]) % 200 + 1
            v = np.sin(np.linspace(seed, seed + 5, self.dim, dtype=np.float32))
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        embeddings = np.array(vectors, dtype=np.float32)

        meta_chunks = []
        for idx, c in enumerate(chunks):
            mc = dict(c)
            mc["embedding_index"] = idx
            mc["source_file"] = source_file
            mc["source_transcript"] = f"{stem}.json"
            mc["language"] = "en"
            meta_chunks.append(mc)

        metadata = {
            "source_file": source_file,
            "language": "en",
            "source_chunk_file": f"{stem}_chunks.json",
            "embedding_model": "mock-all-MiniLM-L6-v2",
            "embedding_dimension": self.dim,
            "chunk_count": num_chunks,
            "normalized": True,
            "chunks": meta_chunks,
        }

        npz_p = self.embeddings_dir / f"{stem}_embeddings.npz"
        json_p = self.embeddings_dir / f"{stem}_metadata.json"

        np.savez_compressed(
            npz_p,
            embeddings=embeddings,
            chunk_ids=np.array([c["chunk_id"] for c in meta_chunks], dtype=object),
        )
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def test_dataframe_creation(self):
        """Test 1: Merging chunks and embeddings creates a valid pandas DataFrame."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_correct_row_count(self):
        """Test 2: Total row count matches total number of chunks across all lectures."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)
        # 2 from python + 3 from ml = 5 total
        self.assertEqual(len(df), 5)

    def test_required_columns_present(self):
        """Test 3: All required columns are present in the consolidated DataFrame."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

        for col in REQUIRED_COLUMNS:
            self.assertIn(col, df.columns, f"Required column '{col}' is missing.")

    def test_unique_chunk_ids(self):
        """Test 4: Chunk IDs are unique across all lectures."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

        self.assertTrue(df["chunk_id"].is_unique)
        self.assertEqual(len(df["chunk_id"].unique()), 5)

    def test_correct_embedding_dimension(self):
        """Test 5: Embedding dimension is detected and validated across all rows."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

        summary = validate_dataset(df, expected_dim=self.dim)
        self.assertEqual(summary["embedding_dimension"], self.dim)
        for vec in df["embedding"]:
            self.assertEqual(vec.shape, (self.dim,))

    def test_chunk_embedding_alignment(self):
        """Test 6: Row i embedding vector corresponds strictly to row i chunk metadata."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

        for idx, row in df.iterrows():
            cid = row["chunk_id"]
            vec = row["embedding"]
            self.assertTrue(np.all(np.isfinite(vec)))
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)
            self.assertTrue(len(row["text"]) > 0)
            self.assertTrue(row["source_file"].endswith(".mp3"))

    def test_multiple_lecture_files_distribution(self):
        """Test 7: Merging preserves multi-lecture distributions without mixing."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

        counts = df["source_file"].value_counts().to_dict()
        self.assertEqual(counts.get("python_lecture.mp3"), 2)
        self.assertEqual(counts.get("ml_lecture.mp3"), 3)

    def test_joblib_save(self):
        """Test 8: save_dataset writes a valid serialized file to disk."""
        target_path = self.processed_dir / "test_save.joblib"
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

        res_path = save_dataset(df, target_path, overwrite=True)
        self.assertTrue(target_path.exists())
        self.assertEqual(res_path, target_path)

    def test_joblib_load(self):
        """Test 9: load_dataset reads serialized dataset back into a DataFrame."""
        target_path = self.processed_dir / "test_load.joblib"
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)
        save_dataset(df, target_path, overwrite=True)

        loaded_df = load_dataset(target_path, validate=True)
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), len(df))

    def test_loaded_dataframe_matches_saved_dataframe(self):
        """Test 10: Loaded DataFrame matches saved DataFrame numerically and textually."""
        target_path = self.processed_dir / "test_equality.joblib"
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)
        save_dataset(df, target_path, overwrite=True)

        loaded_df = load_dataset(target_path, validate=True)
        self.assertEqual(list(loaded_df.columns), list(df.columns))
        self.assertEqual(list(loaded_df["chunk_id"]), list(df["chunk_id"]))
        self.assertEqual(list(loaded_df["text"]), list(df["text"]))

        # Check numerical equality of all embedding vectors
        for v_orig, v_loaded in zip(df["embedding"], loaded_df["embedding"]):
            np.testing.assert_allclose(v_orig, v_loaded, rtol=1e-5, atol=1e-5)

    def test_invalid_data_is_rejected(self):
        """Test 11: Validation detects duplicate IDs, empty text, or dimension mismatches."""
        chunks_map = load_chunk_data(self.chunks_dir)
        embeddings_map = load_embedding_data(self.embeddings_dir)
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

        # 1. Duplicate chunk_id
        bad_df_dup = df.copy()
        bad_df_dup.loc[1, "chunk_id"] = bad_df_dup.loc[0, "chunk_id"]
        with self.assertRaises(DatasetBuilderError) as ctx:
            validate_dataset(bad_df_dup)
        self.assertIn("duplicate chunk_id", str(ctx.exception).lower())

        # 2. Empty text
        bad_df_empty = df.copy()
        bad_df_empty.loc[0, "text"] = "   "
        with self.assertRaises(DatasetBuilderError) as ctx:
            validate_dataset(bad_df_empty)
        self.assertIn("empty text", str(ctx.exception).lower())

        # 3. Dimension mismatch
        bad_df_dim = df.copy()
        bad_df_dim.at[0, "embedding"] = np.zeros(16, dtype=np.float32)
        with self.assertRaises(DatasetBuilderError) as ctx:
            validate_dataset(bad_df_dim)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

        # 4. NaN in embedding
        bad_df_nan = df.copy()
        bad_vec = bad_df_nan.at[0, "embedding"].copy()
        bad_vec[0] = np.nan
        bad_df_nan.at[0, "embedding"] = bad_vec
        with self.assertRaises(DatasetBuilderError) as ctx:
            validate_dataset(bad_df_nan)
        self.assertIn("nan", str(ctx.exception).lower())

    def test_module_7_retrieval_with_joblib_dataset(self):
        """Test 12: Module 7 LectureRetriever can retrieve chunks loaded from Joblib dataset."""
        target_path = self.processed_dir / "teacher_embeddings.joblib"
        build_dataset(
            chunks_dir=self.chunks_dir,
            embeddings_dir=self.embeddings_dir,
            dataset_path=target_path,
            overwrite=True,
        )

        class MockModel:
            def __init__(self, dim):
                self.dimension = dim
            def get_embedding_dimension(self):
                return self.dimension
            def encode(self, sentences, **kwargs):
                return np.ones((len(sentences), self.dimension), dtype=np.float32) / np.sqrt(self.dimension)

        mock_model = MockModel(dim=self.dim)
        retriever = LectureRetriever(
            dataset_path=target_path,
            model=mock_model,
        )

        results = retriever.retrieve("Python variables", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rank"], 1)
        self.assertIn("chunk_id", results[0])
        self.assertIn("similarity_score", results[0])


if __name__ == "__main__":
    unittest.main()
