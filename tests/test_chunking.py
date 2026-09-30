"""
tests/test_chunking.py
======================
Unit tests for Module 5: Transcript Chunking with Metadata.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHUNK_DIR, PROCESSED_TRANSCRIPT_DIR
from src.chunking import (
    ChunkingError,
    batch_chunk_transcripts,
    chunk_transcript,
    chunk_transcript_file,
    count_words,
    create_chunks,
    load_chunks,
    save_chunks,
    split_oversized_segments,
)


class TestChunkingModule(unittest.TestCase):
    """Test suite for transcript chunking logic and metadata attachment."""

    def test_count_words(self):
        """Verify word counting handles empty, whitespace, and normal text."""
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words(None), 0)
        self.assertEqual(count_words("   "), 0)
        self.assertEqual(count_words("word"), 1)
        self.assertEqual(count_words("Today we study machine learning."), 5)
        self.assertEqual(count_words("  Multiple   spaces\tand\nnewlines  "), 4)

    def test_small_transcript_creates_expected_chunks(self):
        """Verify a short transcript produces a single chunk containing all segments."""
        segments = [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "Today we are going to study machine learning."},
            {"id": 1, "start": 3.0, "end": 7.0, "text": "Machine learning is a subset of artificial intelligence."},
        ]
        chunks = create_chunks(
            segments=segments,
            source_file="python_lecture.mp3",
            source_transcript="python_lecture.json",
            language="en",
            chunk_size=700,
            chunk_overlap=100,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["chunk_id"], "python_lecture_0001")
        self.assertEqual(chunk["source_file"], "python_lecture.mp3")
        self.assertEqual(chunk["source_transcript"], "python_lecture.json")
        self.assertEqual(chunk["start_time"], 0.0)
        self.assertEqual(chunk["end_time"], 7.0)
        self.assertEqual(chunk["segment_start_id"], 0)
        self.assertEqual(chunk["segment_end_id"], 1)
        self.assertEqual(chunk["word_count"], 16)
        self.assertIn("Today we are going to study", chunk["text"])
        self.assertIn("artificial intelligence.", chunk["text"])

    def test_chunk_size_and_overlap_behavior(self):
        """Verify segment-based chunking respects approximate chunk size and creates overlapping chunks."""
        # Create 10 segments of 50 words each (500 words total)
        segments = []
        for idx in range(10):
            seg_text = f"Segment {idx} content with ten words in this educational lecture."
            segments.append({
                "id": idx,
                "start": float(idx * 10),
                "end": float((idx + 1) * 10),
                "text": seg_text,
            })

        # chunk_size = 25 words (approx 2-3 segments), overlap = 10 words (approx 1 segment)
        chunks = create_chunks(
            segments=segments,
            source_file="test_lecture.mp3",
            source_transcript="test_lecture.json",
            language="en",
            chunk_size=25,
            chunk_overlap=10,
        )

        # Multiple chunks should be created
        self.assertGreater(len(chunks), 1)

        # Check overlap: neighboring chunks should share segments
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            # Next chunk start_time should be <= current chunk end_time
            self.assertLessEqual(next_chunk["start_time"], curr_chunk["end_time"])
            # Next chunk start segment should be <= current chunk end segment
            self.assertLessEqual(next_chunk["segment_start_id"], curr_chunk["segment_end_id"])

    def test_deterministic_chunk_ids(self):
        """Verify the same transcript processed twice yields identical chunk IDs."""
        segments = [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Hello world from the first lecture segment."},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "Moving on to the second lecture segment."},
            {"id": 2, "start": 10.0, "end": 15.0, "text": "Now we are in the third lecture segment."},
        ]
        run_1 = create_chunks(segments, "ai.mp3", "ai.json", chunk_size=10, chunk_overlap=2)
        run_2 = create_chunks(segments, "ai.mp3", "ai.json", chunk_size=10, chunk_overlap=2)

        ids_1 = [c["chunk_id"] for c in run_1]
        ids_2 = [c["chunk_id"] for c in run_2]
        self.assertEqual(ids_1, ids_2)
        self.assertEqual(ids_1, ["ai_0001", "ai_0002"])



    def test_timestamps_from_first_and_last_segments(self):
        """Verify chunk start_time comes from first segment and end_time from last segment."""
        segments = [
            {"id": 5, "start": 120.456, "end": 130.0, "text": "Segment five start."},
            {"id": 6, "start": 130.0, "end": 145.2, "text": "Segment six middle."},
            {"id": 7, "start": 145.2, "end": 168.789, "text": "Segment seven end."},
        ]
        chunks = create_chunks(segments, "sample.mp3", "sample.json", chunk_size=500, chunk_overlap=100)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["start_time"], 120.456)
        self.assertEqual(chunk["end_time"], 168.789)
        self.assertEqual(chunk["segment_start_id"], 5)
        self.assertEqual(chunk["segment_end_id"], 7)

    def test_multiple_transcripts_remain_separate(self):
        """Verify chunks from different transcript files maintain distinct source metadata."""
        seg_a = [{"id": 0, "start": 0.0, "end": 5.0, "text": "Python programming lecture."}]
        seg_b = [{"id": 0, "start": 0.0, "end": 5.0, "text": "Statistics lecture."}]

        chunks_a = create_chunks(seg_a, "python.mp3", "python.json")
        chunks_b = create_chunks(seg_b, "statistics.mp3", "statistics.json")

        self.assertEqual(chunks_a[0]["source_file"], "python.mp3")
        self.assertEqual(chunks_a[0]["source_transcript"], "python.json")
        self.assertEqual(chunks_a[0]["chunk_id"], "python_0001")

        self.assertEqual(chunks_b[0]["source_file"], "statistics.mp3")
        self.assertEqual(chunks_b[0]["source_transcript"], "statistics.json")
        self.assertEqual(chunks_b[0]["chunk_id"], "statistics_0001")

    def test_empty_segments_ignored(self):
        """Verify empty and whitespace-only segments do not produce empty chunks."""
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": ""},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "   \n\t  "},
            {"id": 2, "start": 4.0, "end": 7.0, "text": "Valid text."},
        ]
        chunks = create_chunks(segments, "lecture.mp3", "lecture.json")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Valid text.")
        self.assertEqual(chunks[0]["segment_start_id"], 2)
        self.assertEqual(chunks[0]["segment_end_id"], 2)

    def test_empty_transcript(self):
        """Verify an empty transcript produces 0 chunks without error."""
        chunks = create_chunks([], "empty.mp3", "empty.json")
        self.assertEqual(chunks, [])

    def test_single_oversized_segment_split(self):
        """Verify a single segment longer than chunk_size is split safely with interpolated timestamps."""
        long_words = [f"word{i}" for i in range(150)]
        long_text = " ".join(long_words)
        oversized = [{"id": 10, "start": 0.0, "end": 150.0, "text": long_text}]

        split_segs = split_oversized_segments(oversized, chunk_size=50, chunk_overlap=10)
        self.assertGreater(len(split_segs), 1)

        # Verify all sub-segments are within chunk_size
        for s in split_segs:
            self.assertLessEqual(len(s["text"].split()), 50)
            self.assertGreaterEqual(s["start"], 0.0)
            self.assertLessEqual(s["end"], 150.0)
            self.assertLessEqual(s["start"], s["end"])

    def test_save_and_reload_chunks(self):
        """Verify saving chunks JSON to disk and reading it back preserves schema and data."""
        dataset = {
            "source_file": "demo.mp3",
            "language": "en",
            "chunk_count": 1,
            "chunking_config": {"chunk_size": 700, "chunk_overlap": 100},
            "chunks": [
                {
                    "chunk_id": "demo_0001",
                    "source_file": "demo.mp3",
                    "source_transcript": "demo.json",
                    "language": "en",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "word_count": 5,
                    "segment_start_id": 0,
                    "segment_end_id": 0,
                    "text": "Hello world from demo lecture.",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "demo_chunks.json"
            saved_path = save_chunks(dataset, target_path)
            self.assertEqual(saved_path, target_path)
            self.assertTrue(saved_path.exists())

            # Verify that overwrite=False raises FileExistsError if file exists
            with self.assertRaises(FileExistsError):
                save_chunks(dataset, target_path, overwrite=False)

            # Verify that overwrite=True replaces the existing file cleanly
            updated_dataset = dict(dataset)
            updated_dataset["chunk_count"] = 99
            saved_again = save_chunks(updated_dataset, target_path, overwrite=True)
            self.assertEqual(saved_again, target_path)

            loaded = load_chunks(saved_path)
            self.assertEqual(loaded["chunk_count"], 99)

    def test_batch_chunk_transcripts_skips_and_overwrites(self):
        """Verify batch processor skips existing chunk files unless overwrite=True."""
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            trans_dir = tmp_dir / "transcripts"
            chunk_dir = tmp_dir / "chunks"
            trans_dir.mkdir(parents=True, exist_ok=True)
            chunk_dir.mkdir(parents=True, exist_ok=True)

            t1 = trans_dir / "lecture_1.json"
            t2 = trans_dir / "lecture_2.json"
            t1.write_text(json.dumps({
                "source_file": "l1.mp3",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "L1 speech"}]
            }), encoding="utf-8")
            t2.write_text(json.dumps({
                "source_file": "l2.mp3",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "L2 speech"}]
            }), encoding="utf-8")

            # Pre-create chunk file for lecture_1
            pre_c1 = chunk_dir / "lecture_1_chunks.json"
            pre_c1.write_text(json.dumps({"existing": True}), encoding="utf-8")

            # First run: overwrite=False -> should skip lecture_1, chunk lecture_2
            res1 = batch_chunk_transcripts(
                input_dir=trans_dir,
                output_dir=chunk_dir,
                overwrite=False,
            )
            self.assertEqual(res1["total_found"], 2)
            self.assertTrue(any(p.name == "lecture_1_chunks.json" for p in res1["skipped"]))
            self.assertTrue(any(p.name == "lecture_2_chunks.json" for p in res1["chunked"]))

            # Second run: overwrite=True -> should chunk both
            res2 = batch_chunk_transcripts(
                input_dir=trans_dir,
                output_dir=chunk_dir,
                overwrite=True,
            )
            self.assertEqual(len(res2["chunked"]), 2)
            self.assertEqual(len(res2["skipped"]), 0)


if __name__ == "__main__":
    unittest.main()
