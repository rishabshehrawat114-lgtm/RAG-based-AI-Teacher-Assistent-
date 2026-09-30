"""
tests/test_processor.py
=======================
Unit tests for Module 4: Transcript Processing and Structured JSON Preparation.
"""

import json
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_TRANSCRIPT_DIR, TRANSCRIPT_DIR
from src.transcription import (
    TranscriptProcessingError,
    batch_process_transcripts,
    clean_segment_text,
    process_transcript,
    process_transcript_file,
    save_processed_transcript,
    validate_segment,
)


class TestTranscriptProcessor(unittest.TestCase):
    """Test suite for transcript validation, cleaning, and normalization."""

    def test_clean_segment_text_conservative(self):
        """Verify conservative whitespace normalization and content preservation."""
        sample_input = "   Machine learning   is a\t\tsubset of AI.\n\nToday we study.   "
        expected = "Machine learning is a subset of AI. Today we study."
        cleaned = clean_segment_text(sample_input)
        self.assertEqual(cleaned, expected)

        # None and empty handling
        self.assertEqual(clean_segment_text(None), "")
        self.assertEqual(clean_segment_text("   "), "")

        # Punctuation, capitalization, and numbers are strictly preserved
        tech_text = "PyTorch 2.14, Python 3.11: (Attention Is All You Need)!"
        self.assertEqual(clean_segment_text(tech_text), tech_text)

    def test_validate_segment_valid(self):
        """Verify a valid segment passes validation and normalizes timestamps."""
        seg = {
            "id": 42,
            "start": 12.345678,
            "end": 18.987654,
            "text": "   Valid lecture segment.  ",
        }
        is_valid, reason, normalized = validate_segment(seg)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["start"], 12.346)
        self.assertEqual(normalized["end"], 18.988)
        self.assertEqual(normalized["text"], "Valid lecture segment.")

    def test_validate_segment_empty_text_removed(self):
        """Verify segments with empty text or whitespace-only text are rejected."""
        seg_empty = {"start": 0.0, "end": 2.5, "text": "    \t\n  "}
        is_valid, reason, normalized = validate_segment(seg_empty)
        self.assertFalse(is_valid)
        self.assertIn("empty", reason.lower())
        self.assertIsNone(normalized)

    def test_validate_segment_invalid_timestamps(self):
        """Verify segments with non-numeric, negative, or inverted timestamps are rejected."""
        # Non-numeric start
        seg1 = {"start": "not_a_time", "end": 5.0, "text": "hello"}
        is_valid1, reason1, _ = validate_segment(seg1)
        self.assertFalse(is_valid1)
        self.assertIn("numeric", reason1.lower())

        # Negative start
        seg2 = {"start": -2.0, "end": 5.0, "text": "hello"}
        is_valid2, reason2, _ = validate_segment(seg2)
        self.assertFalse(is_valid2)
        self.assertIn("negative", reason2.lower())

        # End precedes start
        seg3 = {"start": 10.0, "end": 5.0, "text": "hello"}
        is_valid3, reason3, _ = validate_segment(seg3)
        self.assertFalse(is_valid3)
        self.assertIn("precede", reason3.lower())

        # Missing timestamps
        seg4 = {"text": "hello without timestamps"}
        is_valid4, reason4, _ = validate_segment(seg4)
        self.assertFalse(is_valid4)
        self.assertIn("missing", reason4.lower())

    def test_process_transcript_sequential_ids_and_counts(self):
        """Verify sequential ID re-indexing, segment_count accuracy, and invalid segment filtering."""
        raw_transcript = {
            "source_file": "intro_ai.mp3",
            "language": "en",
            "duration": 50.0,
            "segments": [
                {"id": 10, "start": 0.0, "end": 4.5, "text": "First valid segment."},
                {"id": 11, "start": 4.5, "end": 8.0, "text": "   "},  # Invalid: empty text
                {"id": 99, "start": 8.0, "end": 5.0, "text": "Inverted time."},  # Invalid: end < start
                {"id": 500, "start": 8.5, "end": 14.2, "text": "Second valid segment."},
                {"id": 501, "start": "bad", "end": 15.0, "text": "Non-numeric."},  # Invalid: non-numeric
                {"id": 999, "start": 14.2, "end": 20.0, "text": "Third valid segment."},
            ],
        }

        processed, stats = process_transcript(raw_transcript)

        # Check segment counts
        self.assertEqual(stats["valid_segments"], 3)
        self.assertEqual(stats["invalid_segments"], 3)
        self.assertEqual(stats["total_raw_segments"], 6)
        self.assertEqual(processed["segment_count"], 3)
        self.assertEqual(processed["source_file"], "intro_ai.mp3")
        self.assertEqual(processed["language"], "en")
        self.assertEqual(processed["duration"], 20.0)

        # Check sequential 0-indexed IDs
        ids = [seg["id"] for seg in processed["segments"]]
        self.assertEqual(ids, [0, 1, 2])

        # Check texts
        texts = [seg["text"] for seg in processed["segments"]]
        self.assertEqual(texts, [
            "First valid segment.",
            "Second valid segment.",
            "Third valid segment.",
        ])

    def test_save_and_reload_processed_transcript(self):
        """Verify saving processed transcript to JSON and reading it back without data loss."""
        processed_data = {
            "source_file": "sample_lecture.mp3",
            "language": "en",
            "duration": 15.42,
            "segment_count": 2,
            "segments": [
                {"id": 0, "start": 0.0, "end": 7.5, "text": "Welcome to the class."},
                {"id": 1, "start": 7.5, "end": 15.42, "text": "Next topic is neural networks."},
            ],
        }
        test_file = PROCESSED_TRANSCRIPT_DIR / "test_sample_lecture.json"

        try:
            saved_path = save_processed_transcript(processed_data, test_file)
            self.assertEqual(saved_path, test_file)
            self.assertTrue(saved_path.exists())

            with open(saved_path, "r", encoding="utf-8") as f:
                reloaded = json.load(f)

            self.assertEqual(reloaded, processed_data)
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_original_transcript_not_modified(self):
        """Verify that processing a file produces a new file in processed/ without mutating the original."""
        original_file = TRANSCRIPT_DIR / "immutable_test.json"
        original_content = {
            "source_file": "immutable.mp3",
            "language": "en",
            "duration": 5.0,
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.0, "text": "   Original   content.   "}
            ],
        }
        original_json_str = json.dumps(original_content, indent=2)
        original_file.write_text(original_json_str, encoding="utf-8")

        target_file = PROCESSED_TRANSCRIPT_DIR / "immutable_test.json"

        try:
            _, _, saved_path = process_transcript_file(
                input_path=original_file,
                output_path=target_file,
                overwrite=True,
            )

            # Target exists in processed/
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path, target_file)

            # Original remains completely identical
            current_original = original_file.read_text(encoding="utf-8")
            self.assertEqual(current_original, original_json_str)
        finally:
            if original_file.exists():
                original_file.unlink()
            if target_file.exists():
                target_file.unlink()

    def test_batch_process_skips_and_ignores_processed_dir(self):
        """Verify batch processor skips existing processed files and does not ingest files in processed/."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            proc_dir = tmp_dir / "processed"
            proc_dir.mkdir(parents=True, exist_ok=True)

            raw_1 = tmp_dir / "batch_sample_1.json"
            raw_2 = tmp_dir / "batch_sample_2.json"
            raw_1.write_text(json.dumps({
                "source_file": "b1.mp3",
                "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "B1"}]
            }), encoding="utf-8")
            raw_2.write_text(json.dumps({
                "source_file": "b2.mp3",
                "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "B2"}]
            }), encoding="utf-8")

            # Pre-create processed file for raw_1 in proc_dir
            proc_1 = proc_dir / "batch_sample_1.json"
            proc_1.write_text(json.dumps({"already": "done"}), encoding="utf-8")

            # Create an arbitrary file inside proc_dir that should NOT be picked up as a raw file
            dummy_inside_proc = proc_dir / "should_be_ignored.json"
            dummy_inside_proc.write_text(json.dumps({"ignore": True}), encoding="utf-8")

            results = batch_process_transcripts(
                input_dir=tmp_dir,
                output_dir=proc_dir,
                overwrite=False,
            )

            # Should find exactly 2 raw files (batch_sample_1.json, batch_sample_2.json), ignoring proc_dir
            self.assertEqual(results["total_found"], 2)
            # Should skip batch_sample_1.json because proc_1 already exists
            self.assertTrue(any(p.name == "batch_sample_1.json" for p in results["skipped"]))
            # Should process batch_sample_2.json
            self.assertTrue(any(p.name == "batch_sample_2.json" for p in results["processed"]))
            # should_be_ignored.json was never processed or counted
            self.assertFalse(any(p.name == "should_be_ignored.json" for p in results["processed"]))
            self.assertFalse(any(p.name == "should_be_ignored.json" for p in results["skipped"]))



if __name__ == "__main__":
    unittest.main()
