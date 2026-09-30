"""
tests/test_transcription.py
===========================
Unit tests for Module 3: Whisper-based audio transcription with timestamps.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIO_DIR, DEFAULT_WHISPER_MODEL, SUPPORTED_AUDIO_EXTENSIONS, TRANSCRIPT_DIR
from src.transcription import (
    SUPPORTED_AUDIO_EXTENSIONS,
    TranscriptionError,
    WhisperDependencyError,
    batch_transcribe_audio,
    check_dependencies,
    format_transcription_result,
    get_whisper_model,
    is_ffmpeg_available,
    load_transcript,
    save_transcript,
    transcribe_audio,
)


class TestTranscriptionModule(unittest.TestCase):
    """Test suite for Whisper transcription logic."""

    def test_supported_extensions(self):
        """Verify all required audio formats (.mp3, .wav, .m4a) are supported."""
        required = {".mp3", ".wav", ".m4a"}
        self.assertTrue(required.issubset(SUPPORTED_AUDIO_EXTENSIONS))

    def test_nonexistent_audio_file_raises_error(self):
        """Verify FileNotFoundError is raised when input audio file is missing."""
        nonexistent = AUDIO_DIR / "does_not_exist_lecture.mp3"
        with self.assertRaises(FileNotFoundError):
            transcribe_audio(nonexistent)

    def test_unsupported_audio_format_raises_error(self):
        """Verify ValueError is raised when an unsupported audio format is provided."""
        dummy_file = AUDIO_DIR / "dummy_sample.txt"
        dummy_file.write_text("not an audio file", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                transcribe_audio(dummy_file)
        finally:
            if dummy_file.exists():
                dummy_file.unlink()

    def test_format_transcription_result_structure(self):
        """Verify format_transcription_result produces the exact specified JSON structure."""
        fake_raw = {
            "text": " Today we study machine learning. It is a subset of AI.",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 5.421,
                    "text": " Today we study machine learning.",
                },
                {
                    "id": 1,
                    "seek": 542,
                    "start": 5.421,
                    "end": 12.184,
                    "text": " It is a subset of AI.",
                },
            ],
        }

        formatted = format_transcription_result("machine_learning.mp3", fake_raw)

        # Check top-level keys
        self.assertEqual(formatted["source_file"], "machine_learning.mp3")
        self.assertEqual(formatted["language"], "en")
        self.assertEqual(formatted["duration"], 12.18)
        self.assertEqual(len(formatted["segments"]), 2)

        # Check segment keys and numerical timestamps
        seg0 = formatted["segments"][0]
        self.assertEqual(seg0["id"], 0)
        self.assertEqual(seg0["start"], 0.0)
        self.assertEqual(seg0["end"], 5.42)
        self.assertEqual(seg0["text"], "Today we study machine learning.")

        seg1 = formatted["segments"][1]
        self.assertEqual(seg1["id"], 1)
        self.assertEqual(seg1["start"], 5.42)
        self.assertEqual(seg1["end"], 12.18)
        self.assertEqual(seg1["text"], "It is a subset of AI.")

    def test_save_and_load_transcript(self):
        """Verify saving and reading back a transcript JSON file."""
        sample_data = {
            "source_file": "lecture_sample.mp3",
            "language": "en",
            "duration": 10.5,
            "segments": [
                {"id": 0, "start": 0.0, "end": 10.5, "text": "Sample speech text."}
            ],
        }
        test_out = TRANSCRIPT_DIR / "test_lecture_sample.json"
        try:
            saved_path = save_transcript(sample_data, test_out)
            self.assertEqual(saved_path, test_out)
            self.assertTrue(saved_path.exists())

            loaded = load_transcript(saved_path)
            self.assertEqual(loaded, sample_data)
        finally:
            if test_out.exists():
                test_out.unlink()

    def test_transcribe_audio_mock(self):
        """Verify transcribe_audio invokes Whisper, formats output, and writes JSON."""
        dummy_audio = AUDIO_DIR / "mock_lecture.mp3"
        dummy_audio.write_text("fake audio data", encoding="utf-8")
        target_json = TRANSCRIPT_DIR / "mock_lecture.json"

        mock_whisper_result = {
            "text": " Welcome to python basics.",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 4.5, "text": " Welcome to python basics."}
            ],
        }

        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result

        try:
            with patch("src.transcription.transcriber.check_dependencies"), \
                 patch("src.transcription.transcriber.get_whisper_model", return_value=mock_model):

                res = transcribe_audio(dummy_audio, output_path=target_json, model_name="tiny")

                self.assertEqual(res["source_file"], "mock_lecture.mp3")
                self.assertEqual(res["language"], "en")
                self.assertEqual(res["duration"], 4.5)
                self.assertEqual(res["segments"][0]["text"], "Welcome to python basics.")
                self.assertTrue(target_json.exists())

                # Validate saved JSON content
                with open(target_json, "r", encoding="utf-8") as f:
                    saved_json = json.load(f)
                self.assertEqual(saved_json["source_file"], "mock_lecture.mp3")
        finally:
            if dummy_audio.exists():
                dummy_audio.unlink()
            if target_json.exists():
                target_json.unlink()

    def test_batch_transcribe_skips_existing_and_continues_on_failure(self):
        """Verify batch transcription skips existing transcripts and handles failures gracefully."""
        audio1 = AUDIO_DIR / "batch_test_1.mp3"
        audio2 = AUDIO_DIR / "batch_test_2.wav"
        audio1.write_text("dummy audio 1", encoding="utf-8")
        audio2.write_text("dummy audio 2", encoding="utf-8")

        existing_json = TRANSCRIPT_DIR / "batch_test_1.json"
        existing_json.write_text('{"existing": true}', encoding="utf-8")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Lecture two content.",
            "language": "en",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "Lecture two content."}],
        }

        try:
            with patch("src.transcription.transcriber.check_dependencies"), \
                 patch("src.transcription.transcriber.get_whisper_model", return_value=mock_model):

                results = batch_transcribe_audio(
                    audio_dir=AUDIO_DIR,
                    output_dir=TRANSCRIPT_DIR,
                    overwrite=False,
                )

                # audio1 should be skipped because batch_test_1.json already exists
                self.assertTrue(any(p.name == "batch_test_1.json" for p in results["skipped"]))
                # audio2 should be transcribed
                self.assertTrue(any(p.name == "batch_test_2.json" for p in results["transcribed"]))
        finally:
            for p in [audio1, audio2, existing_json, TRANSCRIPT_DIR / "batch_test_2.json"]:
                if p.exists():
                    p.unlink()


if __name__ == "__main__":
    unittest.main()
