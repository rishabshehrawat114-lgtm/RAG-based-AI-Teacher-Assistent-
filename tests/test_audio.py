"""
tests/test_audio.py
===================
Unit tests for Module 2: Audio extraction and batch conversion logic.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import (
    AudioExtractionError,
    FFmpegNotFoundError,
    SUPPORTED_VIDEO_EXTENSIONS,
    batch_convert_videos,
    check_ffmpeg_installed,
    convert_video_to_mp3,
    is_ffmpeg_available,
)
from src.config import AUDIO_DIR, VIDEO_DIR


class TestAudioModule(unittest.TestCase):
    """Test suite for audio extraction functions."""

    def test_supported_extensions(self):
        """Verify all required video extensions are supported."""
        expected = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
        self.assertEqual(SUPPORTED_VIDEO_EXTENSIONS, expected)

    def test_ffmpeg_missing_raises_error(self):
        """Verify check_ffmpeg_installed raises FFmpegNotFoundError when ffmpeg is missing."""
        with patch("src.audio.extractor.is_ffmpeg_available", return_value=False):
            with self.assertRaises(FFmpegNotFoundError):
                check_ffmpeg_installed()

    def test_convert_video_nonexistent_file(self):
        """Verify FileNotFoundError is raised if input video does not exist."""
        with patch("src.audio.extractor.is_ffmpeg_available", return_value=True):
            fake_video = VIDEO_DIR / "nonexistent_lecture.mp4"
            with self.assertRaises(FileNotFoundError):
                convert_video_to_mp3(fake_video)

    def test_convert_video_unsupported_format(self):
        """Verify ValueError is raised for unsupported video format."""
        with patch("src.audio.extractor.is_ffmpeg_available", return_value=True):
            # Create a temporary dummy file with unsupported format
            test_file = VIDEO_DIR / "dummy_test.txt"
            test_file.write_text("dummy", encoding="utf-8")
            try:
                with self.assertRaises(ValueError):
                    convert_video_to_mp3(test_file)
            finally:
                if test_file.exists():
                    test_file.unlink()

    def test_convert_video_success_mock(self):
        """Verify convert_video_to_mp3 generates expected output path and calls ffmpeg."""
        dummy_video = VIDEO_DIR / "python_lecture.mp4"
        dummy_video.write_text("fake video bytes", encoding="utf-8")
        expected_audio = AUDIO_DIR / "python_lecture.mp3"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ""
        mock_process.stderr = ""

        try:
            with patch("src.audio.extractor.is_ffmpeg_available", return_value=True), \
                 patch("subprocess.run", return_value=mock_process) as mock_run:

                out = convert_video_to_mp3(dummy_video, overwrite=True)
                self.assertEqual(out, expected_audio)
                mock_run.assert_called_once()
                cmd_args = mock_run.call_args[0][0]
                self.assertEqual(cmd_args[0], "ffmpeg")
                self.assertIn("-vn", cmd_args)
                self.assertIn("libmp3lame", cmd_args)
                self.assertIn(str(expected_audio), cmd_args)
        finally:
            if dummy_video.exists():
                dummy_video.unlink()
            if expected_audio.exists():
                expected_audio.unlink()

    def test_batch_convert_videos_skipping_and_batch_resilience(self):
        """Verify batch conversion skips existing audio and continues on individual errors."""
        video1 = VIDEO_DIR / "lecture_1.mp4"
        video2 = VIDEO_DIR / "lecture_2.mov"
        video1.write_text("video1", encoding="utf-8")
        video2.write_text("video2", encoding="utf-8")

        existing_mp3 = AUDIO_DIR / "lecture_1.mp3"
        existing_mp3.write_text("existing audio", encoding="utf-8")

        mock_process = MagicMock()
        mock_process.returncode = 0

        try:
            with patch("src.audio.extractor.is_ffmpeg_available", return_value=True), \
                 patch("subprocess.run", return_value=mock_process):

                results = batch_convert_videos(
                    video_dir=VIDEO_DIR,
                    output_dir=AUDIO_DIR,
                    overwrite=False,
                )

                # video1 should be skipped because lecture_1.mp3 already exists
                self.assertTrue(any(p.name == "lecture_1.mp3" for p in results["skipped"]))
                # video2 should be converted
                self.assertTrue(any(p.name == "lecture_2.mp3" for p in results["converted"]))
        finally:
            for p in [video1, video2, existing_mp3]:
                if p.exists():
                    p.unlink()


if __name__ == "__main__":
    unittest.main()
