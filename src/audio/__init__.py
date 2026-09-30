"""
src/audio
=========
Module responsible for audio extraction from educational video files using FFmpeg.

Provides functions to convert single video files or process entire directories in batch,
producing lightweight MP3 files tailored for Whisper speech-to-text transcription.
"""

from src.audio.extractor import (
    AudioExtractionError,
    DEFAULT_AUDIO_BITRATE,
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    FFmpegNotFoundError,
    SUPPORTED_VIDEO_EXTENSIONS,
    batch_convert_videos,
    check_ffmpeg_installed,
    convert_video_to_mp3,
    get_ffmpeg_version,
    is_ffmpeg_available,
)

__all__ = [
    "convert_video_to_mp3",
    "batch_convert_videos",
    "is_ffmpeg_available",
    "check_ffmpeg_installed",
    "get_ffmpeg_version",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_AUDIO_BITRATE",
    "FFmpegNotFoundError",
    "AudioExtractionError",
]
