"""
src/transcription
=================
Speech-to-text transcription and transcript post-processing module for the RAG AI Teacher Assistant.

Exposes core functions for:
- Transcribing audio files (.mp3, .wav, .m4a) using OpenAI Whisper (Module 3)
- Validating, cleaning, normalizing, and structuring transcripts into processed JSON (Module 4)
"""

from src.transcription.processor import (
    TranscriptProcessingError,
    batch_process_transcripts,
    clean_segment_text,
    process_transcript,
    process_transcript_file,
    save_processed_transcript,
    validate_segment,
)
from src.transcription.transcriber import (
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

__all__ = [
    # Module 3 Whisper Transcription
    "transcribe_audio",
    "batch_transcribe_audio",
    "save_transcript",
    "load_transcript",
    "get_whisper_model",
    "check_dependencies",
    "is_ffmpeg_available",
    "format_transcription_result",
    "TranscriptionError",
    "WhisperDependencyError",
    "SUPPORTED_AUDIO_EXTENSIONS",
    # Module 4 Transcript Processing & Normalization
    "clean_segment_text",
    "validate_segment",
    "process_transcript",
    "process_transcript_file",
    "batch_process_transcripts",
    "save_processed_transcript",
    "TranscriptProcessingError",
]
